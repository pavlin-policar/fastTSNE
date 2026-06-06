import distutils
import os
import platform
import subprocess
import sys
import tempfile
import warnings
from distutils import ccompiler
from distutils.errors import CompileError, LinkError
from distutils.sysconfig import customize_compiler
from os.path import join, isdir

import setuptools
from Cython.Distutils.build_ext import new_build_ext as build_ext
from setuptools import setup, Extension


def optimize_png_bytes(data, colors=256):
    """Palette-quantize a PNG (given as bytes) for the web.

    Matplotlib renders scatter plots as high-entropy RGBA PNGs that compress
    poorly; quantizing them to a 256-colour palette is visually
    indistinguishable but roughly 4x smaller. Falls back to the original bytes
    if Pillow is unavailable or the image cannot be processed, so the
    documentation build never fails just because images could not be shrunk.
    """
    import io

    try:
        from PIL import Image
    except ImportError:
        warnings.warn(
            "Pillow is not installed; documentation PNGs will not be "
            "optimized. Install it (`pip install pillow`) to shrink them.",
        )
        return data

    try:
        with Image.open(io.BytesIO(data)) as im:
            quantized = im.convert("RGB").quantize(
                colors=colors,
                method=Image.Quantize.FASTOCTREE,
                dither=Image.Dither.NONE,
            )
        buffer = io.BytesIO()
        quantized.save(buffer, "PNG", optimize=True)
        return buffer.getvalue()
    except Exception as e:  # pragma: no cover - defensive
        warnings.warn(f"Could not optimize a PNG ({e}); keeping the original.")
        return data


class ConvertNotebooksToDocs(distutils.cmd.Command):
    description = "Convert the example notebooks to reStructuredText that will" \
                  "be available in the documentation."

    user_options = []

    # Notebooks rendered as standalone example pages in the documentation.
    example_notebooks = [
        join("examples", "01_simple_usage.ipynb"),
        join("examples", "02_advanced_usage.ipynb"),
        join("examples", "03_preserving_global_structure.ipynb"),
        join("examples", "04_large_data_sets.ipynb"),
    ]
    # Notebooks that supply figures for a hand-written User Guide page rather
    # than becoming a standalone example page. The guide (e.g.
    # degrees_of_freedom.rst) is authored in rst because it relies on Sphinx
    # features that do not survive a markdown->rst conversion -- admonitions,
    # `:meth:`/`:attr:` cross-references, ref labels -- and is mostly conceptual
    # prose with illustrative (non-runnable) snippets. The notebook contributes
    # only the runnable worked example, so we keep its rendered images and
    # discard the generated .rst.
    figure_only_notebooks = [
        join("examples", "07_degrees_of_freedom.ipynb"),
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import nbconvert

        exporter = nbconvert.RSTExporter()
        writer = nbconvert.writers.FilesWriter()
        target_dir = join("docs", "source", "examples")

        for fname in self.example_notebooks + self.figure_only_notebooks:
            self.announce(f"Converting {fname}...")
            nb_name = os.path.splitext(os.path.basename(fname))[0]
            body, resources = exporter.from_file(fname)

            # Optimize every rendered PNG for the web (see optimize_png_bytes).
            outputs = resources.get("outputs", {})
            for out_name, out_data in list(outputs.items()):
                if out_name.lower().endswith(".png"):
                    outputs[out_name] = optimize_png_bytes(out_data)

            writer.build_directory = join(target_dir, nb_name)
            writer.write(body, resources, nb_name)

            if fname in self.figure_only_notebooks:
                # Drop the generated rst; the figures are referenced from the
                # hand-written guide page instead.
                rst_path = join(writer.build_directory, nb_name + ".rst")
                if os.path.exists(rst_path):
                    os.remove(rst_path)


def get_numpy_include():
    import numpy
    return numpy.get_include()


def get_include_dirs():
    """Get include dirs for the compiler."""
    return (
        os.path.join(sys.prefix, "include"),
        os.path.join(sys.prefix, "Library", "include"),
    )


def get_library_dirs():
    """Get library dirs for the compiler."""
    return (
        os.path.join(sys.prefix, "lib"),
        os.path.join(sys.prefix, "Library", "lib"),
    )


def has_c_library(library, extension=".c", include_dirs=(), library_dirs=(),
                  libraries=(), extra_preargs=(), extra_postargs=()):
    """Check whether a C/C++ library is available on the system to the compiler.

    Parameters
    ----------
    library: str
        The library we want to check for e.g. if we are interested in FFTW3, we
        want to check for `fftw3.h`, so this parameter will be `fftw3`.
    extension: str
        If we want to check for a C library, the extension is `.c`, for C++
        `.cc`, `.cpp` or `.cxx` are accepted.
    include_dirs: Iterable[str]
        Directories to search for headers in addition to the defaults.
    library_dirs: Iterable[str]
        Directories to search for the library in addition to the defaults.
    libraries: Iterable[str]
        Libraries to link the test program against, so that a header found
        without a matching runtime is reported as unavailable.
    extra_preargs, extra_postargs: Iterable[str]
        Extra flags for both the compile and the link step.

    Returns
    -------
    bool
        Whether or not the library is available.

    """
    with tempfile.TemporaryDirectory(dir=".") as directory:
        name = join(directory, "%s%s" % (library, extension))
        with open(name, "w") as f:
            f.write("#include <%s.h>\n" % library)
            f.write("int main() {}\n")

        # Get a compiler instance
        compiler = ccompiler.new_compiler()
        # Configure compiler to do all the platform specific things
        customize_compiler(compiler)
        # Add conda include dirs
        for inc_dir in tuple(get_include_dirs()) + tuple(include_dirs):
            compiler.add_include_dir(inc_dir)
        for lib_dir in library_dirs:
            compiler.add_library_dir(lib_dir)
        assert isinstance(compiler, ccompiler.CCompiler)

        try:
            # Try to compile the file using the C compiler
            objects = compiler.compile(
                [name],
                extra_preargs=list(extra_preargs),
                extra_postargs=list(extra_postargs),
            )
            compiler.link_executable(
                objects,
                name,
                libraries=list(libraries),
                extra_preargs=list(extra_preargs),
                extra_postargs=list(extra_postargs),
            )
            return True
        except (CompileError, LinkError):
            return False


def has_compiler_flag(flag, compiler_type):
    """Check whether the compiler both accepts and acts on a flag.

    Compilers accept unknown `-m` and `-f` flags with nothing more than a
    warning, so a plain compile is not evidence that a flag did anything.
    `-Werror` promotes that warning to an error, which is what makes the answer
    meaningful.
    """
    if compiler_type != "unix":
        return False

    with tempfile.TemporaryDirectory(dir=".") as directory:
        name = join(directory, "flagcheck.c")
        with open(name, "w") as f:
            f.write("int main() {}\n")

        compiler = ccompiler.new_compiler()
        customize_compiler(compiler)
        try:
            compiler.compile([name], extra_postargs=["-Werror", flag])
            return True
        except (CompileError, LinkError):
            return False


def get_openmp_prefixes():
    """Installation prefixes to search for an OpenMP runtime.

    Apple's compiler ships without one, and every way of installing `libomp`
    on macOS puts it somewhere the compiler does not look by default. Without
    this search the build falls back to a single-threaded extension on a
    machine that is perfectly capable of running the parallel one.
    """
    if platform.system() != "Darwin":
        return []

    prefixes = []
    try:
        found = subprocess.run(
            ["brew", "--prefix", "libomp"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30,
        )
        if found.returncode == 0:
            prefixes.append(found.stdout.decode().strip())
    except (OSError, subprocess.SubprocessError):
        pass

    prefixes += [
        "/opt/homebrew/opt/libomp",  # Homebrew, Apple silicon
        "/usr/local/opt/libomp",  # Homebrew, Intel
        "/opt/local",  # MacPorts
    ]
    return [p for p in prefixes if p and isdir(join(p, "include"))]


class CythonBuildExt(build_ext):
    def build_extensions(self):
        extra_compile_args = []
        extra_link_args = []

        # Optimization compiler/linker flags are added appropriately
        compiler = self.compiler.compiler_type
        if compiler == "unix":
            extra_compile_args += ["-O3"]
        elif compiler == "msvc":
            extra_compile_args += ["/Ox", "/fp:precise"]  # can't use fp:fast because we use inf

        if compiler == "unix":
            # https://stackoverflow.com/questions/22931147/stdisinf-does-not-work-with-ffast-math-how-to-check-for-infinity
            extra_compile_args += [
                "-ffast-math",
                "-fno-finite-math-only",  # we use infinity
                "-fno-associative-math",
            ]

        # Annoy specific flags
        annoy_ext = None
        for extension in extensions:
            if "annoy.annoylib" in extension.name:
                annoy_ext = extension
        assert annoy_ext is not None, "Annoy extension not found!"

        if compiler == "unix":
            annoy_ext.extra_compile_args += ["-std=c++14"]
            annoy_ext.extra_compile_args += ["-DANNOYLIB_MULTITHREADED_BUILD"]
        elif compiler == "msvc":
            annoy_ext.extra_compile_args += ["/std:c++14"]

        # Set minimum deployment version for MacOS
        if compiler == "unix" and platform.system() == "Darwin":
            macos_deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "10.12")
            extra_compile_args += [f"-mmacosx-version-min={macos_deployment_target}"]
            extra_link_args += ["-stdlib=libc++", f"-mmacosx-version-min={macos_deployment_target}"]

        # We don't want the compiler to optimize for system architecture if
        # we're building packages to be distributed by conda-forge, but if the
        # package is being built locally, this is desired
        if not ("AZURE_BUILD" in os.environ or "CONDA_BUILD" in os.environ):
            machine = platform.machine().lower()
            if machine in ("x86_64", "amd64", "i386", "i686"):
                tuning = ["-march=native", "-mtune=native"]
            elif machine in ("arm64", "aarch64", "ppc64le", "ppc64"):
                tuning = ["-mcpu=native", "-mtune=native"]
            else:
                tuning = []

            for flag in tuning:
                if has_compiler_flag(flag, compiler):
                    extra_compile_args += [flag]
                    break

        # We will disable openmp flags if the compiler doesn't support it. This
        # is only really an issue with OSX clang
        omp_compile_args, omp_link_args = [], []
        if compiler == "msvc":
            omp_compile_args, omp_link_args = ["/openmp"], ["/openmp"]
            found_openmp = has_c_library("omp")
        elif platform.system() == "Darwin":
            # Apple's clang needs to be told to look at the OpenMP pragmas, and
            # the runtime it links against is not one it ships with
            omp_compile_args = ["-Xpreprocessor", "-fopenmp"]
            omp_link_args = ["-lomp"]
            found_openmp = has_c_library(
                "omp", libraries=["omp"], extra_postargs=omp_compile_args
            )
            for prefix in [] if found_openmp else get_openmp_prefixes():
                include_dir, library_dir = join(prefix, "include"), join(prefix, "lib")
                if has_c_library(
                    "omp",
                    include_dirs=[include_dir],
                    library_dirs=[library_dir],
                    libraries=["omp"],
                    extra_postargs=omp_compile_args,
                ):
                    print("Found openmp in %s" % prefix)
                    omp_compile_args += ["-I" + include_dir]
                    omp_link_args += ["-L" + library_dir, "-Wl,-rpath," + library_dir]
                    found_openmp = True
                    break
        else:
            omp_compile_args, omp_link_args = ["-fopenmp"], ["-fopenmp"]
            found_openmp = has_c_library(
                "omp", libraries=["gomp"], extra_postargs=omp_compile_args
            ) or has_c_library("omp", extra_postargs=omp_compile_args)

        if found_openmp:
            print("Found openmp. Compiling with openmp flags...")
            extra_compile_args += omp_compile_args
            extra_link_args += omp_link_args
        else:
            warnings.warn(
                "You appear to be using a compiler which does not support "
                "openMP, meaning that the library will not be able to run on "
                "multiple cores. Please install/enable openMP to use multiple "
                "cores."
            )

        for extension in self.extensions:
            extension.extra_compile_args += extra_compile_args
            extension.extra_link_args += extra_link_args

        # Add numpy and system include directories
        for extension in self.extensions:
            extension.include_dirs.extend(get_include_dirs())
            extension.include_dirs.append(get_numpy_include())

        # Add numpy and system include directories
        for extension in self.extensions:
            extension.library_dirs.extend(get_library_dirs())

        super().build_extensions()


# Prepare the Annoy extension
# Adapted from annoy setup.py
# Various platform-dependent extras
extra_compile_args = []
extra_link_args = []

annoy_path = "openTSNE/dependencies/annoy/"
annoy = Extension(
    "openTSNE.dependencies.annoy.annoylib",
    [annoy_path + "annoymodule.cc"],
    depends=[annoy_path + f for f in ["annoylib.h", "kissrandom.h", "mman.h"]],
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

# Other extensions
extensions = [
    Extension("openTSNE.quad_tree", ["openTSNE/quad_tree.pyx"], language="c++"),
    Extension("openTSNE._tsne", ["openTSNE/_tsne.pyx"], language="c++"),
    Extension("openTSNE.kl_divergence", ["openTSNE/kl_divergence.pyx"], language="c++"),
    annoy,
]


# Check if we have access to FFTW3 and if so, use that implementation
if has_c_library("fftw3"):
    print("FFTW3 header files found. Using FFTW implementation of FFT.")
    extension_ = Extension(
        "openTSNE._matrix_mul.matrix_mul",
        ["openTSNE/_matrix_mul/matrix_mul_fftw3.pyx"],
        libraries=["fftw3"],
        language="c++",
    )
    extensions.append(extension_)
else:
    print("FFTW3 header files not found. Using numpy implementation of FFT.")
    extension_ = Extension(
        "openTSNE._matrix_mul.matrix_mul",
        ["openTSNE/_matrix_mul/matrix_mul_numpy.pyx"],
        language="c++",
    )
    extensions.append(extension_)


def readme():
    with open("README.rst", encoding="utf-8") as f:
        return f.read()


# Read in version
__version__: str = ""  # This is overridden by the next line
exec(open(os.path.join("openTSNE", "version.py")).read())

setup(
    name="openTSNE",
    description="Extensible, parallel implementations of t-SNE",
    long_description=readme(),
    version=__version__,
    license="BSD-3-Clause",

    author="Pavlin Poličar",
    author_email="pavlin.g.p@gmail.com",
    url="https://github.com/pavlin-policar/openTSNE",
    project_urls={
        "Documentation": "https://opentsne.readthedocs.io/",
        "Source": "https://github.com/pavlin-policar/openTSNE",
        "Issue Tracker": "https://github.com/pavlin-policar/openTSNE/issues",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "License :: OSI Approved",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    packages=setuptools.find_packages(include=["openTSNE", "openTSNE.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.16.6",
        "scikit-learn>=0.20",
        "scipy>=1.4",
    ],
    extras_require={
        "hnsw": "hnswlib>=0.5.0",
        "pynndescent": "pynndescent~=0.5.0",
    },
    ext_modules=extensions,
    cmdclass={"build_ext": CythonBuildExt, "convert_notebooks": ConvertNotebooksToDocs},
)
