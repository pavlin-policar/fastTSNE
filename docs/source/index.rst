openTSNE: Extensible, parallel implementations of t-SNE
=======================================================

openTSNE is a modular Python implementation of t-Distributed Stochasitc Neighbor Embedding (t-SNE) [1]_, a popular dimensionality-reduction algorithm for visualizing high-dimensional data sets. openTSNE incorporates the latest improvements to the t-SNE algorithm, including the ability to add new data points to existing embeddings [2]_, massive speed improvements [3]_ [4]_, enabling t-SNE to scale to millions of data points and various tricks to improve global alignment of the resulting visualizations [5]_.

.. figure:: images/macosko_2015.png
   :width: 500px
   :align: center
   :alt: Macosko 2015 mouse retina t-SNE embedding

   A visualization of 44,808 single cell transcriptomes obtained from the mouse retina [6]_ embedded using the multiscale kernel trick to better preserve the global aligment of the clusters.

.. toctree::
    :maxdepth: 2
    :caption: User Guide

    installation
    examples/index
    tsne_algorithm
    parameters
    affinities
    degrees_of_freedom
    benchmarks


.. toctree::
    :maxdepth: 2
    :caption: API Reference

    api/index

References
----------

.. [1] Van der Maaten, Laurens, and Hinton, Geoffrey. `“Visualizing data using t-SNE” <http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf>`__, Journal of Machine Learning Research (2008).

.. [2] Poličar, Pavlin G., Martin Stražar, and Blaž Zupan. `“Embedding to Reference t-SNE Space Addresses Batch Effects in Single-Cell Classification” <https://link.springer.com/article/10.1007/s10994-021-06043-1>`__, Machine Learning (2021).

.. [3] Van der Maaten, Laurens. `“Accelerating t-SNE using tree-based algorithms” <http://www.jmlr.org/papers/volume15/vandermaaten14a/vandermaaten14a.pdf>`__, Journal of Machine Learning Research (2014).

.. [4] Linderman, George C., et al. `"Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data" <https://www.nature.com/articles/s41592-018-0308-4>`__, Nature Methods (2019).

.. [5] Kobak, Dmitry, and Berens, Philipp. `“The art of using t-SNE for single-cell transcriptomics” <https://www.nature.com/articles/s41467-019-13056-x>`__, Nature Communications (2019).

.. [6] Macosko, Evan Z., et al. \ `“Highly parallel genome-wide expression profiling of individual cells using nanoliter droplets” <https://www.sciencedirect.com/science/article/pii/S0092867415005498>`__, Cell (2015).
