import numpy as np

from openTSNE.tsne import TSNEEmbedding


def pBIC(embedding: TSNEEmbedding) -> float:
    if not hasattr(embedding.affinities, "perplexity"):
        raise TypeError("The embedding affinity matrix has no attribute `perplexity`")
    n_samples = embedding.shape[0]

    return 2 * embedding.kl_divergence_ + np.log(n_samples) * \
        embedding.affinities.perplexity / n_samples
