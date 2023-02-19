"""
===============================
Compare under-sampling samplers
===============================

The following example attends to make a qualitative comparison between the
different under-sampling algorithms available in the imbalanced-learn package.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

# %%
print(__doc__)

import seaborn as sns

#sns.set_context("poster")

# %% [markdown]
# The following function will be used to create toy dataset. It uses the
# :func:`~sklearn.datasets.make_classification` from scikit-learn but fixing
# some parameters.


# %%
from sklearn.datasets import make_classification


def create_dataset(
    n_samples=1000,
    weights=(0.01, 0.01, 0.98),
    n_classes=3,
    class_sep=0.8,
    n_clusters=1,
):
    return make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters,
        weights=list(weights),
        class_sep=class_sep,
        random_state=0,
    )


# %% [markdown]
# The following function will be used to plot the sample space after resampling
# to illustrate the specificities of an algorithm.


# %%
def plot_resampling(X, y, sampler, ax, title=None):
    X_res, y_res = sampler.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor="k")
    if title is None:
        title = f"Resampling with {sampler.__class__.__name__}"
    ax.set_title(title)
    sns.despine(ax=ax, offset=10)


# %% [markdown]
# The following function will be used to plot the decision function of a
# classifier given some data.


# %%
import numpy as np


def plot_decision_function(X, y, clf, ax, title=None):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor="k")
    if title is not None:
        ax.set_title(title)


# %%
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()


# %% [markdown]
# Prototype generation: under-sampling by generating new samples
# --------------------------------------------------------------
#
# :class:`~imblearn.under_sampling.ClusterCentroids` under-samples by replacing
# the original samples by the centroids of the cluster found.

# %%
import matplotlib.pyplot as plt
from imblearn import FunctionSampler
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import ClusterCentroids

X, y = create_dataset(n_samples=400, weights=(0.05, 0.15, 0.8), class_sep=0.8)

samplers = {
    FunctionSampler(),  # identity resampler
    ClusterCentroids(random_state=0),
}

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
for ax, sampler in zip(axs, samplers):
    model = make_pipeline(sampler, clf).fit(X, y)
    plot_decision_function(
        X, y, model, ax[0], title=f"Decision function with {sampler.__class__.__name__}"
    )
    plot_resampling(X, y, sampler, ax[1])

fig.tight_layout()

