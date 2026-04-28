import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mapsy.results import PCAAnalysisResult


def plot_pca_scree(result: PCAAnalysisResult) -> tuple[Figure, Axes, Axes]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    xaxis = range(1, len(result.cumulative_explained_variance) + 1)
    ax1.plot(xaxis, result.cumulative_explained_variance, marker="o")
    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("Cumulative Explained Variance")
    ax1.grid(True)
    ax1.set_title("Optimal Number of Components in PCA")
    ax1.axhline(y=0.98, color="r", linestyle="--")

    ax2.plot(xaxis, result.explained_variance_ratio, marker="o")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_title("Scree Plot")
    ax2.grid(True)
    return fig, ax1, ax2
