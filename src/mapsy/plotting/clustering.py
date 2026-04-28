import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mapsy.results import ClusterScreeningResult


def plot_cluster_screening(result: ClusterScreeningResult) -> tuple[Figure, Axes, Axes]:
    table = result.table
    best_db = result.best_by_db
    best_sil = result.best_by_silhouette

    fig, ax1 = plt.subplots()
    ax1.scatter(
        table["nclusters"],
        table["silhouette_score"],
        color="b",
        marker="o",
        label="Silhouette Score",
    )
    ax1.plot(best_db["nclusters"], best_db["silhouette_score"], "-", color="b")
    ax1.plot(best_sil["nclusters"], best_sil["silhouette_score"], ":", color="b")
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Silhouette Score", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.scatter(
        table["nclusters"],
        table["db_index"],
        color="r",
        marker="s",
        label="Davies-Bouldin Index",
    )
    ax2.plot(best_db["nclusters"], best_db["db_index"], "-", color="r")
    ax2.plot(best_sil["nclusters"], best_sil["db_index"], ":", color="r")
    ax2.set_ylabel("Davies-Bouldin Index", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    ax1.set_title(
        f"Silhouette Score and Davies-Bouldin Index vs. Number of Clusters ({result.method})"
    )
    ax1.grid(True)
    return fig, ax1, ax2
