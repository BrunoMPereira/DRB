import polars as pl
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def cluster_by_kmeans(dataframe, features, n_clusters=4):
    features_to_cluster = dataframe.select(features)

    # Convert to numpy
    X = features_to_cluster.to_numpy()

    # --- Run KMeans ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)


    # --- Add cluster labels back to Polars DataFrame ---
    new_dataframe = dataframe.with_columns(pl.Series("cluster", labels))

    return new_dataframe


def plot_kmeans(data, features, colors=None):
    if colors == None:
        what_to_color = data["cluster"]
    else:
        what_to_color = colors
    plt.figure(figsize=(8,6))
    plt.scatter(
        data[features[0]],
        data[features[1]],
        c = what_to_color,
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="k"
    )

    # Add red dotted line y=x
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    lims = [min(x_min, y_min), max(x_max, y_max)]
    plt.plot(lims, lims, "r--", linewidth=2, label="y = x")

    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title("KMeans Clustering of Users")
    if colors == None:
        plt.colorbar(label="Cluster")
        plt.legend()
    else:
        plt.legend(handles=[
            mpatches.Patch(color="blue", label="Churn risk"),
            mpatches.Patch(color="grey", label="Not at risk"),
            mpatches.Patch(color="red", label="y = x", linestyle="--")
        ])
    plt.show()


if __name__ == "__main__":
    data = pl.read_csv("users_drb.csv")
    # filter only interesting data, no outliers, people that have at least one charging from the last month
    data = data.filter(pl.col("DAYS_FROM_LAST_SESSION") < 30)
    df = data.with_columns(
        (data["DAYS_FROM_LAST_SESSION"] > 2 * data["AVG_DAYS_BETWEEN_SESSIONS"]).alias("churn_risk")
    )
    colors = df["churn_risk"].map_elements(lambda x: "blue" if x else "grey", return_dtype=pl.String).to_list()
    feats = ['AVG_DAYS_BETWEEN_SESSIONS','DAYS_FROM_LAST_SESSION']

    data_with_clusters = cluster_by_kmeans(df,feats)
    
    plot_kmeans(data_with_clusters,feats,colors)
    plot_kmeans(data_with_clusters,feats)