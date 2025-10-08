import os
import polars as pl
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -------------------------
# Outlier Removal using IQR
# -------------------------
def remove_outliers_iqr(df, columns, k=1.5):
    """
    Remove rows outside [Q1 - k*IQR, Q3 + k*IQR] for the given columns.
    k=1.5 is standard, k=3 for more lenient filtering.
    """
    filtered_df = df.clone()
    for col in columns:
        # Compute Q1, Q3, and IQR
        q1 = filtered_df[col].quantile(0.25)
        q3 = filtered_df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - k * iqr
        upper = q3 + k * iqr

        # Filter out rows outside the IQR bounds
        filtered_df = filtered_df.filter(
            (pl.col(col) >= lower) & (pl.col(col) <= upper)
        )

        print(
            f"Filtering {col}: "
            f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
            f"Lower={lower:.2f}, Upper={upper:.2f}"
        )
        print(f"✅ Remaining rows after filtering {col}: {filtered_df.shape[0]}")

    return filtered_df


# -------------------------
# Boxplots for outlier detection
# -------------------------
def plot_boxplots(df, columns, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    for col in columns:
        plt.figure(figsize=(6, 5))
        sns.boxplot(
            y=df[col].to_numpy(),
            color="skyblue"
        )
        plt.title(f"Boxplot of {col}")
        plt.ylabel(col)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        save_path = os.path.join(save_dir, f"boxplot_{col}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved boxplot: {save_path}")


# -------------------------
#  KMeans Clustering
# -------------------------
def cluster_by_kmeans(dataframe, features, n_clusters=4):
    X = dataframe.select(features).to_numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return dataframe.with_columns(pl.Series("cluster", labels))


# -------------------------
#  Elbow Method
# -------------------------
def plot_elbow_method(dataframe, features, k_min=1, k_max=10, save_path=None):
    X = dataframe.select(features).to_numpy()
    inertias = []
    k_range = range(k_min, k_max + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), inertias, "o-", linewidth=2)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (sum of squared distances)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# -------------------------
#  2-D Scatter Plot
# -------------------------
def plot_kmeans(data, features, colors=None, save_path=None):
    if colors is None:
        what_to_color = data["cluster"].to_numpy()
    else:
        what_to_color = colors

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        data[features[0]].to_numpy(),
        data[features[1]].to_numpy(),
        c=what_to_color,
        cmap="viridis" if colors is None else None,
        s=100,
        alpha=0.7,
        edgecolors="k"
    )

    # Add red dotted y = x line
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    lims = [min(x_min, y_min), max(x_max, y_max)]
    ax.plot(lims, lims, "r--", linewidth=2)

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title(f"KMeans Clustering: {features[0]} vs {features[1]}")

    if colors is None:
        plt.colorbar(scatter, label="Cluster", ax=ax)
    else:
        line_legend = mlines.Line2D([], [], color="red", linestyle="--", label="y = x")
        ax.legend(handles=[
            mpatches.Patch(color="blue", label="Churn risk"),
            mpatches.Patch(color="grey", label="Not at risk"),
            line_legend
        ])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# -------------------------
#  3-D Scatter Plot
# -------------------------
def plot_3d_kmeans(data, features, save_path=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        data[features[0]].to_numpy(),
        data[features[1]].to_numpy(),
        data[features[2]].to_numpy(),
        c=data["cluster"].to_numpy(),
        cmap="viridis",
        s=80,
        alpha=0.7,
        edgecolors="k"
    )

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    ax.set_title("3D KMeans Clusters")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# -------------------------
#  Churn & Revenue Insights
# -------------------------
def churn_revenue_summary(df, features):
    """
    Summarize churn risk and revenue by cluster
    """
    summary = (
        df.group_by("cluster")
          .agg([
              pl.len().alias("cluster_size"),
              pl.col("churn_risk").mean().alias("churn_rate"),
              pl.col("CHARGED_AMOUNT").mean().alias("mean_charged_amount"),
              pl.col("REVENUE_PER_SESSION").mean().alias("mean_revenue_per_session"),
              pl.col(features[3]).mean().alias("mean_ac_share"),
              pl.col(features[4]).mean().alias("mean_avg_session_minutes")
          ])
          .sort("cluster")
    )

    print("\n=== Churn & Revenue Insights by Cluster ===")
    print(summary)
    return summary


# -------------------------
#  Main Script
# -------------------------
if __name__ == "__main__":
    # Read CSVs
    data = pl.read_csv("users_drb.csv")
    data_amounts = pl.read_csv(
        "users_charged_amount.csv",
        schema_overrides={
            "TOKEN_ID": pl.Utf8,
            "CHARGED_AMOUNT": pl.Float64,
            "DISCOUNTED_AMOUNT": pl.Float64,
            "DIF_CHARGING_STATIONS": pl.Float64,
        },
    )
    data_acdc = pl.read_csv(
        "ac_dc_sessions.csv",
        schema_overrides={
            "TOKEN_ID": pl.Utf8,
            "AVG_AC_MINUTES": pl.Float64,
            "AVG_DC_MINUTES": pl.Float64,
            "SESSIONS_AC": pl.Int64,
            "SESSIONS_DC": pl.Int64,
            "SESSIONS_UNKNOWN": pl.Int64
        },
    )

    # Filter data
    data = data.filter(pl.col("DAYS_FROM_LAST_SESSION") < 30)

    # Merge and feature engineering
    df_final = (
        data
        .join(data_amounts, how="inner", on="TOKEN_ID")
        .join(data_acdc, how="inner", on="TOKEN_ID")
        .with_columns([
            (pl.col("SESSIONS_AC") + pl.col("SESSIONS_DC")).alias("TOTAL_SESSIONS"),
            (pl.col("SESSIONS_AC") / (pl.col("SESSIONS_AC") + pl.col("SESSIONS_DC"))).alias("AC_SHARE"),
            (
                (
                    pl.col("SESSIONS_AC") * pl.col("AVG_AC_MINUTES")
                    + pl.col("SESSIONS_DC") * pl.col("AVG_DC_MINUTES")
                ) / (pl.col("SESSIONS_AC") + pl.col("SESSIONS_DC"))
            ).alias("AVG_SESSION_MINUTES"),
            (pl.col("CHARGED_AMOUNT") / (pl.col("SESSIONS_AC") + pl.col("SESSIONS_DC"))).alias("REVENUE_PER_SESSION"),
        ])
        .drop_nulls()
    )

    # Flag churn risk
    df_final = df_final.with_columns(
        (pl.col("DAYS_FROM_LAST_SESSION") > 2 * pl.col("AVG_DAYS_BETWEEN_SESSIONS"))
        .alias("churn_risk")
    )



    # --- Boxplots to check outliers ---
    plot_boxplots(df_final, ["REVENUE_PER_SESSION", "AVG_SESSION_MINUTES"])

    # ---- Remove outliers ----
    df_final = remove_outliers_iqr(df_final, ["REVENUE_PER_SESSION", "AVG_SESSION_MINUTES"], k=1.5)

    plot_boxplots(df_final, ["REVENUE_PER_SESSION", "AVG_SESSION_MINUTES"])

    colors = df_final["churn_risk"].map_elements(
        lambda x: "blue" if x else "grey", return_dtype=pl.String
    ).to_list()


    # Features for clustering
    feats = [
        "AVG_DAYS_BETWEEN_SESSIONS",
        "DAYS_FROM_LAST_SESSION",
        "CHARGED_AMOUNT",
        "AC_SHARE",
        "AVG_SESSION_MINUTES",
        "DIF_CHARGING_STATIONS"
    ]

    # -------------------------
    # Elbow method first
    # -------------------------
    os.makedirs("plots", exist_ok=True)
    plot_elbow_method(df_final, feats, k_min=1, k_max=10,
                      save_path="plots/elbow_method.png")

    # -------------------------
    # Cluster
    # -------------------------
    data_with_clusters = cluster_by_kmeans(df_final, feats, n_clusters=4)

    # Pairwise plots
    pairs = [
        (feats[0], feats[1]),
        (feats[0], feats[2]),
        (feats[0], feats[3]),
        (feats[0], feats[4]),
        (feats[2], feats[4])
    ]

    for f1, f2 in pairs:
        path1 = f"plots/{f1}_vs_{f2}_churnrisk.png"
        path2 = f"plots/{f1}_vs_{f2}_clusters.png"

        # With churn risk colors
        plot_kmeans(data_with_clusters, [f1, f2], colors=colors, save_path=path1)

        # With cluster colors
        plot_kmeans(data_with_clusters, [f1, f2], save_path=path2)

    # 3D plot
    # plot_3d_kmeans(data_with_clusters, feats, save_path="plots/3D_clusters.png")

    # Show all figures
    plt.show()

    # -------------------------
    # Churn & Revenue Insights
    # -------------------------

    churn_revenue_summary(data_with_clusters, feats)

    os.makedirs("outputs", exist_ok=True)
    data_with_clusters.write_csv("outputs/users_with_clusters.csv")
    print("\n✅ Saved enriched dataframe to outputs/users_with_clusters.csv")
