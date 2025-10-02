import streamlit as st
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="EV User Clustering Dashboard",
    layout="wide"
)

st.title("🚗 EV User Analysis & Clustering Dashboard")
st.markdown("""
Explore **EDA first**, then select **features to run K-Means clustering** interactively.
""")

# --------------------------
# Load data
# --------------------------
@st.cache_data
def load_data(path="outputs/users_with_clusters.csv"):
    # load original dataset before clustering
    df = pl.read_csv(path,schema_overrides={
        "TOKEN_ID": pl.Utf8})
    return df.to_pandas()

df = load_data()

if df.empty:
    st.error("No data found. Please ensure preprocessing script saved a CSV in outputs/.")
    st.stop()

# Remove any existing cluster column if present
if "cluster" in df.columns:
    df = df.drop(columns=["cluster"])

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.header("Controls")

# Variable to plot for EDA
eda_var = st.sidebar.selectbox(
    "Select variable for EDA:",
    options=["REVENUE_PER_SESSION", "AVG_SESSION_MINUTES","AC_SHARE","AVG_DAYS_BETWEEN_SESSIONS","CHARGED_AMOUNT"]
)

# --------------------------
# EDA Section
# --------------------------
st.subheader("📊 Exploratory Data Analysis")

st.write("**Dataset preview:**")
st.dataframe(df.head())

st.markdown("### Distribution Overview")

# Histogram
fig_hist, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df[eda_var], bins=30, kde=True, color="skyblue", ax=ax)
ax.set_title(f"Histogram of {eda_var}")
ax.set_xlabel(eda_var)
st.pyplot(fig_hist)

# Boxplot
fig_box, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(y=df[eda_var], color="lightgreen", ax=ax)
ax.set_title(f"Boxplot of {eda_var}")
ax.set_ylabel(eda_var)
st.pyplot(fig_box)

# Boxplot by churn risk or cluster
st.markdown("### Compare by churn risk")
fig_box_group, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x=df["churn_risk"], y=df[eda_var], palette="Set2", ax=ax)
ax.set_title(f"{eda_var} by Churn Risk")
ax.set_xlabel("Churn Risk")
st.pyplot(fig_box_group)

st.markdown("---")

# --------------------------
# Feature selection for clustering
# --------------------------
st.subheader("🔑 Feature Selection for Clustering")

default_feats = [
    "AVG_DAYS_BETWEEN_SESSIONS",
    "DAYS_FROM_LAST_SESSION",
    "CHARGED_AMOUNT",
    "AC_SHARE",
    "AVG_SESSION_MINUTES"
]

available_features = [c for c in default_feats if c in df.columns]

selected_features = st.multiselect(
    "Select features for clustering:",
    options=available_features,
    default=available_features
)

n_clusters = st.slider("Select number of clusters (k):", min_value=2, max_value=10, value=4)

st.info("Select features and press **Run Clustering** to cluster.")

if st.button("▶️ Run Clustering"):
    if len(selected_features) < 2:
        st.error("Select at least two features for clustering.")
    else:
        # --------------------------
        # Run KMeans
        # --------------------------
        X = df[selected_features].to_numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        df["cluster"] = labels

        st.success(f"K-Means completed with **{n_clusters} clusters**.")

        # Cluster summary
        st.subheader("📈 Cluster Summary")
        cluster_summary = (
            df.groupby("cluster")
              .agg({
                  "REVENUE_PER_SESSION": "mean",
                  "AVG_SESSION_MINUTES": "mean",
                  "CHARGED_AMOUNT": "mean",
                  "AC_SHARE": "mean",
                  "churn_risk": "mean"
              })
              .reset_index()
        )
        st.dataframe(cluster_summary)

        # 2-D scatter plots for first two selected features
        if len(selected_features) >= 2:
            st.subheader("📌 Scatter Plot (first two selected features)")
            fig_scatter, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                df[selected_features[0]],
                df[selected_features[1]],
                c=df["cluster"],
                cmap="viridis",
                s=80,
                alpha=0.7,
                edgecolors="k"
            )
            ax.set_xlabel(selected_features[0])
            ax.set_ylabel(selected_features[1])
            ax.set_title(f"K-Means Clusters ({selected_features[0]} vs {selected_features[1]})")
            plt.colorbar(scatter, ax=ax, label="Cluster")
            st.pyplot(fig_scatter)

        # Download clustered data
        csv_data = df.to_csv(index=False)
        st.download_button(
            "💾 Download Clustered Data as CSV",
            data=csv_data,
            file_name="clustered_users.csv",
            mime="text/csv"
        )
