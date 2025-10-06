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
    df = pl.read_csv(path, schema_overrides={"TOKEN_ID": pl.Utf8})
    return df.to_pandas()

df = load_data()

if df.empty:
    st.error("No data found. Please ensure preprocessing script saved a CSV in outputs/.")
    st.stop()

if "cluster" in df.columns:
    df = df.drop(columns=["cluster"])

# --------------------------
# Human-readable names mapping
# --------------------------
var_labels = {
    "REVENUE_PER_SESSION": "Average Session Value",
    "AVG_SESSION_MINUTES": "Average Session Duration (min)",
    "AC_SHARE": "Sessions in AC (%)",
    "AVG_DAYS_BETWEEN_SESSIONS": "Average Days between Sessions",
    "CHARGED_AMOUNT": "Total Value",
    "DAYS_FROM_LAST_SESSION": "Days from Last Session"
}

# Reverse mapping (friendly -> original column)
inv_labels = {v: k for k, v in var_labels.items()}

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.header("Controls")

eda_var_friendly = st.sidebar.selectbox(
    "Select variable for EDA:",
    options=[var_labels[k] for k in ["REVENUE_PER_SESSION",
                                     "AVG_SESSION_MINUTES",
                                     "AC_SHARE",
                                     "AVG_DAYS_BETWEEN_SESSIONS",
                                     "CHARGED_AMOUNT"]]
)
eda_var = inv_labels[eda_var_friendly]

# --------------------------
# EDA Section
# --------------------------
st.subheader("📊 Exploratory Data Analysis")

st.write("**Dataset preview:**")
st.dataframe(df.head())

# --- High-level KPIs ---
total_users = len(df)
at_risk = int(df["churn_risk"].sum())
risk_pct = (at_risk / total_users) * 100

col1, col2 = st.columns(2)
col1.metric("Total Tokens", f"{total_users:,}")
col2.metric("Churn Risk Tokens", f"{at_risk:,}  ({risk_pct:.1f}%)")

st.caption(f"ℹ️ Approximately **{risk_pct:.1f}%** of all tokens are currently at risk of dropping out.")
st.markdown("---")

# --------------------------
# Dynamic stats for selected variable
# --------------------------
mean_val = df[eda_var].mean()
median_val = df[eda_var].median()
std_val = df[eda_var].std()

# IQR outliers
q1 = df[eda_var].quantile(0.25)
q3 = df[eda_var].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outlier_pct = ((df[eda_var] < lower) | (df[eda_var] > upper)).mean() * 100

# Compare churn groups
median_churn = df[df["churn_risk"] == 1][eda_var].median()
median_nonchurn = df[df["churn_risk"] == 0][eda_var].median()

st.markdown(f"### 📈 Insights for **{eda_var_friendly}**")
st.write(
    f"- **Mean:** {mean_val:,.2f}  "
    f"**Median:** {median_val:,.2f}  "
    f"**Std:** {std_val:,.2f}"
)
st.write(
    f"- About **{outlier_pct:.1f}%** of tokens are outliers "
    f"(IQR bounds: {lower:,.2f} – {upper:,.2f})."
)
st.write(
    f"- Median for **churn-risk** users: {median_churn:,.2f}  "
    f"vs **not-at-risk**: {median_nonchurn:,.2f}"
)

if median_churn > median_nonchurn:
    st.info(f"💡 **Insight:** Churn-risk users tend to have **higher {eda_var_friendly}** than non-risk users.")
else:
    st.info(f"💡 **Insight:** Churn-risk users tend to have **lower {eda_var_friendly}** than non-risk users.")

st.markdown("---")
st.markdown("### Distribution Overview")

# Histogram
fig_hist, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df[eda_var], bins=30, kde=True, color="skyblue", ax=ax)
ax.set_title(f"Histogram of {eda_var_friendly}")
ax.set_xlabel(eda_var_friendly)
st.pyplot(fig_hist)

# Boxplot
fig_box, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(y=df[eda_var], color="lightgreen", ax=ax)
ax.set_title(f"Boxplot of {eda_var_friendly}")
ax.set_ylabel(eda_var_friendly)
st.pyplot(fig_box)

# Boxplot by churn risk
st.markdown("### Compare by Churn Risk")
fig_box_group, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x=df["churn_risk"], y=df[eda_var], palette="Set2", ax=ax)
ax.set_title(f"{eda_var_friendly} by Churn Risk")
ax.set_xlabel("Churn Risk (0 = Not at risk, 1 = At risk)")
ax.set_ylabel(eda_var_friendly)
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

# Use human-readable names in multiselect
available_features_friendly = [var_labels[c] for c in available_features]

selected_features_friendly = st.multiselect(
    "Select features for clustering:",
    options=available_features_friendly,
    default=available_features_friendly
)

# Map back to original column names
selected_features = [inv_labels[f] for f in selected_features_friendly]

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

        # Rename columns in summary
        cluster_summary = cluster_summary.rename(columns=var_labels)
        st.dataframe(cluster_summary)

        # --- Dynamic cluster-level insights ---
        risk_by_cluster = df.groupby("cluster")["churn_risk"].mean() * 100
        top_risk_cluster = risk_by_cluster.idxmax()
        top_risk_pct = risk_by_cluster.max()

        revenue_by_cluster = df.groupby("cluster")["REVENUE_PER_SESSION"].mean()
        top_rev_cluster = revenue_by_cluster.idxmax()

        st.markdown("---")
        st.subheader("💡 Insights from Clustering")
        st.success(
            f"Cluster **{top_risk_cluster}** shows the **highest churn risk ({top_risk_pct:.1f}%)**, "
            f"while Cluster **{top_rev_cluster}** has the highest **Average Session Value**."
        )

        # Scatter plot for first two features
        if len(selected_features) >= 2:
            st.subheader("📌 Scatter Plot (first two selected features)")
            f1_friendly = var_labels[selected_features[0]]
            f2_friendly = var_labels[selected_features[1]]

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
            ax.set_xlabel(f1_friendly)
            ax.set_ylabel(f2_friendly)
            ax.set_title(f"K-Means Clusters ({f1_friendly} vs {f2_friendly})")
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
