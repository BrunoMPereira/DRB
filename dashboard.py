import streamlit as st
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from openai import OpenAI

# --------------------------
# OpenAI helper
# --------------------------
def get_ai_recommendations(insights_text, client):
    """
    Sends insights to OpenAI model and retrieves recommendations.
    """
    prompt = f"""
    A Daloop é uma empresa especializada em soluções de software para a mobilidade elétrica, com foco na gestão inteligente de carregamentos de veículos elétricos (EVs). O seu principal produto é uma plataforma digital composta por duas interfaces: uma plataforma web (Backoffice) e uma aplicação móvel (app Galp Frotas).
    A plataforma web é utilizada por operadores de pontos de carregamento (CPOs), gestores de frota e outros stakeholders corporativos, permitindo-lhes gerir utilizadores, postos de carregamento (EVSEs), sessões, consumos e tarifas. Suporta ações remotas, gestão de permissões, relatórios de carregamentos e integração com redes públicas ou privadas.
    A app móvel é orientada para o utilizador final (como condutores de frotas), permitindo iniciar carregamentos, consultar históricos, localizar postos e gerir os seus consumos. Ambos os canais estão integrados e suportam fluxos como carregamento via RFID, gestão de limites, e visibilidade de postos (privados/públicos).
    A Daloop opera em conformidade com os principais protocolos de interoperabilidade do setor da mobilidade elétrica, como OCPP e OCPI, permitindo integração com infraestruturas de terceiros e plataformas de roaming elétrico.

    O objetivo é sugerir ações concretas para aumentar o engagement, reduzir o churn de utilizadores e manter/aumentar os carregamentos da app Galp Frotas, com base na análise de clusters de tokens.
    As únicas ações que podem ser recomendadas (limitadas às capacidades atuais da Daloop) são:
        * Enviar notificação push via app com mensagem customizada.
        * Enviar email com conteúdo personalizado.
        * Atribuir código promocional (ex: desconto em carregamentos, minutos gratuitos).
        * Criar campanha temporária com incentivo (ex: “carrega 2 vezes esta semana e ganha 30 min grátis”).
        * Recomendar postos com menor ocupação ou mais baratos (usando push/email).
        * Sugerir carregamentos fora das horas de pico com incentivo.
        * Alertar utilizadores com poucos carregamentos recentes.
        * Incentivar a primeira utilização para tokens com registo mas sem carregamentos.
        * Reativar utilizadores com sessões antigas ou interrupções longas.
        * Recolher feedback após carregamento ou inatividade.

    Os dados que te passamos de seguida são de clusters de carregamentos de veículos elétricos. Estes clusters já foram previamente agrupados através de um algoritmo de k-means e com dados da nossa plataforma. A primeira coluna identifica o id do cluster e a última coluna identifica o risco de churn dos tokens dentro desse cluster. As colunas entre a primeira e a última são propriedades / métricas de cada cluster.
    O formato dos dados é semelhante a um csv, com a separação das colunas por vírgula “,”, sendo que a primeira linha é o nome das colunas e as seguintes são os dados.
    Exemplo de colunas que podem estar presentes nos dados e seu significado:
    Number of Tokens - número de cartões do cluster
    Average Distinct Charging Stations - número médio de postos de carregamento distintos
    Average Session Value - custo médio, em euros, das sessões de carregamento desse cluster
    Average Session Duration - duração média, em minutos, das sessões de carregamento desse cluster
    Total Value - custo total, em euros, das sessões de carregamento desse cluster
    Sessions in AC (%) - percentagem de sessões de carregamento desse cluster cujo carregamento foi realizado em corrente alternada (AC)
    
    Seguem os dados dos clusters:
    {insights_text}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """A tua especialidade é analisar os dados e sugerir recomendações.
                                            Instruções de saída:
                                             * Gera até 10 recomendações curtas e acionáveis (podes dar menos).
                                             * Cada recomendação deve ter no máximo 20 palavras.
                                             * Ordena por relevância decrescente (impacto no churn e engagement).
                                             * Coloca no início de cada linha um número entre 0–10 indicando importância, 0 = pouca importância, 10 = extremamente importante.
                                             * Formata em lista com bullet points.
                                             * A linguagem deve ser inglês americano"""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error while getting recommendations: {e}"


# --------------------------
# Initialize OpenAI client
# --------------------------
api_key = st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key) if api_key else None

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="EV User Clustering Dashboard", layout="wide")

st.title("🚗 EV User Analysis & Clustering Dashboard")
st.markdown("Explore **EDA first**, then select **features to run K-Means clustering** interactively.")

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

# Remove any pre-existing cluster column
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
    "DAYS_FROM_LAST_SESSION": "Days from Last Session",
    "churn_risk": "Churn Risk",
    "DIF_CHARGING_STATIONS": "Different Charging Stations Used",
    "AVG_DIF_CHARGING_STATIONS": "Average Distinct Charging Stations"
}
inv_labels = {v: k for k, v in var_labels.items()}

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.header("Controls")

eda_var_friendly = st.sidebar.selectbox(
    "Select variable for EDA:",
    options=[var_labels[k] for k in [
        "REVENUE_PER_SESSION",
        "AVG_SESSION_MINUTES",
        "AC_SHARE",
        "AVG_DAYS_BETWEEN_SESSIONS",
        "CHARGED_AMOUNT",
        "DIF_CHARGING_STATIONS"
    ]]
)
eda_var = inv_labels[eda_var_friendly]

# --------------------------
# EDA Section
# --------------------------
st.subheader("📊 Exploratory Data Analysis")

st.write("**Dataset preview:**")
st.dataframe(df.head())

# High-level KPIs
total_users = len(df)
at_risk = int(df["churn_risk"].sum())
risk_pct = (at_risk / total_users) * 100

col1, col2 = st.columns(2)
col1.metric("Total Tokens", f"{total_users:,}")
col2.metric("Churn Risk Tokens", f"{at_risk:,}  ({risk_pct:.1f}%)")

st.caption(f"ℹ️ Approximately **{risk_pct:.1f}%** of all tokens are currently at risk of dropping out.")
st.markdown("---")

# Stats for selected variable
mean_val = df[eda_var].mean()
median_val = df[eda_var].median()
std_val = df[eda_var].std()

q1 = df[eda_var].quantile(0.25)
q3 = df[eda_var].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outlier_pct = ((df[eda_var] < lower) | (df[eda_var] > upper)).mean() * 100

median_churn = df[df["churn_risk"] == 1][eda_var].median()
median_nonchurn = df[df["churn_risk"] == 0][eda_var].median()

st.markdown(f"### 📈 Insights for **{eda_var_friendly}**")
st.write(
    f"- **Mean:** {mean_val:,.2f}  "
    f"**Median:** {median_val:,.2f}  "
    f"**Std:** {std_val:,.2f}"
)
st.write(f"- About **{outlier_pct:.1f}%** of tokens are outliers (IQR bounds: {lower:,.2f} – {upper:,.2f}).")
st.write(f"- Median for **churn-risk** users: {median_churn:,.2f}  vs **not-at-risk**: {median_nonchurn:,.2f}")

if median_churn > median_nonchurn:
    st.info(f"💡 **Insight:** Churn-risk users tend to have **higher {eda_var_friendly}** than non-risk users.")
else:
    st.info(f"💡 **Insight:** Churn-risk users tend to have **lower {eda_var_friendly}** than non-risk users.")

st.markdown("---")
st.markdown("### Distribution Overview")

# Histogram
fig_hist, ax = plt.subplots(figsize=(5, 4))
sns.histplot(df[eda_var], bins=30, kde=True, color="skyblue", ax=ax)
ax.set_title(f"Histogram of {eda_var_friendly}")
ax.set_xlabel(eda_var_friendly)
st.pyplot(fig_hist,use_container_width=False)

# Boxplot
fig_box, ax = plt.subplots(figsize=(5, 3))
sns.boxplot(y=df[eda_var], color="lightgreen", ax=ax)
ax.set_title(f"Boxplot of {eda_var_friendly}")
ax.set_ylabel(eda_var_friendly)
st.pyplot(fig_box,use_container_width=False)

# Boxplot by churn risk
st.markdown("### Compare by Churn Risk")
fig_box_group, ax = plt.subplots(figsize=(5, 4))
sns.boxplot(x=df["churn_risk"], y=df[eda_var], palette="Set2", ax=ax)
ax.set_title(f"{eda_var_friendly} by Churn Risk")
ax.set_xlabel("Churn Risk (0 = Not at risk, 1 = At risk)")
ax.set_ylabel(eda_var_friendly)
st.pyplot(fig_box_group,use_container_width=False)

st.markdown("---")

# --------------------------
# Session state
# --------------------------
if "clustered_df" not in st.session_state:
    st.session_state.clustered_df = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# --------------------------
# Clustering Section
# --------------------------
st.subheader("🔑 Feature Selection for Clustering")

default_feats = [
    "AVG_DAYS_BETWEEN_SESSIONS",
    "DAYS_FROM_LAST_SESSION",
    "CHARGED_AMOUNT",
    "AC_SHARE",
    "AVG_SESSION_MINUTES",
    "DIF_CHARGING_STATIONS"
]

available_features = [c for c in default_feats if c in df.columns]
available_features_friendly = [var_labels[c] for c in available_features]

selected_features_friendly = st.multiselect(
    "Select features for clustering:",
    options=available_features_friendly,
    default=available_features_friendly
)

selected_features = [inv_labels[f] for f in selected_features_friendly]

# --- Elbow Method ---
if len(selected_features) >= 2:
    X = df[selected_features].to_numpy()

    inertias = []
    k_values = range(1, 11)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Detect best k by largest drop ("elbow")
    diffs = np.diff(inertias)
    best_k = np.argmin(diffs[1:] - diffs[:-1]) + 2  # crude elbow estimate

    with st.expander("📉 View Elbow Method Plot"):
        st.markdown("The elbow curve helps identify the optimal number of clusters.")
        fig_elbow, ax = plt.subplots(figsize=(6, 4))
        ax.plot(k_values, inertias, marker='o')
        ax.axvline(x=best_k, color='red', linestyle='--', label=f"Suggested k = {best_k}")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method for Optimal k")
        ax.legend()
        st.pyplot(fig_elbow)

    st.caption(f"💡 Based on the elbow method, a good starting choice for k is **{best_k}**.")

else:
    st.info("Select at least two features to calculate elbow method.")


# --- Cluster slider with info tooltip ---
n_clusters = st.slider(
    "Select number of clusters (k):",
    min_value=2,
    max_value=10,
    value=best_k if 'best_k' in locals() else 4,
    help="The elbow method suggests a k where inertia starts to decrease slowly. "
         "This often indicates the best balance between simplicity and accuracy."
)

st.info("Select features and press **Run Clustering** to cluster.")

if st.button("▶️ Run Clustering"):
    if len(selected_features) < 2:
        st.error("Select at least two features for clustering.")
    else:
        # Run KMeans
        X = df[selected_features].to_numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        df["cluster"] = labels

        # Save clustered data in session state
        st.session_state.clustered_df = df.copy()
        st.session_state.recommendations = None
        st.success(f"K-Means completed with **{n_clusters} clusters** ✅")

# --------------------------
# Show clustering results if available
# --------------------------
if st.session_state.clustered_df is not None:
    clustered_df = st.session_state.clustered_df

    st.subheader("📈 Cluster Summary")
    total_tokens = len(clustered_df)

    cluster_summary = (
        clustered_df.groupby("cluster")
        .agg({
            "TOKEN_ID": "count",
            "DIF_CHARGING_STATIONS": "mean",
            "REVENUE_PER_SESSION": "mean",
            "AVG_SESSION_MINUTES": "mean",
            "CHARGED_AMOUNT": "mean",
            "AC_SHARE": "mean",
            "churn_risk": "mean"
        })
        .reset_index()
    )

    cluster_summary = cluster_summary.rename(columns={"TOKEN_ID": "N_TOKENS",
                                                      "DIF_CHARGING_STATIONS": "AVG_DIF_CHARGING_STATIONS"})
    cluster_summary["PCT_TOKENS"] = (cluster_summary["N_TOKENS"] / total_tokens * 100).round(2)
    cluster_summary["TOKENS_INFO"] = cluster_summary.apply(
        lambda row: f"{row['N_TOKENS']} ({row['PCT_TOKENS']}%)", axis=1
    )

    cols = [
        "cluster", "TOKENS_INFO", "AVG_DIF_CHARGING_STATIONS",
        "REVENUE_PER_SESSION", "AVG_SESSION_MINUTES",
        "CHARGED_AMOUNT", "AC_SHARE", "churn_risk"
    ]
    cluster_summary = cluster_summary[cols]
    cluster_summary = cluster_summary.rename(columns={"TOKENS_INFO": "Number of Tokens"})
    cluster_summary = cluster_summary.rename(columns=var_labels)

    st.dataframe(cluster_summary)

    # Cluster-level insights
    risk_by_cluster = clustered_df.groupby("cluster")["churn_risk"].mean() * 100
    top_risk_cluster = risk_by_cluster.idxmax()
    top_risk_pct = risk_by_cluster.max()

    revenue_by_cluster = clustered_df.groupby("cluster")["REVENUE_PER_SESSION"].mean()
    top_rev_cluster = revenue_by_cluster.idxmax()

    st.markdown("---")
    st.subheader("💡 Insights from Clustering")
    st.success(
        f"Cluster **{top_risk_cluster}** shows the **highest churn risk ({top_risk_pct:.1f}%)**, "
        f"while Cluster **{top_rev_cluster}** has the highest **Average Session Value**."
    )

    if len(selected_features) >= 2:
        st.subheader("📌 Scatter Plot (first two selected features)")
        f1_friendly = var_labels[selected_features[0]]
        f2_friendly = var_labels[selected_features[1]]

        fig_scatter, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            clustered_df[selected_features[0]],
            clustered_df[selected_features[1]],
            c=clustered_df["cluster"],
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

    st.markdown("---")
    st.subheader("🤖 AI-Powered Recommendations")

    if client:
        insights_text = f"""
        {cluster_summary.to_string(index=False)}"""


        if st.button("Generate Recommendations"):
            with st.spinner("Fetching recommendations..."):
                st.session_state.recommendations = get_ai_recommendations(insights_text, client)

    else:
        st.info("ℹ️ Add your OpenAI API key in `.streamlit/secrets.toml` to enable recommendations.")

    if st.session_state.recommendations:
        st.success("✅ Recommendations generated:")
        st.write(st.session_state.recommendations)

    # Download clustered data
    csv_data = clustered_df.to_csv(index=False)
    st.download_button(
        "💾 Download Clustered Data as CSV",
        data=csv_data,
        file_name="clustered_users.csv",
        mime="text/csv"
    )
