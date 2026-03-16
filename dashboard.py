"""
dashboard.py
------------
Streamlit decision dashboard for trade-finance risk analysis.

Usage
-----
    streamlit run dashboard.py

Prerequisite: run generate_dataset.py and analyse.py first.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import build_cluster_labels

matplotlib.use("Agg")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trade Finance Risk Dashboard",
    page_icon="🏦",
    layout="wide",
)

# ── Load data ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    if not os.path.exists("data/entreprises.csv"):
        st.error("Dataset manquant. Lancez d'abord : python generate_dataset.py")
        st.stop()
    df_ent = pd.read_csv("data/entreprises.csv")
    df_trx = pd.read_csv("data/transactions.csv")
    df = df_trx.merge(df_ent, on="entreprise_id", how="left")
    return df_ent, df_trx, df


@st.cache_data
def build_models(df_ent, df_trx, df):
    le_ville = LabelEncoder()
    le_hist = LabelEncoder()
    le_tech = LabelEncoder()
    le_sect = LabelEncoder()

    df_ent = df_ent.copy()
    df = df.copy()

    df_ent["ville_enc"] = le_ville.fit_transform(df_ent["ville"])
    df_ent["hist_enc"] = le_hist.fit_transform(df_ent["historique_paiement"])
    df_ent["sect_enc"] = le_sect.fit_transform(df_ent["secteur_activite"])

    df["tech_enc"] = le_tech.fit_transform(df["technique_paiement"])
    df["hist_enc"] = le_hist.transform(df["historique_paiement"])
    df["ville_enc"] = le_ville.transform(df["ville"])
    df["sect_enc"] = le_sect.transform(df["secteur_activite"])

    # Clustering
    agg = (
        df.groupby("entreprise_id")
        .agg(
            nb_transactions=("transaction_id", "count"),
            montant_moyen=("montant", "mean"),
            taux_retard=("is_late", "mean"),
            retard_moyen=("retard", "mean"),
        )
        .reset_index()
    )
    agg = agg.merge(df_ent, on="entreprise_id")

    features_clust = [
        "ligne_de_credit", "pct_utilisation_credit", "nb_transactions_historique",
        "nb_transactions", "montant_moyen", "taux_retard", "retard_moyen",
        "hist_enc", "ville_enc", "sect_enc",
    ]
    X_clust = agg[features_clust].fillna(0)
    scaler_clust = StandardScaler()
    X_clust_scaled = scaler_clust.fit_transform(X_clust)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    agg["cluster"] = km.fit_predict(X_clust_scaled)

    # Heuristic labels
    profile_summary = (
        agg.groupby("cluster")
        .agg(taux_retard_moy=("taux_retard", "mean"), ligne_credit_moy=("ligne_de_credit", "mean"),
             utilisation_moy=("pct_utilisation_credit", "mean"))
        .round(2)
    )
    cluster_labels = build_cluster_labels(profile_summary)
    agg["profil"] = agg["cluster"].map(cluster_labels)

    # Risk model
    features_rf = [
        "montant", "delai_prevu", "tech_enc", "hist_enc", "ville_enc", "sect_enc",
        "ligne_de_credit", "pct_utilisation_credit", "nb_transactions_historique",
    ]
    X = df[features_rf].fillna(0)
    y = df["is_late"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    df["risque_predit"] = rf.predict(X)
    df["proba_retard"] = rf.predict_proba(X)[:, 1]

    # Anomaly detection
    features_iso = ["montant", "delai_reel", "retard", "ligne_de_credit", "pct_utilisation_credit"]
    X_iso = df[features_iso].fillna(0)
    scaler_iso = StandardScaler()
    X_iso_scaled = scaler_iso.fit_transform(X_iso)
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["anomalie_flag"] = (iso.fit_predict(X_iso_scaled) == -1).astype(int)

    return df, agg, rf, features_rf


# ── Data loading ───────────────────────────────────────────────────────────────
df_ent, df_trx, df_merged = load_data()
df, agg, rf, features_rf = build_models(df_ent, df_trx, df_merged)

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filtres")
villes = st.sidebar.multiselect("Ville", df["ville"].unique().tolist(), default=df["ville"].unique().tolist())
techniques = st.sidebar.multiselect(
    "Technique de paiement",
    df["technique_paiement"].unique().tolist(),
    default=df["technique_paiement"].unique().tolist(),
)
montant_range = st.sidebar.slider(
    "Montant (MAD)",
    int(df["montant"].min()),
    int(df["montant"].max()),
    (int(df["montant"].min()), int(df["montant"].max())),
    step=10_000,
)

df_f = df[
    df["ville"].isin(villes)
    & df["technique_paiement"].isin(techniques)
    & df["montant"].between(*montant_range)
]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏦 Dashboard Risque Trade Finance")
st.caption("Analyse des flux de commerce international – inspiré de Bank of Africa (BOA Maroc)")

# ── KPIs ───────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Transactions", f"{len(df_f):,}")
k2.metric("Montant total (MAD)", f"{df_f['montant'].sum()/1e6:.1f} M")
k3.metric("Taux de retard", f"{df_f['is_late'].mean()*100:.1f} %")
k4.metric("Anomalies", f"{df_f['anomalie_flag'].sum()}")
k5.metric("Retard moyen (j)", f"{df_f.loc[df_f['retard']>0,'retard'].mean():.1f}")

st.divider()

# ── Row 1: Distribution charts ─────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Techniques de paiement")
    fig = px.pie(
        df_f,
        names="technique_paiement",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.4,
    )
    fig.update_layout(showlegend=True, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

with c2:
    st.subheader("Distribution des montants")
    fig = px.histogram(
        df_f,
        x="montant",
        nbins=40,
        color_discrete_sequence=["steelblue"],
        labels={"montant": "Montant (MAD)"},
    )
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

with c3:
    st.subheader("Retards par pays")
    retard_pays = (
        df_f.groupby("pays")["is_late"].mean().mul(100).round(1).reset_index()
    )
    retard_pays.columns = ["Pays", "Taux retard (%)"]
    retard_pays = retard_pays.sort_values("Taux retard (%)", ascending=False)
    fig = px.bar(
        retard_pays,
        x="Taux retard (%)",
        y="Pays",
        orientation="h",
        color="Taux retard (%)",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

# ── Row 2: Profils + Anomalies ─────────────────────────────────────────────────
c4, c5 = st.columns(2)

with c4:
    st.subheader("Profils des entreprises (K-Means)")
    profil_count = agg["profil"].value_counts().reset_index()
    profil_count.columns = ["Profil", "Nombre"]
    fig = px.bar(
        profil_count,
        x="Profil",
        y="Nombre",
        color="Profil",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text="Nombre",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

with c5:
    st.subheader("Montants : normal vs anomalie")
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df_f.loc[df_f["anomalie_flag"] == 0, "montant"],
            name="Normal",
            opacity=0.65,
            marker_color="steelblue",
            nbinsx=40,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=df_f.loc[df_f["anomalie_flag"] == 1, "montant"],
            name="Anomalie",
            opacity=0.85,
            marker_color="tomato",
            nbinsx=40,
        )
    )
    fig.update_layout(barmode="overlay", margin=dict(t=10, b=10), xaxis_title="Montant (MAD)")
    st.plotly_chart(fig, width="stretch")

# ── Row 3: Risk model importances + cluster scatter ───────────────────────────
c6, c7 = st.columns(2)

with c6:
    st.subheader("Importance des variables – modèle de risque")
    importances = pd.Series(rf.feature_importances_, index=features_rf).sort_values()
    fig = px.bar(
        importances.reset_index(),
        x=0,
        y="index",
        orientation="h",
        labels={"index": "Variable", 0: "Importance"},
        color=0,
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

with c7:
    st.subheader("Profils entreprises – taux retard vs utilisation crédit")
    fig = px.scatter(
        agg,
        x="pct_utilisation_credit",
        y="taux_retard",
        color="profil",
        size="montant_moyen",
        hover_data=["entreprise_id", "ligne_de_credit", "nb_transactions"],
        labels={
            "pct_utilisation_credit": "Utilisation crédit (%)",
            "taux_retard": "Taux de retard",
            "profil": "Profil",
        },
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

# ── Row 4: Transactions table ──────────────────────────────────────────────────
st.subheader("📋 Transactions à risque élevé (proba retard > 50 %)")
high_risk = df_f[df_f["proba_retard"] > 0.5][
    [
        "transaction_id", "entreprise_id", "pays", "montant",
        "technique_paiement", "date_ouverture", "date_prevue",
        "delai_prevu", "retard", "proba_retard", "anomalie_flag",
    ]
].sort_values("proba_retard", ascending=False).head(50)
high_risk["proba_retard"] = high_risk["proba_retard"].mul(100).round(1).astype(str) + " %"
high_risk["montant"] = high_risk["montant"].apply(lambda x: f"{x:,.0f}")
st.dataframe(high_risk, width="stretch", hide_index=True)
