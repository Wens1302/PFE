"""
dashboard.py
------------
Streamlit decision dashboard for trade-finance risk analysis.

Sections
--------
  1. KPIs globaux
  2. Flux & Techniques de paiement
  3. Risque pays (notation OECD/COFACE)
  4. Profils entreprises (K-Means)
  5. Modèle de risque – retard
  6. Détection Fraude / AML
  7. Détection d'anomalies (Isolation Forest)
  8. Profil Client 360 – évolution des montants
  9. Tableau des alertes

Usage
-----
    streamlit run dashboard.py

Prerequisite: run generate_dataset.py first.
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

from utils import NOTATION_LABELS, NOTATION_PAYS, build_cluster_labels

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
    le_hist  = LabelEncoder()
    le_tech  = LabelEncoder()
    le_sect  = LabelEncoder()
    le_pays  = LabelEncoder()

    df_ent = df_ent.copy()
    df     = df.copy()

    df_ent["ville_enc"] = le_ville.fit_transform(df_ent["ville"])
    df_ent["hist_enc"]  = le_hist.fit_transform(df_ent["historique_paiement"])
    df_ent["sect_enc"]  = le_sect.fit_transform(df_ent["secteur_activite"])

    df["tech_enc"]  = le_tech.fit_transform(df["technique_paiement"])
    df["hist_enc"]  = le_hist.transform(df["historique_paiement"])
    df["ville_enc"] = le_ville.transform(df["ville"])
    df["sect_enc"]  = le_sect.transform(df["secteur_activite"])
    df["pays_enc"]  = le_pays.fit_transform(df["pays"])

    # ── Clustering ─────────────────────────────────────────────────────────────
    agg = (
        df.groupby("entreprise_id")
        .agg(
            nb_transactions  = ("transaction_id", "count"),
            montant_moyen    = ("montant", "mean"),
            taux_retard      = ("is_late", "mean"),
            retard_moyen     = ("retard", "mean"),
            taux_fraude      = ("fraude_suspectee", "mean"),
            score_aml_moyen  = ("score_aml", "mean"),
        )
        .reset_index()
    )
    agg = agg.merge(df_ent, on="entreprise_id")

    features_clust = [
        "ligne_de_credit", "pct_utilisation_credit", "nb_transactions_historique",
        "nb_transactions", "montant_moyen", "taux_retard", "retard_moyen",
        "taux_fraude", "score_aml_moyen", "hist_enc", "ville_enc", "sect_enc",
    ]
    X_clust = agg[features_clust].fillna(0)
    scaler_clust = StandardScaler()
    X_clust_scaled = scaler_clust.fit_transform(X_clust)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    agg["cluster"] = km.fit_predict(X_clust_scaled)

    profile_summary = (
        agg.groupby("cluster")
        .agg(
            taux_retard_moy  = ("taux_retard", "mean"),
            ligne_credit_moy = ("ligne_de_credit", "mean"),
            utilisation_moy  = ("pct_utilisation_credit", "mean"),
        )
        .round(2)
    )
    cluster_labels = build_cluster_labels(profile_summary)
    agg["profil"] = agg["cluster"].map(cluster_labels)

    # ── Retard risk model ──────────────────────────────────────────────────────
    features_rf = [
        "montant", "delai_prevu", "notation_pays", "score_risque_pays",
        "score_aml", "tech_enc", "hist_enc", "ville_enc", "sect_enc",
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
    df["proba_retard"]  = rf.predict_proba(X)[:, 1]

    # ── Fraud / AML model ──────────────────────────────────────────────────────
    features_fraud = [
        "montant", "notation_pays", "score_risque_pays", "score_aml",
        "tech_enc", "hist_enc", "pays_enc", "ligne_de_credit",
        "pct_utilisation_credit", "risque_operationnel",
        "evolution_montant_pct", "rang_transaction_entreprise",
    ]
    df_f = df.copy()
    df_f["evolution_montant_pct"] = df_f["evolution_montant_pct"].fillna(0)
    X_fr = df_f[features_fraud].fillna(0)
    y_fr = df_f["fraude_suspectee"]
    X_ftr, X_fte, y_ftr, y_fte = train_test_split(
        X_fr, y_fr, test_size=0.25, random_state=42, stratify=y_fr
    )
    rf_fraud = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    rf_fraud.fit(X_ftr, y_ftr)
    df["proba_fraude"] = rf_fraud.predict_proba(X_fr)[:, 1]

    # ── Anomaly detection ──────────────────────────────────────────────────────
    features_iso = [
        "montant", "delai_reel", "retard", "score_aml",
        "notation_pays", "ligne_de_credit", "pct_utilisation_credit",
        "evolution_montant_pct",
    ]
    df_iso = df.copy()
    df_iso["evolution_montant_pct"] = df_iso["evolution_montant_pct"].fillna(0)
    X_iso = df_iso[features_iso].fillna(0)
    scaler_iso = StandardScaler()
    X_iso_scaled = scaler_iso.fit_transform(X_iso)
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["anomalie_flag"] = (iso.fit_predict(X_iso_scaled) == -1).astype(int)

    return df, agg, rf, features_rf, rf_fraud, features_fraud


# ── Data loading ───────────────────────────────────────────────────────────────
df_ent, df_trx, df_merged = load_data()
df, agg, rf, features_rf, rf_fraud, features_fraud = build_models(df_ent, df_trx, df_merged)

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filtres")
villes = st.sidebar.multiselect(
    "Ville", df["ville"].unique().tolist(), default=df["ville"].unique().tolist()
)
techniques = st.sidebar.multiselect(
    "Technique de paiement",
    df["technique_paiement"].unique().tolist(),
    default=df["technique_paiement"].unique().tolist(),
)
notations = st.sidebar.multiselect(
    "Notation risque pays (1=très faible … 4=élevé)",
    sorted(df["notation_pays"].unique().tolist()),
    default=sorted(df["notation_pays"].unique().tolist()),
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
    & df["notation_pays"].isin(notations)
    & df["montant"].between(*montant_range)
]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏦 Dashboard Risque Trade Finance")
st.caption(
    "Analyse data-driven des flux de commerce international – "
    "lutte contre la fraude, le blanchiment et les risques opérationnels "
    "(inspiré de Bank of Africa – BOA Maroc)"
)

# ══════════════════════════════════════════════════════════════════════════════
# KPIs
# ══════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Transactions",          f"{len(df_f):,}")
k2.metric("Montant total (MAD)",   f"{df_f['montant'].sum()/1e6:.1f} M")
k3.metric("Taux de retard",        f"{df_f['is_late'].mean()*100:.1f} %")
k4.metric("Anomalies détectées",   f"{df_f['anomalie_flag'].sum()}")
k5.metric("Fraudes suspectées",    f"{df_f['fraude_suspectee'].sum()}")
k6.metric("Risques opérationnels", f"{df_f['risque_operationnel'].sum()}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 – Flux & Techniques de paiement
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("💳 Flux & Techniques de paiement")
c1, c2, c3 = st.columns(3)

with c1:
    fig = px.pie(
        df_f, names="technique_paiement",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.4, title="Techniques de paiement",
    )
    fig.update_layout(showlegend=True, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

with c2:
    fig = px.histogram(
        df_f, x="montant", nbins=40,
        color_discrete_sequence=["steelblue"],
        labels={"montant": "Montant (MAD)"},
        title="Distribution des montants",
    )
    fig.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

with c3:
    retard_pays = (
        df_f.groupby("pays")["is_late"].mean().mul(100).round(1).reset_index()
    )
    retard_pays.columns = ["Pays", "Taux retard (%)"]
    retard_pays = retard_pays.sort_values("Taux retard (%)", ascending=False)
    fig = px.bar(
        retard_pays, x="Taux retard (%)", y="Pays", orientation="h",
        color="Taux retard (%)", color_continuous_scale="RdYlGn_r",
        title="Taux de retard par pays",
    )
    fig.update_layout(margin=dict(t=40, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 – Risque pays
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🌍 Risque pays (notation OECD/COFACE)")
ca, cb = st.columns(2)

with ca:
    country_risk = (
        df_f.groupby("pays")
        .agg(
            notation_pays   = ("notation_pays", "first"),
            score_aml_moyen = ("score_aml", "mean"),
            taux_fraude     = ("fraude_suspectee", "mean"),
            nb_transactions = ("transaction_id", "count"),
        )
        .reset_index()
        .sort_values("notation_pays", ascending=True)
    )
    country_risk["libelle"] = country_risk["notation_pays"].map(NOTATION_LABELS)
    fig = px.bar(
        country_risk, x="pays", y="score_aml_moyen",
        color="notation_pays",
        color_continuous_scale=["#4caf50", "#8bc34a", "#ff9800", "#f44336", "#9c27b0"],
        range_color=[1, 5],
        hover_data=["libelle", "taux_fraude", "nb_transactions"],
        labels={"pays": "Pays", "score_aml_moyen": "Score AML moyen", "notation_pays": "Notation"},
        title="Score AML moyen par pays partenaire",
    )
    fig.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

with cb:
    fig = px.scatter(
        country_risk,
        x="score_aml_moyen",
        y="taux_fraude",
        size="nb_transactions",
        color="notation_pays",
        text="pays",
        color_continuous_scale=["#4caf50", "#ff9800", "#f44336"],
        range_color=[1, 4],
        labels={
            "score_aml_moyen": "Score AML moyen",
            "taux_fraude":     "Taux de fraude",
            "notation_pays":   "Notation",
        },
        title="Corrélation notation pays ↔ fraude",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 – Profils entreprises
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🏢 Profils des entreprises (K-Means)")
cd, ce = st.columns(2)

with cd:
    profil_count = agg["profil"].value_counts().reset_index()
    profil_count.columns = ["Profil", "Nombre"]
    fig = px.bar(
        profil_count, x="Profil", y="Nombre",
        color="Profil",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text="Nombre", title="Répartition des profils clients",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

with ce:
    fig = px.scatter(
        agg,
        x="pct_utilisation_credit",
        y="taux_retard",
        color="profil",
        size="montant_moyen",
        hover_data=["entreprise_id", "ligne_de_credit", "nb_transactions", "taux_fraude"],
        labels={
            "pct_utilisation_credit": "Utilisation crédit (%)",
            "taux_retard":            "Taux de retard",
            "profil":                 "Profil",
        },
        color_discrete_sequence=px.colors.qualitative.Set1,
        title="Profils entreprises – retard vs utilisation crédit",
    )
    fig.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 – Modèle de risque (retard)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("⏱️ Modèle de risque transactionnel – retard paiement")
cf, cg = st.columns(2)

with cf:
    importances = pd.Series(rf.feature_importances_, index=features_rf).sort_values()
    fig = px.bar(
        importances.reset_index(),
        x=0, y="index", orientation="h",
        labels={"index": "Variable", 0: "Importance"},
        color=0, color_continuous_scale="Blues",
        title="Importance des variables – modèle retard",
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

with cg:
    # Score AML distribution by is_late
    fig = px.box(
        df_f, x="is_late", y="score_aml",
        color="is_late",
        color_discrete_map={0: "steelblue", 1: "tomato"},
        labels={"is_late": "En retard (0=Non, 1=Oui)", "score_aml": "Score AML"},
        title="Score AML selon le statut de retard",
    )
    fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 5 – Détection Fraude / AML
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🚨 Détection Fraude & Blanchiment (AML)")
ch, ci = st.columns(2)

with ch:
    imp_fraud = pd.Series(rf_fraud.feature_importances_, index=features_fraud).sort_values()
    fig = px.bar(
        imp_fraud.reset_index(),
        x=0, y="index", orientation="h",
        labels={"index": "Variable", 0: "Importance"},
        color=0, color_continuous_scale="Reds",
        title="Importance des variables – modèle fraude/AML",
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

with ci:
    fraud_tech = (
        df_f.groupby("technique_paiement")["fraude_suspectee"].mean()
        .mul(100).round(1).reset_index()
    )
    fraud_tech.columns = ["Technique", "Taux fraude (%)"]
    fig = px.bar(
        fraud_tech, x="Technique", y="Taux fraude (%)",
        color="Taux fraude (%)", color_continuous_scale="OrRd",
        text="Taux fraude (%)",
        title="Taux de fraude par technique de paiement",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

# Score AML histogram
fig = px.histogram(
    df_f, x="score_aml", color="fraude_suspectee",
    nbins=40, barmode="overlay", opacity=0.75,
    color_discrete_map={0: "steelblue", 1: "tomato"},
    labels={"score_aml": "Score AML", "fraude_suspectee": "Fraude"},
    title="Distribution du score AML (normal vs fraude suspectée)",
)
fig.update_layout(margin=dict(t=40, b=10))
st.plotly_chart(fig, width="stretch")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 6 – Anomalies (Isolation Forest)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🔍 Détection d'anomalies (Isolation Forest)")
cj, ck = st.columns(2)

with cj:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_f.loc[df_f["anomalie_flag"] == 0, "montant"],
        name="Normal", opacity=0.65, marker_color="steelblue", nbinsx=40,
    ))
    fig.add_trace(go.Histogram(
        x=df_f.loc[df_f["anomalie_flag"] == 1, "montant"],
        name="Anomalie", opacity=0.85, marker_color="tomato", nbinsx=40,
    ))
    fig.update_layout(
        barmode="overlay", margin=dict(t=40, b=10),
        xaxis_title="Montant (MAD)", title="Montants : normal vs anomalie",
    )
    st.plotly_chart(fig, width="stretch")

with ck:
    # Anomalies by country risk
    anomaly_pays = (
        df_f.groupby("pays")["anomalie_flag"].mean()
        .mul(100).round(1).reset_index()
    )
    anomaly_pays.columns = ["Pays", "Taux anomalie (%)"]
    anomaly_pays = anomaly_pays.sort_values("Taux anomalie (%)", ascending=False)
    fig = px.bar(
        anomaly_pays, x="Taux anomalie (%)", y="Pays", orientation="h",
        color="Taux anomalie (%)", color_continuous_scale="YlOrRd",
        title="Taux d'anomalies par pays partenaire",
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 7 – Profil Client 360 – Évolution des montants
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("👤 Profil Client 360 – Évolution des montants")

all_companies = sorted(df["entreprise_id"].unique().tolist())
selected_company = st.selectbox("Sélectionner une entreprise", all_companies)

sub = (
    df[df["entreprise_id"] == selected_company]
    .sort_values("date_ouverture")
    .reset_index(drop=True)
)

# Info cards
ci1, ci2, ci3, ci4, ci5 = st.columns(5)
company_info = df_ent[df_ent["entreprise_id"] == selected_company].iloc[0]
ci1.metric("Secteur",            company_info["secteur_activite"])
ci2.metric("Historique paiement", company_info["historique_paiement"])
ci3.metric("Ligne de crédit",    f"{company_info['ligne_de_credit']/1e6:.1f} M MAD")
ci4.metric("Taux de retard",     f"{sub['is_late'].mean()*100:.0f} %")
ci5.metric("Fraudes suspectées", f"{sub['fraude_suspectee'].sum()}")

# Transaction evolution line chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=sub.index + 1,
    y=sub["montant"] / 1_000,
    mode="lines+markers",
    name="Montant (k MAD)",
    line=dict(color="steelblue", width=2),
    marker=dict(size=6),
))

# Flag fraud transactions
fraud_sub = sub[sub["fraude_suspectee"] == 1]
if not fraud_sub.empty:
    fig.add_trace(go.Scatter(
        x=fraud_sub.index + 1,
        y=fraud_sub["montant"] / 1_000,
        mode="markers",
        name="Fraude suspectée",
        marker=dict(color="tomato", size=12, symbol="x"),
    ))

# Flag anomalies
anom_sub = sub[sub["anomalie_flag"] == 1]
if not anom_sub.empty:
    fig.add_trace(go.Scatter(
        x=anom_sub.index + 1,
        y=anom_sub["montant"] / 1_000,
        mode="markers",
        name="Anomalie",
        marker=dict(color="orange", size=10, symbol="triangle-up"),
    ))

fig.update_layout(
    xaxis_title="Rang transaction",
    yaxis_title="Montant (k MAD)",
    title=f"Évolution des montants – {selected_company}",
    margin=dict(t=40, b=10),
)
st.plotly_chart(fig, width="stretch")

# Transaction history table
st.dataframe(
    sub[[
        "transaction_id", "pays", "notation_pays", "montant",
        "technique_paiement", "date_ouverture", "delai_prevu",
        "retard", "is_late", "score_aml", "fraude_suspectee",
        "risque_operationnel", "anomalie_flag",
        "evolution_montant_pct",
    ]].rename(columns={"evolution_montant_pct": "évol. montant (%)"}),
    width="stretch", hide_index=True,
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 8 – Tableau des alertes
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📋 Tableau des alertes – transactions à risque élevé")

tab1, tab2, tab3 = st.tabs(["⏱️ Retard élevé (>50 %)", "🚨 Fraude suspectée", "⚠️ Risque opérationnel"])

with tab1:
    high_risk = df_f[df_f["proba_retard"] > 0.5][[
        "transaction_id", "entreprise_id", "pays", "notation_pays", "montant",
        "technique_paiement", "date_ouverture", "retard", "proba_retard",
        "score_aml", "anomalie_flag",
    ]].sort_values("proba_retard", ascending=False).head(50).copy()
    high_risk["proba_retard"] = high_risk["proba_retard"].mul(100).round(1).astype(str) + " %"
    high_risk["montant"]      = high_risk["montant"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(high_risk, width="stretch", hide_index=True)

with tab2:
    fraud_alerts = df_f[df_f["fraude_suspectee"] == 1][[
        "transaction_id", "entreprise_id", "pays", "notation_pays", "montant",
        "technique_paiement", "date_ouverture", "score_aml",
        "proba_fraude", "anomalie_flag",
    ]].sort_values("proba_fraude", ascending=False).head(50).copy()
    fraud_alerts["proba_fraude"] = fraud_alerts["proba_fraude"].mul(100).round(1).astype(str) + " %"
    fraud_alerts["montant"]      = fraud_alerts["montant"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(fraud_alerts, width="stretch", hide_index=True)

with tab3:
    op_alerts = df_f[df_f["risque_operationnel"] == 1][[
        "transaction_id", "entreprise_id", "pays", "notation_pays", "montant",
        "technique_paiement", "date_ouverture", "retard",
        "score_aml", "fraude_suspectee", "anomalie_flag",
    ]].sort_values("score_aml", ascending=False).head(50).copy()
    op_alerts["montant"] = op_alerts["montant"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(op_alerts, width="stretch", hide_index=True)
