"""
analyse.py
----------
Trade-finance risk analysis:
  1. Company eco-profile identification (K-Means clustering)
  2. Transactional risk modelling (Random Forest – predict is_late)
  3. Fraud / AML detection (Random Forest – predict fraude_suspectee)
  4. Anomaly detection (Isolation Forest)
  5. Transaction-amount evolution per company

Usage
-----
    python analyse.py

Prerequisite: run generate_dataset.py first (or just run it from here).
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import NOTATION_LABELS, build_cluster_labels

matplotlib.use("Agg")  # non-interactive backend
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading datasets …")
df_ent = pd.read_csv("data/entreprises.csv")
df_trx = pd.read_csv("data/transactions.csv")

# Merge for enriched view
df = df_trx.merge(df_ent, on="entreprise_id", how="left")

# ── Encode categoricals ────────────────────────────────────────────────────────

le_ville = LabelEncoder()
le_hist  = LabelEncoder()
le_tech  = LabelEncoder()
le_sect  = LabelEncoder()
le_pays  = LabelEncoder()

df_ent["ville_enc"] = le_ville.fit_transform(df_ent["ville"])
df_ent["hist_enc"]  = le_hist.fit_transform(df_ent["historique_paiement"])
df_ent["sect_enc"]  = le_sect.fit_transform(df_ent["secteur_activite"])

df["tech_enc"]  = le_tech.fit_transform(df["technique_paiement"])
df["hist_enc"]  = le_hist.transform(df["historique_paiement"])
df["ville_enc"] = le_ville.transform(df["ville"])
df["sect_enc"]  = le_sect.transform(df["secteur_activite"])
df["pays_enc"]  = le_pays.fit_transform(df["pays"])

# ══════════════════════════════════════════════════════════════════════════════
# 1.  COMPANY ECO-PROFILE IDENTIFICATION  (K-Means)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 1. Company eco-profiles (K-Means) ─────────────────────────────────")

# Per-company aggregates
agg = (
    df.groupby("entreprise_id")
    .agg(
        nb_transactions       = ("transaction_id", "count"),
        montant_moyen         = ("montant", "mean"),
        taux_retard           = ("is_late", "mean"),
        retard_moyen          = ("retard", "mean"),
        taux_fraude           = ("fraude_suspectee", "mean"),
        taux_risque_op        = ("risque_operationnel", "mean"),
        score_aml_moyen       = ("score_aml", "mean"),
        notation_pays_moyenne = ("notation_pays", "mean"),
    )
    .reset_index()
)
agg = agg.merge(df_ent, on="entreprise_id")

features_clust = [
    "ligne_de_credit",
    "pct_utilisation_credit",
    "nb_transactions_historique",
    "nb_transactions",
    "montant_moyen",
    "taux_retard",
    "retard_moyen",
    "taux_fraude",
    "score_aml_moyen",
    "hist_enc",
    "ville_enc",
    "sect_enc",
]

X_clust = agg[features_clust].fillna(0)
scaler_clust = StandardScaler()
X_clust_scaled = scaler_clust.fit_transform(X_clust)

# Elbow method
inertias = []
k_range = range(2, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_clust_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(k_range), inertias, marker="o")
ax.set_xlabel("Nombre de clusters (k)")
ax.set_ylabel("Inertie")
ax.set_title("Méthode du coude – profils entreprises")
fig.tight_layout()
fig.savefig("outputs/elbow_kmeans.png", dpi=120)
plt.close(fig)

# Fit with k=4
K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10)
agg["cluster"] = km.fit_predict(X_clust_scaled)

profile_summary = (
    agg.groupby("cluster")
    .agg(
        nb_entreprises    = ("entreprise_id", "count"),
        ligne_credit_moy  = ("ligne_de_credit", "mean"),
        utilisation_moy   = ("pct_utilisation_credit", "mean"),
        taux_retard_moy   = ("taux_retard", "mean"),
        taux_fraude_moy   = ("taux_fraude", "mean"),
        score_aml_moy     = ("score_aml_moyen", "mean"),
        montant_moyen     = ("montant_moyen", "mean"),
    )
    .round(2)
)
print(profile_summary.to_string())
profile_summary.to_csv("outputs/profils_entreprises.csv")

# Cluster label map
cluster_labels = build_cluster_labels(profile_summary)

agg["profil"] = agg["cluster"].map(cluster_labels)
agg[["entreprise_id", "cluster", "profil"]].to_csv(
    "outputs/entreprises_profils.csv", index=False
)
print("\nProfils attribués :", agg["profil"].value_counts().to_dict())

# ══════════════════════════════════════════════════════════════════════════════
# 2.  TRANSACTIONAL RISK MODELLING  (Random Forest – is_late)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 2. Transactional risk model (Random Forest – retard) ──────────────")

features_rf = [
    "montant",
    "delai_prevu",
    "notation_pays",
    "score_risque_pays",
    "score_aml",
    "tech_enc",
    "hist_enc",
    "ville_enc",
    "sect_enc",
    "ligne_de_credit",
    "pct_utilisation_credit",
    "nb_transactions_historique",
]

X = df[features_rf].fillna(0)
y = df["is_late"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["À temps", "En retard"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["À temps", "En retard"],
    yticklabels=["À temps", "En retard"],
    ax=ax,
)
ax.set_xlabel("Prédit")
ax.set_ylabel("Réel")
ax.set_title("Matrice de confusion – modèle de retard")
fig.tight_layout()
fig.savefig("outputs/confusion_matrix.png", dpi=120)
plt.close(fig)

# Feature importances
importances = pd.Series(rf.feature_importances_, index=features_rf).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 5))
importances.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Importance des variables – risque transactionnel (retard)")
ax.set_xlabel("Importance (Gini)")
fig.tight_layout()
fig.savefig("outputs/feature_importances.png", dpi=120)
plt.close(fig)

# Save predictions
df_test = X_test.copy()
df_test["is_late_reel"]   = y_test.values
df_test["is_late_predit"] = y_pred
df_test.to_csv("outputs/predictions_risque.csv", index=False)
print(f"Predictions saved → outputs/predictions_risque.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  FRAUD / AML MODEL  (Random Forest – fraude_suspectee)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 3. Fraud / AML model (Random Forest – fraude_suspectee) ──────────")

features_fraud = [
    "montant",
    "notation_pays",
    "score_risque_pays",
    "score_aml",
    "tech_enc",
    "hist_enc",
    "pays_enc",
    "ligne_de_credit",
    "pct_utilisation_credit",
    "risque_operationnel",
    "evolution_montant_pct",
    "rang_transaction_entreprise",
]

# evolution_montant_pct is NaN for first transactions → fill with 0
df_fraud = df.copy()
df_fraud["evolution_montant_pct"] = df_fraud["evolution_montant_pct"].fillna(0)

X_f = df_fraud[features_fraud].fillna(0)
y_f = df_fraud["fraude_suspectee"]

X_ftrain, X_ftest, y_ftrain, y_ftest = train_test_split(
    X_f, y_f, test_size=0.25, random_state=42, stratify=y_f
)

rf_fraud = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
rf_fraud.fit(X_ftrain, y_ftrain)
y_fpred = rf_fraud.predict(X_ftest)

print(classification_report(y_ftest, y_fpred, target_names=["Normal", "Fraude suspectée"]))

# Fraud feature importances
imp_fraud = pd.Series(rf_fraud.feature_importances_, index=features_fraud).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 5))
imp_fraud.plot(kind="barh", ax=ax, color="tomato")
ax.set_title("Importance des variables – modèle fraude / AML")
ax.set_xlabel("Importance (Gini)")
fig.tight_layout()
fig.savefig("outputs/feature_importances_fraude.png", dpi=120)
plt.close(fig)

# AML score by country
aml_by_country = (
    df.groupby("pays")
    .agg(
        score_aml_moyen  = ("score_aml", "mean"),
        taux_fraude      = ("fraude_suspectee", "mean"),
        notation_pays    = ("notation_pays", "first"),
    )
    .sort_values("score_aml_moyen", ascending=True)
    .round(2)
)
aml_by_country["libelle_risque"] = aml_by_country["notation_pays"].map(NOTATION_LABELS)
print("\nAML score moyen et taux de fraude par pays :")
print(aml_by_country.to_string())
aml_by_country.to_csv("outputs/aml_par_pays.csv")

fig, ax = plt.subplots(figsize=(10, 5))
colors = aml_by_country["notation_pays"].map(
    {1: "#4caf50", 2: "#8bc34a", 3: "#ff9800", 4: "#f44336", 5: "#9c27b0"}
)
ax.barh(aml_by_country.index, aml_by_country["score_aml_moyen"], color=colors)
ax.set_xlabel("Score AML moyen")
ax.set_title("Score AML moyen et risque pays")
fig.tight_layout()
fig.savefig("outputs/aml_par_pays.png", dpi=120)
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 4.  ANOMALY DETECTION  (Isolation Forest)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 4. Anomaly detection (Isolation Forest) ───────────────────────────")

features_iso = [
    "montant",
    "delai_reel",
    "retard",
    "score_aml",
    "notation_pays",
    "ligne_de_credit",
    "pct_utilisation_credit",
    "evolution_montant_pct",
]
df_iso = df.copy()
df_iso["evolution_montant_pct"] = df_iso["evolution_montant_pct"].fillna(0)

X_iso = df_iso[features_iso].fillna(0)
scaler_iso = StandardScaler()
X_iso_scaled = scaler_iso.fit_transform(X_iso)

iso = IsolationForest(contamination=0.05, random_state=42)
df["anomalie"]      = iso.fit_predict(X_iso_scaled)   # -1 = anomaly, 1 = normal
df["anomalie_flag"] = (df["anomalie"] == -1).astype(int)

n_anomalies = df["anomalie_flag"].sum()
print(f"Anomalies détectées : {n_anomalies} / {len(df)} transactions ({n_anomalies/len(df)*100:.1f} %)")

anomalies = df[df["anomalie_flag"] == 1][
    [
        "transaction_id", "entreprise_id", "pays", "montant",
        "technique_paiement", "notation_pays", "score_aml",
        "delai_reel", "retard", "historique_paiement", "ville",
        "fraude_suspectee", "risque_operationnel",
    ]
]
anomalies.to_csv("outputs/anomalies.csv", index=False)
print(f"Anomaly details saved → outputs/anomalies.csv")

# Amount distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df.loc[df["anomalie_flag"] == 0, "montant"], bins=50, alpha=0.6, label="Normal",   color="steelblue")
ax.hist(df.loc[df["anomalie_flag"] == 1, "montant"], bins=50, alpha=0.8, label="Anomalie", color="tomato")
ax.set_xlabel("Montant (MAD)")
ax.set_ylabel("Fréquence")
ax.set_title("Distribution des montants – normal vs anomalie")
ax.legend()
ax.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
)
fig.tight_layout()
fig.savefig("outputs/anomalies_montants.png", dpi=120)
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRANSACTION-AMOUNT EVOLUTION  (per-company trend)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 5. Transaction-amount evolution ───────────────────────────────────")

# Companies with most transactions
top_companies = (
    df.groupby("entreprise_id")["transaction_id"].count()
    .nlargest(6)
    .index.tolist()
)

fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
axes = axes.flatten()

for idx, eid in enumerate(top_companies):
    sub = (
        df[df["entreprise_id"] == eid]
        .sort_values("date_ouverture")
        .reset_index(drop=True)
    )
    axes[idx].plot(
        range(1, len(sub) + 1),
        sub["montant"] / 1_000,
        marker="o",
        linewidth=1.5,
        color="steelblue",
    )
    # Mark fraud / anomaly points
    fraud_idx  = sub.index[sub["fraude_suspectee"] == 1].tolist()
    anomaly_idx = sub.index[sub["anomalie_flag"] == 1].tolist()
    if fraud_idx:
        axes[idx].scatter(
            [i + 1 for i in fraud_idx],
            sub.loc[fraud_idx, "montant"] / 1_000,
            color="tomato", zorder=5, label="Fraude", s=60,
        )
    if anomaly_idx:
        axes[idx].scatter(
            [i + 1 for i in anomaly_idx],
            sub.loc[anomaly_idx, "montant"] / 1_000,
            color="orange", zorder=4, label="Anomalie", s=40, marker="^",
        )
    axes[idx].set_title(f"{eid} ({len(sub)} txn)")
    axes[idx].set_xlabel("Rang transaction")
    axes[idx].set_ylabel("Montant (k MAD)")
    axes[idx].legend(fontsize=7)

fig.suptitle("Évolution des montants de transaction par entreprise", fontsize=12)
fig.tight_layout()
fig.savefig("outputs/evolution_montants.png", dpi=120)
plt.close(fig)
print(f"Evolution plot saved → outputs/evolution_montants.png")

# ── Summary ────────────────────────────────────────────────────────────────────

print("\n══ Analyse complète ══════════════════════════════════════════════════")
print(f"  Entreprises analysées   : {len(df_ent)}")
print(f"  Transactions analysées  : {len(df_trx)}")
print(f"  Profils identifiés      : {K}")
print(f"  Anomalies détectées     : {n_anomalies} ({n_anomalies/len(df)*100:.1f} %)")
print(f"  Fraudes suspectées      : {df['fraude_suspectee'].sum()} ({df['fraude_suspectee'].mean()*100:.1f} %)")
print(f"  Risques opérationnels   : {df['risque_operationnel'].sum()} ({df['risque_operationnel'].mean()*100:.1f} %)")
print("  Outputs → outputs/")
