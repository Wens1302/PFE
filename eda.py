"""
eda.py
------
Analyse Exploratoire des Données (EDA) – Trade Finance Risk (BOA Maroc)

Sections
--------
  1. Vue d'ensemble (forme, valeurs manquantes, statistiques descriptives)
  2. Profil des entreprises (univarié)
  3. Transactions – distributions univariées
  4. Variables cibles  (is_late, fraude_suspectee, risque_operationnel, score_aml)
  5. Matrice de corrélation
  6. Analyse risque pays (notation ↔ retard, fraude, risque opérationnel)
  7. Évolution temporelle
  8. Évolution des montants par entreprise (Customer 360)
  9. Analyse croisée multi-variables

Usage
-----
    python eda.py

Prerequisite: run generate_dataset.py first.
Outputs: outputs/eda/  (PNG charts + CSV tables)
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Output directory ───────────────────────────────────────────────────────────
OUT = "outputs/eda"
os.makedirs(OUT, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
BLUE   = "#2196F3"
ORANGE = "#FF9800"
RED    = "#F44336"
GREEN  = "#4CAF50"
PURPLE = "#9C27B0"
GREY   = "#9E9E9E"

NOTATION_LABELS = {1: "Très faible", 2: "Faible", 3: "Modéré", 4: "Élevé", 5: "Très élevé"}
NOTATION_COLORS = {1: GREEN, 2: "#8BC34A", 3: ORANGE, 4: RED, 5: PURPLE}

def fmt_mad(x, _):
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.0f}k"
    return str(int(x))

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading datasets …")
df_ent = pd.read_csv("data/entreprises.csv")
df_trx = pd.read_csv("data/transactions.csv", parse_dates=["date_ouverture", "date_prevue", "date_effective"])
df = df_trx.merge(df_ent, on="entreprise_id", how="left")
df["annee"]  = df["date_ouverture"].dt.year
df["mois"]   = df["date_ouverture"].dt.to_period("M").astype(str)
df["notation_label"] = df["notation_pays"].map(NOTATION_LABELS)

# ══════════════════════════════════════════════════════════════════════════════
# 1. VUE D'ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 1. Vue d'ensemble ──────────────────────────────────────────────────")

# ── Statistiques descriptives ──────────────────────────────────────────────────
for name, frame in [("entreprises", df_ent), ("transactions", df_trx)]:
    stats = frame.describe(include="all").T
    stats.to_csv(f"{OUT}/stats_descriptives_{name}.csv")
    print(f"  Stats {name} → {OUT}/stats_descriptives_{name}.csv")

# ── Valeurs manquantes ─────────────────────────────────────────────────────────
missing = df_trx.isnull().sum().rename("manquants").to_frame()
missing["pct"] = (missing["manquants"] / len(df_trx) * 100).round(1)
missing = missing[missing["manquants"] > 0]
if not missing.empty:
    fig, ax = plt.subplots(figsize=(6, max(2, len(missing) * 0.5)))
    missing["pct"].plot(kind="barh", ax=ax, color=ORANGE)
    ax.set_xlabel("% valeurs manquantes")
    ax.set_title("Valeurs manquantes – transactions")
    for bar in ax.patches:
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/01_valeurs_manquantes.png", dpi=120)
    plt.close(fig)
    print(f"  Missing values plot → {OUT}/01_valeurs_manquantes.png")
else:
    print("  Aucune valeur manquante significative dans transactions.csv")

print(f"\n  Entreprises : {len(df_ent)} lignes × {df_ent.shape[1]} colonnes")
print(f"  Transactions: {len(df_trx)} lignes × {df_trx.shape[1]} colonnes")
print(f"  Transactions/entreprise — min: {df_trx.groupby('entreprise_id').size().min()}, "
      f"moy: {df_trx.groupby('entreprise_id').size().mean():.1f}, "
      f"max: {df_trx.groupby('entreprise_id').size().max()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. PROFIL DES ENTREPRISES  (univarié)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 2. Profil des entreprises ──────────────────────────────────────────")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 2a – Secteur d'activité
sect_counts = df_ent["secteur_activite"].value_counts()
axes[0, 0].barh(sect_counts.index, sect_counts.values, color=BLUE)
axes[0, 0].set_title("Secteurs d'activité")
axes[0, 0].set_xlabel("Nombre d'entreprises")
for i, v in enumerate(sect_counts.values):
    axes[0, 0].text(v + 0.2, i, str(v), va="center", fontsize=8)

# 2b – Ville
ville_counts = df_ent["ville"].value_counts()
axes[0, 1].pie(ville_counts.values, labels=ville_counts.index,
               autopct="%1.1f%%", colors=[BLUE, ORANGE],
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[0, 1].set_title("Répartition par ville")

# 2c – Historique paiement
order_hist = ["Excellent", "Bon", "Moyen", "Mauvais"]
hist_counts = df_ent["historique_paiement"].value_counts().reindex(order_hist)
bar_colors  = [GREEN, BLUE, ORANGE, RED]
axes[0, 2].bar(hist_counts.index, hist_counts.values, color=bar_colors)
axes[0, 2].set_title("Historique de paiement")
axes[0, 2].set_ylabel("Nombre d'entreprises")
for i, v in enumerate(hist_counts.values):
    axes[0, 2].text(i, v + 0.3, str(v), ha="center", fontsize=9, fontweight="bold")

# 2d – Ligne de crédit
axes[1, 0].hist(df_ent["ligne_de_credit"] / 1e6, bins=20, color=BLUE, edgecolor="white")
axes[1, 0].set_title("Distribution ligne de crédit")
axes[1, 0].set_xlabel("Ligne de crédit (M MAD)")
axes[1, 0].set_ylabel("Fréquence")

# 2e – % utilisation crédit
axes[1, 1].hist(df_ent["pct_utilisation_credit"], bins=20, color=ORANGE, edgecolor="white")
axes[1, 1].set_title("% Utilisation du crédit")
axes[1, 1].set_xlabel("Utilisation (%)")
axes[1, 1].set_ylabel("Fréquence")

# 2f – Nb transactions historiques
axes[1, 2].hist(df_ent["nb_transactions_historique"], bins=20, color=PURPLE, edgecolor="white")
axes[1, 2].set_title("Transactions historiques (avant période)")
axes[1, 2].set_xlabel("Nombre de transactions")
axes[1, 2].set_ylabel("Fréquence")

fig.suptitle("Profil des entreprises – analyse univariée", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/02_profil_entreprises.png", dpi=120)
plt.close(fig)
print(f"  → {OUT}/02_profil_entreprises.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRANSACTIONS – distributions univariées
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 3. Transactions – distributions ────────────────────────────────────")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 3a – Montant
axes[0, 0].hist(df_trx["montant"] / 1e6, bins=40, color=BLUE, edgecolor="white")
axes[0, 0].set_title("Distribution des montants")
axes[0, 0].set_xlabel("Montant (M MAD)")
axes[0, 0].set_ylabel("Fréquence")
axes[0, 0].axvline(df_trx["montant"].median() / 1e6, color=RED, linestyle="--",
                   linewidth=1.5, label=f"Médiane {df_trx['montant'].median()/1e6:.2f}M")
axes[0, 0].legend(fontsize=8)

# 3b – Technique de paiement
tech_counts = df_trx["technique_paiement"].value_counts()
axes[0, 1].pie(tech_counts.values, labels=tech_counts.index,
               autopct="%1.1f%%", colors=[BLUE, ORANGE, GREEN],
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[0, 1].set_title("Techniques de paiement")

# 3c – Notation pays
not_counts = df_trx["notation_pays"].value_counts().sort_index()
not_labels = [f"{k} – {NOTATION_LABELS[k]}" for k in not_counts.index]
not_colors = [NOTATION_COLORS.get(k, GREY) for k in not_counts.index]
bars = axes[0, 2].bar(range(len(not_counts)), not_counts.values, color=not_colors, edgecolor="white")
axes[0, 2].set_xticks(range(len(not_counts)))
axes[0, 2].set_xticklabels(not_labels, rotation=20, ha="right", fontsize=8)
axes[0, 2].set_title("Notation risque pays")
axes[0, 2].set_ylabel("Nombre de transactions")
for bar in bars:
    axes[0, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(int(bar.get_height())), ha="center", fontsize=8)

# 3d – Délai prévu vs réel (boxplot)
df_delays = pd.DataFrame({"Délai prévu": df_trx["delai_prevu"],
                           "Délai réel":  df_trx["delai_reel"]})
axes[1, 0].boxplot([df_delays["Délai prévu"], df_delays["Délai réel"]],
                   labels=["Délai prévu", "Délai réel"],
                   patch_artist=True,
                   boxprops=dict(facecolor=BLUE, alpha=0.6),
                   medianprops=dict(color=RED, linewidth=2))
axes[1, 0].set_title("Délais prévus vs réels (jours)")
axes[1, 0].set_ylabel("Jours")

# 3e – Score AML
axes[1, 1].hist(df_trx["score_aml"], bins=30, color=RED, edgecolor="white", alpha=0.8)
axes[1, 1].set_title("Distribution du score AML")
axes[1, 1].set_xlabel("Score AML (0–100)")
axes[1, 1].set_ylabel("Fréquence")
axes[1, 1].axvline(df_trx["score_aml"].mean(), color="black", linestyle="--",
                   linewidth=1.5, label=f"Moyenne {df_trx['score_aml'].mean():.1f}")
axes[1, 1].legend(fontsize=8)

# 3f – Rang transaction par entreprise
axes[1, 2].hist(df_trx["rang_transaction_entreprise"], bins=range(1, 17), color=PURPLE,
                edgecolor="white", rwidth=0.8)
axes[1, 2].set_title("Rang des transactions par entreprise")
axes[1, 2].set_xlabel("Rang (1 = première transaction)")
axes[1, 2].set_ylabel("Nombre de transactions")

fig.suptitle("Transactions – analyse univariée", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/03_transactions_univarie.png", dpi=120)
plt.close(fig)
print(f"  → {OUT}/03_transactions_univarie.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. VARIABLES CIBLES
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 4. Variables cibles ────────────────────────────────────────────────")

targets = {
    "is_late":            ("Retards de paiement",     ORANGE),
    "fraude_suspectee":   ("Fraudes suspectées",       RED),
    "risque_operationnel":("Risques opérationnels",    PURPLE),
}

# 4a – Prévalence des variables cibles
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (col, (label, color)) in zip(axes, targets.items()):
    counts = df_trx[col].value_counts().sort_index()
    labels_pie = ["Normal", label]
    colors_pie = [GREY, color]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels_pie, autopct="%1.1f%%",
        colors=colors_pie, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        startangle=90,
    )
    autotexts[1].set_fontweight("bold")
    n_pos = counts.get(1, 0)
    ax.set_title(f"{label}\n(n={n_pos} / {len(df_trx)})", fontsize=10)

fig.suptitle("Prévalence des variables cibles (classes binaires)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/04a_prevalence_cibles.png", dpi=120)
plt.close(fig)

# 4b – Score AML selon les variables cibles
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (col, (label, color)) in zip(axes, targets.items()):
    data0 = df_trx.loc[df_trx[col] == 0, "score_aml"]
    data1 = df_trx.loc[df_trx[col] == 1, "score_aml"]
    ax.hist(data0, bins=25, alpha=0.6, color=BLUE,  label="0 – Normal",  edgecolor="white")
    ax.hist(data1, bins=25, alpha=0.8, color=color, label=f"1 – {label}", edgecolor="white")
    ax.axvline(data0.mean(), color=BLUE,  linestyle="--", linewidth=1.2)
    ax.axvline(data1.mean(), color=color, linestyle="--", linewidth=1.2)
    ax.set_title(f"Score AML selon {col}")
    ax.set_xlabel("Score AML")
    ax.legend(fontsize=8)

fig.suptitle("Score AML selon les variables cibles", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/04b_aml_vs_cibles.png", dpi=120)
plt.close(fig)

# 4c – Montant selon les variables cibles (boxplot)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (col, (label, color)) in zip(axes, targets.items()):
    data = [df_trx.loc[df_trx[col] == 0, "montant"] / 1e6,
            df_trx.loc[df_trx[col] == 1, "montant"] / 1e6]
    bp = ax.boxplot(data, labels=["Normal", label], patch_artist=True,
                    boxprops=dict(alpha=0.7),
                    medianprops=dict(color=RED, linewidth=2))
    bp["boxes"][0].set_facecolor(BLUE)
    bp["boxes"][1].set_facecolor(color)
    ax.set_title(f"Montant vs {col}")
    ax.set_ylabel("Montant (M MAD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))

fig.suptitle("Montant des transactions selon les variables cibles", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/04c_montant_vs_cibles.png", dpi=120)
plt.close(fig)

print(f"  → {OUT}/04a_prevalence_cibles.png")
print(f"  → {OUT}/04b_aml_vs_cibles.png")
print(f"  → {OUT}/04c_montant_vs_cibles.png")

for col, (label, _) in targets.items():
    pct = df_trx[col].mean() * 100
    print(f"  {label:28s}: {df_trx[col].sum():>4d}  ({pct:.1f} %)")

# ══════════════════════════════════════════════════════════════════════════════
# 5. MATRICE DE CORRÉLATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 5. Matrice de corrélation ──────────────────────────────────────────")

num_cols = [
    "montant", "notation_pays", "score_risque_pays", "score_aml",
    "delai_prevu", "delai_reel", "retard", "is_late",
    "fraude_suspectee", "risque_operationnel",
    "rang_transaction_entreprise", "evolution_montant_pct",
]
corr = df_trx[num_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
    center=0, linewidths=0.5, ax=ax,
    annot_kws={"size": 8}, vmin=-1, vmax=1,
)
ax.set_title("Matrice de corrélation – variables numériques", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/05_correlation_matrix.png", dpi=120)
plt.close(fig)
corr.to_csv(f"{OUT}/05_correlation_matrix.csv")
print(f"  → {OUT}/05_correlation_matrix.png")

# Top correlations with targets
for col in ["is_late", "fraude_suspectee", "risque_operationnel", "score_aml"]:
    top = corr[col].drop(col).abs().sort_values(ascending=False).head(5)
    print(f"\n  Top corrélations avec {col}:")
    for var, r in top.items():
        print(f"    {var:35s}: r = {corr[col][var]:+.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. ANALYSE RISQUE PAYS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 6. Analyse risque pays ─────────────────────────────────────────────")

# Aggregate by pays
country_agg = (
    df.groupby(["pays", "notation_pays"])
    .agg(
        nb_transactions    = ("transaction_id", "count"),
        montant_moyen      = ("montant", "mean"),
        taux_retard        = ("is_late", "mean"),
        taux_fraude        = ("fraude_suspectee", "mean"),
        taux_risque_op     = ("risque_operationnel", "mean"),
        score_aml_moyen    = ("score_aml", "mean"),
        retard_moyen_jours = ("retard", "mean"),
    )
    .reset_index()
    .sort_values("notation_pays")
)
country_agg["libelle_risque"] = country_agg["notation_pays"].map(NOTATION_LABELS)
country_agg.to_csv(f"{OUT}/06_risque_par_pays.csv", index=False)

# 6a – Heatmap pays × indicateurs de risque
fig, ax = plt.subplots(figsize=(10, 6))
heat_data = country_agg.set_index("pays")[
    ["taux_retard", "taux_fraude", "taux_risque_op", "score_aml_moyen"]
].rename(columns={
    "taux_retard":     "Taux retard",
    "taux_fraude":     "Taux fraude",
    "taux_risque_op":  "Risque opérationnel",
    "score_aml_moyen": "Score AML moyen",
}).sort_values("Score AML moyen", ascending=False)

sns.heatmap(
    heat_data, annot=True, fmt=".2f", cmap="YlOrRd",
    linewidths=0.5, ax=ax, annot_kws={"size": 9},
)
ax.set_title("Indicateurs de risque par pays partenaire", fontsize=13, fontweight="bold")
ax.set_xlabel("")
fig.tight_layout()
fig.savefig(f"{OUT}/06a_heatmap_risque_pays.png", dpi=120)
plt.close(fig)

# 6b – Scatter notation ↔ taux fraude + retard (bubble = nb transactions)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (y_col, ylabel, color) in zip(axes, [
    ("taux_fraude", "Taux de fraude",  RED),
    ("taux_retard", "Taux de retard",  ORANGE),
]):
    for _, row in country_agg.iterrows():
        ax.scatter(
            row["notation_pays"] + np.random.uniform(-0.1, 0.1),
            row[y_col],
            s=row["nb_transactions"] * 2,
            color=NOTATION_COLORS.get(int(row["notation_pays"]), GREY),
            alpha=0.75, edgecolors="white", linewidth=0.5,
        )
        ax.annotate(
            row["pays"], (row["notation_pays"], row[y_col]),
            fontsize=7, ha="center", va="bottom",
            xytext=(0, 6), textcoords="offset points",
        )
    xticks = sorted(country_agg["notation_pays"].unique())
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{k}\n({NOTATION_LABELS[k]})" for k in xticks], fontsize=8)
    ax.set_xlabel("Notation risque pays")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Notation pays ↔ {ylabel}\n(taille bulle = nb transactions)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

fig.suptitle("Risque pays – corrélation notation ↔ indicateurs", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/06b_notation_vs_risques.png", dpi=120)
plt.close(fig)

# 6c – Barplot comparatif technique × notation
tech_not = (
    df.groupby(["technique_paiement", "notation_label"])["fraude_suspectee"]
    .mean()
    .mul(100)
    .reset_index()
    .rename(columns={"fraude_suspectee": "taux_fraude_pct"})
)
pivot_tn = tech_not.pivot(index="technique_paiement", columns="notation_label", values="taux_fraude_pct").fillna(0)
fig, ax = plt.subplots(figsize=(10, 5))
pivot_tn.plot(kind="bar", ax=ax, colormap="RdYlGn_r", edgecolor="white")
ax.set_title("Taux de fraude par technique de paiement et notation pays (%)", fontsize=12)
ax.set_xlabel("Technique de paiement")
ax.set_ylabel("Taux de fraude (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
ax.legend(title="Notation pays", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/06c_fraude_technique_notation.png", dpi=120)
plt.close(fig)

print(f"  → {OUT}/06a_heatmap_risque_pays.png")
print(f"  → {OUT}/06b_notation_vs_risques.png")
print(f"  → {OUT}/06c_fraude_technique_notation.png")
print(f"  → {OUT}/06_risque_par_pays.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 7. ÉVOLUTION TEMPORELLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 7. Évolution temporelle ────────────────────────────────────────────")

# Monthly aggregates
monthly = (
    df.groupby("mois")
    .agg(
        nb_transactions    = ("transaction_id", "count"),
        montant_total      = ("montant", "sum"),
        taux_retard        = ("is_late", "mean"),
        taux_fraude        = ("fraude_suspectee", "mean"),
        score_aml_moyen    = ("score_aml", "mean"),
    )
    .reset_index()
)
monthly["mois_dt"] = pd.to_datetime(monthly["mois"])
monthly = monthly.sort_values("mois_dt")

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# 7a – Flux mensuel (volumes)
ax = axes[0]
ax.bar(monthly["mois_dt"], monthly["nb_transactions"], color=BLUE, alpha=0.75, width=20)
ax2 = ax.twinx()
ax2.plot(monthly["mois_dt"], monthly["montant_total"] / 1e6, color=ORANGE, linewidth=2, marker="o", markersize=3)
ax.set_ylabel("Nombre de transactions", color=BLUE)
ax2.set_ylabel("Montant total (M MAD)", color=ORANGE)
ax.set_title("Flux mensuel – volume et montant des transactions")
ax.tick_params(axis="y", labelcolor=BLUE)
ax2.tick_params(axis="y", labelcolor=ORANGE)

# 7b – Taux de retard et fraude mensuel
ax = axes[1]
ax.plot(monthly["mois_dt"], monthly["taux_retard"] * 100, color=ORANGE, linewidth=2, marker="o", markersize=3, label="Taux retard (%)")
ax.plot(monthly["mois_dt"], monthly["taux_fraude"] * 100, color=RED,    linewidth=2, marker="s", markersize=3, label="Taux fraude (%)")
ax.set_ylabel("Taux (%)")
ax.set_title("Évolution mensuelle – taux de retard et de fraude")
ax.legend()
ax.axhline(monthly["taux_retard"].mean() * 100, color=ORANGE, linestyle="--", linewidth=0.8, alpha=0.6)
ax.axhline(monthly["taux_fraude"].mean()  * 100, color=RED,    linestyle="--", linewidth=0.8, alpha=0.6)

# 7c – Score AML moyen mensuel
ax = axes[2]
ax.fill_between(monthly["mois_dt"], monthly["score_aml_moyen"], alpha=0.3, color=PURPLE)
ax.plot(monthly["mois_dt"], monthly["score_aml_moyen"], color=PURPLE, linewidth=2, marker="o", markersize=3)
ax.axhline(monthly["score_aml_moyen"].mean(), color="black", linestyle="--", linewidth=1, label=f"Moyenne {monthly['score_aml_moyen'].mean():.1f}")
ax.set_ylabel("Score AML moyen")
ax.set_xlabel("Mois")
ax.set_title("Évolution mensuelle – Score AML moyen")
ax.legend(fontsize=8)

fig.suptitle("Évolution temporelle (2020–2024)", fontsize=14, fontweight="bold")
fig.autofmt_xdate(rotation=30, ha="right")
fig.tight_layout()
fig.savefig(f"{OUT}/07_evolution_temporelle.png", dpi=120)
plt.close(fig)

# 7d – Heatmap volume par mois × année
df["mois_num"] = df["date_ouverture"].dt.month
pivot_time = df.pivot_table(index="annee", columns="mois_num", values="transaction_id",
                            aggfunc="count", fill_value=0)
pivot_time.columns = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"][:len(pivot_time.columns)]
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(pivot_time, annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax,
            annot_kws={"size": 9})
ax.set_title("Volume de transactions par mois et année", fontsize=12, fontweight="bold")
ax.set_xlabel("Mois")
ax.set_ylabel("Année")
fig.tight_layout()
fig.savefig(f"{OUT}/07b_heatmap_temporal.png", dpi=120)
plt.close(fig)

print(f"  → {OUT}/07_evolution_temporelle.png")
print(f"  → {OUT}/07b_heatmap_temporal.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. ÉVOLUTION DES MONTANTS PAR ENTREPRISE  (Customer 360)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 8. Évolution montants par entreprise ───────────────────────────────")

# Distribution de l'évolution inter-transaction
evo = df["evolution_montant_pct"].dropna()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(evo.clip(-100, 200), bins=50, color=BLUE, edgecolor="white", alpha=0.8)
axes[0].axvline(0,            color="black", linestyle="--", linewidth=1.2, label="0%")
axes[0].axvline(evo.median(), color=RED,   linestyle="--", linewidth=1.5, label=f"Médiane {evo.median():.1f}%")
axes[0].set_title("Distribution de l'évolution des montants\n(clippé à [-100%, +200%])")
axes[0].set_xlabel("Évolution (%)")
axes[0].set_ylabel("Fréquence")
axes[0].legend(fontsize=8)

# Évolution moyenne selon rang
evo_rang = (
    df.groupby("rang_transaction_entreprise")["evolution_montant_pct"]
    .agg(["mean", "median", "std"])
    .reset_index()
)
axes[1].fill_between(evo_rang["rang_transaction_entreprise"],
                     evo_rang["mean"] - evo_rang["std"],
                     evo_rang["mean"] + evo_rang["std"],
                     alpha=0.2, color=BLUE, label="±1 écart-type")
axes[1].plot(evo_rang["rang_transaction_entreprise"], evo_rang["mean"],   color=BLUE,   linewidth=2, marker="o", label="Moyenne")
axes[1].plot(evo_rang["rang_transaction_entreprise"], evo_rang["median"], color=ORANGE, linewidth=1.5, linestyle="--", label="Médiane")
axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[1].set_xlabel("Rang de la transaction (1=première)")
axes[1].set_ylabel("Évolution montant (%)")
axes[1].set_title("Évolution moyenne des montants selon le rang\n(maturité de la relation client)")
axes[1].legend(fontsize=8)
axes[1].set_xticks(evo_rang["rang_transaction_entreprise"])

fig.suptitle("Évolution des montants de transaction par entreprise", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/08a_evolution_montants_distribution.png", dpi=120)
plt.close(fig)

# Évolution % moyen par historique de paiement
fig, ax = plt.subplots(figsize=(10, 5))
for hist_label, color in zip(["Excellent", "Bon", "Moyen", "Mauvais"],
                              [GREEN, BLUE, ORANGE, RED]):
    sub = df[df["historique_paiement"] == hist_label]
    evo_sub = (
        sub.groupby("rang_transaction_entreprise")["evolution_montant_pct"]
        .mean()
    )
    ax.plot(evo_sub.index, evo_sub.values, color=color, linewidth=2,
            marker="o", markersize=4, label=hist_label)

ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_xlabel("Rang de la transaction")
ax.set_ylabel("Évolution montant moyen (%)")
ax.set_title("Évolution des montants selon l'historique de paiement\n(comportement client par rang)")
ax.legend(title="Historique paiement", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/08b_evolution_par_historique.png", dpi=120)
plt.close(fig)

print(f"  → {OUT}/08a_evolution_montants_distribution.png")
print(f"  → {OUT}/08b_evolution_par_historique.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. ANALYSE CROISÉE MULTI-VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 9. Analyse croisée multi-variables ─────────────────────────────────")

# 9a – Historique paiement × variables cibles
cross_hist = (
    df.groupby("historique_paiement")[["is_late", "fraude_suspectee", "risque_operationnel"]]
    .mean()
    .mul(100)
    .reindex(["Excellent", "Bon", "Moyen", "Mauvais"])
    .rename(columns={"is_late": "Retard (%)", "fraude_suspectee": "Fraude (%)", "risque_operationnel": "Risque op. (%)"})
)
cross_hist.to_csv(f"{OUT}/09_cross_hist_cibles.csv")

fig, ax = plt.subplots(figsize=(9, 5))
cross_hist.plot(kind="bar", ax=ax, color=[ORANGE, RED, PURPLE], edgecolor="white", width=0.7)
ax.set_title("Taux de risque selon l'historique de paiement (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("Historique de paiement")
ax.set_ylabel("Taux (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(fontsize=9)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=7.5)
fig.tight_layout()
fig.savefig(f"{OUT}/09a_historique_vs_risques.png", dpi=120)
plt.close(fig)

# 9b – Secteur × montant moyen + taux retard
cross_sect = (
    df.groupby("secteur_activite")
    .agg(montant_moyen=("montant", "mean"), taux_retard=("is_late", "mean"))
    .sort_values("taux_retard", ascending=False)
)
fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()
x = range(len(cross_sect))
ax1.bar(x, cross_sect["montant_moyen"] / 1e6, color=BLUE, alpha=0.7, label="Montant moyen (M MAD)")
ax2.plot(x, cross_sect["taux_retard"] * 100, color=ORANGE, linewidth=2, marker="o", markersize=6, label="Taux retard (%)")
ax1.set_xticks(x)
ax1.set_xticklabels(cross_sect.index, rotation=30, ha="right", fontsize=8)
ax1.set_ylabel("Montant moyen (M MAD)", color=BLUE)
ax2.set_ylabel("Taux de retard (%)", color=ORANGE)
ax1.tick_params(axis="y", labelcolor=BLUE)
ax2.tick_params(axis="y", labelcolor=ORANGE)
ax1.set_title("Secteur d'activité – montant moyen et taux de retard", fontsize=12, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/09b_secteur_vs_risques.png", dpi=120)
plt.close(fig)

# 9c – Violin: montant vs is_late × technique de paiement
fig, ax = plt.subplots(figsize=(12, 5))
plot_data = df.copy()
plot_data["Statut"] = plot_data["is_late"].map({0: "À temps", 1: "En retard"})
parts = ax.violinplot(
    [
        plot_data.loc[(plot_data["technique_paiement"] == t) & (plot_data["is_late"] == s), "montant"] / 1e6
        for t in ["Transfert", "Remise documentaire", "Crédit documentaire"]
        for s in [0, 1]
    ],
    positions=list(range(6)),
    showmedians=True,
    widths=0.75,
)
colors_v = [BLUE, ORANGE] * 3
for pc, c in zip(parts["bodies"], colors_v):
    pc.set_facecolor(c)
    pc.set_alpha(0.6)
ax.set_xticks(range(6))
ax.set_xticklabels([
    "Transfert\nÀ temps", "Transfert\nEn retard",
    "Remise doc.\nÀ temps", "Remise doc.\nEn retard",
    "Crédit doc.\nÀ temps", "Crédit doc.\nEn retard",
], fontsize=8)
ax.set_ylabel("Montant (M MAD)")
ax.set_title("Distribution des montants par technique × statut retard", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/09c_violin_montant_technique_retard.png", dpi=120)
plt.close(fig)

# 9d – Tableau récapitulatif complet par pays
summary_pays = country_agg[[
    "pays", "notation_pays", "libelle_risque", "nb_transactions",
    "montant_moyen", "taux_retard", "taux_fraude", "taux_risque_op",
    "score_aml_moyen", "retard_moyen_jours",
]].copy()
summary_pays["montant_moyen"] = summary_pays["montant_moyen"].round(0).astype(int)
summary_pays["taux_retard"]   = (summary_pays["taux_retard"]  * 100).round(1)
summary_pays["taux_fraude"]   = (summary_pays["taux_fraude"]  * 100).round(1)
summary_pays["taux_risque_op"]= (summary_pays["taux_risque_op"]* 100).round(1)
summary_pays["score_aml_moyen"]= summary_pays["score_aml_moyen"].round(1)
summary_pays["retard_moyen_jours"] = summary_pays["retard_moyen_jours"].round(1)
summary_pays.columns = [
    "Pays", "Notation", "Libellé risque", "Nb transactions",
    "Montant moyen (MAD)", "Taux retard (%)", "Taux fraude (%)",
    "Risque opérationnel (%)", "Score AML moyen", "Retard moyen (j)",
]
summary_pays.to_csv(f"{OUT}/09d_summary_par_pays.csv", index=False)

print(f"  → {OUT}/09a_historique_vs_risques.png")
print(f"  → {OUT}/09b_secteur_vs_risques.png")
print(f"  → {OUT}/09c_violin_montant_technique_retard.png")
print(f"  → {OUT}/09d_summary_par_pays.csv")

# ══════════════════════════════════════════════════════════════════════════════
# RÉCAPITULATIF DES FICHIERS PRODUITS
# ══════════════════════════════════════════════════════════════════════════════
print("\n══ EDA terminée ══════════════════════════════════════════════════════")
produced = sorted(os.listdir(OUT))
print(f"  {len(produced)} fichiers dans {OUT}/")
for f in produced:
    print(f"    {f}")
