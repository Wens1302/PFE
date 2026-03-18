"""
eda.py
------
Exploratory Data Analysis (EDA) – Trade Finance Datasets
=========================================================
Performs a rigorous, structured EDA on the two datasets produced by
generate_dataset.py:
  • data/entreprises.csv   (company-level features)
  • data/transactions.csv  (transaction-level features)

Outputs
-------
All figures and summary tables are written to ``outputs/``:

  eda_01_missing_values.png
  eda_02_entreprises_numeriques.png
  eda_03_entreprises_categoriques.png
  eda_04_transactions_montant.png
  eda_05_transactions_delais.png
  eda_05b_is_late_pie.png
  eda_06_transactions_categoriques.png
  eda_07_time_series.png
  eda_08_correlation_heatmap.png
  eda_09_bivariate_entreprises.png
  eda_10_bivariate_is_late.png
  eda_11_bivariate_montant.png
  eda_12_outliers_boxplots.png
  eda_13_cible_croise.png
  eda_summary.csv
  eda_outliers.csv

Usage
-----
    python generate_dataset.py   # generate data/ first (if not done)
    python eda.py
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Setup ──────────────────────────────────────────────────────────────────────

os.makedirs("outputs", exist_ok=True)

PALETTE = "Set2"
FIG_DPI = 120

sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.0)

# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading datasets …")
df_ent = pd.read_csv("data/entreprises.csv")
df_trx = pd.read_csv("data/transactions.csv", parse_dates=["date_ouverture", "date_prevue", "date_effective"])

# Merged view (transactions enriched with company attributes)
df = df_trx.merge(df_ent, on="entreprise_id", how="left")

print(f"  entreprises  : {df_ent.shape[0]} rows × {df_ent.shape[1]} columns")
print(f"  transactions : {df_trx.shape[0]} rows × {df_trx.shape[1]} columns")
print(f"  merged       : {df.shape[0]} rows × {df.shape[1]} columns")


# ══════════════════════════════════════════════════════════════════════════════
# 0.  DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 0. Dataset overview ───────────────────────────────────────────────")

for name, frame in [("Entreprises", df_ent), ("Transactions", df_trx)]:
    print(f"\n{name}")
    print(f"  Shape          : {frame.shape}")
    print(f"  Dtypes:\n{frame.dtypes.to_string()}")
    print(f"\n  describe():\n{frame.describe(include='all').T.to_string()}")
    dup = frame.duplicated().sum()
    print(f"\n  Duplicate rows : {dup}")


# ── Figure 1: Missing values ────────────────────────────────────────────────

def plot_missing(frame, title, ax):
    missing = frame.isnull().sum().sort_values(ascending=False)
    pct = (missing / len(frame) * 100).round(2)
    colors = ["tomato" if v > 0 else "steelblue" for v in missing]
    ax.barh(missing.index, pct, color=colors)
    ax.set_xlabel("% valeurs manquantes")
    ax.set_title(title)
    for i, (v, p) in enumerate(zip(missing, pct)):
        ax.text(p + 0.2, i, f"{v} ({p:.1f}%)", va="center", fontsize=8)

fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df_ent.columns) * 0.4 + 1)))
plot_missing(df_ent, "Valeurs manquantes – Entreprises", axes[0])
plot_missing(df_trx, "Valeurs manquantes – Transactions", axes[1])
fig.suptitle("Analyse des valeurs manquantes", fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/eda_01_missing_values.png", dpi=FIG_DPI)
plt.close(fig)
print("  → outputs/eda_01_missing_values.png")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ENTREPRISES – UNIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 1. Entreprises – analyse univariée ────────────────────────────────")

num_ent = ["ligne_de_credit", "pct_utilisation_credit", "nb_transactions_historique"]

fig, axes = plt.subplots(2, len(num_ent), figsize=(15, 8))
for j, col in enumerate(num_ent):
    series = df_ent[col].dropna()

    # Histogram + KDE
    ax_hist = axes[0, j]
    ax_hist.hist(series, bins=25, color=sns.color_palette(PALETTE)[j], alpha=0.75, edgecolor="white")
    ax2 = ax_hist.twinx()
    series.plot.kde(ax=ax2, color="black", lw=1.5)
    ax2.set_ylabel("Densité", fontsize=8)
    ax_hist.set_title(col)
    ax_hist.set_xlabel("")
    mu, sigma = series.mean(), series.std()
    ax_hist.axvline(mu, color="red", linestyle="--", lw=1.2, label=f"μ={mu:,.0f}")
    ax_hist.axvline(mu + sigma, color="orange", linestyle=":", lw=1)
    ax_hist.axvline(mu - sigma, color="orange", linestyle=":", lw=1)
    ax_hist.legend(fontsize=7)

    # Boxplot
    ax_box = axes[1, j]
    ax_box.boxplot(series, vert=False, patch_artist=True,
                   boxprops=dict(facecolor=sns.color_palette(PALETTE)[j], alpha=0.7))
    ax_box.set_yticks([])
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    n_out = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
    ax_box.set_xlabel(f"Q1={q1:,.0f}  Q3={q3:,.0f}  IQR={iqr:,.0f}  outliers={n_out}", fontsize=8)

fig.suptitle("Entreprises – Variables numériques (histogramme & boîte à moustaches)", fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/eda_02_entreprises_numeriques.png", dpi=FIG_DPI)
plt.close(fig)
print("  → outputs/eda_02_entreprises_numeriques.png")

# Categorical variables
cat_ent = ["secteur_activite", "historique_paiement", "ville"]

fig, axes = plt.subplots(1, len(cat_ent), figsize=(16, 5))
for j, col in enumerate(cat_ent):
    vc = df_ent[col].value_counts()
    colors = sns.color_palette(PALETTE, len(vc))
    axes[j].barh(vc.index, vc.values, color=colors)
    axes[j].set_title(col)
    axes[j].set_xlabel("Nombre d'entreprises")
    for i, v in enumerate(vc.values):
        axes[j].text(v + 0.3, i, f"{v} ({v/len(df_ent)*100:.1f}%)", va="center", fontsize=8)

fig.suptitle("Entreprises – Variables catégorielles", fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/eda_03_entreprises_categoriques.png", dpi=FIG_DPI)
plt.close(fig)
print("  → outputs/eda_03_entreprises_categoriques.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TRANSACTIONS – UNIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 2. Transactions – analyse univariée ───────────────────────────────")

# ── Montant distribution + Q-Q plot ─────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Histogram
axes[0].hist(df_trx["montant"], bins=40, color="steelblue", alpha=0.8, edgecolor="white")
axes[0].set_title("Distribution du montant (MAD)")
axes[0].set_xlabel("Montant (MAD)")
axes[0].set_ylabel("Fréquence")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
mu_m = df_trx["montant"].mean()
axes[0].axvline(mu_m, color="red", linestyle="--", lw=1.5, label=f"μ = {mu_m:,.0f}")
axes[0].legend(fontsize=8)

# Log-scale histogram
axes[1].hist(np.log1p(df_trx["montant"]), bins=40, color="teal", alpha=0.8, edgecolor="white")
axes[1].set_title("Distribution du montant (log-scale)")
axes[1].set_xlabel("log(1 + Montant)")
axes[1].set_ylabel("Fréquence")

# Q-Q plot against normal
(osm, osr), (slope, intercept, r) = stats.probplot(df_trx["montant"], dist="norm")
axes[2].scatter(osm, osr, s=5, alpha=0.5, color="steelblue")
axes[2].plot(
    [min(osm), max(osm)],
    [slope * min(osm) + intercept, slope * max(osm) + intercept],
    color="red", lw=1.5
)
axes[2].set_title(f"Q-Q Plot – montant vs Normale (r={r:.3f})")
axes[2].set_xlabel("Quantiles théoriques")
axes[2].set_ylabel("Quantiles observés")

fig.suptitle("Transactions – Distribution du montant", fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/eda_04_transactions_montant.png", dpi=FIG_DPI)
plt.close(fig)
print("  → outputs/eda_04_transactions_montant.png")

# ── Delay variables ──────────────────────────────────────────────────────────

delay_cols = ["delai_prevu", "delai_reel", "retard"]
fig, axes = plt.subplots(2, len(delay_cols), figsize=(15, 8))
for j, col in enumerate(delay_cols):
    series = df_trx[col]

    ax_hist = axes[0, j]
    ax_hist.hist(series, bins=30, color=sns.color_palette(PALETTE)[j + 1], alpha=0.8, edgecolor="white")
    ax_hist.set_title(col)
    ax_hist.set_xlabel("Jours")
    ax_hist.set_ylabel("Fréquence")
    mu_d = series.mean()
    ax_hist.axvline(mu_d, color="red", linestyle="--", lw=1.2, label=f"μ={mu_d:.1f}j")
    ax_hist.legend(fontsize=8)

    ax_box = axes[1, j]
    ax_box.boxplot(series, vert=False, patch_artist=True,
                   boxprops=dict(facecolor=sns.color_palette(PALETTE)[j + 1], alpha=0.7))
    ax_box.set_yticks([])
    ax_box.set_xlabel(
        f"min={series.min():.0f}  médiane={series.median():.0f}  max={series.max():.0f}  σ={series.std():.1f}",
        fontsize=8,
    )

# is_late pie on an additional axis
fig2, ax_pie = plt.subplots(figsize=(5, 5))
vc_late = df_trx["is_late"].value_counts()
ax_pie.pie(
    vc_late,
    labels=["À temps", "En retard"],
    autopct="%1.1f%%",
    colors=["steelblue", "tomato"],
    startangle=90,
    wedgeprops={"edgecolor": "white"},
)
ax_pie.set_title("Répartition is_late")
fig2.tight_layout()
fig2.savefig("outputs/eda_05b_is_late_pie.png", dpi=FIG_DPI)
plt.close(fig2)

fig.suptitle("Transactions – Variables de délai", fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/eda_05_transactions_delais.png", dpi=FIG_DPI)
plt.close(fig)
print("  → outputs/eda_05_transactions_delais.png")

# ── Categorical variables ────────────────────────────────────────────────────

cat_trx = ["technique_paiement", "pays"]
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for j, col in enumerate(cat_trx):
    vc = df_trx[col].value_counts()
    colors = sns.color_palette(PALETTE, len(vc))
    axes[j].barh(vc.index, vc.values, color=colors)
    axes[j].set_title(col)
    axes[j].set_xlabel("Nombre de transactions")
    for i, v in enumerate(vc.values):
        axes[j].text(v + 1, i, f"{v} ({v/len(df_trx)*100:.1f}%)", va="center", fontsize=8)

fig.suptitle("Transactions – Variables catégorielles", fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/eda_06_transactions_categoriques.png", dpi=FIG_DPI)
plt.close(fig)
print("  → outputs/eda_06_transactions_categoriques.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TIME-SERIES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 3. Analyse temporelle ─────────────────────────────────────────────")

df_trx["year_month"] = df_trx["date_ouverture"].dt.to_period("M")
monthly = (
    df_trx.groupby("year_month")
    .agg(nb_transactions=("transaction_id", "count"),
         montant_total=("montant", "sum"),
         taux_retard=("is_late", "mean"))
    .reset_index()
)
monthly["year_month_dt"] = monthly["year_month"].dt.to_timestamp()

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].bar(monthly["year_month_dt"], monthly["nb_transactions"], width=20, color="steelblue", alpha=0.8)
axes[0].set_ylabel("Nb transactions")
axes[0].set_title("Volume mensuel de transactions")

axes[1].plot(monthly["year_month_dt"], monthly["montant_total"] / 1e6, color="teal", linewidth=1.5, marker="o", markersize=3)
axes[1].set_ylabel("Montant total (M MAD)")
axes[1].set_title("Montant mensuel total")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

axes[2].plot(monthly["year_month_dt"], monthly["taux_retard"] * 100, color="tomato", linewidth=1.5, marker="o", markersize=3)
axes[2].axhline(df_trx["is_late"].mean() * 100, color="grey", linestyle="--", lw=1, label="Moyenne globale")
axes[2].set_ylabel("Taux de retard (%)")
axes[2].set_title("Taux de retard mensuel")
axes[2].legend(fontsize=8)
axes[2].set_xlabel("Date")

fig.suptitle("Analyse temporelle des transactions (2020–2024)", fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/eda_07_time_series.png", dpi=FIG_DPI)
plt.close(fig)
print("  → outputs/eda_07_time_series.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 4. Corrélations ───────────────────────────────────────────────────")

# Encode categoricals for correlation
df_corr = df.copy()
for col in ["technique_paiement", "historique_paiement", "ville", "secteur_activite", "pays"]:
    df_corr[col + "_enc"] = pd.Categorical(df_corr[col]).codes

num_cols_corr = [
    "montant", "delai_prevu", "delai_reel", "retard", "is_late",
    "ligne_de_credit", "pct_utilisation_credit", "nb_transactions_historique",
    "technique_paiement_enc", "historique_paiement_enc",
]
corr_matrix = df_corr[num_cols_corr].corr()

readable_labels = {
    "montant": "Montant",
    "delai_prevu": "Délai prévu",
    "delai_reel": "Délai réel",
    "retard": "Retard",
    "is_late": "is_late",
    "ligne_de_credit": "Ligne crédit",
    "pct_utilisation_credit": "% Utilisation",
    "nb_transactions_historique": "Hist. transactions",
    "technique_paiement_enc": "Technique pmt",
    "historique_paiement_enc": "Hist. paiement",
}
corr_matrix.rename(index=readable_labels, columns=readable_labels, inplace=True)

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    mask=mask,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 9},
)
ax.set_title("Matrice de corrélation (Pearson) – variables numériques & encodées", fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/eda_08_correlation_heatmap.png", dpi=FIG_DPI)
plt.close(fig)
print("  → outputs/eda_08_correlation_heatmap.png")

# Strongest correlations with is_late
top_corr = (
    corr_matrix["is_late"]
    .drop("is_late")
    .abs()
    .sort_values(ascending=False)
)
print(f"\n  Top corrélations avec is_late:\n{top_corr.to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  BIVARIATE – ENTREPRISES
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 5. Bivariate – Entreprises ────────────────────────────────────────")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Ligne de crédit vs % utilisation (scatter colored by historique)
hist_order = ["Excellent", "Bon", "Moyen", "Mauvais"]
palette_hist = {"Excellent": "#2ecc71", "Bon": "#3498db", "Moyen": "#f39c12", "Mauvais": "#e74c3c"}
for hist_val in hist_order:
    sub = df_ent[df_ent["historique_paiement"] == hist_val]
    axes[0].scatter(
        sub["ligne_de_credit"] / 1e6,
        sub["pct_utilisation_credit"],
        label=hist_val,
        alpha=0.6,
        s=20,
        color=palette_hist[hist_val],
    )
axes[0].set_xlabel("Ligne de crédit (M MAD)")
axes[0].set_ylabel("% Utilisation du crédit")
axes[0].set_title("Ligne crédit vs Utilisation\n(coloré par historique)")
axes[0].legend(fontsize=8)

# Ligne de crédit by secteur_activite
order_sect = df_ent.groupby("secteur_activite")["ligne_de_credit"].median().sort_values()
axes[1].barh(
    order_sect.index,
    order_sect.values / 1e6,
    color=sns.color_palette(PALETTE, len(order_sect)),
)
axes[1].set_xlabel("Ligne de crédit médiane (M MAD)")
axes[1].set_title("Ligne de crédit médiane\npar secteur d'activité")

# nb_transactions_historique by historique_paiement
order_hist_df = df_ent.groupby("historique_paiement")["nb_transactions_historique"].mean().reindex(hist_order)
axes[2].bar(
    hist_order,
    order_hist_df.values,
    color=[palette_hist[h] for h in hist_order],
    alpha=0.85,
)
axes[2].set_xlabel("Historique de paiement")
axes[2].set_ylabel("Nb transactions historique (moy.)")
axes[2].set_title("Transactions historiques\npar qualité de paiement")

fig.suptitle("Entreprises – Analyses bivariées", fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/eda_09_bivariate_entreprises.png", dpi=FIG_DPI)
plt.close(fig)
print("  → outputs/eda_09_bivariate_entreprises.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  BIVARIATE – TARGET VARIABLE (is_late)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 6. Bivariate – Variable cible (is_late) ───────────────────────────")

groupers = ["technique_paiement", "historique_paiement", "pays", "secteur_activite", "ville"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

for idx, grp in enumerate(groupers):
    rate = (
        df.groupby(grp)["is_late"]
        .mean()
        .mul(100)
        .sort_values(ascending=True)
    )
    # Reorder historique by risk logic
    if grp == "historique_paiement":
        rate = rate.reindex(["Excellent", "Bon", "Moyen", "Mauvais"])

    colors_bar = ["tomato" if v > df["is_late"].mean() * 100 else "steelblue" for v in rate.values]
    axes_flat[idx].barh(rate.index, rate.values, color=colors_bar)
    axes_flat[idx].axvline(df["is_late"].mean() * 100, color="black", linestyle="--", lw=1, label="Moyenne")
    axes_flat[idx].set_xlabel("Taux de retard (%)")
    axes_flat[idx].set_title(f"Taux de retard par {grp}")
    axes_flat[idx].legend(fontsize=7)
    for i, v in enumerate(rate.values):
        axes_flat[idx].text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8)

# Hide the unused 6th subplot
axes_flat[-1].set_visible(False)

fig.suptitle("Variable cible is_late – Analyses bivariées", fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/eda_10_bivariate_is_late.png", dpi=FIG_DPI)
plt.close(fig)
print("  → outputs/eda_10_bivariate_is_late.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  BIVARIATE – MONTANT
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 7. Bivariate – Montant ────────────────────────────────────────────")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Boxplot montant by technique_paiement
order_tech = ["Transfert", "Remise documentaire", "Crédit documentaire"]
data_by_tech = [df[df["technique_paiement"] == t]["montant"].values for t in order_tech]
bp = axes[0].boxplot(data_by_tech, labels=order_tech, patch_artist=True, vert=True)
colors_bp = sns.color_palette(PALETTE, len(order_tech))
for patch, color in zip(bp["boxes"], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
axes[0].set_ylabel("Montant (MAD)")
axes[0].set_title("Montant par technique de paiement")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

# Montant moyen by pays (top 10)
pays_montant = df.groupby("pays")["montant"].mean().sort_values(ascending=True)
axes[1].barh(pays_montant.index, pays_montant.values / 1e6, color=sns.color_palette(PALETTE, len(pays_montant)))
axes[1].set_xlabel("Montant moyen (M MAD)")
axes[1].set_title("Montant moyen par pays")

# Montant moyen by historique_paiement
hist_montant = (
    df.groupby("historique_paiement")["montant"]
    .mean()
    .reindex(["Excellent", "Bon", "Moyen", "Mauvais"])
)
axes[2].bar(
    hist_montant.index,
    hist_montant.values / 1e6,
    color=[palette_hist[h] for h in hist_montant.index],
    alpha=0.85,
)
axes[2].set_xlabel("Historique de paiement")
axes[2].set_ylabel("Montant moyen (M MAD)")
axes[2].set_title("Montant moyen\npar historique de paiement")

fig.suptitle("Transactions – Montant par catégorie", fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/eda_11_bivariate_montant.png", dpi=FIG_DPI)
plt.close(fig)
print("  → outputs/eda_11_bivariate_montant.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  OUTLIER DETECTION (IQR method)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 8. Détection des valeurs aberrantes (IQR) ─────────────────────────")

outlier_cols = ["montant", "delai_prevu", "delai_reel", "retard",
                "ligne_de_credit", "pct_utilisation_credit", "nb_transactions_historique"]

outlier_data = []
for col in outlier_cols:
    series = df[col].dropna()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((series < lower) | (series > upper)).sum()
    skew = series.skew()
    kurt = series.kurt()
    outlier_data.append({
        "variable": col,
        "n_outliers": n_out,
        "pct_outliers": round(n_out / len(series) * 100, 2),
        "lower_fence": round(lower, 2),
        "upper_fence": round(upper, 2),
        "skewness": round(skew, 3),
        "kurtosis": round(kurt, 3),
    })
    print(f"  {col:35s}: {n_out:4d} outliers ({n_out/len(series)*100:.1f}%)  skew={skew:.2f}")

df_outliers = pd.DataFrame(outlier_data)
df_outliers.to_csv("outputs/eda_outliers.csv", index=False)

fig, axes = plt.subplots(1, len(outlier_cols), figsize=(22, 4))
for j, col in enumerate(outlier_cols):
    axes[j].boxplot(df[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor=sns.color_palette(PALETTE)[j % 8], alpha=0.7))
    axes[j].set_title(col, fontsize=8)
    axes[j].set_xticks([])

fig.suptitle("Boîtes à moustaches – détection des outliers (IQR ×1.5)", fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/eda_12_outliers_boxplots.png", dpi=FIG_DPI)
plt.close(fig)
print("  → outputs/eda_12_outliers_boxplots.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  CROSS-DATASET ANALYSIS – SECTEUR × TARGET
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 9. Analyse croisée – secteur × retard ─────────────────────────────")

sect_stats = (
    df.groupby("secteur_activite")
    .agg(
        nb_transactions=("transaction_id", "count"),
        montant_moyen=("montant", "mean"),
        taux_retard=("is_late", "mean"),
        retard_moyen=("retard", "mean"),
    )
    .sort_values("taux_retard", ascending=False)
    .round(2)
)
print(sect_stats.to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Taux de retard by secteur
axes[0].barh(
    sect_stats.index,
    sect_stats["taux_retard"] * 100,
    color=[
        "tomato" if v > df["is_late"].mean() else "steelblue"
        for v in sect_stats["taux_retard"]
    ],
)
axes[0].axvline(df["is_late"].mean() * 100, color="black", linestyle="--", lw=1)
axes[0].set_xlabel("Taux de retard (%)")
axes[0].set_title("Taux de retard\npar secteur")

# Montant moyen by secteur
axes[1].barh(
    sect_stats.index,
    sect_stats["montant_moyen"] / 1e6,
    color=sns.color_palette(PALETTE, len(sect_stats)),
)
axes[1].set_xlabel("Montant moyen (M MAD)")
axes[1].set_title("Montant moyen\npar secteur")

# Volume by secteur
axes[2].barh(
    sect_stats.index,
    sect_stats["nb_transactions"],
    color=sns.color_palette("Set3", len(sect_stats)),
)
axes[2].set_xlabel("Nombre de transactions")
axes[2].set_title("Volume de transactions\npar secteur")

fig.suptitle("Analyse croisée – Secteur d'activité × Performance", fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/eda_13_cible_croise.png", dpi=FIG_DPI)
plt.close(fig)
print("  → outputs/eda_13_cible_croise.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 10. Résumé statistique ────────────────────────────────────────────")

summary_rows = []

# Entreprises
for col in df_ent.select_dtypes(include="number").columns:
    s = df_ent[col]
    summary_rows.append({
        "dataset": "entreprises",
        "variable": col,
        "type": "numérique",
        "count": int(s.count()),
        "missing": int(s.isna().sum()),
        "mean": round(s.mean(), 2),
        "std": round(s.std(), 2),
        "min": round(s.min(), 2),
        "q25": round(s.quantile(0.25), 2),
        "median": round(s.median(), 2),
        "q75": round(s.quantile(0.75), 2),
        "max": round(s.max(), 2),
        "skewness": round(s.skew(), 3),
        "kurtosis": round(s.kurt(), 3),
    })
for col in df_ent.select_dtypes(exclude="number").columns:
    s = df_ent[col]
    top_val = s.value_counts().idxmax()
    summary_rows.append({
        "dataset": "entreprises",
        "variable": col,
        "type": "catégorielle",
        "count": int(s.count()),
        "missing": int(s.isna().sum()),
        "mean": top_val,
        "std": None,
        "min": None,
        "q25": None,
        "median": None,
        "q75": None,
        "max": s.value_counts().max(),
        "skewness": None,
        "kurtosis": None,
    })

# Transactions
for col in df_trx.select_dtypes(include="number").columns:
    s = df_trx[col]
    summary_rows.append({
        "dataset": "transactions",
        "variable": col,
        "type": "numérique",
        "count": int(s.count()),
        "missing": int(s.isna().sum()),
        "mean": round(s.mean(), 2),
        "std": round(s.std(), 2),
        "min": round(s.min(), 2),
        "q25": round(s.quantile(0.25), 2),
        "median": round(s.median(), 2),
        "q75": round(s.quantile(0.75), 2),
        "max": round(s.max(), 2),
        "skewness": round(s.skew(), 3),
        "kurtosis": round(s.kurt(), 3),
    })
for col in df_trx.select_dtypes(include="object").columns:
    s = df_trx[col]
    top_val = s.value_counts().idxmax()
    summary_rows.append({
        "dataset": "transactions",
        "variable": col,
        "type": "catégorielle",
        "count": int(s.count()),
        "missing": int(s.isna().sum()),
        "mean": top_val,
        "std": None,
        "min": None,
        "q25": None,
        "median": None,
        "q75": None,
        "max": s.value_counts().max(),
        "skewness": None,
        "kurtosis": None,
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv("outputs/eda_summary.csv", index=False)
print(f"  Summary table saved → outputs/eda_summary.csv  ({len(df_summary)} variables)")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

print("\n══ EDA complète ══════════════════════════════════════════════════════")
print(f"  Entreprises : {len(df_ent)} lignes | {df_ent.shape[1]} variables")
print(f"  Transactions: {len(df_trx)} lignes | {df_trx.shape[1]} variables")
print(f"  Taux retard global : {df_trx['is_late'].mean()*100:.1f} %")
print(f"  Montant moyen      : {df_trx['montant'].mean():,.0f} MAD")
print(f"  Période couverte   : {df_trx['date_ouverture'].min().date()} → {df_trx['date_ouverture'].max().date()}")
print("\n  Outputs enregistrés → outputs/eda_*.png  +  outputs/eda_summary.csv")
print("  Terminé.")
