"""
generate_dataset.py
-------------------
Generates synthetic datasets for trade-finance risk modelling at a Moroccan bank
(inspired by Bank of Africa – BOA Morocco).

Two CSV files are produced inside the ``data/`` folder:
  - data/entreprises.csv   : company-level features
  - data/transactions.csv  : transaction-level features

New in v2
---------
  • Country risk notation (OECD/COFACE-inspired, 1–5) drives transactional,
    operational and fraud risk probabilities.
  • Each company is guaranteed at least MIN_TRX_PER_COMPANY transactions so
    that transaction-amount evolution can be tracked.
  • New transaction fields: notation_pays, score_risque_pays, risque_operationnel,
    fraude_suspectee, score_aml, rang_transaction_entreprise,
    montant_precedent, evolution_montant_pct.

Run
---
    python generate_dataset.py

Requirements
------------
    pip install pandas numpy faker
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from utils import NOTATION_PAYS, PAYS_RISQUE_PARAMS

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

fake = Faker("fr_FR")
fake.seed_instance(SEED)

# ── Constants ──────────────────────────────────────────────────────────────────
N_ENTREPRISES = 150
N_TRANSACTIONS = 1_500
MIN_TRX_PER_COMPANY = 3          # guarantees transaction-history depth

VILLES = ["Fès", "Meknès"]

SECTEURS = [
    "Textile",
    "Agroalimentaire",
    "Chimie & Parachimie",
    "Cuir & Maroquinerie",
    "Tourisme",
    "Matériaux de construction",
    "Minerais & Métaux",
    "Electronique",
    "Pharmaceutique",
    "Commerce général",
]

# Partner countries for BOA Morocco trade corridors
PAYS = list(NOTATION_PAYS.keys())

# Payment-technique distribution  (transfert > 50 %, then remise, then credoc)
TECHNIQUES = ["Transfert", "Remise documentaire", "Crédit documentaire"]
TECHNIQUE_WEIGHTS = [0.55, 0.30, 0.15]

# Delay ranges in days per payment technique  {technique: (min, max)}
DELAI_RANGES = {
    "Transfert": (1, 10),
    "Remise documentaire": (10, 30),
    "Crédit documentaire": (20, 60),
}

# Historique paiement labels
HISTORIQUE_PAIEMENT_LABELS = ["Excellent", "Bon", "Moyen", "Mauvais"]
HISTORIQUE_PAIEMENT_WEIGHTS = [0.25, 0.40, 0.25, 0.10]

# Amount distribution (MAD) – normal, mean ≈ 800 000, std ≈ 400 000
AMOUNT_MEAN = 800_000
AMOUNT_STD  = 400_000
AMOUNT_MIN  = 50_000
AMOUNT_MAX  = 8_000_000

# Payment-history → base late probability
HIST_LATE_PROB = {
    "Excellent": 0.05,
    "Bon":       0.15,
    "Moyen":     0.30,
    "Mauvais":   0.60,
}

# Payment-history → AML component (0–100)
HIST_AML_SCORE = {"Excellent": 0, "Bon": 25, "Moyen": 60, "Mauvais": 100}

# Payment-technique → AML component (transfers have less documentation)
TECH_AML_SCORE = {
    "Transfert": 70,
    "Remise documentaire": 30,
    "Crédit documentaire": 10,
}

START_DATE = datetime(2020, 1, 1)
END_DATE   = datetime(2024, 12, 31)


# ── Helper functions ───────────────────────────────────────────────────────────

def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def random_montant() -> float:
    """Draw from a truncated normal distribution (in MAD)."""
    while True:
        v = np.random.normal(AMOUNT_MEAN, AMOUNT_STD)
        if AMOUNT_MIN <= v <= AMOUNT_MAX:
            return round(v, 2)


def compute_score_aml(
    notation: int,
    montant: float,
    hist_paiement: str,
    technique: str,
    evolution_pct: float | None,
) -> float:
    """Compute an AML risk score (0–100).

    Components
    ----------
    Country risk (30 %)  : derived from OECD/COFACE notation.
    Amount (20 %)        : higher amounts → higher score.
    Payment history (20 %): Mauvais → max score.
    Technique (15 %)     : Transfert has least documentation.
    Evolution (15 %)     : abnormal amount jumps or drops.
    """
    country_c   = (notation - 1) / 4 * 100
    amount_c    = min(montant / AMOUNT_MAX, 1.0) * 100
    history_c   = HIST_AML_SCORE.get(hist_paiement, 0)
    technique_c = TECH_AML_SCORE.get(technique, 0)

    if evolution_pct is not None and not np.isnan(evolution_pct):
        evolution_c = min(abs(evolution_pct) / 200, 1.0) * 100
    else:
        evolution_c = 0.0

    score = (
        0.30 * country_c
        + 0.20 * amount_c
        + 0.20 * history_c
        + 0.15 * technique_c
        + 0.15 * evolution_c
    )
    return round(min(score, 100), 1)


# ── 1. Entreprises dataset ─────────────────────────────────────────────────────

def generate_entreprises(n: int = N_ENTREPRISES) -> pd.DataFrame:
    records = []
    for i in range(1, n + 1):
        ville    = random.choice(VILLES)
        secteur  = random.choice(SECTEURS)

        # Credit line: between 500 000 and 20 000 000 MAD
        ligne_credit   = round(random.uniform(500_000, 20_000_000), 2)

        # Utilisation rate: 0–100 %
        pct_utilisation = round(random.uniform(5, 95), 2)

        # Payment history
        hist_paiement = random.choices(
            HISTORIQUE_PAIEMENT_LABELS, weights=HISTORIQUE_PAIEMENT_WEIGHTS, k=1
        )[0]

        # Number of historical transactions (before dataset period)
        hist_transactions = random.randint(1, 150)

        records.append(
            {
                "entreprise_id":              f"E{i:04d}",
                "secteur_activite":           secteur,
                "ligne_de_credit":            ligne_credit,
                "pct_utilisation_credit":     pct_utilisation,
                "historique_paiement":        hist_paiement,
                "ville":                      ville,
                "nb_transactions_historique": hist_transactions,
            }
        )

    return pd.DataFrame(records)


# ── 2. Transactions dataset ────────────────────────────────────────────────────

def generate_transactions(
    entreprises: pd.DataFrame, n: int = N_TRANSACTIONS
) -> pd.DataFrame:
    entreprise_ids = entreprises["entreprise_id"].tolist()

    # Build a lookup {entreprise_id → hist_paiement} for fast access
    hist_map = dict(zip(entreprises["entreprise_id"], entreprises["historique_paiement"]))

    # ── Assign companies to transaction slots ───────────────────────────────────
    # Guarantee every company appears at least MIN_TRX_PER_COMPANY times
    guaranteed = entreprise_ids * MIN_TRX_PER_COMPANY
    remaining  = random.choices(entreprise_ids, k=n - len(guaranteed))
    all_eids   = guaranteed + remaining
    random.shuffle(all_eids)

    records = []
    for i, entreprise_id in enumerate(all_eids, start=1):
        pays       = random.choice(PAYS)
        notation   = NOTATION_PAYS[pays]
        params     = PAYS_RISQUE_PARAMS[notation]
        montant    = random_montant()
        technique  = random.choices(TECHNIQUES, weights=TECHNIQUE_WEIGHTS, k=1)[0]
        hist_pay   = hist_map[entreprise_id]

        date_ouverture = random_date(START_DATE, END_DATE - timedelta(days=90))

        # Expected delay
        lo, hi       = DELAI_RANGES[technique]
        delai_prevu  = random.randint(lo, hi)
        date_prevue  = date_ouverture + timedelta(days=delai_prevu)

        # ── is_late ────────────────────────────────────────────────────────────
        # P(late) = base (from historique) + country addition
        base_late   = HIST_LATE_PROB[hist_pay]
        late_prob   = min(1.0, base_late + params["late_add"])
        is_late     = random.random() < late_prob
        retard_jours = random.randint(1, 45) if is_late else 0

        date_effective = date_prevue + timedelta(days=retard_jours)
        delai_reel     = (date_effective - date_ouverture).days

        # ── risque_operationnel ────────────────────────────────────────────────
        op_prob = params["op_risk_base"]
        if notation >= 3 and technique in ("Crédit documentaire", "Remise documentaire"):
            op_prob = min(1.0, op_prob + 0.15)
        if montant > 1_500_000:
            op_prob = min(1.0, op_prob + 0.10)
        risque_operationnel = int(random.random() < op_prob)

        # ── fraude_suspectee ───────────────────────────────────────────────────
        fraud_prob = params["fraud_base"]
        if hist_pay == "Mauvais":
            fraud_prob = min(1.0, fraud_prob + 0.15)
        elif hist_pay == "Moyen":
            fraud_prob = min(1.0, fraud_prob + 0.05)
        if montant > 2_000_000:
            fraud_prob = min(1.0, fraud_prob + 0.10)
        if technique == "Transfert" and notation >= 3:
            fraud_prob = min(1.0, fraud_prob + 0.08)
        fraude_suspectee = int(random.random() < fraud_prob)

        records.append(
            {
                "transaction_id":     f"T{i:06d}",
                "entreprise_id":      entreprise_id,
                "pays":               pays,
                "notation_pays":      notation,
                "score_risque_pays":  round((notation - 1) / 4 * 100, 1),
                "montant":            montant,
                "technique_paiement": technique,
                "date_ouverture":     date_ouverture.strftime("%Y-%m-%d"),
                "date_prevue":        date_prevue.strftime("%Y-%m-%d"),
                "date_effective":     date_effective.strftime("%Y-%m-%d"),
                "delai_prevu":        delai_prevu,
                "delai_reel":         delai_reel,
                "retard":             retard_jours,
                "is_late":            int(is_late),
                "risque_operationnel": risque_operationnel,
                "fraude_suspectee":   fraude_suspectee,
            }
        )

    df = pd.DataFrame(records)

    # ── Post-processing: transaction history per company ───────────────────────
    df = df.sort_values(["entreprise_id", "date_ouverture"]).reset_index(drop=True)

    df["rang_transaction_entreprise"] = (
        df.groupby("entreprise_id").cumcount() + 1
    )
    df["montant_precedent"] = df.groupby("entreprise_id")["montant"].shift(1)
    df["evolution_montant_pct"] = (
        (df["montant"] - df["montant_precedent"]) / df["montant_precedent"] * 100
    ).round(2)

    # ── score_aml (requires evolution, so computed last) ──────────────────────
    # Join historique_paiement from entreprises for scoring
    df = df.merge(
        entreprises[["entreprise_id", "historique_paiement"]],
        on="entreprise_id",
        how="left",
    )
    df["score_aml"] = df.apply(
        lambda r: compute_score_aml(
            r["notation_pays"],
            r["montant"],
            r["historique_paiement"],
            r["technique_paiement"],
            r["evolution_montant_pct"],
        ),
        axis=1,
    )
    # Drop the denormalized column (already in entreprises.csv)
    df = df.drop(columns=["historique_paiement"])

    # Restore original transaction_id order for readability
    df = df.sort_values("transaction_id").reset_index(drop=True)

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)

    print("Generating entreprises dataset …")
    df_entreprises = generate_entreprises()
    df_entreprises.to_csv("data/entreprises.csv", index=False)
    print(f"  → {len(df_entreprises)} rows written to data/entreprises.csv")

    print("Generating transactions dataset …")
    df_transactions = generate_transactions(df_entreprises)
    df_transactions.to_csv("data/transactions.csv", index=False)
    print(f"  → {len(df_transactions)} rows written to data/transactions.csv")

    # ── Quick stats ────────────────────────────────────────────────────────────
    print("\n── Entreprises ────────────────────────────────────────")
    print(df_entreprises.describe(include="all").T.to_string())

    print("\n── Transactions ───────────────────────────────────────")
    print(df_transactions.describe().T.to_string())

    print("\nPayment-technique distribution:")
    print(
        df_transactions["technique_paiement"]
        .value_counts(normalize=True).mul(100).round(1).to_string()
    )

    print("\nCountry risk distribution:")
    print(
        df_transactions.groupby("notation_pays")["transaction_id"]
        .count()
        .rename("nb_transactions")
        .to_string()
    )

    print("\nFraude suspectée :", df_transactions["fraude_suspectee"].sum(),
          f"({df_transactions['fraude_suspectee'].mean()*100:.1f} %)")
    print("Risque opérationnel :", df_transactions["risque_operationnel"].sum(),
          f"({df_transactions['risque_operationnel'].mean()*100:.1f} %)")

    trx_per_ent = df_transactions.groupby("entreprise_id")["transaction_id"].count()
    print(f"\nTransactions/entreprise — min: {trx_per_ent.min()}, "
          f"mean: {trx_per_ent.mean():.1f}, max: {trx_per_ent.max()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
