"""
generate_dataset.py
-------------------
Generates synthetic datasets for trade-finance risk modelling at a Moroccan bank
(inspired by Bank of Africa – BOA Morocco).

Two CSV files are produced inside the ``data/`` folder:
  - data/entreprises.csv   : company-level features
  - data/transactions.csv  : transaction-level features

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

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

fake = Faker("fr_FR")
fake.seed_instance(SEED)

# ── Constants ──────────────────────────────────────────────────────────────────
N_ENTREPRISES = 200
N_TRANSACTIONS = 1_200

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
PAYS = [
    "France",
    "Espagne",
    "Italie",
    "Allemagne",
    "Chine",
    "États-Unis",
    "Turquie",
    "Émirats arabes unis",
    "Sénégal",
    "Côte d'Ivoire",
    "Pays-Bas",
    "Belgique",
]

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
AMOUNT_STD = 400_000
AMOUNT_MIN = 50_000
AMOUNT_MAX = 8_000_000

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2024, 12, 31)


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


# ── 1. Entreprises dataset ─────────────────────────────────────────────────────

def generate_entreprises(n: int = N_ENTREPRISES) -> pd.DataFrame:
    records = []
    for i in range(1, n + 1):
        ville = random.choice(VILLES)
        secteur = random.choice(SECTEURS)

        # Credit line: between 500 000 and 20 000 000 MAD
        ligne_credit = round(random.uniform(500_000, 20_000_000), 2)

        # Utilisation rate: 0–100 %
        pct_utilisation = round(random.uniform(5, 95), 2)

        # Payment history
        hist_paiement = random.choices(
            HISTORIQUE_PAIEMENT_LABELS, weights=HISTORIQUE_PAIEMENT_WEIGHTS, k=1
        )[0]

        # Number of historical transactions
        hist_transactions = random.randint(1, 150)

        records.append(
            {
                "entreprise_id": f"E{i:04d}",
                "secteur_activite": secteur,
                "ligne_de_credit": ligne_credit,
                "pct_utilisation_credit": pct_utilisation,
                "historique_paiement": hist_paiement,
                "ville": ville,
                "nb_transactions_historique": hist_transactions,
            }
        )

    return pd.DataFrame(records)


# ── 2. Transactions dataset ────────────────────────────────────────────────────

def generate_transactions(
    entreprises: pd.DataFrame, n: int = N_TRANSACTIONS
) -> pd.DataFrame:
    entreprise_ids = entreprises["entreprise_id"].tolist()

    records = []
    for i in range(1, n + 1):
        entreprise_id = random.choice(entreprise_ids)
        pays = random.choice(PAYS)
        montant = random_montant()
        technique = random.choices(TECHNIQUES, weights=TECHNIQUE_WEIGHTS, k=1)[0]

        date_ouverture = random_date(START_DATE, END_DATE - timedelta(days=90))

        # Expected processing time depends on payment technique
        lo, hi = DELAI_RANGES[technique]
        delai_prevu = random.randint(lo, hi)

        date_prevue = date_ouverture + timedelta(days=delai_prevu)

        # Actual delay: most on-time, some late
        # Companies with bad payment history more likely to be late
        hist_paiement = entreprises.loc[
            entreprises["entreprise_id"] == entreprise_id, "historique_paiement"
        ].values[0]

        retard_prob = {
            "Excellent": 0.05,
            "Bon": 0.15,
            "Moyen": 0.30,
            "Mauvais": 0.60,
        }[hist_paiement]

        is_late = random.random() < retard_prob
        if is_late:
            retard_jours = random.randint(1, 45)
        else:
            retard_jours = 0

        date_effective = date_prevue + timedelta(days=retard_jours)
        delai_reel = (date_effective - date_ouverture).days

        records.append(
            {
                "transaction_id": f"T{i:06d}",
                "entreprise_id": entreprise_id,
                "pays": pays,
                "montant": montant,
                "technique_paiement": technique,
                "date_ouverture": date_ouverture.strftime("%Y-%m-%d"),
                "date_prevue": date_prevue.strftime("%Y-%m-%d"),
                "date_effective": date_effective.strftime("%Y-%m-%d"),
                "delai_prevu": delai_prevu,
                "delai_reel": delai_reel,
                "retard": retard_jours,
                "is_late": int(is_late),
            }
        )

    return pd.DataFrame(records)


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

    # Quick stats
    print("\n── Entreprises ────────────────────────────────────────")
    print(df_entreprises.describe(include="all").T.to_string())

    print("\n── Transactions ───────────────────────────────────────")
    print(df_transactions.describe().T.to_string())
    print("\nPayment-technique distribution:")
    print(
        df_transactions["technique_paiement"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .to_string()
    )
    print("\nCity distribution (entreprises):")
    print(
        df_entreprises["ville"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .to_string()
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
