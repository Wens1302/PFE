"""
utils.py
--------
Shared utilities for trade-finance analysis and dashboard.
"""

# ── Country risk ratings ────────────────────────────────────────────────────────
# Scale 1–5 (1 = very low risk, 5 = very high risk).
# Source: OECD Country Risk Classification + COFACE + FATF (2024, indicative values).
NOTATION_PAYS = {
    "France": 1,
    "Espagne": 1,
    "Italie": 1,
    "Allemagne": 1,
    "Pays-Bas": 1,
    "Belgique": 1,
    "États-Unis": 1,
    "Émirats arabes unis": 2,
    "Chine": 3,
    "Turquie": 3,
    "Sénégal": 4,
    "Côte d'Ivoire": 4,
}

NOTATION_LABELS = {
    1: "Très faible",
    2: "Faible",
    3: "Modéré",
    4: "Élevé",
    5: "Très élevé",
}

# Per-notation risk parameters
# late_add        : additional probability of payment delay
# fraud_base      : base probability of suspected fraud
# op_risk_base    : base probability of operational incident
PAYS_RISQUE_PARAMS = {
    1: {"late_add": 0.00, "fraud_base": 0.02, "op_risk_base": 0.03},
    2: {"late_add": 0.05, "fraud_base": 0.05, "op_risk_base": 0.08},
    3: {"late_add": 0.10, "fraud_base": 0.10, "op_risk_base": 0.15},
    4: {"late_add": 0.18, "fraud_base": 0.18, "op_risk_base": 0.25},
    5: {"late_add": 0.25, "fraud_base": 0.28, "op_risk_base": 0.35},
}

# ── Cluster labelling thresholds ───────────────────────────────────────────────
SEUIL_TAUX_RETARD_ELEVE = 0.30        # >= → "Risque élevé"
SEUIL_LIGNE_CREDIT_GRANDE = 12_000_000  # >= → "Grande entreprise"
SEUIL_UTILISATION_INTENSIVE = 60      # >= → "Utilisation intensive"


def label_cluster(taux_retard_moy: float, ligne_credit_moy: float, utilisation_moy: float) -> str:
    """Return a human-readable label for a K-Means cluster."""
    if taux_retard_moy >= SEUIL_TAUX_RETARD_ELEVE:
        return "Risque élevé"
    if ligne_credit_moy >= SEUIL_LIGNE_CREDIT_GRANDE:
        return "Grande entreprise"
    if utilisation_moy >= SEUIL_UTILISATION_INTENSIVE:
        return "Utilisation intensive"
    return "PME standard"


def build_cluster_labels(profile_summary) -> dict:
    """Build a {cluster_id: label} mapping from a profile-summary DataFrame.

    Parameters
    ----------
    profile_summary : pd.DataFrame
        Must have columns: taux_retard_moy, ligne_credit_moy, utilisation_moy.
        Index should be integer cluster IDs.

    Returns
    -------
    dict[int, str]
    """
    return {
        c: label_cluster(
            row["taux_retard_moy"],
            row["ligne_credit_moy"],
            row["utilisation_moy"],
        )
        for c, row in profile_summary.iterrows()
    }
