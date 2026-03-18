"""
utils.py
--------
Shared utilities for trade-finance analysis and dashboard.
"""

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
