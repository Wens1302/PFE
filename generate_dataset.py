"""
Synthetic International Banking Transactions Dataset Generator
Business Center: Fes-Meknes, Morocco
Generates a realistic Big-Data-ready dataset for EDA, clustering, and profiling.

SCOPE: ONLY international trade transactions (import/export).
Payment methods: Wire Transfer, Documentary Collection, Letter of Credit.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import warnings

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
N_COMPANIES = 500
N_TRANSACTIONS = 120_000
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2024, 12, 31)
DATE_RANGE_DAYS = (END_DATE - START_DATE).days

# Only Fes and Meknes (business center region)
CITIES = {
    "Fes": 0.55,
    "Meknes": 0.45,
}

SECTORS = ["agriculture", "textile", "industry", "services", "trade"]
SECTOR_WEIGHTS = [0.18, 0.20, 0.25, 0.17, 0.20]

# SME = small+medium companies, Large = large companies
SIZES = ["SME", "Large"]
SIZE_WEIGHTS = [0.85, 0.15]

# ONLY international trade transaction types
TRANSACTION_TYPES = ["import", "export"]

# Strictly required payment methods for international trade
PAYMENT_METHODS = ["Wire Transfer", "Documentary Collection", "Letter of Credit"]

# EUR, USD, CNY as primary currencies
EXCHANGE_RATES = {
    "EUR": 10.80,
    "USD": 10.10,
    "CNY": 1.40,
}

# Countries & their trade regions
COUNTRIES = {
    "France":      "Europe",
    "Spain":       "Europe",
    "Germany":     "Europe",
    "Italy":       "Europe",
    "Turkey":      "Europe",
    "China":       "Asia",
    "India":       "Asia",
    "UAE":         "Middle East",
    "USA":         "America",
}

MOROCCO = "Morocco"

# Partner country distribution per sector — Morocco excluded (all trades are international)
# Imports: China dominates; Exports: EU countries dominate
SECTOR_IMPORT_COUNTRIES = {
    "agriculture": {"China": 0.25, "France": 0.20, "Spain": 0.18, "Germany": 0.12,
                    "Italy": 0.10, "Turkey": 0.08, "UAE": 0.04, "USA": 0.03},
    "textile":     {"China": 0.55, "India": 0.18, "Turkey": 0.12, "Germany": 0.07,
                    "France": 0.05, "Italy": 0.03},
    "industry":    {"China": 0.50, "Germany": 0.15, "France": 0.10, "Spain": 0.08,
                    "Italy": 0.07, "USA": 0.06, "UAE": 0.04},
    "services":    {"France": 0.30, "Spain": 0.22, "Germany": 0.18, "USA": 0.15,
                    "UAE": 0.10, "Italy": 0.05},
    "trade":       {"China": 0.40, "France": 0.14, "Spain": 0.12, "Germany": 0.10,
                    "Italy": 0.08, "Turkey": 0.08, "UAE": 0.05, "USA": 0.03},
}

SECTOR_EXPORT_COUNTRIES = {
    "agriculture": {"France": 0.30, "Spain": 0.25, "Germany": 0.15, "Italy": 0.15,
                    "Turkey": 0.10, "UAE": 0.05},
    "textile":     {"France": 0.28, "Spain": 0.22, "Germany": 0.18, "Italy": 0.18,
                    "Turkey": 0.14},
    "industry":    {"France": 0.25, "Germany": 0.22, "Spain": 0.18, "Italy": 0.15,
                    "Turkey": 0.12, "UAE": 0.05, "USA": 0.03},
    "services":    {"France": 0.35, "Spain": 0.25, "Germany": 0.20, "Italy": 0.12,
                    "Turkey": 0.08},
    "trade":       {"France": 0.22, "Spain": 0.20, "Germany": 0.18, "Italy": 0.16,
                    "Turkey": 0.14, "UAE": 0.06, "USA": 0.04},
}

# Currency by partner country
COUNTRY_CURRENCY = {
    "France":   {"EUR": 0.90, "USD": 0.10},
    "Spain":    {"EUR": 0.90, "USD": 0.10},
    "Germany":  {"EUR": 0.90, "USD": 0.10},
    "Italy":    {"EUR": 0.88, "USD": 0.12},
    "Turkey":   {"EUR": 0.70, "USD": 0.25, "CNY": 0.05},
    "China":    {"USD": 0.55, "CNY": 0.40, "EUR": 0.05},
    "India":    {"USD": 0.70, "EUR": 0.20, "CNY": 0.10},
    "UAE":      {"USD": 0.65, "EUR": 0.30, "CNY": 0.05},
    "USA":      {"USD": 0.90, "EUR": 0.10},
}

# Risk scores by country (notation 1–5, score 0–100)
COUNTRY_RISK = {
    "France":  (1, 10), "Spain":   (1, 12), "Germany": (1,  8),
    "Italy":   (2, 20), "Turkey":  (2, 28),
    "China":   (2, 30), "India":   (2, 28),
    "UAE":     (2, 22), "USA":     (1, 10),
}

# Sector import/export tendency weights
SECTOR_TRADE_TYPE = {
    "agriculture": {"import": 0.35, "export": 0.65},
    "textile":     {"import": 0.55, "export": 0.45},
    "industry":    {"import": 0.60, "export": 0.40},
    "services":    {"import": 0.45, "export": 0.55},
    "trade":       {"import": 0.50, "export": 0.50},
}

# Ramadan approximate start months (March 2020, Apr 2021, Apr 2022, Mar 2023, Mar 2024)
RAMADAN_MONTHS = {2020: 4, 2021: 4, 2022: 4, 2023: 3, 2024: 3}


# ── Helper functions ──────────────────────────────────────────────────────────

def _weighted_choice(mapping: dict) -> str:
    keys = list(mapping.keys())
    weights = list(mapping.values())
    total = sum(weights)
    probs = [w / total for w in weights]
    return np.random.choice(keys, p=probs)


def _seasonal_multiplier(date: datetime) -> float:
    """Return a multiplier (>1 means more transactions)."""
    month = date.month
    year = date.year
    # Summer peak (July–August)
    if month in (7, 8):
        return 1.40
    # Ramadan peak (varies by year)
    ramadan_month = RAMADAN_MONTHS.get(year, 4)
    if month == ramadan_month:
        return 1.25
    # End of year
    if month == 12:
        return 1.15
    # Low season (January–February)
    if month in (1, 2):
        return 0.80
    return 1.0


def _amount_for_sector_size(sector: str, size: str) -> float:
    """Generate a realistic international trade transaction amount."""
    base_ranges = {
        ("agriculture", "SME"):   (20_000,  400_000),
        ("agriculture", "Large"): (300_000, 3_000_000),
        ("textile",     "SME"):   (15_000,  300_000),
        ("textile",     "Large"): (200_000, 2_500_000),
        ("industry",    "SME"):   (30_000,  600_000),
        ("industry",    "Large"): (500_000, 6_000_000),
        ("services",    "SME"):   (10_000,  150_000),
        ("services",    "Large"): (100_000, 1_000_000),
        ("trade",       "SME"):   (20_000,  500_000),
        ("trade",       "Large"): (300_000, 5_000_000),
    }
    lo, hi = base_ranges.get((sector, size), (20_000, 400_000))
    mean = (lo + hi) / 2
    sigma = 0.7
    val = np.random.lognormal(np.log(mean), sigma)
    return float(np.clip(val, lo / 10, hi * 3))


def _transaction_type_for_sector(sector: str) -> str:
    return _weighted_choice(SECTOR_TRADE_TYPE.get(sector, {"import": 0.5, "export": 0.5}))


def _payment_method_for_type(t_type: str, size: str) -> str:
    """
    Wire Transfer: majority
    Documentary Collection: moderate
    Letter of Credit: smaller but significant, more common for large companies
    """
    if size == "Large":
        mapping = {
            "import": {"Wire Transfer": 0.55, "Documentary Collection": 0.25,
                       "Letter of Credit": 0.20},
            "export": {"Wire Transfer": 0.50, "Documentary Collection": 0.28,
                       "Letter of Credit": 0.22},
        }
    else:
        mapping = {
            "import": {"Wire Transfer": 0.65, "Documentary Collection": 0.25,
                       "Letter of Credit": 0.10},
            "export": {"Wire Transfer": 0.68, "Documentary Collection": 0.24,
                       "Letter of Credit": 0.08},
        }
    return _weighted_choice(mapping.get(t_type, {"Wire Transfer": 0.65,
                                                  "Documentary Collection": 0.25,
                                                  "Letter of Credit": 0.10}))


# ── 1. Generate Companies ─────────────────────────────────────────────────────

def generate_companies(n: int = N_COMPANIES) -> pd.DataFrame:
    cities = list(CITIES.keys())
    city_weights = [CITIES[c] for c in cities]
    city_weights_norm = np.array(city_weights) / sum(city_weights)

    records = []
    for i in range(1, n + 1):
        sector = np.random.choice(SECTORS, p=SECTOR_WEIGHTS)
        size = np.random.choice(SIZES, p=SIZE_WEIGHTS)
        city = np.random.choice(cities, p=city_weights_norm)
        age = int(np.random.choice(range(1, 31), p=np.array(
            [1 / (k + 1) for k in range(30)]) / sum([1 / (k + 1) for k in range(30)])))

        # Base transaction frequency (monthly) driven by size
        base_freq = {"SME": 8, "Large": 40}[size]
        freq = max(1, int(np.random.lognormal(np.log(base_freq), 0.4)))

        # Risk score correlated with sector
        sector_base_risk = {
            "agriculture": 0.20, "textile": 0.25, "industry": 0.30,
            "services": 0.15, "trade": 0.35,
        }
        risk = float(np.clip(
            np.random.beta(2, 6) + sector_base_risk[sector] - 0.20, 0.02, 0.99
        ))

        records.append({
            "company_id": f"CMP{i:04d}",
            "company_sector": sector,
            "company_size": size,
            "company_age": age,
            "city": city,
            "transaction_frequency": freq,
            "risk_score_base": risk,
        })

    return pd.DataFrame(records)


# ── 2. Generate Transactions ──────────────────────────────────────────────────

def generate_transactions(companies: pd.DataFrame,
                          n: int = N_TRANSACTIONS) -> pd.DataFrame:
    # Larger companies → more transactions
    size_tx_weight = {"SME": 1, "Large": 8}
    tx_weights = companies["company_size"].map(size_tx_weight).values.astype(float)
    tx_weights /= tx_weights.sum()

    chosen_companies = np.random.choice(
        companies.index, size=n, p=tx_weights
    )

    records = []
    for txn_idx, comp_idx in enumerate(chosen_companies):
        comp = companies.iloc[comp_idx]
        cid = comp["company_id"]
        sector = comp["company_sector"]
        size = comp["company_size"]
        risk_base = comp["risk_score_base"]

        # Date – with seasonal weighting
        attempts = 0
        while True:
            candidate = START_DATE + timedelta(days=int(np.random.uniform(0, DATE_RANGE_DAYS)))
            mult = _seasonal_multiplier(candidate)
            if np.random.random() < mult / 1.5:
                tx_date = candidate
                break
            attempts += 1
            if attempts > 20:
                tx_date = candidate
                break

        # Transaction type (import or export only)
        t_type = _transaction_type_for_sector(sector)

        # Partner country based on transaction direction
        if t_type == "import":
            partner_map = SECTOR_IMPORT_COUNTRIES.get(sector, {"China": 0.5, "France": 0.5})
            partner_country = _weighted_choice(partner_map)
            origin_country = partner_country
            destination_country = MOROCCO
        else:  # export
            partner_map = SECTOR_EXPORT_COUNTRIES.get(sector, {"France": 0.5, "Spain": 0.5})
            partner_country = _weighted_choice(partner_map)
            origin_country = MOROCCO
            destination_country = partner_country

        trade_region = COUNTRIES.get(partner_country, "Europe")

        # Currency (always foreign, based on partner country)
        curr_map = COUNTRY_CURRENCY.get(partner_country, {"EUR": 0.6, "USD": 0.4})
        currency = _weighted_choice(curr_map)

        # Exchange rate (add small noise)
        base_rate = EXCHANGE_RATES.get(currency, 10.10)
        exchange_rate = base_rate * np.random.uniform(0.97, 1.03)

        # Amount
        amount = _amount_for_sector_size(sector, size)

        # Outlier transactions (~1 % of rows)
        if np.random.random() < 0.01:
            amount *= np.random.uniform(10, 50)

        amount_mad = amount * exchange_rate

        # Payment method (size-aware)
        payment = _payment_method_for_type(t_type, size)

        # Risk
        country_risk_notation, country_risk_score = COUNTRY_RISK.get(partner_country, (2, 30))
        risk_operational = float(np.clip(
            np.random.beta(2, 8) + 0.05 * country_risk_notation, 0.01, 0.99
        ))
        aml_score = float(np.clip(
            risk_base * 40 + country_risk_score * 0.3 + np.random.normal(0, 5), 0, 100
        ))
        fraude_suspectee = int(aml_score > 75 and np.random.random() < 0.15)

        records.append({
            "transaction_id":         f"TXN{txn_idx + 1:07d}",
            "company_id":             cid,
            "transaction_date":       tx_date.strftime("%Y-%m-%d"),
            "transaction_type":       t_type,
            "payment_method":         payment,
            "amount":                 round(amount, 2),
            "currency":               currency,
            "exchange_rate_to_MAD":   round(exchange_rate, 4),
            "amount_MAD":             round(amount_mad, 2),
            "origin_country":         origin_country,
            "destination_country":    destination_country,
            "trade_region":           trade_region,
            "company_sector":         sector,
            "company_size":           size,
            "company_age":            int(comp["company_age"]),
            "company_city":           comp["city"],
            "transaction_frequency":  int(comp["transaction_frequency"]),
            "avg_transaction_amount": None,   # filled post-hoc
            "partner_concentration_index": None,  # filled post-hoc
            "risk_score":             round(risk_base, 4),
            # Extended fields
            "notation_pays":          country_risk_notation,
            "score_risque_pays":      country_risk_score,
            "risque_operationnel":    round(risk_operational, 4),
            "fraude_suspectee":       fraude_suspectee,
            "score_aml":              round(aml_score, 2),
            "rang_transaction_entreprise": None,  # filled post-hoc
            "montant_precedent":      None,        # filled post-hoc
            "evolution_montant_pct":  None,        # filled post-hoc
        })

    df = pd.DataFrame(records)

    # ── Post-hoc derived columns ──────────────────────────────────────────────
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df.sort_values(["company_id", "transaction_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # avg_transaction_amount – per company mean of amount_MAD
    avg_map = df.groupby("company_id")["amount_MAD"].mean().round(2)
    df["avg_transaction_amount"] = df["company_id"].map(avg_map)

    # partner_concentration_index – Herfindahl index of partner country share per company
    def _herfindahl(series):
        counts = series.value_counts(normalize=True)
        return round(float((counts ** 2).sum()), 4)

    partner_col = df["origin_country"].where(
        df["transaction_type"] == "import", df["destination_country"]
    )
    pci_map = partner_col.groupby(df["company_id"]).agg(_herfindahl)
    df["partner_concentration_index"] = df["company_id"].map(pci_map)

    # rang_transaction_entreprise – rank within company
    df["rang_transaction_entreprise"] = (
        df.groupby("company_id").cumcount() + 1
    )

    # montant_precedent + evolution_montant_pct
    df["montant_precedent"] = df.groupby("company_id")["amount_MAD"].shift(1)
    df["evolution_montant_pct"] = (
        (df["amount_MAD"] - df["montant_precedent"]) / df["montant_precedent"] * 100
    ).round(2)

    # Introduce missing values (~1–2 % of select columns)
    for col in ["exchange_rate_to_MAD", "origin_country", "company_age",
                "montant_precedent", "evolution_montant_pct"]:
        mask = np.random.random(len(df)) < 0.015
        df.loc[mask, col] = np.nan

    # Reset date to string for CSV
    df["transaction_date"] = df["transaction_date"].dt.strftime("%Y-%m-%d")

    return df


# ── 3. Generate Company Profiles (aggregated) ─────────────────────────────────

def generate_company_profiles(transactions: pd.DataFrame,
                               companies: pd.DataFrame) -> pd.DataFrame:
    grp = transactions.groupby("company_id")

    profiles = pd.DataFrame()
    profiles["company_id"]         = grp["company_id"].first()
    profiles["total_transactions"]  = grp["transaction_id"].count()
    profiles["total_volume_MAD"]    = grp["amount_MAD"].sum().round(2)
    profiles["avg_transaction_MAD"] = grp["amount_MAD"].mean().round(2)
    profiles["max_transaction_MAD"] = grp["amount_MAD"].max().round(2)
    profiles["min_transaction_MAD"] = grp["amount_MAD"].min().round(2)
    profiles["std_transaction_MAD"] = grp["amount_MAD"].std().round(2)

    # Bonus fields required by problem statement
    imports_df = transactions[transactions["transaction_type"] == "import"]
    exports_df = transactions[transactions["transaction_type"] == "export"]

    total_import_volume = imports_df.groupby("company_id")["amount_MAD"].sum().round(2)
    total_export_volume = exports_df.groupby("company_id")["amount_MAD"].sum().round(2)
    profiles["total_import_volume"] = total_import_volume
    profiles["total_export_volume"] = total_export_volume

    # trade_balance_ratio = exports / (imports + exports); 0.5 = balanced
    total_trade = (total_import_volume.fillna(0) + total_export_volume.fillna(0)).replace(0, np.nan)
    profiles["trade_balance_ratio"] = (
        total_export_volume.fillna(0) / total_trade
    ).round(4)

    # main_partner_region – most common trade_region per company
    profiles["main_partner_region"] = (
        transactions.groupby("company_id")["trade_region"]
        .agg(lambda x: x.mode()[0] if len(x) > 0 else "Europe")
    )

    # dominant_payment_method – most used payment method per company
    profiles["dominant_payment_method"] = (
        transactions.groupby("company_id")["payment_method"]
        .agg(lambda x: x.mode()[0])
    )

    # Main partner country (by count, excluding Morocco)
    partner_series = transactions["origin_country"].where(
        transactions["transaction_type"] == "import",
        transactions["destination_country"]
    )
    profiles["main_partner_country"] = (
        partner_series[partner_series != MOROCCO]
        .groupby(transactions["company_id"])
        .agg(lambda x: x.mode()[0] if len(x) > 0 else "France")
    )
    profiles["main_partner_country"] = profiles["main_partner_country"].fillna("France")

    # Import / Export ratio
    profiles["import_export_ratio"] = (
        total_import_volume / total_export_volume.replace(0, np.nan)
    ).round(4)

    # Risk aggregates
    profiles["avg_risk_score"]   = grp["risk_score"].mean().round(4)
    profiles["avg_aml_score"]    = grp["score_aml"].mean().round(2)
    profiles["fraud_flag_count"] = grp["fraude_suspectee"].sum()
    profiles["avg_notation_pays"] = grp["notation_pays"].mean().round(2)

    # Partner concentration
    profiles["avg_partner_concentration"] = grp["partner_concentration_index"].mean().round(4)

    # Activity span
    tx_dates = pd.to_datetime(transactions["transaction_date"])
    first_tx = tx_dates.groupby(transactions["company_id"]).min()
    last_tx  = tx_dates.groupby(transactions["company_id"]).max()
    profiles["first_transaction"] = first_tx.dt.strftime("%Y-%m-%d")
    profiles["last_transaction"]  = last_tx.dt.strftime("%Y-%m-%d")
    profiles["active_days"]       = (last_tx - first_tx).dt.days

    # Merge static company info
    comp_static = companies.set_index("company_id")[
        ["company_sector", "company_size", "company_age", "city",
         "transaction_frequency"]
    ]
    profiles = profiles.merge(comp_static, left_index=True, right_index=True, how="left")

    profiles.reset_index(drop=True, inplace=True)
    return profiles


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating companies …")
    companies = generate_companies(N_COMPANIES)

    print(f"Generating {N_TRANSACTIONS:,} transactions …")
    transactions = generate_transactions(companies, N_TRANSACTIONS)

    print("Generating company profiles …")
    profiles = generate_company_profiles(transactions, companies)

    # Save
    transactions.to_csv("transactions.csv", index=False)
    profiles.to_csv("company_profiles.csv", index=False)

    print(f"\nDone!")
    print(f"  transactions.csv  → {len(transactions):,} rows × {len(transactions.columns)} cols")
    print(f"  company_profiles.csv → {len(profiles):,} rows × {len(profiles.columns)} cols")

    # Preview
    print("\n── First 20 rows of transactions ──────────────────────────────────")
    preview_cols = [
        "transaction_id", "company_id", "transaction_date", "transaction_type",
        "payment_method", "amount", "currency", "amount_MAD",
        "origin_country", "destination_country", "trade_region",
        "company_sector", "company_size", "company_city", "risk_score",
    ]
    print(transactions[preview_cols].head(20).to_string(index=False))

    print("\n── First 10 rows of company_profiles ──────────────────────────────")
    print(profiles.head(10).to_string(index=False))
