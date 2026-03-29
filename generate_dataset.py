"""
Synthetic International Banking Transactions Dataset Generator
Business Center: Fes-Meknes, Morocco
Generates a realistic Big-Data-ready dataset for EDA, clustering, and profiling.
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

CITIES = {
    "Fes": 0.45,
    "Meknes": 0.35,
    "Ifrane": 0.05,
    "Sefrou": 0.05,
    "Khénifra": 0.05,
    "Azrou": 0.05,
}

SECTORS = ["agriculture", "textile", "industry", "services", "tech", "trade"]
SECTOR_WEIGHTS = [0.15, 0.18, 0.22, 0.20, 0.10, 0.15]

SIZES = ["small", "medium", "large"]
SIZE_WEIGHTS = [0.55, 0.35, 0.10]

TRANSACTION_TYPES = ["import", "export", "domestic", "transfer"]
PAYMENT_METHODS = ["wire", "cash", "check", "mobile_payment", "card"]

CURRENCIES = ["MAD", "EUR", "USD", "CNY", "GBP", "SAR", "XOF"]
# EUR ↔ MAD ≈ 10.8, USD ↔ MAD ≈ 10.1, CNY ↔ MAD ≈ 1.40, GBP ↔ MAD ≈ 12.7
# SAR ↔ MAD ≈ 2.70, XOF ↔ MAD ≈ 0.016
EXCHANGE_RATES = {
    "MAD": 1.0,
    "EUR": 10.80,
    "USD": 10.10,
    "CNY": 1.40,
    "GBP": 12.70,
    "SAR": 2.70,
    "XOF": 0.016,
}

# Countries & their regions
COUNTRIES = {
    "France":        "Europe",
    "Spain":         "Europe",
    "Germany":       "Europe",
    "Italy":         "Europe",
    "Netherlands":   "Europe",
    "Senegal":       "Africa",
    "Ivory Coast":   "Africa",
    "Mali":          "Africa",
    "Nigeria":       "Africa",
    "Egypt":         "Africa",
    "Morocco":       "Africa",
    "China":         "Asia",
    "Japan":         "Asia",
    "India":         "Asia",
    "UAE":           "Middle East",
    "Saudi Arabia":  "Middle East",
    "USA":           "America",
    "Canada":        "America",
    "Brazil":        "America",
    "UK":            "Europe",
}

# Preferred partners per sector
SECTOR_PARTNER_COUNTRIES = {
    "agriculture": {"France": 0.25, "Spain": 0.20, "Senegal": 0.15, "Ivory Coast": 0.15,
                    "Netherlands": 0.10, "Morocco": 0.05, "Germany": 0.05, "UK": 0.05},
    "textile":     {"France": 0.20, "Spain": 0.15, "Germany": 0.15, "Italy": 0.15,
                    "China": 0.20, "Morocco": 0.05, "UK": 0.05, "USA": 0.05},
    "industry":    {"China": 0.30, "France": 0.15, "Germany": 0.15, "Spain": 0.10,
                    "UAE": 0.10, "USA": 0.10, "Morocco": 0.05, "Italy": 0.05},
    "services":    {"Morocco": 0.50, "France": 0.15, "Spain": 0.10, "Senegal": 0.10,
                    "UAE": 0.05, "UK": 0.05, "USA": 0.05},
    "tech":        {"USA": 0.25, "France": 0.15, "Germany": 0.15, "UK": 0.15,
                    "India": 0.15, "China": 0.10, "Morocco": 0.05},
    "trade":       {"China": 0.20, "France": 0.15, "Spain": 0.15, "UAE": 0.10,
                    "Morocco": 0.10, "Senegal": 0.10, "USA": 0.10, "Germany": 0.10},
}

# Preferred currency per destination region
REGION_CURRENCY = {
    "Europe":      {"EUR": 0.75, "MAD": 0.10, "USD": 0.10, "GBP": 0.05},
    "Africa":      {"MAD": 0.30, "EUR": 0.25, "USD": 0.25, "XOF": 0.20},
    "Asia":        {"USD": 0.50, "CNY": 0.30, "EUR": 0.15, "MAD": 0.05},
    "Middle East": {"USD": 0.40, "SAR": 0.35, "EUR": 0.15, "MAD": 0.10},
    "America":     {"USD": 0.80, "EUR": 0.10, "CAD": 0.05, "MAD": 0.05},
}

# Moroccan country code (domestic)
MOROCCO = "Morocco"

# Risk scores by country (notation 1–5 and score 0–100)
COUNTRY_RISK = {
    "France":       (1, 10), "Spain":       (1, 12), "Germany":    (1,  8),
    "Italy":        (2, 20), "Netherlands": (1,  9), "UK":         (1, 11),
    "Senegal":      (3, 45), "Ivory Coast": (3, 50), "Mali":       (4, 65),
    "Nigeria":      (4, 70), "Egypt":       (3, 48), "Morocco":    (2, 25),
    "China":        (2, 30), "Japan":       (1, 10), "India":      (2, 28),
    "UAE":          (2, 22), "Saudi Arabia":(2, 25), "USA":        (1, 10),
    "Canada":       (1,  9), "Brazil":      (3, 40),
}

# Ramadan approximate start months (March 2020, Apr 2021, Apr 2022, Mar 2023, Mar 2024)
RAMADAN_MONTHS = {2020: 4, 2021: 4, 2022: 4, 2023: 3, 2024: 3}  # month number


# ── Helper functions ──────────────────────────────────────────────────────────

def _weighted_choice(mapping: dict) -> str:
    keys = list(mapping.keys())
    weights = list(mapping.values())
    # normalise
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
    """Generate a realistic transaction amount."""
    base_ranges = {
        ("agriculture", "small"):   (5_000,  80_000),
        ("agriculture", "medium"):  (50_000, 500_000),
        ("agriculture", "large"):   (200_000, 3_000_000),
        ("textile",     "small"):   (3_000,  50_000),
        ("textile",     "medium"):  (30_000, 300_000),
        ("textile",     "large"):   (150_000, 2_000_000),
        ("industry",    "small"):   (10_000, 150_000),
        ("industry",    "medium"):  (80_000, 800_000),
        ("industry",    "large"):   (300_000, 5_000_000),
        ("services",    "small"):   (1_000,  20_000),
        ("services",    "medium"):  (10_000, 100_000),
        ("services",    "large"):   (50_000, 500_000),
        ("tech",        "small"):   (5_000,  60_000),
        ("tech",        "medium"):  (30_000, 400_000),
        ("tech",        "large"):   (100_000, 2_000_000),
        ("trade",       "small"):   (5_000,  80_000),
        ("trade",       "medium"):  (40_000, 600_000),
        ("trade",       "large"):   (200_000, 4_000_000),
    }
    lo, hi = base_ranges.get((sector, size), (10_000, 200_000))
    # Log-normal distribution for realistic heavy tail
    mean = (lo + hi) / 2
    sigma = 0.8
    val = np.random.lognormal(np.log(mean), sigma)
    # Clip at 3× hi to limit extreme outliers (but keep some)
    return float(np.clip(val, lo / 10, hi * 3))


def _transaction_type_for_sector(sector: str) -> str:
    mapping = {
        "agriculture": {"export": 0.55, "domestic": 0.25, "import": 0.10, "transfer": 0.10},
        "textile":     {"export": 0.40, "import": 0.30, "domestic": 0.20, "transfer": 0.10},
        "industry":    {"import": 0.40, "export": 0.30, "domestic": 0.20, "transfer": 0.10},
        "services":    {"domestic": 0.60, "transfer": 0.20, "import": 0.10, "export": 0.10},
        "tech":        {"domestic": 0.40, "import": 0.30, "export": 0.15, "transfer": 0.15},
        "trade":       {"import": 0.35, "export": 0.35, "domestic": 0.20, "transfer": 0.10},
    }
    return _weighted_choice(mapping.get(sector, {"domestic": 0.5, "import": 0.25, "export": 0.25}))


def _payment_method_for_type(t_type: str) -> str:
    mapping = {
        "import":   {"wire": 0.60, "card": 0.15, "check": 0.15, "cash": 0.05, "mobile_payment": 0.05},
        "export":   {"wire": 0.65, "check": 0.15, "card": 0.10, "cash": 0.05, "mobile_payment": 0.05},
        "domestic": {"wire": 0.30, "cash": 0.25, "check": 0.20, "mobile_payment": 0.15, "card": 0.10},
        "transfer": {"wire": 0.70, "mobile_payment": 0.20, "card": 0.05, "cash": 0.03, "check": 0.02},
    }
    return _weighted_choice(mapping.get(t_type, {"wire": 0.5, "cash": 0.5}))


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
        base_freq = {"small": 5, "medium": 15, "large": 40}[size]
        freq = max(1, int(np.random.lognormal(np.log(base_freq), 0.4)))

        # Risk score correlated with sector
        sector_base_risk = {
            "agriculture": 0.20, "textile": 0.25, "industry": 0.30,
            "services": 0.15, "tech": 0.18, "trade": 0.35,
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
    # Build cumulative weights per company (larger companies → more transactions)
    size_tx_weight = {"small": 1, "medium": 3, "large": 8}
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

        # Transaction type
        t_type = _transaction_type_for_sector(sector)

        # Destination country
        partner_map = SECTOR_PARTNER_COUNTRIES.get(sector, {"Morocco": 1.0})
        dest_country = _weighted_choice(partner_map)

        # Origin country (Morocco exports → Morocco origin; imports → foreign origin)
        if t_type == "export":
            origin_country = MOROCCO
        elif t_type == "import":
            origin_country = dest_country
            dest_country = MOROCCO
        elif t_type == "domestic":
            origin_country = MOROCCO
            dest_country = MOROCCO
        else:  # transfer
            origin_country = MOROCCO
            dest_country = _weighted_choice(partner_map)

        region = COUNTRIES.get(dest_country if dest_country != MOROCCO else origin_country, "Africa")
        is_international = int(dest_country != MOROCCO or origin_country != MOROCCO)

        # Currency
        if not is_international:
            currency = "MAD"
        else:
            region_key = COUNTRIES.get(
                dest_country if dest_country != MOROCCO else origin_country, "Europe"
            )
            curr_map = REGION_CURRENCY.get(region_key, {"EUR": 0.5, "USD": 0.5})
            currency = _weighted_choice(curr_map)

        # Exchange rate (add small noise)
        base_rate = EXCHANGE_RATES.get(currency, 1.0)
        exchange_rate = base_rate * np.random.uniform(0.97, 1.03)

        # Amount
        amount = _amount_for_sector_size(sector, size)

        # Add outlier transactions (~1 % of rows)
        if np.random.random() < 0.01:
            amount *= np.random.uniform(10, 50)

        amount_mad = amount * exchange_rate

        # Payment method
        payment = _payment_method_for_type(t_type)

        # Risk
        country_risk_notation, country_risk_score = COUNTRY_RISK.get(
            dest_country if dest_country != MOROCCO else origin_country,
            (3, 40)
        )
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
            "destination_country":    dest_country,
            "region":                 region,
            "company_sector":         sector,
            "company_size":           size,
            "company_age":            int(comp["company_age"]),
            "city":                   comp["city"],
            "transaction_frequency":  int(comp["transaction_frequency"]),
            "avg_transaction_amount": None,           # filled post-hoc
            "risk_score":             round(risk_base, 4),
            "is_international":       is_international,
            # Extended fields
            "notation_pays":          country_risk_notation,
            "score_risque_pays":      country_risk_score,
            "risque_operationnel":    round(risk_operational, 4),
            "fraude_suspectee":       fraude_suspectee,
            "score_aml":              round(aml_score, 2),
            "rang_transaction_entreprise": None,      # filled post-hoc
            "montant_precedent":      None,           # filled post-hoc
            "evolution_montant_pct":  None,           # filled post-hoc
        })

    df = pd.DataFrame(records)

    # ── Post-hoc derived columns ──────────────────────────────────────────────
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df.sort_values(["company_id", "transaction_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # avg_transaction_amount – per company mean of amount_MAD
    avg_map = df.groupby("company_id")["amount_MAD"].mean().round(2)
    df["avg_transaction_amount"] = df["company_id"].map(avg_map)

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
    profiles["company_id"]          = grp["company_id"].first()
    profiles["total_transactions"]   = grp["transaction_id"].count()
    profiles["total_volume_MAD"]     = grp["amount_MAD"].sum().round(2)
    profiles["avg_transaction_MAD"]  = grp["amount_MAD"].mean().round(2)
    profiles["max_transaction_MAD"]  = grp["amount_MAD"].max().round(2)
    profiles["min_transaction_MAD"]  = grp["amount_MAD"].min().round(2)
    profiles["std_transaction_MAD"]  = grp["amount_MAD"].std().round(2)

    # Main partner country (by count)
    profiles["main_partner_country"] = (
        transactions[transactions["destination_country"] != "Morocco"]
        .groupby("company_id")["destination_country"]
        .agg(lambda x: x.mode()[0] if len(x) > 0 else "Morocco")
    )
    profiles["main_partner_country"].fillna("Morocco", inplace=True)

    # Import / Export ratio
    imports = transactions[transactions["transaction_type"] == "import"] \
                .groupby("company_id")["amount_MAD"].sum()
    exports = transactions[transactions["transaction_type"] == "export"] \
                .groupby("company_id")["amount_MAD"].sum()
    profiles["total_imports_MAD"]   = imports.round(2)
    profiles["total_exports_MAD"]   = exports.round(2)
    profiles["import_export_ratio"] = (imports / exports.replace(0, np.nan)).round(4)

    # International share
    intl = transactions.groupby("company_id")["is_international"].mean().round(4)
    profiles["international_share"] = intl

    # Risk aggregates
    profiles["avg_risk_score"]      = grp["risk_score"].mean().round(4)
    profiles["avg_aml_score"]       = grp["score_aml"].mean().round(2)
    profiles["fraud_flag_count"]    = grp["fraude_suspectee"].sum()
    profiles["avg_notation_pays"]   = grp["notation_pays"].mean().round(2)

    # Activity span
    tx_dates = pd.to_datetime(transactions["transaction_date"])
    first_tx = tx_dates.groupby(transactions["company_id"]).min()
    last_tx  = tx_dates.groupby(transactions["company_id"]).max()
    profiles["first_transaction"]   = first_tx.dt.strftime("%Y-%m-%d")
    profiles["last_transaction"]    = last_tx.dt.strftime("%Y-%m-%d")
    profiles["active_days"]         = (last_tx - first_tx).dt.days

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
        "amount", "currency", "amount_MAD", "origin_country", "destination_country",
        "region", "company_sector", "risk_score", "is_international",
    ]
    print(transactions[preview_cols].head(20).to_string(index=False))

    print("\n── First 10 rows of company_profiles ──────────────────────────────")
    print(profiles.head(10).to_string(index=False))
