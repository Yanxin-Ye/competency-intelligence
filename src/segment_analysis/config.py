DATA_API = "hf://datasets/pointe77/credit-card-transaction/"

# Date range of the analysis
START_DATE_TY = "2020-01-01"
START_DATE_LY = "2019-01-01"
END_DATE_TY = "2020-05-31"
END_DATE_LY = "2019-05-31"

DIMENSION_COLS = [
    "generation",
    "gender",
    "category",
    "state",
    "market",
]  # Please put the desired dimension order here, as it will be used in the final learned outcomes formatting

RANDOM_SEED = 177

STATE_TO_MARKET = {
    # West
    "AK": "West",
    "AZ": "West",
    "CA": "West",
    "CO": "West",
    "HI": "West",
    "ID": "West",
    "MT": "West",
    "NM": "West",
    "NV": "West",
    "OR": "West",
    "UT": "West",
    "WA": "West",
    "WY": "West",
    # Midwest
    "IA": "Midwest",
    "IL": "Midwest",
    "IN": "Midwest",
    "KS": "Midwest",
    "MI": "Midwest",
    "MN": "Midwest",
    "MO": "Midwest",
    "ND": "Midwest",
    "NE": "Midwest",
    "OH": "Midwest",
    "SD": "Midwest",
    "WI": "Midwest",
    # South
    "AL": "South",
    "AR": "South",
    "DC": "South",
    "DE": "South",
    "FL": "South",
    "GA": "South",
    "KY": "South",
    "LA": "South",
    "MD": "South",
    "MS": "South",
    "NC": "South",
    "OK": "South",
    "SC": "South",
    "TN": "South",
    "TX": "South",
    "VA": "South",
    "WV": "South",
    # Northeast
    "CT": "Northeast",
    "MA": "Northeast",
    "ME": "Northeast",
    "NH": "Northeast",
    "NJ": "Northeast",
    "NY": "Northeast",
    "PA": "Northeast",
    "RI": "Northeast",
    "VT": "Northeast",
}
