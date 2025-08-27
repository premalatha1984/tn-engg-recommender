import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from math import radians, sin, cos, asin, sqrt

CATEGORY_ORDER = ["OC", "BC", "MBC", "SC", "ST"]

# rough district centroid lat/long (synthetic small set for demo)
DISTRICT_COORDS = {
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Salem": (11.6643, 78.1460),
    "Tirunelveli": (8.7139, 77.7567),
    "Erode": (11.3410, 77.7172),
    "Vellore": (12.9165, 79.1325),
    "Thanjavur": (10.7867, 79.1378),
}

def haversine_km(lat1, lon1, lat2, lon2):
    # Earth radius in km
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def distance_between_districts(d1: str, d2: str) -> float:
    if d1 not in DISTRICT_COORDS or d2 not in DISTRICT_COORDS:
        return 150.0  # fallback
    (a_lat, a_lon) = DISTRICT_COORDS[d1]
    (b_lat, b_lon) = DISTRICT_COORDS[d2]
    return haversine_km(a_lat, a_lon, b_lat, b_lon)

def normalize_series(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    min_v, max_v = s.min(), s.max()
    if max_v - min_v == 0:
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - min_v) / (max_v - min_v)

def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    colleges = pd.read_csv(f"{data_dir}/colleges.csv")
    programs = pd.read_csv(f"{data_dir}/programs.csv")
    cutoffs = pd.read_csv(f"{data_dir}/cutoffs.csv")
    return colleges, programs, cutoffs

def get_required_cutoff(cutoffs_df: pd.DataFrame, college_code: str, branch: str, category: str) -> float:
    row = cutoffs_df[(cutoffs_df.college_code==college_code) & (cutoffs_df.branch==branch)]
    if row.empty:
        # conservative: require higher cutoff if unknown
        return 180.0
    # pick category column or fallback to OC
    cat_col = category if category in row.columns else "OC"
    return float(row.iloc[0][cat_col])

def score_row(row, weights) -> Tuple[float, Dict[str, float], List[str]]:
    components = {
        "affordability": row["affordability_norm"],
        "proximity": row["proximity_norm"],
        "placements": row["placement_rate_norm"],
        "quality": row["quality_score_norm"],
        "rural_support": row["rural_support_norm"],
        "hostel": row["hostel_norm"],
        "branch_priority": row["branch_priority_norm"],
    }
    total = sum(getattr(weights, k) * components[k] for k in components.keys())
    notes = []
    if row.get("budget_stretch", 0) > 0:
        total *= 0.9
        notes.append("Budget slightly above limit: applied 10% penalty.")
    return total, components, notes

def recommend(profile, weights, colleges, programs, cutoffs) -> List[Dict]:
    # Merge programs with college info
    df = programs.merge(colleges, on="college_code", how="left", suffixes=("_prog",""))
    # Eligibility filter by cutoff
    def eligible(rec):
        req = get_required_cutoff(cutoffs, rec["college_code"], rec["branch"], profile.category)
        return profile.cutoff >= req, req
    elig_list, req_list = [], []
    for _, r in df.iterrows():
        ok, req = eligible(r)
        elig_list.append(ok)
        req_list.append(req)
    df["eligible"] = elig_list
    df["required_cutoff"] = req_list
    df = df[df["eligible"]]

    if df.empty:
        return []

    # Budget
    df["budget_ok"] = df["annual_fee"] <= profile.budget
    df["budget_stretch"] = (~df["budget_ok"]).astype(int)

    # Distance
    df["distance_km"] = df.apply(lambda r: distance_between_districts(profile.district, r["district"]), axis=1)

    # Preferred branches
    preferred = set([b.strip().upper() for b in profile.preferred_branches]) if profile.preferred_branches else set()
    df["branch_priority"] = df["branch"].str.upper().apply(lambda b: 1.0 if b in preferred else 0.5 if len(preferred)>0 else 0.7)

    # Hostel need
    df["hostel_ok"] = df["hostel_available"].astype(int)
    if profile.need_hostel:
        df = df[df["hostel_ok"] == 1]
        if df.empty:
            return []

    # Rural/first-gen boost as a metric (not a hard filter)
    df["rural_support_score"] = df["rural_support"].astype(int)

    # Normalize metrics
    # Affordability: lower fee is better => invert before normalize
    df["affordability"] = -df["annual_fee"]
    df["affordability_norm"] = normalize_series(df["affordability"])

    # Proximity: lower distance is better => invert
    df["proximity"] = -df["distance_km"]
    df["proximity_norm"] = normalize_series(df["proximity"])

    df["placement_rate_norm"] = normalize_series(df["placement_rate"])
    df["quality_score_norm"] = normalize_series(df["quality_score"])
    df["rural_support_norm"] = normalize_series(df["rural_support_score"])
    df["hostel_norm"] = normalize_series(df["hostel_ok"])
    df["branch_priority_norm"] = normalize_series(df["branch_priority"])

    # Compute eligibility margin
    df["eligibility_margin"] = profile.cutoff - df["required_cutoff"]

    # Score
    totals, comps, notes_list = [], [], []
    for _, r in df.iterrows():
        total, components, notes = score_row(r, weights)
        totals.append(total)
        comps.append(components)
        notes_list.append(notes)
    df["total_score"] = totals
    df["components"] = comps
    df["notes"] = notes_list

    # Tie-breakers: total_score desc, eligibility_margin desc, ownership pref (Govt > Govt-Aided > Private), proximity asc
    ownership_rank = {"Government": 3, "Government-Aided": 2, "Private": 1}
    df["ownership_rank"] = df["ownership"].map(ownership_rank).fillna(0)

    df = df.sort_values(
        by=["total_score", "eligibility_margin", "ownership_rank", "proximity"],
        ascending=[False, False, False, False]
    )

    # Build output
    out = []
    for i, r in enumerate(df.itertuples(), start=1):
        out.append({
            "rank": i,
            "college_code": r.college_code,
            "college_name": r.college_name,
            "district": r.district,
            "ownership": r.ownership,
            "program": r.branch,
            "annual_fee": int(r.annual_fee),
            "placement_rate": float(r.placement_rate),
            "quality_score": float(r.quality_score),
            "distance_km": float(r.distance_km),
            "eligibility_margin": float(r.eligibility_margin),
            "total_score": float(r.total_score),
            "explanation": {
                "components": r.components,
                "notes": r.notes,
            }
        })
    return out
