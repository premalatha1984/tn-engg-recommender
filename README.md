# TN Engineering Recommender

An end-to-end, transparent recommendation system to help +2 students in Tamil Nadu
choose suitable engineering colleges & courses during counselling.

## Stack
- **Backend:** FastAPI (Python), in-memory pandas (CSV seed), transparent scoring engine
- **Frontend:** Static HTML + JS (fetch API) with lightweight UX and Tamil language support
- **Data:** Sample CSVs (synthetic) with clear schema; swap with official data when available

## Features
- Student profile input: cutoff, category (OC/BC/MBC/SC/ST), preferred branches, district, budget
- Smart eligibility filter (category-wise), realistic budget check
- Multi-criteria scoring with configurable weights (affordability, proximity, placements, quality, diversity support)
- Rural/first-generation support flag
- Personalized ranking with **explanations per recommendation** (transparent breakdown)
- Tamil language toggle for parents
- Simple, auditable code; replace sample data with real data

## Quick Start (Dev)

### 1) Python backend
```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
