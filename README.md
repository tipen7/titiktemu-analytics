# titiktemu-analytics

Offline batch pipeline: GWR + XGBoost + LLM narrative synthesis, per the updated PRD's
"GIS & Spatial Analytics Pipeline" and "Technology Architecture" sections. This is a
**batch script, not a hosted service** — see our repo-topology discussion for why.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in DATABASE_URL at minimum
```

## Running

```bash
.venv/bin/python run_pipeline.py
```
