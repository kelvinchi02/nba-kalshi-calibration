# nba-kalshi-calibration

Data pipeline for the STSCI 4981/5981 final project: are Kalshi NBA game prices before tipoff well calibrated, and are popular teams overpriced?

Setup and run instructions will be filled in as the pipeline stages are built.

## Environment

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

## Secrets (optional)

```bash
cp .env.example .env
python scripts/check_secrets.py   # reports what is set, never prints values
```

All Kalshi endpoints this pipeline uses (series, events, markets, trades, candlesticks and the
`/historical/*` endpoints) are public, so an API key is not required. If you add one, put the key
ID in `.env` and the private key at `secrets/kalshi_private_key.pem`. Both are gitignored.
