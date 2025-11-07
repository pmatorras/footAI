# footAI

> **v0.1 - Elo Engine** | **v0.2 - ML Predictions** (In Development)

Calculate and visualize **Elo rankings** for football teams across major European leagues. 
This tool automatically downloads match data, computes dynamic Elo ratings for each team, 
and generates interactive visualizations of team performance over time. 
Supports multi-season analysis with configurable decay factors.

## Quick Start

```bash
# Download data for Spanish La Liga Division 1, seasons 2022-2025
python main.py download --country SP --div SP1 --season-start 22,23,24,25 -m

# Calculate Elo rankings (single season)
python main.py elo --country SP --div SP1 --season-start 24

# Calculate Elo rankings (multiple seasons with decay factor)
python main.py elo --country SP --div SP1 --season-start 22,23,24,25 -m --decay-factor 0.95

# Plot the results

python main.py plot --country SP --div SP1 --season-start 24
```



## Supported Countries

| Code | Country |
|------|---------|
| SP | Spain (La Liga) |
| EN | England (Premier League) |
| IT | Italy (Serie A) |
| DE | Germany (Bundesliga) |
| FR | France (Ligue 1) |




## Command Options

All subcommands (`download`, `elo`, `plot`) support these options:

| Flag | Description | Example |
|------|-------------|---------|
| `--country` | Country code (default: SP) | `--country EN` |
| `--div` | Division(s), comma-separated | `--div SP1,SP2` |
| `--season-start` | Season start year(s), comma-separated | `--season-start 22,23,24` |
| `-m, --multiseason` | Calculate across multiple seasons | `-m` |
| `-v, --verbose` | Show detailed output | `-v` |
| `--decay-factor` | Elo decay factor 0-1 (default: 0.95) | `--decay-factor 0.9` |
| `--raw-dir` | Directory for raw data (default: `football_data`) | `--raw-dir my_data` |
| `--processed-dir` | Directory for processed data | `--processed-dir my_output` |



## Elo Rating

Uses the official [Elo rating formula](https://en.wikipedia.org/wiki/Elo_rating_system):

- Initial rating: 1500
- K-factor: 32 (volatility per match)
- Supports both single matches and season progression


## Pipeline

1. **Download** – Fetch match data from [football-data.co.uk](https://football-data.co.uk)
2. **Calculate** – Compute Elo ratings for all teams per match
3. **Plot** – Visualize team progression as interactive charts
4. **Dash** – Interactive dashboard

## Files

| Module | Purpose |
|--------|---------|
| `main.py` | CLI entry point |
| `download_data.py` | Fetch raw CSV from web |
| `calculate_elo.py` | Compute Elo ratings |
| `plot_elo.py` | Generate interactive charts |
| `common.py` | Shared config & helpers |
| `app.py` |  Interactive dashboard |

## Requirements

```
pip install pandas plotly requests
```

## Data Structure

```
data/
├── raw/                                            # Downloaded, unmodified
│   └── {country}_{division}_{season}.csv           # e.g SP1_2024-25.csv
└── processed/                                      # With Elo calculated
│   └── {country}_{division}_{season}_elo.csv       # e.g SP1_2024-25_elo.csv
│   └── {country}_{division}_{season}_elo_multi.csv # e.g SP1_2024-25_elo_multi.csv
figures/                                            # interactive plots
├─── {country}_{division}_{season}_elo.csv          # e.g SP1_2024-25_elo.
└── {country}_{division}_{season}_elo_multi.html    # e.g SP1_2024-25_elo_multi.html
```

