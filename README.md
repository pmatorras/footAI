# footAI

Calculate and visualize Elo rankings for Spanish football divisions (LaLiga SP1 and SP2).

## Quick Start

```
# Download raw data and calculate Elo ratings

python main.py --help
python main.py 2024 --div 1

# Plot the results

python -c "from plot_elo import plot_elo_rankings; plot_elo_rankings('data/processed/laliga_with_elo_SP1_2024-25.csv').show()"
```

## Pipeline

1. **Download** – Fetch match data from [football-data.co.uk](https://football-data.co.uk)
2. **Calculate** – Compute Elo ratings for all teams per match
3. **Plot** – Visualize team progression as interactive charts
4. **Dash** – Interactive dashboard (coming soon)

## Files

| Module | Purpose |
|--------|---------|
| `main.py` | CLI entry point |
| `download_data.py` | Fetch raw CSV from web |
| `calculate_elo.py` | Compute Elo ratings |
| `plot_elo.py` | Generate interactive charts |
| `common.py` | Shared config & helpers |

## Requirements

```
pip install pandas plotly requests
```

## Data Structure

```
data/
├── raw/                              \# Downloaded, unmodified
│   └── SP1_2024-25.csv
└── processed/                        \# With Elo calculated
└── laliga_with_elo_SP1_2024-25.csv
```

## Elo Rating

Uses the official [Elo rating formula](https://en.wikipedia.org/wiki/Elo_rating_system):

- Initial rating: 1500
- K-factor: 32 (volatility per match)
- Supports both single matches and season progression