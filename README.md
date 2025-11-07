# footAI

> **v0.1 - Elo Engine** | **v0.2 - ML Predictions** (In Development)

Calculate and visualize **Elo rankings** for football teams across major European leagues. 
This tool automatically downloads match data, computes dynamic Elo ratings for each team, 
and generates interactive visualizations of team performance over time. 
Supports multi-season analysis with configurable decay factors.

## Table of Contents

- [About](#about)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Files & Outputs](#files-and-outputs)
- [Supported Countries](#supported-countries)
- [Roadmap](#roadmap)
- [Requirements](#requirements)
- [License](#license)

## About
This project takes the fetches football data from the main European leages, from [**football-data.co.uk**](https://football-data.co.uk/), and calculates Elo ratings using the [standard formula](https://en.wikipedia.org/wiki/Elo_rating_system):

- Initial rating: 1500
- K-factor: 32 (volatility per match)
- Supports both single matches and season progression
- Carries elo seasons between seasons (with a `decay-factor` set to 0.95 by default)
- Assigns to newly promoted teams the Elo ranking from last season's demoted teams.

It also produces plots for each given season, and an interactive dashboard
## Quick Start

```bash
# Download data for Spanish La Liga Division 1, seasons 2022-2025
footai download --country SP --div SP1 --season-start 22,23,24,25 -m

# Calculate Elo rankings (single season)
footai elo --country SP --div SP1 --season-start 24

# Calculate Elo rankings (multiple seasons with decay factor) and transfering the elo between promoted and relegated teams
footai elo --country SP --div SP1 --season-start 22,23,24,25 --multiseason --elo-transfer --decay-factor 0.95

# Plot the results
footai plot --country SP --div SP1 --season-start 24 --multiseason --elo-transfer
```

## Installation

```bash
# Clone repository
git clone https://github.com/pmatorras/footAI.git
cd footAI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  \# On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .

```

## Usage
footAI provides four main commands to download data, calculate Elo ratings, track team movements, and visualize results.

### Available Commands

**download** - Fetch match data from football-data.co.uk
```bash
footai download --country SP --season-start 24
footai download --country EN --season-start 23,24 -m  # Multiple seasons
```

**elo** - Calculate Elo rankings for teams
```bash
footai elo --country SP --season-start 24
footai elo --season-start 24 -m --decay-factor 0.95  # Multi-season with decay
```

**promotion-relegation** - Identify promoted/relegated teams between seasons
```bash
footai promotion-relegation --country SP --season-start 23,24
```

**plot** - Generate interactive visualizations of Elo progression
```bash
footai plot --country SP --season-start 24
```


### Command Options

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

### Examples

```bash
# Download Spanish La Liga seasons 2022-2025
footai download --country SP --season-start 22,23,24,25 -m

# Calculate Elo for Premier League 2024-25 with custom directory
footai elo --country EN --season-start 24 --processed-dir my_output

# Generate multi-season plot with Elo decay
footai plot --country SP --season-start 22,23,24,25 -m --decay-factor 0.95

# Track team movements with verbose output
footai promotion-relegation --country SP --season-start 23,24 -v --elo-transfer
```

## Roadmap

**Current Features (v0.1)**
- ✅ Elo calculations
- ✅ Multi-season analysis

**In Development (v0.2)**
- 🔄 ML predictions
- 🔄 Feature engineering




## Pipeline

1. **Download** – Fetch match data from [football-data.co.uk](https://football-data.co.uk)
2. **Calculate** – Compute Elo ratings for all teams per match
3. **Plot** – Visualize team progression as interactive charts
4. **Dash** – Interactive dashboard


## Project Structure

```bash
src/footai/
├── init.py
├── main.py # Entry point for python -m footai
├── cli.py # Argument parser setup
├── main.py # Business logic & command dispatch
│
├── core/ # Domain business logic
│ ├── init.py
│ ├── config.py # Configuration & constants
│ ├── elo.py # Elo rating calculations
│ ├── team_movements.py # Promotion/relegation tracking
│ ├── validators.py # Input validation
│ └── utils.py # Utility functions
│
├── data/ # Data acquisition & processing
│ ├── init.py
│ └── downloader.py # Download match data from football-data.co.uk
│
└── viz/ # Visualization & UI
├── init.py
├── plotter.py # Interactive Plotly charts
├── dashboard.py # Dash web dashboard
└── themes.py # Color palettes & styling
```



## Files and outputs

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

## Supported Countries

| Code | Country |
|------|---------|
| SP | Spain (La Liga) |
| EN | England (Premier League) |
| IT | Italy (Serie A) |
| DE | Germany (Bundesliga) |
| FR | France (Ligue 1) |

## Requirements

- Python 3.12+
- pandas, plotly, requests

## License

MIT License - see LICENSE file

---

**Maintainer:** [@pmatorras](https://github.com/pmatorras)