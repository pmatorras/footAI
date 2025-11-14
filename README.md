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
- [Model Configuration](#model-configuration)
- [Supported Countries](#supported-countries--divisions)
- [Files & Outputs](#files-and-outputs)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Roadmap](#roadmap)
- [License](#license)

## About
This project fetches football data from the main European leages, from [**football-data.co.uk**](https://football-data.co.uk/), and calculates Elo ratings using the [standard formula](https://en.wikipedia.org/wiki/Elo_rating_system):

- Initial rating: 1500
- K-factor: 32 (volatility per match)
- Supports both single matches and season progression
- Carries Elo across seasons (with a `decay-factor` set to 0.95 by default)
- Assigns newly promoted teams the Elo ranking from last season's demoted teams.

It also produces plots for each given season and an interactive dashboard. The ML component trains RandomForest models on engineered features (Elo, odds, L5 form, draw-specific signals) for outcome prediction (H/D/A), with v1.0 locking the `draw_optimized` set for ~55% accuracy and ~33% draw recall on SP1 data (2122-2425 seasons). [file:321]

## Requirements

- **Python**: 3.12+
- **Core dependencies**: pandas, plotly, requests
- **ML dependencies**: scikit-learn, xgboost, lightgbm, joblib

For full dependency list and versions, see [`pyproject.toml`](pyproject.toml).

## Quick Start

```bash
# Download data for Spanish La Liga Division 1, seasons 2022-2025
footai download --country SP --div SP1 --season-start 22,23,24,25 -m

# Calculate Elo rankings (single season)
footai elo --country SP --div SP1 --season-start 24

# Get promotions and relegations per season
footai promotion-relegation --country SP --season-start 23,24

# Calculate Elo rankings (multiple seasons with decay factor) and transfering the elo between promoted and relegated teams
footai elo --country SP --div SP1 --season-start 22,23,24,25 --multiseason --elo-transfer --decay-factor 0.95

# Train ML model with draw_optimized features (v1.0 default)
footai train --country SP --div SP1 --season-start 23,24 --elo-transfer --features-set draw_optimized -m 

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

### Commands

**download** - Fetch match data from football-data.co.uk
```bash
footai download --country SP --season-start 24
footai download --country EN --season-start 23,24 -m  # Multiple seasons
```

**promotion-relegation** - Identify promoted/relegated teams between seasons
```bash
footai promotion-relegation --country SP --season-start 23,24
```

**elo** - Calculate Elo rankings for teams
```bash
footai elo --country SP --season-start 23,24
footai elo --season-start 23,24 -m --decay-factor 0.95  # Multi-season with decay
```
**features** necessary for the ML training 
```bash
footai features --country SP --div SP1 --season-start 22,23,24,25 -m
```
**train** - Train ML models (RandomForest default; supports multi-season, multi-division, Elo transfer)
```bash
footai train --country SP --div SP1,SP2 --season-start 23,24 --elo-transfer  -m 
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
| `-v, --verbose` | Show detailed output | `-v` |
| `--processed-dir` | Directory for processed data | `--processed-dir my_output` |

### Elo-Specific Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--decay-factor` | Elo decay factor 0-1 (default: 0.95) | `--decay-factor 0.9` |

| `--raw-dir` | Directory for raw data (default: `football_data`) | `--raw-dir my_data` |

### Training-Specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Chose what model to train | `rf` |
| `--features-set` | ML features (`baseline/extended/draw_optimized`; default: `draw_optimized`) | `--features-set baseline` |
| `-m, --multiseason` | Calculate across multiple seasons | `-m` |
| `--multi-division` | Train on combined divisions (e.g., I1+I2) | `False` |
| `--nostats` | Suppress detailed statistics output | `False` |


## Model Configuration

The ML pipeline uses a RandomForestClassifier (`scikit-learn`) with balanced class weights for outcome prediction (H/D/A). Key v1.0 decisions (Nov 11, 2025; see [docs/feature_configuration_decision.md](docs/feature_configuration_decision.md)):

- **Default Features**: `draw_optimized` (~28 features: baseline Elo/odds/form + extended L5 stats + skimmed draw signals like consensus probs, under_2_5_zscore, AH diffs). Improves draw recall to ~33% vs. ~25% baseline, at 55% overall acc.
- **Performance (SP1 2122-2425, ~760 matches)**: Accuracy of 55.0% (test), draw recall 0.333, F1_draw ~0.328. (~3.1% test-CV gap).
- **Training**: 3-fold temporal CV; `n_estimators=100`, `max_depth=10`. 
- **CLI**: Use `--features-set draw_optimized` for v1.0; alternatives: `baseline` (12 features, lean), `extended` (~18 features).

---

## Supported Countries & Divisions


The tool supports the following leagues, organized by country code and division identifier:

| Country | Code | Division | Division Code | League Name          |
|---------|------|----------|---------------|----------------------|
| Spain   | `SP` | Top tier | `SP1`         | La Liga              |
|         |      | Second   | `SP2`         | Segunda              |
| Italy   | `IT` | Top tier | `I1`          | Serie A              |
|         |      | Second   | `I2`          | Serie B              |
| 󠁧England | `EN` | Top tier | `E0`          | Premier League       |
|         |      | Second   | `E1`          | Championship         |
|         |      | Third    | `E2`          | EFL League 1         |
|         |      | Fourth   | `E3`          | EFL League 2         |
|         |      | Fifth    | `EC`          | National League      |
| Germany | `DE` | Top tier | `D1`          | Bundesliga           |
|         |      | Second   | `D2`          | 2. Bundesliga        |
| France  | `FR` | Top tier | `FR1`         | Ligue 1              |
|         |      | Second   | `FR2`         | Ligue 2              |

**Usage:** Specify the country code with `--country` and division code(s) with `--div`:

## Files and outputs

```bash
├── data/
│ ├── raw/{COUNTRY}/ # SP/, IT/, EN/, etc.
│ ├── processed/{COUNTRY}/ # Elo-enhanced data
│   └── promottions/ # Data for each promoted and relegated team per season
│ └── features/{COUNTRY}/ # ML-ready features
├── models/{COUNTRY}/ # Trained models
├── results/{COUNTRY}/ # Training logs & JSON metrics
└── figures/ # Elo visualizations
```
### File Naming

Files follow the pattern: `{COUNTRY}_{SEASON}_{DIVISION}{SUFFIX}`

Examples:
- `SP_2324_SP1_elo.csv` → Spain, 2023/24, La Liga, Elo data
- `IT_2324_I1_feat.csv` → Italy, 2023/24, Serie A, features
- `EN_2223_E0_rf.pkl` → England, 2022/23, Premier League, model

**Seasons:** `2324` = 2023/24, `2223` = 2022/23, etc.

### Training Results

Each run creates two files in `results/{COUNTRY}/`:
- `.txt` → Full output (CV, confusion matrix, feature importance)
- `.json` → Structured metrics (accuracy, precision/recall, CV stats)

## Project Structure

```bash
src/footai/
├── init.py
├── main.py # Entry point for python -m footai
│
├── cli/ # Command handlers
│ ├── parser.py # Argument parser setup
│ ├── train.py # Training command orchestration
│ ├── download.py # Download command handler
│ ├── elo.py # Elo command handler
│ ├── promotion.py # Promotion/relegation handler
│ ├── features.py # Feature engineering handler
│ └── plot.py # Plotting command handler
|
├── core/ # Domain business logic
│ ├── elo.py # Elo rating calculations
│ └── team_movements.py # Promotion/relegation tracking
│
├── data/ # Data acquisition & processing
│ ├── init.py
│ └── downloader.py # Download match data from football-data.co.uk
│
├── ml/                      # Machine Learning (NEW)
│ ├── feature_engineering.py # Rolling features, odds normalization
│ ├── models.py              # Model training (RandomForest, XGBoost)
│ └── evaluation.py          # Results summary, benchmarks, metrics
│
├── utils/ # Shared infrastructure
│ ├── config.py # Constants, feature sets, directories
│ ├── paths.py # File path construction
│ ├── validators.py # Input validation
│ └── logger.py # Training run logging (stdout -> file)
│
├── viz/ # Visualization & UI
├── init.py
├── plotter.py # Interactive Plotly charts
├── dashboard.py # Dash web dashboard
└── themes.py # Color palettes & styling
```


## Pipeline

1. **Download** – Fetch match data from [football-data.co.uk](https://football-data.co.uk)
2. **Promotion** - Identify promoted/relegated teams between seasons
3. **Elo** – Compute Elo ratings for all teams per match
4. **Features** - Engineer ML features (rolling stats, Elo, odds normalization)
5. **Train** – Engineer features and train ML model for predictions
6. **Plot** – Visualize team progression as interactive charts
7. **Dash** – Interactive dashboard (WIP)

## Roadmap

### ✅ Current Status (v1.0 - November 2025)

**Completed:**
- Elo rating engine with multi-season support and decay
- Promotion/relegation tracking with Elo transfer
- Feature engineering: `baseline` (12), `extended` (18), `draw_optimized` (28)
- RandomForest training with temporal cross-validation
- Training result logging (`.txt` + `.json` metrics)
- Multi-country support (SP, IT, EN, DE, FR)
- Multi-division training (e.g., Serie A + Serie B combined)

**Performance Benchmarks (SP1, 2122-2425, ~760 matches):**
- Test accuracy: **55.0%**
- Draw recall: **33.3%** (vs. 25% baseline)
- Draw F1: **0.328**
- CV gap: ~3.1%

---

### 🔄 In Development (v1.1 - Q1 2026)

**Model Improvements:**
- Hyperparameter tuning (grid search for `max_depth`, `n_estimators`, `min_samples_split`)
- Compare GradientBoosting vs. XGBoost vs. LightGBM on `draw_optimized` features
- Probability calibration (Platt scaling, isotonic regression) for better confidence estimates
- SHAP explainability (feature importance per match, counterfactual analysis)

**Multi-League Optimization:**
- Validate `draw_optimized` on IT (Serie A), EN (Premier League), DE (Bundesliga)
- Per-league feature tuning (e.g., "low scoring" bias for Serie A vs. Premier League)
- Cross-league transfer learning (train on SP+IT, test on EN)

**Infrastructure:**
- Add `footai backtest` command for historical strategy evaluation
- Add `footai compare` command to compare feature sets side-by-side
- Results dashboard (Streamlit/Dash) to visualize training runs from `results/*.json`

---

### Future Work (v1.2+)

**Advanced Features:**
- Squad value integration (Transfermarkt API) for team strength indicators
- Player-level features (injuries, suspensions, form) for deeper model inputs
- xG (expected goals) data from advanced sources (Understat, FBref)

**Predictions & Deployment:**
- Live match predictions (consume real-time APIs like API-Football)
- Monte Carlo season simulations (predict league table, top 4, relegation probabilities)
- Web API (FastAPI) for serving model predictions
- Betting strategy backtesting with Kelly criterion and ROI tracking

**Ethics & Transparency:**
- Betting disclaimers and responsible gambling warnings
- Model card documentation (data sources, biases, limitations)
- Fairness audits (class imbalance handling, minority league representation)

 
## License

MIT License - see LICENSE file

---

**Maintainer:** [@pmatorras](https://github.com/pmatorras)