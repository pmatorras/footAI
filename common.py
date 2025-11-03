from pathlib import Path
import os 
ROOT_DIR = Path(__file__).resolve().parents[0] #to be changed if this goes to /src/
DATA_DIR = ROOT_DIR / 'data/'
RAW_DIR = DATA_DIR / 'raw/'
PROCESSED_DIR = DATA_DIR /'processed/'

COUNTRIES = {
    "SP" : {
        "name" : "Spain",
        "divisions": {
            "SP1" : "La Liga",
            "SP2" : "Segunda"
            },
        },
    "EN" : {
        "name" : "England",
        "divisions": {
            "E0" : "Premier League",
            "E1" : "Championship",
            "E2" : "EFL League 1",
            "E3" : "EFL League 2",
            "EC" : "National League"
            },
        },
    "IT" : {
        "name" : "Italy",
        "divisions": {
            "I1" : "Serie A",
            "I2" : "Serie B"
            },
        },
    "DE" : {
        "name" : "Germany",
        "divisions": {
            "D1" : "Bundesliga",
            "D2" : "2. Bundesliga"
            },
        },
    "FR" : {
        "name" : "France",
        "divisions": {
        "FR1" : "Ligue 1",
        "FR2" : "Ligue 2"
        }
    }
}
def season_to_season_str(season, country, division):
    # Convert season format (2024 -> 2425 if needed)
    if isinstance(season, int) and season > 1000:
        next_year = (season % 100) + 1
        season_str = f"{season %100}{next_year:02d}"
    else:
        season_str = str(season)
    print(season, season_str)

    return season_str
def get_data_loc(season_str, division, country, data_dir = DATA_DIR, elo=False):
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    elo_suffix = '_elo' if elo else ''

    # Create filename
    filename = f"{country.upper()}_{season_str}_{division}{elo_suffix}.csv"
    filepath =  os.path.join(data_dir, filename)
    print(f"Chosen file{elo_suffix}: {filepath}")
    return filepath

def get_elo_loc(season_str, division, country, data_dir = PROCESSED_DIR):
    # Create elo directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Create filename
    filename = f"{country.upper()}_{season_str}_D{division}.csv"
    return os.path.join(data_dir, filename)