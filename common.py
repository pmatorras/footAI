from pathlib import Path
import os 
ROOT_DIR = Path(__file__).resolve().parents[0] #to be changed if this goes to /src/
DATA_DIR = ROOT_DIR / 'data/'
RAW_DIR = DATA_DIR / 'raw/'
PROCESSED_DIR = DATA_DIR /'processed/'
FIG_DIR = ROOT_DIR / 'figures/'
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
def year_to_season_code(season):
    """Convert starting year to compact season format (e.g., 2024 -> '2425' for 2024-25)"""
    if isinstance(season, int) and season > 1000:
        next_year = (season % 100) + 1
        season = f"{season %100}{next_year:02d}"
    else:
        season = str(season)
    return season

def get_data_loc(season, division, country, file_dir = None, is_elo=False, is_fig=False, multi='', verbose=False):
    # Create data directory if it doesn't exist
    Path(file_dir).mkdir(parents=True, exist_ok=True)
    elo_suffix = '_elo' if is_elo else ''
    file_type = '.html' if is_fig else '.csv'
    # Create filename
    filename = f"{country.upper()}_{season}_{division}{elo_suffix}{multi}{file_type}"
    filepath =  os.path.join(file_dir, filename)
    if verbose: print(f"Chosen file: {filepath}")
    return filepath
