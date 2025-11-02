DATA_DIR='football_data'
COUNTRY = 'SP'
DIV=1
from pathlib import Path
import os 

def season_to_season_str(season, country, division):
    # Convert season format (2024 -> 2425 if needed)
    if isinstance(season, int) and season > 1000:
        next_year = (season % 100) + 1
        season_str = f"{season %100}{next_year:02d}"
    else:
        season_str = str(season)
    print(season, season_str)

    return season_str
def get_data_loc(season_str, division=DIV, country=COUNTRY, data_dir = DATA_DIR):
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Create filename
    filename = f"{country.upper()}_{season_str}_D{division}.csv"
    return os.path.join(data_dir, filename)
