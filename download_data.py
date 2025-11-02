import requests
from common import DATA_DIR, COUNTRY, DIV

def download_football_data(season_str, division=DIV, country=COUNTRY, filepath=DATA_DIR):
    """
    Download football match data from football-data.co.uk
    
    Parameters:
    - season (int or str): Season year (e.g., 2024 for 2024-25 season)
    - division (int): League division, 1 or 2 (default: 1)
    - country (str): Country code (default: 'SP' for Spain/La Liga)
    - data_dir (str): Directory to save CSV files (default: 'football_data')
    
    Returns:
    - tuple: (filepath, success_bool, message)
    """
    # Build the URL
    division_code = f"{country.upper()}{division}"
    url = f"https://www.football-data.co.uk/mmz4281/{season_str}/{division_code}.csv"
    
    print(f"URL: {url}")
    print(f"Output: {filepath}")

    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            #Remove byte order mark
            content = response.content.decode('utf-8-sig')
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Count matches
            lines = response.text.split('\n')
            num_matches = len([l for l in lines if l.strip() and l != lines[0]])
            
            message = f"SUCCESS: Downloaded {num_matches} matches."
            return True, message
        else:
            message = f"FAILED: Status code {response.status_code}."
            return False, message
    
    except requests.exceptions.RequestException as e:
        message = f"ERROR: {str(e)}"
        return False, message