import requests
from footai.core.config import DATA_DIR

def download_football_data(season_str, division, filepath=DATA_DIR):
    """
    Download football match data from football-data.co.uk
    
    Parameters:
    - season (int or str): Season year (e.g., 2024 for 2024-25 season)
    - division (int): League division
    - filepath (str): Directory to save CSV files (default: 'football_data')
    
    Returns:
    - tuple: (filepath, success_bool, message)
    """
    # Build the URL
    url = f"https://www.football-data.co.uk/mmz4281/{season_str}/{division}.csv"
    
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