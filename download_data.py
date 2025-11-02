import argparse
import requests
from common import DATA_DIR, COUNTRY, DIV, season_to_season_str, get_data_loc

def download_football_data(season, division=DIV, country=COUNTRY, data_dir=DATA_DIR):
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
    season_str = season_to_season_str(season, division, country)
    # Build the URL
    division_code = f"{country.upper()}{division}"
    url = f"https://www.football-data.co.uk/mmz4281/{season_str}/{division_code}.csv"
    
    filepath = get_data_loc(season_str, division, country, data_dir)
    print(f"URL: {url}")
    print(f"Output: {filepath}")

    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Count matches
            lines = response.text.split('\n')
            num_matches = len([l for l in lines if l.strip() and l != lines[0]])
            
            message = f"SUCCESS: Downloaded {num_matches} matches."
            return filepath, True, message
        else:
            message = f"FAILED: Status code {response.status_code}."
            return filepath, False, message
    
    except requests.exceptions.RequestException as e:
        message = f"ERROR: {str(e)}"
        return filepath, False, message


def main():
    parser = argparse.ArgumentParser(
        description='Download football match data from football-data.co.uk',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_football.py 2024                    # La Liga 2024-25, Division 1
  python download_football.py 2024 --division 2       # La Liga 2024-25, Division 2
  python download_football.py 2021 --country SP       # La Liga 2021-22
  python download_football.py 2024 --data-dir my_data # Save to custom directory
        """
    )
    
    parser.add_argument( 'season', type=int, help='Season year (e.g., 2024 for 2024-25 season)')
    
    parser.add_argument( '--division', type=int, default=1, choices=[1, 2], help='League division (default: 1)')
    
    parser.add_argument( '--country', type=str, default='SP', help='Country code (default: SP for Spain/La Liga)')
    
    parser.add_argument( '--data-dir', type=str, default='football_data', help='Directory to save CSV files (default: football_data)')
    
    args = parser.parse_args()
    
    filepath, success, message = download_football_data(
        season=args.season,
        division=args.division,
        country=args.country,
        data_dir=args.data_dir
    )
    
    print(message)
    
    if success:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
