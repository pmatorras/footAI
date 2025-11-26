"""download command handler for footAI."""
from footai.utils.paths import get_season_paths
from footai.data.team_colors import update_team_colors
from footai.data.match_data import download_football_data

          
def execute(countries, seasons, divisions, args, dirs):
    # Determine if run data, colors, or both 
    run_data = True
    run_colors = True

    if args.only_data:
        run_colors = False
    elif args.only_colors:
        run_data = False
    if run_data:
        print(f"Downloading match data for: {', '.join(countries)}")
        for country in countries:
            for season in seasons:
                for division in divisions[country]:
                    paths = get_season_paths(country, season, division, dirs, args)
                    download_football_data(season, division, paths['raw'])
    if run_colors:
        print(f"Updating team colors for: {', '.join(countries)}")
        
        for country in countries:
            # Path to the raw data directory for this country
            update_team_colors(country, dirs[country]['raw'], output_dir=dirs[country]['col'])
