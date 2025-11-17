"""download command handler for footAI."""

from footai.utils.paths import get_season_paths
from footai.data.downloader import download_football_data

def execute(countries, seasons, divisions, args, dirs):
    for country in countries:
        for season in seasons:
            for division in divisions[country]:
                paths = get_season_paths(country, season, division, dirs, args)
                download_football_data(season, division, paths['raw'])