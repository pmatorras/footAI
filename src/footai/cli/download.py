"""download command handler for footAI."""

from footai.utils.paths import get_season_paths
from footai.data.downloader import download_football_data

def execute(seasons, divisions, args, dirs):
    for season in seasons:
        for division in divisions:
            paths = get_season_paths(season, division, dirs, args)
            download_football_data(season, division, paths['raw'])