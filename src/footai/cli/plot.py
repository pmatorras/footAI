"""Plotting command handler for footAI."""

from footai.utils.paths import get_season_paths
from footai.utils.config import COUNTRIES
from footai.viz.plotter import plot_elo_rankings

def execute(seasons, divisions, args, dirs):    
    for season in seasons:
        for division in divisions:
            paths = get_season_paths(season, division, dirs, args)
            fig = plot_elo_rankings(paths['proc'], division=division, custom_title=f"for {COUNTRIES[args.country]['divisions'][division]} ({COUNTRIES[args.country]["name"]}, season {season})")
            fig.write_html(paths['fig'])
            print(f"{season} / {division} saved to {paths['fig']}")