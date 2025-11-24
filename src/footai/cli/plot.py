"""Plotting command handler for footAI."""
from pathlib import Path
from footai.utils.paths import get_season_paths
from footai.utils.config import COUNTRIES
from footai.viz.plotter import plot_elo_rankings
from footai.viz.model_viz import generate_model_visualizations

def execute(countries, seasons, divisions, args, dirs):
    """
    Generate visualizations (Elo rankings or model performance).
    
    If --results-json is provided, generates model visualizations.
    Otherwise, generates Elo ranking plots for specified seasons/divisions.
    """
    # MODEL VISUALIZATION MODE
    if hasattr(args, 'results_json') and args.results_json:
        generate_model_visualizations(args.results_json, output_dir=args.output_dir, top_n=args.top_n)
        return 0
    
    #ELO PLOTTER
    for season in seasons:
        for division in divisions:
            paths = get_season_paths(season, division, dirs, args)
            fig = plot_elo_rankings(paths['proc'], division=division, custom_title=f"for {COUNTRIES[args.country]['divisions'][division]} ({COUNTRIES[args.country]["name"]}, season {season})")
            fig.write_html(paths['fig'])
            print(f"{season} / {division} saved to {paths['fig']}")