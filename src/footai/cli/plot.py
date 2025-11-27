"""Plotting command handler for footAI."""

from footai.utils.paths import (
    get_data_loc,
    get_season_paths,
    get_multiseason_path
)
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
    for country in countries:
        if args.multi_season:
            suffix = '_transfer' if args.elo_transfer else '_multi'  
            for division in divisions[country]:
                path = get_multiseason_path(dirs[country]['proc'], division, seasons[0],seasons[-1], args)
                fig = plot_elo_rankings(path, division=division, country=country, selected_seasons=seasons, custom_title=f"for {COUNTRIES[country]['divisions'][division]} ({COUNTRIES[country]['name']}, seasons {seasons[0]}-{seasons[-1]})")
                fig_dir = get_data_loc(f"{seasons[0]}to{seasons[-1]}", division, country, dirs[country]['fig'], file_type='fig', suffix=suffix, verbose=args.verbose)
                print("Save figure to ", fig_dir)
                fig.write_html(fig_dir)
        else:
            for season in seasons:
                for division in divisions:
                        paths = get_season_paths(country, season, division, dirs, args)
                        fig = plot_elo_rankings(paths['proc'], division=division, country=country, custom_title=f"for {COUNTRIES[country]['divisions'][division]} ({COUNTRIES[country]['name']}, season {season})")
                        fig.write_html(paths['fig'])
                        print(f"{season} / {division} saved to {paths['fig']}")