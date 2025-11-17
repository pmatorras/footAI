"""Promotion relegation command handler for footAI."""

from footai.utils.paths import get_previous_season
from footai.core.team_movements import identify_promotions_relegations_for_season, save_promotion_relegation

def execute(seasons, divisions, args, dirs):
    for country in args.countries:
        for season_idx, season in enumerate(seasons):
            if season_idx == 0:
                # First season - no previous season to compare
                print(f"Skipping promotion-relegation for first season ({season})")
                continue
            prev_season = get_previous_season(season)
            results = identify_promotions_relegations_for_season(season, country, prev_season, dirs, args)
            save_promotion_relegation(results, season, country, dirs)
            print(f"Saved promotion/relegation data for {prev_season} -> {season}") 