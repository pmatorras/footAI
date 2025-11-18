"""Logging utilities for capturing command outputs to files."""

import sys
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from footai.utils.paths import format_season_list 

class TeeLogger:
    """Prints to stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


@contextmanager
def log_training_run(country, divisions, feature_set, seasons, model='rf', multidiv=False, multicountry=False, tier=None, results_dir='results'):
    """
    Context manager for logging training runs to file.
    
    Args:
        country: Country code
        divisions: List of divisions
        feature_set: Feature set name
        seasons: List of season codes
        model: Model type
        multidiv: Multi-division training
        multicountry: Multi-country training
        tier: 'tier1', 'tier2', or None
        results_dir: Base results directory
    
    Yields:
        Path: JSON file path for writing structured metrics
    """
    timestamp = datetime.now().strftime('%Y%m%d')
    country_str = '_'.join(country) if multicountry else country
    season_str = f"{seasons[0]}_to_{seasons[-1]}" if len(seasons) > 1 else seasons[0]
    # Format division string for filename and header
    if tier:
        # Tier-specific training
        all_divs = [d for divs in divisions.values() for d in divs]
        div_str = tier
        header_div = f"{tier.capitalize()} ({', '.join(all_divs)})"
    elif multicountry:
        # Multi-country: list all divisions
        all_divs = [d for divs in divisions.values() for d in divs]
        div_str = f"{country_str}_multicountry"
        header_div = f"{', '.join(all_divs)}"
    elif multidiv:
        # Multi-division (single country): list divisions
        div_str = f"{country_str}_multidiv"
        header_div = f"{', '.join(divisions)}"
    else:
        # Single division
        div_str = divisions[0] if isinstance(divisions, list) else str(divisions)
        header_div = div_str
    

    
    out_dir = Path(results_dir) / country_str
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{div_str}_{season_str}_{feature_set}_{model}_{timestamp}"
    txt_path = out_dir / f"{filename}.txt"
    json_path = txt_path.with_suffix('.json')
    
    logger = TeeLogger(txt_path)
    old_stdout = sys.stdout
    sys.stdout = logger
    
    try:
        # Print header
        print("="*70)
        print(f"TRAINING: {header_div} ({format_season_list(seasons)})")
        print("-"*70)        
        print(f"Feature set: {feature_set}, Model: {model}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("="*70)
        yield json_path
    finally:
        sys.stdout = old_stdout
        logger.close()
        print(f"\nResults saved: {txt_path}\n")
        if json_path.exists():
            print(f"Metrics saved: {json_path}\n")
