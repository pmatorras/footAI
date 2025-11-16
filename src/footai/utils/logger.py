"""Logging utilities for capturing command outputs to files."""

import sys
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager


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
def log_training_run(country, divisions, feature_set, seasons, model='rf', multidiv=False, results_dir='results'):
    """
    Context manager for logging training runs to file.
    
    Args:
        country: Country code
        divisions: List of divisions
        feature_set: Feature set name
        seasons: List of season codes
        model: Model type
        results_dir: Base results directory
    
    Yields:
        Path: JSON file path for writing structured metrics
    """
    timestamp = datetime.now().strftime('%Y%m%d')
    div_str = country+"_multidiv" if multidiv else str(divisions)
    season_str = f"{seasons[0]}_to_{seasons[-1]}" if len(seasons) > 1 else seasons[0]
    
    out_dir = Path(results_dir) / country
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
        print(f"TRAINING: {divisions} ({seasons})")
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
