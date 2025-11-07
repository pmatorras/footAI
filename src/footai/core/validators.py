import argparse
from footai.core.config import COUNTRIES


class ValidateDivisionAction(argparse.Action):
    """Validate that provided divisions exist for the selected country."""
    def __call__(self, parser, namespace, values, option_string=None):
        country = namespace.country
        print(COUNTRIES.keys())
        divisions = [d.strip() for d in values.split(',')]
        
        for div in divisions:
            if div not in COUNTRIES[country]['divisions']:
                valid = ', '.join(COUNTRIES[country]['divisions'].keys())
                parser.error(f"Invalid division '{div}' for {country}. Choose from: {valid}")
        
        setattr(namespace, self.dest, divisions)

def validate_decay_factor(value):
    """
    Validate that decay factor is between 0 and 1 (inclusive).
    
    Args:
        value: String value from command line
        
    Returns:
        float: Validated decay factor
        
    Raises:
        argparse.ArgumentTypeError: If value is not in valid range
    """
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Decay factor must be a number, got '{value}'")
    
    if fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError(f"Decay factor must be between 0 and 1 (inclusive), got {fvalue}")
    
    return fvalue
