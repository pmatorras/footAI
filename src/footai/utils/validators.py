import argparse
from footai.utils.config import COUNTRIES, get_default_divisions

class ValidateDivisionAction(argparse.Action):
    """Validate that provided divisions exist for the selected country."""
    def __call__(self, parser, namespace, values, option_string=None):

        
        countries = namespace.countries.split(',')

        if values and values.lower() == 'tier1':
            divisions = {
                country: [list(COUNTRIES[country]['divisions'].keys())[0]]
                for country in countries
            }
            is_tier_specific = True
            setattr(namespace, 'tier', 'tier1')
            setattr(namespace, self.dest, divisions)
            return
        
        if values and values.lower() == 'tier2':
            divisions = {
                country: [list(COUNTRIES[country]['divisions'].keys())[1]]
                for country in countries
                if len(COUNTRIES[country]['divisions']) >= 2
            }
            is_tier_specific = True
            setattr(namespace, 'tier', 'tier2')
            setattr(namespace, self.dest, divisions)
            return
        
        # If no divisions specified, use defaults
        setattr(namespace, 'tier', None)
        if values is None:
            divisions = get_default_divisions(countries)
        else:
            # Parse user-provided divisions
            if isinstance(values, str):
                div_list = [d.strip() for d in values.split(',')]
            else:
                div_list = values
            
            # Map divisions to their countries
            divisions = {country: [] for country in countries}
            
            for div in div_list:
                # Find which country this division belongs to
                found = False
                for country in countries:
                    if div in COUNTRIES[country]['divisions']:
                        divisions[country].append(div)
                        found = True
                        break
                
                if not found:
                    # Division doesn't belong to ANY specified country
                    valid_divs = []
                    for c in countries:
                        valid_divs.extend(COUNTRIES[c]['divisions'].keys())
                    parser.error(
                        f"Invalid division '{div}'. "
                        f"Choose from: {', '.join(valid_divs)} "
                        f"(for countries: {', '.join(countries)})"
                    )
        
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
