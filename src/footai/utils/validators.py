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



def validate_decay_factors(value):
    """
    Validate that decay factor(s) are between 0 and 1 (inclusive). Accepts single float or comma-separated list.
    
    Args:
        value: String value from command line

    Accepts:
        - Single float: "0.95" -> [0.95, 0.95] (Replicates value for tier 2)
        - List string: "0.95,0.90" -> [0.95, 0.90]
        - Space separated: "0.95 0.90" -> [0.95, 0.90]

    Returns:
        list[float]: A list of validated decay factors
        
    Raises:
        argparse.ArgumentTypeError: If any of the values is not in valid [0,1] range
    """
    factors = []
    try:
        #save if it comes already as list
        if isinstance(value, list):
            parts = value
        else:
            # Replace comma with space to handle "0.9,0.8" and "0.9 0.8"
            parts = str(value).replace(',', ' ').split()

        factors = [float(p) for p in parts]

        for fvalue in factors:
            if fvalue < 0 or fvalue > 1:
                    raise argparse.ArgumentTypeError(f"Decay factor must be between 0 and 1 (inclusive), got {fvalue}")

        if len(factors) > 2:
            raise argparse.ArgumentTypeError("Decay factor accepts at most two values (for tier1 and tier2)")
        #Normalise to dictionary
        if len(factors) == 1:
            # Single value applies to both tiers
            return {'tier1': factors[0], 'tier2': factors[0]}
        
        if len(factors) == 2:
            # First is tier1, second is tier2
            return {'tier1': factors[0], 'tier2': factors[1]}
        return factors

    except ValueError:
        raise argparse.ArgumentTypeError(f"Decay factor must be a number, got '{value}'")
    
    
    
    return fvalue
