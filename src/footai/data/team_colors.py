"""
Scrape team colors from teamcolours.netlify.app JSON API

Usage:
    python scripts/update_team_colors.py --country SP
"""

import json
import re
import unicodedata
import argparse
from pathlib import Path
from typing import Dict, Set, Optional
import pandas as pd
import requests


def normalize_for_matching(team_name: str) -> str:
    """
    Normalize team name for fuzzy matching.
    'Málaga CF' -> 'malaga'
    'FC Köln' -> 'koln'
    'Ath Madrid' -> 'atletico madrid'
    """
    name = team_name.strip()
    
    # Expand abbreviations
    expansions = {
        r'\bAth\s+Bilbao\b': 'Athletic Bilbao',
        r'\bMan\s+City\b': 'Manchester City',
        r'\bMan\s+United\b': 'Manchester United',
        r"\bNott'?m\s+Forest\b": 'Nottingham Forest',
        r'\bQPR\b': 'Queens Park Rangers',
        r'\bEin\s+Frankfurt\b': 'Eintracht Frankfurt',
        r'\bFrankfurt\s+FSV\b': 'FSV Frankfurt',
        r"\bM'?gladbach\b": 'Borussia Monchengladbach',
        r'\bMunich\s+1860\b': 'TSV 1860 Munich',
        r'\bParis\s+SG\b': 'Paris Saint-Germain',
        r'\bUlm\b': 'SSV Ulm',
        r'\bAth\b': 'Atletico',
        r'\bSp\b': 'Sporting',
    }
    for pattern, replacement in expansions.items():
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    
    # Remove accents (Köln -> Koln, Málaga -> Malaga)
    name = unicodedata.normalize('NFKD', name)
    name = ''.join([c for c in name if not unicodedata.combining(c)])
    
    # Remove prefixes/suffixes
    remove_patterns = [
        r'\bFC\b', r'\bCF\b', r'\bCD\b', r'\bSD\b', r'\bAC\b',
        r'\bAS\b', r'\bSC\b', r'\bRC\b', r'\bUD\b', r'\bReal\b',
        r'\bClub\b', r'\bSA\.?D\.?\b',
    ]
    for pattern in remove_patterns:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    
    # Clean up
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name.lower()


def fetch_all_team_colors() -> Dict[str, str]:
    """
    Fetch all team colors from the JSON API.
    
    Returns:
        Dictionary mapping team names to hex colors
    """
    url = "https://teamcolours.netlify.app/data.json"
    
    print(f"Fetching team colors from {url}...")
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        teams_dict = {}
        
        for team in data:
            team_short = team.get('TeamShort', '')
            team_long = team.get('TeamLong', '')
            colors = team.get('TeamColours', [])
            
            if colors and len(colors) > 0:
                colors = team.get('TeamColours', [])
                if not colors: continue
                
                # Normalize all to upper case
                colors = [c.upper() for c in colors]
                # Store both short and long names
                if team_short:
                    teams_dict[team_short] = colors
                if team_long:
                    teams_dict[team_long] = colors
        
        print(f"Successfully fetched {len(teams_dict)} team color entries")
        return teams_dict
        
    except Exception as e:
        print(f"ERROR fetching team colors: {e}")
        return {}


def find_best_match(csv_team_name: str, scraped_teams: Dict[str, str]) -> Optional[tuple]:
    """
    Find best matching team from scraped data.
    
    Returns:
        (matched_name, color) or None
    """
    normalized_search = normalize_for_matching(csv_team_name)
    
    # Exclusion list for teams that cause false matches
    exclusions = {
        'espanol': ['spain', 'española', 'español'],  # Don't match Espanyol to Spain/Española teams
    }
    
    # Try exact normalized match first
    for team_name, color in scraped_teams.items():
        normalized_scraped = normalize_for_matching(team_name)
        
        if normalized_search == normalized_scraped:
            return (team_name, color)
    
    # Try substring matching
    for team_name, color in scraped_teams.items():
        # Skip national teams
        if 'national' in team_name.lower():
            continue
        
        normalized_scraped = normalize_for_matching(team_name)
        
        # Check exclusion list
        if normalized_search in exclusions:
            skip = False
            for excluded_term in exclusions[normalized_search]:
                if excluded_term in normalized_scraped:
                    skip = True
                    break
            if skip:
                continue
        
        if len(normalized_search) >= 4 and len(normalized_scraped) >= 4:
            if normalized_search in normalized_scraped or normalized_scraped in normalized_search:
                return (team_name, color)
    
    return None



def extract_teams_from_csvs(data_dir: Path) -> Set[str]:
    """Extract all unique team names from CSV files."""
    teams = set()
    
    for csv_file in data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if 'HomeTeam' in df.columns:
                teams.update(df['HomeTeam'].dropna().unique())
                teams.update(df['AwayTeam'].dropna().unique())
        except Exception as e:
            print(f"WARNING: Error reading {csv_file.name}: {e}")
    
    return teams


def load_color_json(json_path: Path) -> Dict[str, str]:
    """Load existing team colors from JSON."""
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'colors' in data:
                return data['colors']
            return data
    return {}


def save_color_json(colors: Dict[str, str], json_path: Path):
    """Save team colors to JSON with metadata."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        "_metadata": {
            "source": "https://teamcolours.netlify.app/",
            "attribution": "Team colors curated by FC Python",
            "note": "Primary team colors in hex format"
        },
        "colors": colors
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, sort_keys=False)


def update_team_colors(country: str, data_dir: Path, output_dir: Path):
    """Update team colors JSON by fetching from API and matching."""
    
    print(f"\n{'='*60}")
    print(f"Updating team colors for {country}")
    print(f"{'='*60}\n")
    
    json_path = output_dir / f"{country}_colors.json"
    
    # Load existing colors
    colors = load_color_json(json_path)
    initial_count = len(colors)
    print(f"Existing colors in JSON: {initial_count} teams")
    
    # Extract teams from CSVs
    csv_teams = extract_teams_from_csvs(data_dir)
    print(f"Teams found in CSV files: {len(csv_teams)}\n")
    
    # Fetch all colors from API
    scraped_colors = fetch_all_team_colors()
    
    if not scraped_colors:
        print("ERROR: Failed to fetch any colors. Aborting.")
        return
    
    print(f"\nMatching CSV teams to fetched data...\n")
    
    matched = 0
    not_found = []
    
    for csv_team in sorted(csv_teams):
        # Skip if already have non-placeholder color
        if csv_team in colors and colors[csv_team] != "#CCCCCC":
            continue
        
        # Try to find match
        match = find_best_match(csv_team, scraped_colors)
        
        if match:
            matched_name, color = match
            colors[csv_team] = color
            matched += 1
            print(f"  MATCH: '{csv_team}' -> '{matched_name}' ({color})")
        else:
            colors[csv_team] = "#CCCCCC"
            not_found.append(csv_team)
            print(f"  NOT FOUND: '{csv_team}' (added placeholder)")
    
    # Save
    save_color_json(colors, json_path)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Saved to: {json_path}")
    print(f"  Total teams: {len(colors)}")
    print(f"  Newly matched: {matched}")
    print(f"  Not found: {len(not_found)}")
    
    if not_found:
        print(f"\nTeams needing manual entry:")
        for team in not_found:
            print(f"  - {team}")

