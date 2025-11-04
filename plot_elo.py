import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from common import COUNTRIES
from colors import get_team_colors_dict

def plot_elo_rankings(csv_path='laliga_with_elo.csv', division=None, custom_title=None):
def plot_elo_rankings(csv_path='laliga_with_elo.csv', division=None, custom_title=None):
    """
    Plot Elo rankings as line chart.
    
    Parameters:
    -----------
    csv_path : path to a csv file, with Columns: ['team', 'matchday', 'elo', 'division']
    division : str, optional
        Filter to specific division (e.g., 'SP1', 'SP2')
    custom_title : str, optional
    custom_title : str, optional
        Custom title
        
    Raises:
    -------
    FileNotFoundError: If csv_path does not exist
    ValueError: If DataFrame is empty or required columns are missing
    """
    
    title = f"Elo Rankings {custom_title}"
    
    title = f"Elo Rankings {custom_title}"
    df = pd.read_csv(csv_path)
    teams = df['HomeTeam'].unique()
    team_colors = get_team_colors_dict(teams)

    if not Path(csv_path).exists(): raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if df.empty: raise ValueError(f"CSV file is empty: {csv_path}")


    if division:
        df = df[df['Div'] == division]
        if df.empty: raise ValueError(f"No data found for division: {division}")
    else:
        df = df
    
    # Reshape: create separate rows for home and away teams
    home_rows = df[['Date', 'HomeTeam', 'HomeElo']].rename(
        columns={'HomeTeam': 'team', 'HomeElo': 'elo'}
    )
    away_rows = df[['Date', 'AwayTeam', 'AwayElo']].rename(
        columns={'AwayTeam': 'team', 'AwayElo': 'elo'}
    )
    
    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    if long_df.empty: raise ValueError("Failed to reshape data: resulting DataFrame is empty")
    long_df = long_df.sort_values(['team', 'Date']).reset_index(drop=True)
    
    # Add matchday (sequential order per team)
    long_df['matchday'] = long_df.groupby('team').cumcount() + 1
    
    fig = go.Figure()

    # Get final Elo for each team to sort legend
    team_final_elo = {}
    for team in long_df['team'].unique():
        team_data = long_df[long_df['team'] == team]
        final_elo = team_data.iloc[-1]['elo']
        team_final_elo[team] = final_elo

    # Sort by final Elo (highest first)
    sorted_teams = sorted(team_final_elo.items(), key=lambda x: x[1], reverse=True)

    for team, final_elo in sorted_teams:
        team_data = long_df[long_df['team'] == team]
        fig.add_trace(go.Scatter(
            x=team_data['matchday'],
            y=team_data['elo'],
            mode='lines+markers',
            name=team,
            line=dict(color=team_colors[team])  

        ))
    
    fig.update_layout(
        title=title,
        title=title,
        xaxis_title="Matchday",
        yaxis_title="Elo Rating",
        hovermode='x unified',
        height=600,
        template='plotly_white'
    )
    
    return fig

