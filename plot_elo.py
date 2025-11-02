import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

def plot_elo_rankings(csv_path='laliga_with_elo.csv', division=None, title=None):
    """
    Plot Elo rankings as line chart.
    
    Parameters:
    -----------
    csv_path : path to a csv file, with Columns: ['team', 'matchday', 'elo', 'division']
    division : str, optional
        Filter to specific division (e.g., 'SP1', 'SP2')
    title : str, optional
        Custom title
        
    Raises:
    -------
    FileNotFoundError: If csv_path does not exist
    ValueError: If DataFrame is empty or required columns are missing
    """

    
    df = pd.read_csv(csv_path)

    if not Path(csv_path).exists(): raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if df.empty: raise ValueError(f"CSV file is empty: {csv_path}")


    if division:
        df = df[df['Div'] == division]
        default_title = f"Elo Rankings - {division}"
        if df.empty: raise ValueError(f"No data found for division: {division}")
    else:
        df = df
        default_title = "Elo Rankings"
    
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
    
    for team in sorted(long_df['team'].unique()):
        team_data = long_df[long_df['team'] == team]
        fig.add_trace(go.Scatter(
            x=team_data['matchday'],
            y=team_data['elo'],
            mode='lines+markers',
            name=team
        ))
    
    fig.update_layout(
        title=title or default_title,
        xaxis_title="Matchday",
        yaxis_title="Elo Rating",
        hovermode='x unified',
        height=600,
        template='plotly_white'
    )
    
    return fig

