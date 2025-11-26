import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from footai.viz.themes import get_team_colors_dict

def add_breaks_for_gaps(df, gap_threshold_days=120):
    """
    Insert None/NaN rows where time gaps between matches exceed threshold.
    This forces Plotly to break the line for cases where the team returns to the given tier after having promoted/relegated to a different tier.
    """
    new_rows = []
    
    for team in df['team'].unique():
        team_df = df[df['team'] == team].sort_values('Date')
        
        # Calculate time difference between consecutive matches
        team_df['delta'] = team_df['Date'].diff().dt.days
        gaps = team_df[team_df['delta'] > gap_threshold_days]
        
        if not gaps.empty:
            # For every gap found, create a "break" row
            for idx, row in gaps.iterrows():
                # Create a dummy row between the previous match and this one
                break_row = row.copy()
                # Set Date to slightly before the new match (or midpoint)
                break_row['Date'] = row['Date'] - pd.Timedelta(days=1) 
                break_row['elo'] = None
                new_rows.append(break_row)
    
    if new_rows:
        # Combine original data with break rows
        return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True).sort_values(['team', 'Date'])
    
    return df


def plot_elo_rankings(csv_path='laliga_with_elo.csv', division=None, country='SP',  selected_seasons = None, custom_title=None):
    """
    Plot Elo rankings as line chart.
    
    Parameters:
    -----------
    csv_path : path to a csv file, with Columns: ['team', 'matchday', 'elo', 'division']
    division : str, optional
        Filter to specific division (e.g., 'SP1', 'SP2')
    custom_title : str, optional
        Custom title
        
    Raises:
    -------
    FileNotFoundError: If csv_path does not exist
    ValueError: If DataFrame is empty or required columns are missing
    """
    
    title = f"Elo Rankings {custom_title}"
    df = pd.read_csv(csv_path)
    teams = df['HomeTeam'].unique()
    team_colors = get_team_colors_dict(teams, country=country)

    if not Path(csv_path).exists(): raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if df.empty: raise ValueError(f"CSV file is empty: {csv_path}")


    if division:
        df = df[df['Div'] == division]
        if df.empty: raise ValueError(f"No data found for division: {division}")
    if selected_seasons:
        df['Season'] = df['Season'].astype(str)
        df = df[df['Season'].isin(selected_seasons)]
        if df.empty:
            raise ValueError(f"No data for selected seasons: {selected_seasons}")
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
    long_df['Date'] = pd.to_datetime(long_df['Date'])
    if selected_seasons and len(selected_seasons) > 1:
        long_df = add_breaks_for_gaps(long_df)
        # Multi-season: Use actual Date
        x_column = 'Date'
        x_label = "Date"
    else:
        print("or on the contrary")
        # Single season: Use Matchday count (1-38) for cleaner look
        long_df['matchday'] = long_df.groupby('team').cumcount() + 1
        x_column = 'matchday'
        x_label = "Matchday"

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
            x=team_data[x_column],
            y=team_data['elo'],
            mode='lines',
            name=team,
            line=dict(color=team_colors[team])  

        ))
    
    fig.update_layout(
        title=title,
        xaxis_title= x_label,
        yaxis_title="Elo Rating",
        hovermode='x unified',
        height=600,
        template='plotly_white'
    )
    
    return fig