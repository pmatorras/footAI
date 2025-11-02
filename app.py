from dash import Dash, dcc, html, Input, Output
from plot_elo import plot_elo_rankings
from common import get_data_loc, season_to_season_str, PROCESSED_DIR, COUNTRY, DIV

app = Dash(__name__)

SEASON = 2024


app.layout = html.Div([
    html.H1("Elo Rankings - La Liga"),

    dcc.Dropdown(
        id='division-dropdown',
        options=[
            {'label': 'SP1 (Primera)', 'value': 'SP1'},
            {'label': 'SP2 (Segunda)', 'value': 'SP2'}
        ],
        value='SP1'
    ),
    dcc.Graph(id='elo-graph')
])

@app.callback(
    Output('elo-graph', 'figure'),
    Input('division-dropdown', 'value')
)

def update_graph(division):
    div_int = int(division[-1])  # Extract the "1" or "2" from options
    season_str = season_to_season_str(SEASON, COUNTRY, div_int)
    elo_path = get_data_loc(season_str, division=div_int, country=COUNTRY, data_dir=PROCESSED_DIR, elo=True)
    div_plot = str(COUNTRY) + str(div_int)
    return plot_elo_rankings(csv_path=elo_path, division=div_plot)

if __name__ == '__main__':
    app.run(debug=True, port=8050)