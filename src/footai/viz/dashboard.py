from dash import Dash, dcc, html, Input, Output
from footai.viz.plotter import plot_elo_rankings
from footai.utils.config import  PROCESSED_DIR
from footai.utils.paths import get_data_loc, PROCESSED_DIR

app = Dash(__name__)

SEASON = 2425
COUNTRY = "SP"
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
    season_str = SEASON
    elo_path = get_data_loc(season_str, division=div_int, country=COUNTRY, data_dir=PROCESSED_DIR, elo=True)
    div_plot = str(COUNTRY) + str(div_int)
    return plot_elo_rankings(csv_path=elo_path, division=div_plot)

if __name__ == '__main__':
    app.run(debug=True, port=8050)