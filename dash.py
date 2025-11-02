from dash import Dash, dcc, html, Input, Output
from plot_elo import plot_elo_rankings

app = Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='division-dropdown',
        options=[
            {'label': 'SP1', 'value': 'SP1'},
            {'label': 'SP2', 'value': 'SP2'}
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
    return plot_elo_rankings(division=division)

if __name__ == '__main__':
    app.run_server(debug=True)
