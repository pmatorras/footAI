import types
from dash import Dash, dcc, html, Input, Output
from footai.viz.plotter import plot_elo_rankings
from footai.utils.config import  setup_directories, COUNTRIES
from footai.utils.paths import get_data_loc, parse_start_years, get_multiseason_path

app = Dash(__name__)

SEASON = [1516]
COUNTRY = ["SP"]
app.layout = html.Div([

    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': c['name'], 'value': code} for code, c in COUNTRIES.items()],
        value='SP'
    ),

    dcc.Dropdown(
        id='division-dropdown',
        options = [
            {
                'label': f"{division_name} ({country_info['name']})",
                'value': division_code
            }
            for country_info in COUNTRIES.values()
            for division_code, division_name in country_info['divisions'].items()
        ],
        value='SP1'
    ),
    dcc.Graph(id='elo-graph')
])

@app.callback(
    [Output('division-dropdown', 'options'),   # Update options
     Output('division-dropdown', 'value')],    # Update selected value
    [Input('country-dropdown', 'value')]
)

def set_division_options(selected_country):
    divisions = COUNTRIES[selected_country]['divisions']
    options= [
        {'label': name, 'value': code}
        for code, name in COUNTRIES[selected_country]['divisions'].items()
    ]
    first_division = list(divisions.keys())[0]  # Get the first division code
    return options, first_division


@app.callback(
    Output('elo-graph', 'figure'),
    [Input('country-dropdown', 'value'),   # Listens to country changes
     Input('division-dropdown', 'value')]  # Listens to division changes)
)
def update_graph(country=COUNTRY, division='SP1'):

    args = types.SimpleNamespace(
        countries = [country],
        season_start = '15-25',
        elo_transfer = True
)
    seasons = parse_start_years(args.season_start)
    dirs = setup_directories(args)
    print(seasons)
    if len(seasons)==1:
        season_str = seasons[0]
        elo_path = get_data_loc(season_str, division=division, country=country, file_dir=dirs[country]['proc'], file_type='elo')
        div_plot = str(division)
        return plot_elo_rankings(csv_path=elo_path, division=div_plot, custom_title=f"({COUNTRIES[country]['divisions'][division]})")
    else:
        print("wip", country, seasons[0],seasons[-1], division, dirs)
        print("multiseason", dirs[country]['proc'], division)
        path = get_multiseason_path(dirs[country]['proc'], division, seasons[0],seasons[-1], args)
        print(path)
        return plot_elo_rankings(path, division=division, custom_title=f"({COUNTRIES[country]['divisions'][division]})")
if __name__ == '__main__':
    app.run(debug=True, port=8050)