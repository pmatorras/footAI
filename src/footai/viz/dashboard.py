import types
from dash import Dash, dcc, html, Input, Output
from footai.viz.plotter import plot_elo_rankings
from footai.utils.config import  setup_directories, COUNTRIES
from footai.utils.paths import get_data_loc, parse_start_years, get_multiseason_path

app = Dash(__name__)
server = app.server

SEASONS = '15-25'
season_list = parse_start_years(SEASONS)

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
    dcc.RangeSlider(
        id='season-slider',
        min=0,
        max=len(season_list)-1,
        step=1,
        marks={i: f"{season[:2]}-{season[2:]}" for i, season in enumerate(season_list)},
        value=[0, len(season_list)-1],  # Default: all seasons
        tooltip={"placement": "bottom", "always_visible": False}
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
    [Input('country-dropdown', 'value'),
     Input('division-dropdown', 'value'),
     Input('season-slider', 'value')]

)
def update_graph(country='SP', division='SP1', season_range = 25):

    args = types.SimpleNamespace(
        countries = [country],
        elo_transfer = True
    )
    dirs = setup_directories(args)

    if len(season_list)==1:
        season_str = season_list[0]
        elo_path = get_data_loc(season_str, division=division, country=country, file_dir=dirs[country]['proc'], file_type='elo')
        div_plot = str(division)
        return plot_elo_rankings(csv_path=elo_path, division=div_plot, custom_title=f"({COUNTRIES[country]['divisions'][division]})")
    else:
        selected_seasons = season_list[season_range[0]:season_range[1]+1]
        path = get_multiseason_path(dirs[country]['proc'], division, season_list[0],season_list[-1], args)
        return plot_elo_rankings(path, division=division, selected_seasons=selected_seasons, custom_title=f"({COUNTRIES[country]['divisions'][division]})")
if __name__ == '__main__':
    app.run(debug=True, port=8050)