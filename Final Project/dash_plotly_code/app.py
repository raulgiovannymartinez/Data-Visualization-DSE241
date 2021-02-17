import warnings
warnings.filterwarnings('ignore')

from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from os.path import join
import pandas as pd
import numpy as np
import pathlib

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import dash


# initialize app

app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])
server = app.server


# load data

APP_PATH = str(pathlib.Path(__file__).parent.resolve())

df_data = pd.read_pickle(join(APP_PATH, join('data', 'data_test.pkl'))) 
df_coord = pd.read_pickle(join(APP_PATH, join('data', 'coordinates_test.pkl'))) 

end_date = max(df_data.Time)
start_date = end_date - relativedelta(years=1)


# define helper functions

# main title and buttons
def main_title_buttons():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.Div([
                        html.H4(children="Rate of US Poison-Induced Deaths"),
                        html.P(
                            children="Deaths are classified using the International Classification of Diseases, \
                            Tenth Revision (ICD–10). Drug-poisoning deaths are defined as having ICD–10 underlying \
                            cause-of-death codes X40–X44 (unintentional), X60–X64 (suicide), X85 (homicide), or Y10–Y14 \
                            (undetermined intent).",
                        ),
                        html.Div([
                            html.H6('Select Date Range:'),
                            dcc.DatePickerRange(
                                id='date-range',
                                start_date_placeholder_text="Start Period",
                                end_date_placeholder_text="End Period",
                                calendar_orientation='vertical',
                                min_date_allowed=min(df_data.Time),
                                max_date_allowed=max(df_data.Time),
                                start_date=start_date,
                                end_date=end_date,
                                day_size=45,
                            )
                        ], className="four columns"),
                        html.Div([
                            html.H6('Group By:'),
                            dcc.Dropdown(
                                id="group-by",
                                options=[
                                    {'label': 'Year', 'value': 'year'},
                                    {'label': 'Month', 'value': 'month'},
                                    {'label': 'Day', 'value': 'day'}
                                ],
                                multi=False,
                                clearable=False,
                                value='day'
                            )  
                        ], className="two columns"),
                        html.Div([
                            html.H6('Select Metric:'),
                            dcc.Dropdown(
                                id="metric-select",
                                options=[
                                    {'label': 'Dose Rate', 'value': 'Dose_Rate'},
                                    {'label': 'Gamma Count', 'value': 'Gamma_Count'}
                                ],
                                multi=False,
                                clearable=False,
                                value='Dose_Rate'
                            )  
                        ], className="two columns")
                    ], className="row"),
                ], style={'textAlign': 'left', 'margin': '30px'}) 
            ])
        ),
    ])


# timeseries analysis buttons
def timeseries_title_buttons():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.Div([
                        html.H4(children="Rate of US Poison-Induced Deaths"),
                        html.P(
                            children="Deaths are classified using the International Classification of Diseases, \
                            Tenth Revision (ICD–10). Drug-poisoning deaths are defined as having ICD–10 underlying \
                            cause-of-death codes X40–X44 (unintentional), X60–X64 (suicide), X85 (homicide), or Y10–Y14 \
                            (undetermined intent).",
                        ),
                        html.Div([
                            html.H6('Select States:'),
                            dcc.Dropdown(
                                id="state-options",
                                options=[{'label': i, 'value': i} for i in set(df_data.State)],
                                multi=True,
                                value=['CA', 'NY'],
                                searchable=True
                            )  
                        ], className="six columns"),
                        html.Div([
                            html.H6('Moving Average Window Size:'),
                            html.Div(
                                children=[
                                    dcc.Input(id='moving-average-window', type='text', style={"margin-right": "15px"}),
                                    html.Button('Run', id='submit-val', n_clicks=0)
                                ]
                            )
                        ], className="three columns")
                    ], className="row"),
                ], style={'textAlign': 'left', 'margin': '30px'}) 
            ])
        ),
    ])


# functions for plots

def map_plot():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="map-plot"
                ) 
            ])
        ),  
    ])

def timeseries_plot():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="timeseries-plot"
                ) 
            ])
        ),  
    ])
    
def treemap_plot():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="treemap-plot"
                ) 
            ])
        ),  
    ])
       
def correlation_plot():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="correlation-plot"
                ) 
            ])
        ),  
    ])
    
def bar_plot():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    id="bar-plot"
                )
            ]), 
        ),  
    ])
    
def display_logo():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.Img(id="logo", src=app.get_asset_url("dash-logo.png")),
                ], style={'margin': '30px'}) 
            ])
        ),
    ])



# App layout

app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    main_title_buttons()
                ], width=9),
                dbc.Col([
                    display_logo()
                ], width=3),
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    map_plot() 
                ], width=8),
                dbc.Col([
                    bar_plot()
                ], width=4),
            ], align='center'), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    treemap_plot()
                ], width=6),
                dbc.Col([
                    correlation_plot()
                ], width=6),
            ], align='center'),    
            html.Br(),
            dbc.Row([
                dbc.Col([
                    timeseries_title_buttons(),
                    timeseries_plot()
                ], width=12)
            ], align='center'),  
        ]), color = 'dark'
    ),
    # Hidden div inside the app that stores the intermediate value
    html.Div(id='agg-df', style={'display': 'none'})
])


# callback functions

@app.callback(
    Output("map-plot", "figure"),
    [
        Input("agg-df", "children"),
        Input('metric-select', 'value'),
        Input('group-by', 'value')
    ]
)
def display_map_plot(data, metric, groupby_value):
    if data:
        
        dff = pd.read_json(data, orient='split')
        
        dff = dff.dropna(subset=[metric])
        
        if dff.empty: raise PreventUpdate
        
        fig = px.scatter_geo(dff, color=metric, size=metric,
                            lat = 'lat', lon = 'lng', animation_frame = groupby_value)

        fig.update_layout(
                height=600,
                margin={"r":5,"t":5,"l":5,"b":5},
                template='plotly_dark',
                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                paper_bgcolor= 'rgba(0, 0, 0, 0)',
                geo = dict(
                    scope = 'usa',
                    landcolor = 'rgb(217, 217, 217)',
                )
            )
        
        return fig
    else:
        raise PreventUpdate


@app.callback(
    Output("bar-plot", "figure"),
    [
        Input("agg-df", "children"),
        Input('metric-select', 'value')
    ]
)
def display_bar_plot(data, metric):
    if data:
        
        dff = pd.read_json(data, orient='split')
        dff = dff.dropna(subset=[metric]).groupby(['City'])[metric].median().sort_values(ascending=True).reset_index()
        
        if dff.empty: raise PreventUpdate

        fig = px.bar(dff, x=metric, y="City", orientation='h', title = 'add title')

        fig.update_layout(
                height=600,
                margin={"r":50,"t":50,"l":50,"b":50},
                template='plotly_dark',
                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                paper_bgcolor= 'rgba(0, 0, 0, 0)'
            )
        
        return fig
    else:
        raise PreventUpdate
    

@app.callback(
    Output("treemap-plot", "figure"),
    [
        Input("agg-df", "children"),
        Input('metric-select', 'value')
    ]
)
def display_treemap_plot(data, metric):
    if data:
        
        dff = pd.read_json(data, orient='split')
        
        dff = dff.dropna(subset=[metric])
        
        dff = dff.groupby(['State', 'City', 'Elevation'], as_index=False)[metric].median()
        
        if dff.empty: raise PreventUpdate
        
        fig = px.treemap(dff, path=['State', 'City'], values=metric, color=metric, title = 'add title')

        fig.update_layout(
                height=400,
                margin={"r":50,"t":50,"l":50,"b":50},
                template='plotly_dark'
            )
        
        return fig
    else:
        raise PreventUpdate
    
    
@app.callback(
    Output("correlation-plot", "figure"),
    [
        Input("agg-df", "children"),
        Input('metric-select', 'value')
    ]
)
def display_correlation_plot(data, metric):
    if data:
        
        dff = pd.read_json(data, orient='split')
        if dff.empty: raise PreventUpdate
        
        dff = dff.dropna(subset=[metric])
        
        dff = dff.groupby(['City','State','Elevation'], as_index=False)[metric].median()
        
        fig = px.scatter(dff, x=metric, y='Elevation', marginal_y="box", title = 'add title',
                        marginal_x="box", template="simple_white", hover_data=['City'])

        fig.update_layout(
                height=400,
                margin={"r":50,"t":50,"l":50,"b":50},
                template='plotly_dark',
                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                paper_bgcolor= 'rgba(0, 0, 0, 0)'
            )
        
        return fig
    else:
        raise PreventUpdate
    

@app.callback(
    Output("timeseries-plot", "figure"),
    [
        Input('submit-val', 'n_clicks')
    ],
    [
        State('metric-select', 'value'),
        State('moving-average-window', 'value'),
        State("state-options", "value")
    ]
)
def display_timeseries_plot(n_clicks, metric, ma_window_size, states_list):
    if not df_data.empty and n_clicks != 0 and ma_window_size != None:
        
        dff = df_data.merge(pd.DataFrame({'State':states_list}), how='inner')
        
        # when input is not an integer and no states records match
        if not ma_window_size.isdigit() or len(dff)==0: 
            raise PreventUpdate
                        
        dff = dff.set_index(['Time','State', 'City'])
        dff = dff.rolling(int(ma_window_size)).mean().reset_index()
        
        dff = dff.dropna(subset=[metric])
        
        fig = px.line(dff, x='Time', y=metric, title='Time Series with Range Slider and Selectors',
                    hover_name="State", color='City')

        fig.update_layout(
                height=500,
                margin={"r":50,"t":50,"l":50,"b":50},
                template='plotly_dark',
                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                paper_bgcolor= 'rgba(0, 0, 0, 0)'
            )
                
        return fig

    else:
        raise PreventUpdate


@app.callback(
    Output("agg-df", "children"),
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('group-by', 'value')
    ]
)
def get_agg_df(start_date, end_date, groupby_value):

    dff = df_data.loc[(df_data.Time > start_date) & (df_data.Time <= end_date)]
    dff[groupby_value] = [getattr(i, groupby_value) for i in dff.Time] 
    
    cols = ['State', 'City', groupby_value]    
    dff = dff.groupby(cols, as_index=False)['Dose_Rate', 'Gamma_Count'].median()

    return merge_coordinates(dff).to_json(date_format='iso', orient='split')


def merge_coordinates(data):
    dff = pd.merge(data, df_coord, on=['City', 'State'], how='inner')
    return dff


if __name__ == "__main__":
    app.run_server(debug=True)