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

df_data = pd.read_pickle(join(APP_PATH, join('data', 'data.pkl'))) 
df_coord = pd.read_pickle(join(APP_PATH, join('data', 'coordinates.pkl'))) 

end_date = df_data.Time.max()
start_date = end_date - relativedelta(years=1)


# define helper functions

# main title and buttons
def main_title_buttons():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.Div([
                        html.Div([
                            html.H1(children="Environmental Radiation Monitoring in the US"),
                            html.P(
                                children="This dashboard has been created to serve as a platform for radiation monitoring data exploration. \
                                All data can be obtained directly from United States Environmental Protection Agency (EPA). \
                                    Web Link: https://www.epa.gov/radnet/radnet-csv-file-downloads"),
                            html.P(
                                children="All of the data presented here is part of the EPA's RadNet program. The RadNet program monitors \
                                    environmental radiation in air, rain, and drinking water. Scientist can and have used this information \
                                        to track variations in background radiation, atmospheric nuclear weapons, and nuclear reactor accidents."),
                            html.P('Information presented in this map is grouped by the selecion on the "Group By" option. The map and charts will \
                                    update accordingly.'),
                        ], className="ten columns"),
                        html.Div([
                            html.Img(id="logo", src=app.get_asset_url("radiation_logo.png")),
                        ], className="two columns", style={'margin-left': '4%'}),
                        html.Div([
                            html.P('Select Date Range:', style={'font-weight': 'bold'}),
                            dcc.DatePickerRange(
                                id='date-range',
                                start_date_placeholder_text="Start Period",
                                end_date_placeholder_text="End Period",
                                calendar_orientation='vertical',
                                min_date_allowed=df_data.Time.min(),
                                max_date_allowed=df_data.Time.max(),
                                start_date=start_date,
                                end_date=end_date,
                                day_size=45,
                            )
                        ], style={'width': '21%', 'margin-left': '4%'}),
                        html.Div([
                            html.P('Group By:', style={'font-weight': 'bold'}),
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
                            html.P('Select Radiation Metric:', style={'font-weight': 'bold'}),
                            dcc.Dropdown(
                                id="metric-select",
                                options=[
                                    {'label': 'Dose Rate [nSv/h]', 'value': 'Dose_Rate'},
                                    {'label': 'Gamma Count Rate [cpm]', 'value': 'Gamma_Count'}
                                ],
                                multi=False,
                                clearable=False,
                                value='Dose_Rate'
                            )  
                        ], style={'width': '18%', 'margin-left': '5%'})
                    ], className="row"),
                ], style={'textAlign': 'left', 'margin': '30px'}) 
            ])
        ),
    ])


init_ca_cities = ['ANAHEIM', 'BAKERSFIELD', 'EUREKA', 'FRESNO', 'LOS ANGELES',
                'RIVERSIDE', 'SACRAMENTO', 'SAN BERNARDINO', 'SAN DIEGO',
                'SAN FRANCISCO', 'SAN JOSE']
# timeseries analysis buttons
def timeseries_title_buttons():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.Div([
                        html.H1(children="Radiation Monitoring Time Series"),
                        html.P(
                            children="The time-series data below includes all raw data with near-real-time (per hour) measurements, \
                                a smoothing feature is available by defining a moving average window size. There is no limit with the \
                                number of cities to plot but please be mindful that higher latency for plotting might be experienced \
                                as more data is requested."
                        ),
                        html.P(
                            children="Note: Both Date Range and Radiation Metric filters from above are applied to the visualization below."
                        )
                    ], className="row"),
                    html.Div([
                        html.Div([
                            html.P('Select States:', style={'font-weight': 'bold'}),
                            dcc.Dropdown(
                                id="state-options",
                                options=[{'label': i, 'value': i} for i in df_data.State.unique()],
                                multi=True,
                                value=['CA'],
                                searchable=True
                            )  
                        ], className="three columns"),
                        html.Div([
                            html.P('Select Cities:', style={'font-weight': 'bold'}),
                            dcc.Dropdown(
                                id="city-options",
                                multi=True,
                                options=[{'label': i, 'value': i} for i in init_ca_cities],
                                value=['LOS ANGELES','SAN DIEGO','ANAHEIM', 'SAN JOSE', 'BAKERSFIELD'],
                                searchable=True
                            )  
                        ], className="four columns"),
                        html.Div([
                            html.P('Moving Average Window Size [hours]:', style={'font-weight': 'bold'}),
                            html.Div(
                                children=[
                                    dcc.Input(
                                        id='moving-average-window', 
                                        type='text', 
                                        style={"margin-right": "15px"}, 
                                        value='24'
                                    ),
                                    html.Button('Plot', id='submit-val', n_clicks=0)
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


# App layout

app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    main_title_buttons()
                ], width=12),
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
                ], width=12)
            ], align='center'),  
            html.Br(),
            dbc.Row([
                dbc.Col([
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
    Output("city-options", "options"),
    [
        Input("agg-df", "children"),
        Input('metric-select', 'value'),
        Input('state-options', 'value')
    ]
)
def update_city_options(data, metric, states_list):
    if data:
        
        dff = pd.read_json(data, orient='split')
        dff = dff.dropna(subset=[metric])
        dff = dff.merge(pd.DataFrame({'State':states_list}), how='inner')
        
        return [{'label': i, 'value': i} for i in dff.City.unique()] 

    else:
        raise PreventUpdate


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
        
        units = 'Dose Rate: nSv/h' if metric == 'Dose_Rate' else 'Counts per Minute'
        #xaxis = 'Dose Rate [nSv/h]' if metric == 'Dose_Rate' else 'Gamma Count Rate [cpm]'

        dff = pd.read_json(data, orient='split')
        
        dff = dff.dropna(subset=[metric])
        
        if dff.empty: raise PreventUpdate
                
        fig = px.scatter_geo(dff, color=metric, size=metric, range_color=[dff[metric].min(), dff[metric].max()],
                            lat = 'lat', lon = 'lng', animation_frame = groupby_value, color_continuous_scale='bluered',
                            hover_data=['State','City'], title = '<b>US Background Gamma Radiation - {}</b>'.format(units))

        fig.update_layout(
                font_color='#f7cc00',
                height=600,
                margin={"r":5,"t":50,"l":5,"b":5},
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

        xaxis = 'Dose Rate [nSv/h]' if metric == 'Dose_Rate' else 'Gamma Count Rate [cpm]'
        
        dff = pd.read_json(data, orient='split')
        dff = dff.dropna(subset=[metric]).groupby(['City', 'State'])[metric].median()
        # dff = dff.reset_index().sort_values(by=['State', metric], ascending=[False, True])
        dff = dff.reset_index().sort_values(by=metric, ascending=True)
        
        if dff.empty: raise PreventUpdate

        fig = px.bar(dff, x=metric, y="City", orientation='h', title = '<b>Highest Background Radiation Levels by City</b>',
                    hover_data=['State'])

        fig.update_layout(
                legend_traceorder="reversed",
                font_color='#f7cc00',
                height=600,
                margin={"r":50,"t":50,"l":50,"b":50},
                template='plotly_dark',
                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                paper_bgcolor= 'rgba(0, 0, 0, 0)',
                xaxis_title= xaxis       
        )
        
        fig.update_traces(marker={'color':'#f7cc00', 'line':{'color':'#808080', 'width':1.5}})

        fig.update_xaxes(gridcolor='#696969')
        fig.update_yaxes(showgrid=False)
        
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
        
        units = 'Dose Rate: nSv/h' if metric == 'Dose_Rate' else 'Counts per Minute'

        dff = pd.read_json(data, orient='split')
        
        dff = dff.dropna(subset=[metric])
                
        dff = dff.groupby(['State', 'City', 'Elevation'], as_index=False)[metric].median()
                
        if dff.empty: raise PreventUpdate
                        
        #dff['US'] = 'US' 
        #fig = px.treemap(dff, path=['US', 'State', 'City'], values=metric, color=metric, title = '<b> add title </b>')
        fig = px.treemap(dff, path=['State', 'City'], values=metric, color=metric, title = '<b> Radiation Sensor Distribution Across All States - {}</b>'.format(units))

        fig.update_layout(
            
                font_color='#f7cc00',
                height=400,
                margin={"r":50,"t":50,"l":50,"b":50},
                template='plotly_dark',
                
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
        
        xaxis = 'Dose Rate [nSv/h]' if metric == 'Dose_Rate' else 'Gamma Count Rate [cpm]'

        dff = pd.read_json(data, orient='split')
        if dff.empty: raise PreventUpdate
        
        dff = dff.dropna(subset=[metric])
        
        dff = dff.groupby(['City','State','Elevation'], as_index=False)[metric].median()
        
        fig = px.scatter(dff, x=metric, y='Elevation', marginal_y="box", title = '<b> City Elevation vs Gamma Metric </b>',
                        marginal_x="box", template="simple_white", hover_data=['City'])

        fig.update_layout(
                font_color='#f7cc00',
                height=400,
                margin={"r":50,"t":50,"l":50,"b":50},
                template='plotly_dark',
                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                paper_bgcolor= 'rgba(0, 0, 0, 0)',
                xaxis_title= xaxis,
                yaxis_title='Elevation [m]'
          
        )
        
        fig.update_traces(marker={'color':'#f7cc00', 'size':10, 'line':{'color':'#808080', 'width':1.5}})
        
        fig.update_xaxes(zeroline=True, zerolinecolor='#696969', showgrid=False)
        fig.update_yaxes(zeroline=True, zerolinecolor='#696969', showgrid=False)
        
        return fig
    else:
        raise PreventUpdate
    

@app.callback(
    Output("timeseries-plot", "figure"),
    [
        Input('submit-val', 'n_clicks')
    ],
    [
        State('date-range', 'start_date'),
        State('date-range', 'end_date'),
        State('metric-select', 'value'),
        State('moving-average-window', 'value'),
        State("state-options", "value"),
        State("city-options", "value")
    ]
)
def display_timeseries_plot(n_clicks, start_date, end_date, metric, ma_window_size, states_list, cities_list):
    # if not df_data.empty and n_clicks != 0 and ma_window_size != None:
    if not df_data.empty and ma_window_size != None:

        dff = df_data.merge(pd.DataFrame({'State':states_list}), how='inner')
        dff = df_data.merge(pd.DataFrame({'City':cities_list}), how='inner')
        dff = dff.loc[(dff.Time > start_date) & (dff.Time <= end_date)]
        
        # when input is not an integer and no states records match
        if not ma_window_size.isdigit() or len(dff)==0: 
            raise PreventUpdate
                        
        dff = dff.set_index(['Time','State', 'City'])
        dff = dff.rolling(int(ma_window_size)).mean().reset_index()
        
        dff = dff.dropna(subset=[metric])
        
        fig = px.line(dff, x='Time', y=metric, hover_name="State", color='City')

        fig.update_layout(
                font_color='#f7cc00',
                height=500,
                margin={"r":50,"t":50,"l":50,"b":50},
                template='plotly_dark',
                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                paper_bgcolor= 'rgba(0, 0, 0, 0)'
        )
        
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor='#696969')
                
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
    
    dff[groupby_value] = getattr(dff.Time.dt, groupby_value)
    
    cols = ['State', 'City', groupby_value]   
    dff = dff.groupby(cols, as_index=False)['Dose_Rate', 'Gamma_Count'].median()

    return merge_coordinates(dff).to_json(date_format='iso', orient='split')


def merge_coordinates(data):
    dff = pd.merge(data, df_coord, on=['City', 'State'], how='inner')
    return dff


if __name__ == "__main__":
    app.run_server(debug=True)