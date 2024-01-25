import numpy as np
import options
import pandas as pd
# Plotly and Dash Imports
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, dcc, html, Input, Output, callback, Patch, clientside_callback, ALL, MATCH
import dash_bootstrap_components as dbc
import dash_bootstrap_templates as dbt

import logging
# Create logger and set level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create console handler and set level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Add formatter to ch
ch.setFormatter(formatter)
# Add ch to logger
logger.addHandler(ch)

fig = go.Figure()
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY, dbc.icons.FONT_AWESOME], title="Options Pricing")

dbt.load_figure_template(['minty_dark', 'minty'])

color_mode_switch = dbt.ThemeSwitchAIO(aio_id='theme', themes=[dbc.themes.MINTY, dbc.themes.CYBORG], switch_props={'persistence': True})

component_price_rangeslider = dcc.RangeSlider(min=1, max=5000, count=1, value=[2000,3500], id='price-range', tooltip={'placement':'bottom', 'always_visible':True})
component_time_rangeslider = dcc.RangeSlider(min=1, max=100, count=1, value=[1,90], id='time-range', tooltip={'placement':'bottom', 'always_visible':True})

component_strike_price = dbc.Input(id='strike-price', type='number', placeholder="Strike", value=3000, inputMode='numeric', debounce=True)
component_time = dbc.Input(id='amount-time', type='number', placeholder="Number of Days", value=7, inputMode='numeric', debounce=True)
component_volatility = dbc.Input(id='volatility', type='number', placeholder="Volatility", value=70, inputMode='numeric', debounce=True)
component_rate = dbc.Input(id='rate', type='number', placeholder="Rate", value=15, inputMode='numeric', debounce=True)
component_dividend = dbc.Input(id='dividend', type='number', placeholder="Dividend", value=0.1, inputMode='numeric', debounce=True)
component_option_type = dbc.RadioItems(id='option-type', options=['Put', 'Call'], value='Call', inline=False)

component_option_price_plot = dcc.Graph(id="price-plot", responsive=True)
component_option_delta_plot = dcc.Graph(id='delta-plot', responsive=True)

options_container = dbc.Container(children=[
    dbc.Button("Add Option", id="add-option-btn", n_clicks=0),
    html.Div(id='container-div', children=[]),
    html.Div(id='container-output-div'),
], fluid=True)

tab_plot = dbc.Tab(id='plot-tab', label="Plot Tab", children=[
    dbc.Container(children=[
    html.P(),
    dbc.Row([component_price_rangeslider], justify='center'),
    html.P(),
    dbc.Row([component_time_rangeslider], justify='center'),
    html.P(),
    dbc.Row([
        dbc.Col(["Strike Price ($): ", component_strike_price]),
        dbc.Col(["Volatility (%): ", component_volatility]), 
        dbc.Col(["Rate (%): ", component_rate]), 
        dbc.Col(["Dividend (%): ", component_dividend]),
        dbc.Col(["# of Time Components", component_time]),
        dbc.Col([component_option_type], align='center'),
    ], justify='center', align='center'),
    html.P(),
    dbc.Row(children=[component_option_price_plot,
                      component_option_delta_plot,
                      ], justify='center', align='center')],
                  fluid=True)])

tab_options = dbc.Tab(id='option-tab', label="Options Tab", children=[
    options_container])

tabs = dbc.Tabs(id='tabs', children=[
    tab_options,
    tab_plot,], persistence=True, persistence_type='memory')

app.layout = dbc.Container(children=[
    dbc.Row(children=[color_mode_switch], justify='center'), 
    tabs
], fluid=True, className='m-4 dbc')

def make_new_option(n_clicks):
    logger.info(f'Making new option index={n_clicks}')
    return dbc.Container(children=[
        f"Option #{n_clicks}: ",
        dbc.Form(children=[
            dbc.Input(id={'type': 'price', "index": n_clicks}, persistence=True, persistence_type='memory', type='number', inputmode='numeric', placeholder='Price ($)', min=0),
            dbc.Input(id={'type': 'strike', "index": n_clicks}, persistence=True, persistence_type='memory', type='number', inputmode='numeric', placeholder='Strike ($)', min=0),
            dbc.Input(id={'type': 'time', "index": n_clicks}, persistence=True, persistence_type='memory', type='number', inputmode='numeric', placeholder='Time (Days)', min=0),
            dbc.Input(id={'type': 'vol', "index": n_clicks}, persistence=True, persistence_type='memory', type='number', inputmode='numeric', placeholder='Vol (%)', min=0),
            dbc.Input(id={'type': 'rate', "index": n_clicks}, persistence=True, persistence_type='memory', type='number', inputmode='numeric', placeholder='Rate (%)'),
            dbc.Input(id={'type': 'dividend', "index": n_clicks}, persistence=True, persistence_type='memory', type='number', inputmode='numeric', placeholder='Dividend (%)'),
            dbc.RadioItems(id={'type': 'option-type', 'index': n_clicks}, options=['Call', 'Put'], value='Call', inline=True),
        ], id={'type': 'option-form', 'index': n_clicks}),
        dbc.Textarea(id={'type': 'option-price-text-area', 'index': n_clicks}, readOnly=True, rows=1),
        dbc.Textarea(id={'type': 'greeks-text-area', 'index': n_clicks}, readOnly=True, rows=5),
        dbc.Button(children="Delete Child", id={'type': 'delete', 'index': n_clicks}, value=n_clicks, type='button', active=True, size='sm'),
        html.P(),
    #], id={'type': 'option-row', 'index': n_clicks}, align='start')
    ], id={'type': 'option-container', 'index': n_clicks}, fluid=True)

@callback(Output('container-div', 'children', allow_duplicate=True),
          Input('add-option-btn', 'n_clicks'),
          prevent_initial_call=True)
def add_option(n_clicks):
    logger.info(f'add_option arg: {n_clicks}')
    patched_children = Patch()
    new_option = make_new_option(n_clicks)
    patched_children.append(new_option)
    return patched_children

@callback(Output({'type': 'option-container', 'index': MATCH}, 'children'),
          [Input({'type': 'delete', 'index': MATCH}, 'n_clicks'),
           Input({'type': 'delete', 'index': MATCH}, 'value')],
          prevent_initial_call=True)
def delete_option(n_clicks, value):
    logger.info(f'Clicked delete on option #{value}')
    return None

@callback([Output({'type': 'option-price-text-area', 'index': MATCH}, 'value'),
           Output({'type': 'greeks-text-area', 'index': MATCH}, 'value')],
          [Input({'type': 'price', 'index': MATCH}, 'value'),
           Input({'type': 'strike', 'index': MATCH}, 'value'),
           Input({'type': 'time', 'index': MATCH}, 'value'),
           Input({'type': 'vol', 'index': MATCH}, 'value'),
           Input({'type': 'rate', 'index': MATCH}, 'value'),
           Input({'type': 'dividend', 'index': MATCH}, 'value'),
           Input({'type': 'option-type', 'index': MATCH}, 'value')])
def options_calculator(price, strike, time, vol, rate, dividend, option_type):
    if not all([price, strike, time, vol, rate, dividend]):
        return "Fill All Fields"
    if option_type.lower() == 'call':
        option = options.Call()
    elif option_type.lower() == 'put':
        option = options.Put()
    optionfn = option.optionfn
    greeksfn = option.greeks
    option_price = optionfn(price, strike, time, vol/100, rate/100, dividend/100)
    greeks = greeksfn(price, strike, time, vol/100, rate/100, dividend/100)
    logger.info('(price,strike,time,vol,rate,dividend,option_type)={}'.format((price, strike, time, vol, rate, dividend, option_type)))
    logger.info(f'Options price={option_price}')
    logger.info(f'Greeks={greeks}')
    return (f'{option_price:.3g}', f'{greeks}')

def create_option_dataframe(price_range, strike, time_range, vol, rate, dividend, option_type, amount_time):
    price = np.linspace(*price_range, 500)
    time_range = np.linspace(*time_range,int(amount_time),dtype=int)
    if option_type.lower() == 'call':
        optionfn = options.Call().optionfn
    else:
        optionfn = options.Put().optionfn
    df = pd.DataFrame({f"{time:d}d": optionfn(price, strike, time, vol/100, rate/100, dividend/100) for time in time_range}, index=price)
    return df

@callback(Output("price-plot", "figure", allow_duplicate=True),
          [Input("price-range","value"),
           Input("strike-price","value"),
           Input("time-range","value"),
           Input("volatility","value"),
           Input("rate","value"),
           Input("dividend","value"),
           Input("option-type","value"),
           Input("amount-time","value"),
           Input(dbt.ThemeSwitchAIO.ids.switch('theme'), 'value')],
          prevent_initial_call='initial_duplicate')
def render_plot(price_range, strike, time_range, vol, rate, dividend, option_type, amount_time, theme):
    df = create_option_dataframe(price_range, strike, time_range, vol, rate,
                                 dividend, option_type, amount_time)
    fig = px.scatter(df, labels='label', template='minty' if theme else 'minty_dark')
    hover_template = "<br>".join(["Asset Price: $%{x}", "Option Price: $%{y}"]) + "<extra></extra>"
    fig.update_layout(xaxis_title="Asset Price ($)", yaxis_title="Option Price ($)", transition_duration=200)
    fig.update_legends(title={'text':'Days to Expiry'})
    return fig

@callback([Output("time-range",'min'),
           Input("time-range",'value')])
def update_time_rangeslider_min(child):
    return [max(0.5*child[0], 1)]

@callback([Output("time-range",'max'),
           Input("time-range",'value')])
def update_time_rangeslider_max(child):
    return [2*child[1]]

@callback([Output("price-range",'min'),
           Input("price-range",'value')])
def update_price_rangeslider_min(child):
    return [max(0.5*child[0], 1)]

@callback([Output("price-range",'max'),
           Input("price-range",'value')])
def update_price_rangeslider_max(child):
    return [2*child[1]]

@callback(Output("price-plot", "figure", allow_duplicate=True),
          Input(dbt.ThemeSwitchAIO.ids.switch('theme'), 'value'),
          prevent_initial_call=True)
def update_figure_template(switch_on):
    template = pio.templates["minty"] if switch_on else pio.templates["minty_dark"]
    patch_figure = Patch()
    patch_figure["layout"]["template"] = template
    return patch_figure

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
