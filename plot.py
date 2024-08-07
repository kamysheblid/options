import numpy as np
import pandas as pd
# Plotly and Dash Imports
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, dcc, html, Input, Output, callback, Patch, ALL, MATCH
import dash_bootstrap_components as dbc
import dash_bootstrap_templates as dbt
import datetime

# Import from local folder
import options
from lib import binance

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

@callback(Output("price-graph", "figure", allow_duplicate=True),
          Input(dbt.ThemeSwitchAIO.ids.switch('theme'), 'value'),
          prevent_initial_call=True)
def update_figure_template(switch_on):
    template = pio.templates["minty"] if switch_on else pio.templates["minty_dark"]
    patch_figure = Patch()
    patch_figure["layout"]["template"] = template
    return patch_figure

component_price_rangeslider = dcc.RangeSlider(min=1, max=5000, count=1, value=[2000,3500], id='price-range', tooltip={'placement':'bottom', 'always_visible':True})
component_time_rangeslider = dcc.RangeSlider(min=1, max=100, count=1, value=[1,90], id='time-range', tooltip={'placement':'bottom', 'always_visible':True})

component_strike_price = dbc.Input(id='strike-price', type='number', placeholder="Strike", value=3000, inputMode='numeric', debounce=True)
component_time = dbc.Input(id='amount-time', type='number', placeholder="Number of Days", value=7, inputMode='numeric', debounce=True)
component_volatility = dbc.Input(id='volatility', type='number', placeholder="Volatility", value=70, inputMode='numeric', debounce=True)
component_rate = dbc.Input(id='rate', type='number', placeholder="Rate", value=15, inputMode='numeric', debounce=True)
component_dividend = dbc.Input(id='dividend', type='number', placeholder="Dividend", value=0.1, inputMode='numeric', debounce=True)
component_option_type = dbc.RadioItems(id='option-type', options=['Put', 'Call'], value='Call', inline=False)

component_graph = dcc.Graph(id="price-graph", responsive=True)

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
        dbc.Col([component_option_type], align='center')
    ], justify='center', align='center'),
    html.P(),
    dbc.Row(component_graph, justify='center', align='center')],
                  fluid=True)])

def create_option_dataframe(price_range, strike, time_range, vol, rate, dividend, option_type, amount_time):
    price = np.linspace(*price_range, 500)
    time_range = np.linspace(*time_range,int(amount_time),dtype=int)
    if option_type.lower() == 'call':
        optionfn = options.Call().optionfn
    else:
        optionfn = options.Put().optionfn
    df = pd.DataFrame({f"{time:d}d": optionfn(price, strike, time, vol/100, rate/100, dividend/100) for time in time_range}, index=price)
    return df

@callback(Output("price-graph", "figure", allow_duplicate=True),
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

ethusdt_interval_selector = dbc.RadioItems(id='ethusdt-interval-selector', inline=True,
                                           value='1d',
                                           options=['1m', '1h', '4h', '1d', '1w', '1M'])

ethusdt_container = dbc.Container(children=[html.H4("ETHUSDT Binance"), 
                                            ethusdt_interval_selector,
                                            dcc.Graph(id='ethusdt-candlestick-plot', 
                                                      responsive=True)],
                                  fluid=True)

tab_ethusdt = dbc.Tab(id='ethusdt-tab', label='ETHUSDT Graph', children=[ethusdt_container])

@callback(output=Output('ethusdt-candlestick-plot', 'figure'),
          inputs=[Input('ethusdt-interval-selector', 'value'),
                  Input(dbt.ThemeSwitchAIO.ids.switch('theme'), 'value')])
def display_eth_candlestick(value, theme):
    if not binance.ping():
        raise Exception("ERROR: Cannot contact binance. Try setting proxy")

    if value not in binance.INTERVALS.keys():
        raise Exception(f'INTERVAL ERROR: interval is not valid: {value}')
    ethusdt_df = binance.main(symbol='ETHUSDT', interval=value)
    config = config={'modeBarButtonsToAdd':
                     ['drawline','drawopenpath',
                      'drawclosedpath', 'drawcircle',
                      'drawrect'], 'scrollZoom': True,
                     'dragmode': 'pan', 'responsive': False,
                     'displaylogo': False}

    fig = go.Figure(go.Candlestick(x=ethusdt_df.index,
                                   open=ethusdt_df.Open,
                                   close=ethusdt_df.Close,
                                   low=ethusdt_df.Low,
                                   high=ethusdt_df.High))

    fig.update_layout(dragmode='pan', template='minty' if theme else
                      'minty_dark',
                      modebar_add=['drawline','drawopenpath', 'eraseshape',
                                   'drawclosedpath', 'drawcircle', 'drawrect'])

    #fig.layout.template = template='minty' if theme else 'minty_dark'
    return fig

@callback(Output("ethusdt-candlestick-plot", "figure", allow_duplicate=True),
          Input(dbt.ThemeSwitchAIO.ids.switch('theme'), 'value'),
          prevent_initial_call=True)
def update_figure_template(switch_on):
    template = pio.templates["minty"] if switch_on else pio.templates["minty_dark"]
    patch_figure = Patch()
    patch_figure["layout"]["template"] = template
    return patch_figure

btcusdt_interval_selector = dbc.RadioItems(id='btcusdt-interval-selector', inline=True,
                                           value='1d',
                                           options=['1m', '1h', '4h', '1d', '1w', '1M'])

btcusdt_container = dbc.Container(children=[html.H4("BTCUSDT Binance"), 
                                            btcusdt_interval_selector,
                                            dcc.Graph(id='btcusdt-candlestick-plot', 
                                                      responsive=True)],
                                  fluid=True)

tab_btcusdt = dbc.Tab(id='btcusdt-tab', label='BTCUSDT Graph', children=[btcusdt_container])

@callback(Output('btcusdt-candlestick-plot', 'figure'),
          Input('btcusdt-interval-selector', 'value'),
          Input(dbt.ThemeSwitchAIO.ids.switch('theme'), 'value'))
def display_btc_candlestick(value, theme):
    if not binance.ping():
        raise Exception("ERROR: Cannot contact binance. Try setting proxy")

    if value not in binance.INTERVALS.keys():
        raise Exception(f'INTERVAL ERROR: interval is not valid: {value}')
    btcusdt_df = binance.main(symbol='BTCUSDT', interval=value)
    fig = go.Figure(go.Candlestick(x=btcusdt_df.index,
                                   open=btcusdt_df.Open,
                                   close=btcusdt_df.Close,
                                   low=btcusdt_df.Low,
                                   high=btcusdt_df.High))
    fig.layout.template = template='minty' if theme else 'minty_dark'
    return fig

@callback(Output("btcusdt-candlestick-plot", "figure", allow_duplicate=True),
          Input(dbt.ThemeSwitchAIO.ids.switch('theme'), 'value'),
          prevent_initial_call=True)
def update_figure_template(switch_on):
    template = pio.templates["minty"] if switch_on else pio.templates["minty_dark"]
    patch_figure = Patch()
    patch_figure["layout"]["template"] = template
    return patch_figure

solusdt_interval_selector = dbc.RadioItems(id='solusdt-interval-selector', inline=True,
                                           value='1d',
                                           options=['1m', '1h', '4h', '1d', '1w', '1M'])

solusdt_container = dbc.Container(children=[html.H4("SOLUSDT Binance"), 
                                            solusdt_interval_selector,
                                            dcc.Graph(id='solusdt-candlestick-plot', 
                                                      responsive=True)],
                                  fluid=True)

tab_solusdt = dbc.Tab(id='solusdt-tab', label='SOLUSDT Graph', children=[solusdt_container])

@callback(Output('solusdt-candlestick-plot', 'figure'),
          Input('solusdt-interval-selector', 'value'),
          Input(dbt.ThemeSwitchAIO.ids.switch('theme'), 'value'))
def display_sol_candlestick(value, theme):
    if not binance.ping():
        raise Exception("ERROR: Cannot contact binance. Try setting proxy")

    if value not in binance.INTERVALS.keys():
        raise Exception(f'INTERVAL ERROR: interval is not valid: {value}')
    solusdt_df = binance.main(symbol='SOLUSDT', interval=value)
    fig = go.Figure(go.Candlestick(x=solusdt_df.index,
                                   open=solusdt_df.Open,
                                   close=solusdt_df.Close,
                                   low=solusdt_df.Low,
                                   high=solusdt_df.High))
    fig.layout.template = template='minty' if theme else 'minty_dark'
    return fig

@callback(Output("solusdt-candlestick-plot", "figure", allow_duplicate=True),
          Input(dbt.ThemeSwitchAIO.ids.switch('theme'), 'value'),
          prevent_initial_call=True)
def update_figure_template(switch_on):
    template = pio.templates["minty"] if switch_on else pio.templates["minty_dark"]
    patch_figure = Patch()
    patch_figure["layout"]["template"] = template
    return patch_figure

component_initial_date = dcc.DatePickerSingle(id='initial-date',
                                              display_format='DD/MM/YYYY',
                                              clearable=True,
                                              persistence=True,
                                              date=datetime.date.today())

def find_next_friday(day = datetime.date.today()):
    '''Finds next nearest friday.'''
    while day.isoweekday() != 5:
        day += datetime.timedelta(1)
    return day

def find_last_friday_of_month(year=datetime.date.today().year, month=datetime.date.today().month):
    if month<12:
        date = datetime.date(year,month+1,1)
    else:
        date = datetime.date(year+1,1,1)
    match date.weekday():
        case 0|1|2|3:
            delta = date.weekday() + 3
        case 4:
            delta = 0
        case 5|6:
            delta = date.weekday() - 4
    date -= datetime.timedelta(days=delta)
    return date

def find_quarterly_fridays(year):
    '''Find important fridays. These months are important because a
    huge amount of options expire on these dates.
    '''
    return [find_last_friday_of_month(year, month) for month in [3,6,9,12]]

this_year = datetime.date.today().year
important_dates = [date for date in
                   [find_last_friday_of_month()]+find_quarterly_fridays(this_year)+find_quarterly_fridays(this_year+1)+find_quarterly_fridays(this_year+2)
                   if date > datetime.date.today()][:6]

date_picker_components = [[dcc.DatePickerSingle(id=f'date-picker-{i}',
                                                display_format='DD/MM/YYYY',
                                                clearable=True,
                                                persistence=True,
                                                date=important_dates[i]),
                           html.Div(children=f'Placeholder text {i}', id=f'number-of-days-{i}'), html.P()]
                                               for i in
                          range(len(important_dates))]

# (defun flatten (tree)
#   (cond ((null tree) nil)
# 	((listp tree) (append (flatten (car tree)) (flatten (cdr tree))))
# 	(t (list tree))))

def flatten(tree):
    if isinstance(tree, list):
        if len(tree) == 0:
            return []
        elif len(tree) == 1:
            return flatten(tree[0])
        elif len(tree) > 1:
            return flatten(tree[0]) + flatten(tree[1:])
    else:
        return [tree]    

days_calc_container = dbc.Container(children=[html.H4('Dates Calculator'),
                                              html.Div(children=['Initial Date: ', component_initial_date]),
                                              html.P()] + flatten(date_picker_components))

tab_days_calc = dbc.Tab(id='dates-calc-tab', label='Days Calculator',
                        children=[days_calc_container])

@callback(output=Output(date_picker_components[0][1].id, 'children'),
          inputs=[Input('initial-date', 'date'),
                  Input(date_picker_components[0][0].id, 'date')])
def calculate_days_interval_0(date_str_1, date_str_2):
  if (not date_str_1) or (not date_str_2):
    return 'Choose Dates'
  date1 = datetime.date.fromisoformat(date_str_1)
  date2 = datetime.date.fromisoformat(date_str_2)
  return f'{abs((date1-date2).days)} days'

@callback(output=Output(date_picker_components[1][1].id, 'children'),
          inputs=[Input('initial-date', 'date'),
                  Input(date_picker_components[1][0].id, 'date')])
def calculate_days_interval_1(date_str_1, date_str_2):
  if (not date_str_1) or (not date_str_2):
    return 'Choose Dates'
  date1 = datetime.date.fromisoformat(date_str_1)
  date2 = datetime.date.fromisoformat(date_str_2)
  return f'{abs((date1-date2).days)} days'

@callback(output=Output(date_picker_components[2][1].id, 'children'),
          inputs=[Input('initial-date', 'date'),
                  Input(date_picker_components[2][0].id, 'date')])
def calculate_days_interval_2(date_str_1, date_str_2):
  if (not date_str_1) or (not date_str_2):
    return 'Choose Dates'
  date1 = datetime.date.fromisoformat(date_str_1)
  date2 = datetime.date.fromisoformat(date_str_2)
  return f'{abs((date1-date2).days)} days'

@callback(output=Output(date_picker_components[3][1].id, 'children'),
          inputs=[Input('initial-date', 'date'),
                  Input(date_picker_components[3][0].id, 'date')])
def calculate_days_interval_3(date_str_1, date_str_2):
  if (not date_str_1) or (not date_str_2):
    return 'Choose Dates'
  date1 = datetime.date.fromisoformat(date_str_1)
  date2 = datetime.date.fromisoformat(date_str_2)
  return f'{abs((date1-date2).days)} days'

@callback(output=Output(date_picker_components[4][1].id, 'children'),
          inputs=[Input('initial-date', 'date'),
                  Input(date_picker_components[4][0].id, 'date')])
def calculate_days_interval_4(date_str_1, date_str_2):
  if (not date_str_1) or (not date_str_2):
    return 'Choose Dates'
  date1 = datetime.date.fromisoformat(date_str_1)
  date2 = datetime.date.fromisoformat(date_str_2)
  return f'{abs((date1-date2).days)} days'

@callback(output=Output(date_picker_components[5][1].id, 'children'),
          inputs=[Input('initial-date', 'date'),
                  Input(date_picker_components[5][0].id, 'date')])
def calculate_days_interval_5(date_str_1, date_str_2):
  if (not date_str_1) or (not date_str_2):
    return 'Choose Dates'
  date1 = datetime.date.fromisoformat(date_str_1)
  date2 = datetime.date.fromisoformat(date_str_2)
  return f'{abs((date1-date2).days)} days'

options_container = dbc.Container(children=[
    dbc.Button("Add Option", id="add-option-btn", n_clicks=0),
    html.Div(id='container-div', children=[]),
    html.Div(id='container-output-div'),
], fluid=True)

tab_options = dbc.Tab(id='option-tab', label="Options Tab", children=[
    options_container])

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
        dbc.Textarea(id={'type': 'text-area', 'index': n_clicks}, readOnly=True, rows=1),
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

@callback(Output({'type': 'text-area', 'index': MATCH}, 'value'),
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

    normalized_parameters = (price, strike, time, vol/100, rate/100, dividend/100)
    option_price = option.optionfn(*normalized_parameters)
    greeks = option.greeks(*normalized_parameters)
    delta, gamma, theta, vega, rho = greeks.values()
    leverage = price*delta/option_price

    logger.info('(price,strike,time,vol,rate,dividend,option_type)={}'.format((price, strike, time, vol, rate, dividend, option_type)))
    logger.info(f'(price,strike,time,vol,rate,dividend,option_type)={normalized_parameters}')
    logger.info(f'Options price={option_price}')
    logger.info(f'Greeks: {greeks}')

    return f'{option_price:.3g}; Δ: {100*delta:.1f}; γ: {10000*gamma:.2f}; θ: {theta:.2f}; ν: {vega:.1f}; ρ: {rho:0.1f}; leverage = {leverage:.1f}x'

tabs = dbc.Tabs(id='tabs', children=[
    tab_options,
    tab_plot, tab_btcusdt, tab_ethusdt, tab_solusdt, tab_days_calc], persistence=True, persistence_type='session')

app.layout = dbc.Container(children=[
    dbc.Row(children=[color_mode_switch], justify='center'), 
    tabs
], fluid=False, className='m-4 dbc')

def main(vals):
    component_price_rangeslider.value = vals.price_range
    component_time_rangeslider.max = vals.time
    component_strike_price.value = vals.strike
    component_volatility.value = vals.volatility
    component_rate.value = vals.rate
    component_dividend.value = vals.dividend
    component_option_type.value = 'Call' if vals.option_type == 'c' else 'Put'
    app.run(debug=vals.debug_mode, host='0.0.0.0', port=vals.server_port )
    return None

if __name__ == '__main__':
    from argparse import Namespace
    args = Namespace(price_range=[3000,4000],
                     strike_price=4000,
                     time=270,
                     volatility=70,
                     rate=0, dividend=0,
                     option_type='c',
                     server_port='0.0.0.0',
                     debug_mode=True)
    
    main(args)
