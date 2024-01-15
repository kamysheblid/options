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
dbt.load_figure_template(['minty', 'minty_dark'])
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY, dbc.icons.FONT_AWESOME], title="Options Pricing")

color_mode_switch = html.Span(
    [dbc.Label(className="fa fa-moon", html_for="color_mode_switch"),
     dbc.Switch(id="color-mode-switch", value=True, className='d-inline-block ms-1', persistence=True),
     dbc.Label(className='fa fa-sun', html_for="color-mode-switch")])

component_price_rangeslider = dcc.RangeSlider(min=1, max=5000, count=1, value=[2500,3500], id='price-range', tooltip={'placement':'bottom', 'always_visible':True})
component_time_rangeslider = dcc.RangeSlider(min=1, max=100, count=1, value=[1,60], id='time-range', tooltip={'placement':'bottom', 'always_visible':True})

component_strike_price = dcc.Input(id='strike-price', type='number', placeholder="Strike", value=3000, inputMode='numeric', debounce=True)
component_time = dcc.Input(id='amount-time', type='number', placeholder="Number of Days", value=7, inputMode='numeric', debounce=True)
component_volatility = dcc.Input(id='volatility', type='number', placeholder="Volatility", value=70, inputMode='numeric', debounce=True)
component_rate = dcc.Input(id='rate', type='number', placeholder="Rate", value=40, inputMode='numeric', debounce=True)
component_dividend = dcc.Input(id='dividend', type='number', placeholder="Dividend", value=40, inputMode='numeric', debounce=True)
component_option_type = dcc.RadioItems(id='option-type', options=['Put', 'Call'], value='Call', inline=False)

component_graph = dcc.Graph(id="price-graph", responsive=True)#, animate=True, animate_options={transition_duration=200})

options_container = dbc.Container(children=[
    dbc.Button("Add Option", id="add-option-btn", n_clicks=0),
    html.Div(id='container-div', children=[]),
    html.Div(id='container-output-div'),
], fluid=True)

tab_plot = dcc.Tab(id={'type': 'tab', 'index': 'plot'}, label="Plot Tab", children=[
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
    dbc.Row(component_graph, justify='center', align='center'),
],
                   selected_style={'borderTop': '2px solid green', 'border': '1px solid blue', 'color': 'black'}, 
                   disabled_style={'borderTop': '1px solid blue', 'border': '1px solid white', 'color': 'purple'}
                   )
tab_options = dcc.Tab(id={'type': 'tab', 'index': 'options'}, label="Options Tab", children=[
    options_container])
tabs = dcc.Tabs(id='tabs', children=[
    tab_options,
    tab_plot,])

app.layout = dbc.Container(children=[
    dbc.Row(children=[color_mode_switch], justify='center'), 
    tabs
], fluid=True)

def make_new_option(n_clicks):
    logger.info(f'Making new option index={n_clicks}')
    return dbc.Container(children=[
        f"Option #{n_clicks}: ",
        dbc.Form(children=[
            dcc.Input(id={'type': 'price', "index": n_clicks}, persistence=True, persistence_type='memory', type='number', placeholder='Price ($)', min=0),
            dcc.Input(id={'type': 'strike', "index": n_clicks}, persistence=True, persistence_type='memory', type='number', placeholder='Strike ($)', min=0),
            dcc.Input(id={'type': 'time', "index": n_clicks}, persistence=True, persistence_type='memory', type='number', placeholder='Time (Days)', min=0),
            dcc.Input(id={'type': 'vol', "index": n_clicks}, persistence=True, persistence_type='memory', type='number', placeholder='Vol (%)', min=0),
            dcc.Input(id={'type': 'rate', "index": n_clicks}, persistence=True, persistence_type='memory', type='number', placeholder='Rate (%)'),
            dcc.Input(id={'type': 'dividend', "index": n_clicks}, persistence=True, persistence_type='memory', type='number', placeholder='Dividend (%)'),
            dcc.RadioItems(id={'type': 'option-type', 'index': n_clicks}, options=['Call', 'Put'], value='Call', inline=True),
        ], id={'type': 'option-form', 'index': n_clicks}),
        dcc.Textarea(id={'type': 'text-area', 'index': n_clicks}, readOnly=True, rows=1),
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
def options_calculator(*vals):
    if not all(vals):
        return
    price,strike,time,vol,rate,dividend,option_type = vals
    if option_type.lower() == 'call':
        optionfn = options.Call().optionfn
    else:
        optionfn = options.Put().optionfn
    option_price = optionfn(price, strike, time, vol/100, rate/100, dividend/100)
    logger.info(f'Calculating options price using ({vals}): {option_price}')
    return f'{option_price}'

@callback(Output("price-graph", "figure", allow_duplicate=True),
          [Input("{}".format(_), "value") for _ in ['price-range', 'strike-price', 'time-range', 'volatility', 'rate', 'dividend', 'option-type', 'amount-time']],
          prevent_initial_call='initial_duplicate')
def render_plot(*vals):
    logger.info(f'render_plot input args: {vals}')
    price = np.linspace(*vals[0], 500)
    strike = vals[1]
    time_range = np.linspace(*vals[2],int(vals[7]),dtype=int)
    volatility = vals[3]/100
    rate = vals[4]/100
    dividend = vals[5]/100
    option_type = vals[6]
    if option_type.lower() == 'call':
        optionfn = options.Call().optionfn
    else:
        optionfn = options.Put().optionfn
    df = pd.DataFrame({f"{time:d}d": optionfn(price, strike, time, volatility, rate, dividend) for time in time_range}, index=price)
    return df

@callback(Output("price-graph", "figure", allow_duplicate=True),
          [Input("{}".format(_), "value") for _ in ['price-range', 'strike-price', 'time-range', 'volatility', 'rate', 'dividend', 'option-type', 'amount-time']],
          prevent_initial_call='initial_duplicate')
def render_plot(*vals):
    logger.info(f'render_plot input args: {vals}')
    df = create_option_dataframe(vals)
    fig = px.line(df, template="minty", labels='label')
    hover_template = "<br>".join(["Asset Price: $%{x}", "Option Price: $%{y}"]) + "<extra></extra>"
    fig.update_layout(yaxis={'type': 'log'}, xaxis_title="Asset Price ($)", yaxis_title="Option Price ($)", transition_duration=250, template='plotly_dark', hovermode='x unified')
    fig.update_legends(title={'text':'Days to Expiry'})
    #fig.update_traces(hovertemplate=hover_template)
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

clientside_callback("""(SwitchOn) => {
SwitchOn
? document.documentElement.setAttribute('data-bs-theme', 'light')
: document.documentElement.setAttribute('data-bs-theme', 'dark')
return window.dash_clientside.no_update
}""",
                    Output('color-mode-switch', 'id'),
                    Input('color-mode-switch', 'value'))

@callback(Output({'type': 'tab'}, 'style'),
          Input('color-mode-switch', 'value'))
def update_tabs_darkmode(switch_on):
    patch = Patch()
    patch_styles = Patch()

    style = "border: 1px solid rgb(229,229,229); border-style: solid; border-radius: 0; border-width: 0px 0px 1px 0px;"
    selected_style = 'border-top: 2px solid #2186f4; background-color: #ffffff; border: 1px solid rgb(229,229,229); color: #3f3f3f'
    darkmode_style="border-bottom: 1px solid rgb(214, 214, 214); padding: 6px; font-weight: bold;"
    darkmode_selected_style="border-top: 1px solid rgb(214, 214, 214); border-bottom: 1px solid rgb(214, 214, 214); background-color: rgb(17, 157, 255); color: white; padding: 6px;"


    disabled_style = {'borderTop': '1px solid black', 'border': '1px solid green', 'color': 'blue'}
    selected_style = {'borderTop': '2px solid blue', 'border': '1px solid black', 'color': 'green'}
    darkmode_disabled_style = {'borderTop': '1px solid blue', 'border': '1px solid white', 'color': 'purple'}
    darkmode_selected_style = {'borderTop': '2px solid green', 'border': '1px solid blue', 'color': 'black'}

    if switch_on:
        patch['props']['disabled_style'] = darkmode_disabled_style
        patch['props']['selected_style'] = darkmode_selected_style
        patch_styles = {'disabled_style': darkmode_disabled_style, 'selected_style': darkmode_selected_style}
    else:
        patch['props']['disabled_style'] = disabled_style
        patch['props']['selected_style'] = selected_style
        patch_styles = {'disabled_style': disabled_style, 'selected_style': selected_style}
    return patch_styles

@callback(Output("price-graph", "figure", allow_duplicate=True),
          Input("color-mode-switch", "value"),
          prevent_initial_call='initial_duplicate')
def update_figure_template(switch_on):
    template = pio.templates["minty"] if switch_on else pio.templates["minty_dark"]
    patch_figure = Patch()
    patch_figure["layout"]["template"] = template
    return patch_figure

app.run(debug=False, host='0.0.0.0')
