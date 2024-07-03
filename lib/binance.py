"""Author: Kamy

This is the api code. It is for communicating with the binance server
and getting information from them.

Date Info: Must be in format YY/MM/DD HH:MM

Time Info (for development): Binance wants time information in milliseconds since epoch. But
datetime gives it in seconds. So there has to be some jiggering to fix
that in communications with binance.

Proxy Info: To set a proxy just do this in the shell 

```$ export ALL_PROXY=PROXY_HOST:PROXY_PORT``` 

This program uses requests library which automatically uses
environment proxies.
"""

import logging
import requests
import json
import time, datetime
import numpy as np
import pandas as pd
import os

import icecream
from icecream import ic

def getTime():
    return str(datetime.datetime.now())
ic.configureOutput(prefix=getTime, includeContext=True)

## Logger Setup ##
logger = logging.getLogger('binance-api')
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    # "filters": {...},
    # "formatters": {...},
    # "handlers": {...},
    # "loggers": {...}
}

## Global Variables ##
PROXY = {'all': '0.0.0.0:2081'}
SITE = 'https://api.binance.com'
DAY_DELTA = datetime.timedelta(days=1)
INTERVALS = {'1s': datetime.timedelta(seconds=1),
             '1m': datetime.timedelta(seconds=60),
             '3m': datetime.timedelta(seconds=180),
             '5m': datetime.timedelta(seconds=300),
             '15m': datetime.timedelta(seconds=900),
             '30m': datetime.timedelta(seconds=1800),
             '1h': datetime.timedelta(seconds=3600),
             '2h': datetime.timedelta(seconds=7200),
             '4h': datetime.timedelta(seconds=14400),
             '6h': datetime.timedelta(seconds=21600),
             '8h': datetime.timedelta(seconds=28800),
             '12h': datetime.timedelta(seconds=43200),
             '1d': datetime.timedelta(days=1),
             '3d': datetime.timedelta(days=3),
             '1w': datetime.timedelta(days=7),
             '1M': datetime.timedelta(weeks=4)}
SYMBOLS = None

## Miscellaneous ##
def ping():
    '''Check to see if we can contact the server'''
    req = requests.get(SITE+'/api/v3/ping')
    if req.status_code != 200:
        raise Exception(f'Error: cannot contact the server: {req.status_code}')
    return True

def get_symbols():
    '''Get an np.array of all symbols'''
    req = requests.get(SITE+'/api/v3/exchangeInfo')
    if req.status_code != 200:
        raise Exception(f"Error: contact with server failed:\t {req.status_code}")
    ic(req)
    return np.array([elt['symbol'] for elt in json.loads(req.text)['symbols']])

def ensure_symbol(symbol):
    global SYMBOLS
    if SYMBOLS is None:
        SYMBOLS= get_symbols()
    if symbol not in SYMBOLS:
        raise Exception(f'Error: Symbol {symbol} not available')
    return True

def date_to_epoch(date):
    '''Date is in this format: YY/MM/DD HH:MM

    Example "24/06/26 14:50" returns epoch equivalent to
    datetime(2024, 6, 26, 14, 50) which is equal to 1719431520.0

    >>> date_to_epoch('20/01/01 12:00')
    1577867400.0
    '''
    return datetime.datetime.strptime(date, "%y/%m/%d %H:%M").timestamp()

def datetime_to_milliseconds(date):
    '''Turns datetime object to milliseconds

    >>> datetime_to_milliseconds(datetime.datetime(2024,6,26,14,50))
    1719400800000
    '''
    return int(date.timestamp()*1000)

def date_to_milliseconds(date): 
    '''Date is in this format: YY/MM/DD HH:MM

    >>> date_to_milliseconds("20/01/01 12:00")
    1577867400000.0

    >>> date_to_milliseconds("24/06/26 14:50") #equivalent to datetime(2024, 6, 26, 14, 50)
    1719431520000.0
    '''
    return date_to_epoch(date)*1000

def milliseconds_to_datetime(ms):
    '''Data from binance is in milliseconds. Convert milliseconds to
    datetime object.

    >>> milliseconds_to_datetime(1719417832.0)
    datetime.datetime(2024, 6, 26, 19, 33, 50)
    '''
    seconds_to_epoch = int(ms/1000)
    return datetime.datetime.fromtimestamp(seconds_to_epoch)

def ensure_interval(interval_string):
    '''
    == Interval must be one of the following ==
    seconds 	1s
    minutes 	1m, 3m, 5m, 15m, 30m
    hours 	1h, 2h, 4h, 6h, 8h, 12h
    days 	1d, 3d
    weeks 	1w
    months 	1M
    '''
    if interval_string not in INTERVALS.keys():
        raise Exception(f'Interval incorrect "{interval_string}"')
    return True

import builtins
def normalize_date_to_datetime(date):
    '''Get date and return a datetime object

    >>> normalize_date_to_datetime('24/6/10 10:00')
    datetime.datetime(2024, 6, 10, 10, 0)

    >>> normalize_date_to_datetime(int(normalize_date_to_datetime('24/6/10 10:00').timestamp())*1000)
    datetime.datetime(2024, 6, 10, 10, 0)

    >>> normalize_date_to_datetime(normalize_date_to_datetime('24/6/10 10:00').timestamp()*1000*1000)
    datetime.datetime(2024, 6, 10, 10, 0)
    '''
    match type(date):
        case datetime.datetime:
            return date
        # is in microseconds
        case builtins.float:
            return datetime.datetime.fromtimestamp(int(date/1000/1000))
        # is in milliseconds
        case builtins.int:
            return datetime.datetime.fromtimestamp(int(date/1000))
        case builtins.str:
            return datetime.datetime.strptime(date, '%y/%m/%d %H:%M')

def ensure_dates(startTime, endTime, intervalString='1d'):
    '''Dates are auto-converted, function ensures they are less than
    1000 intervals apart

    >>> check_dates('24/01/01 00:00','22/01/01 00:00','1d')
    True

    >>> check_dates('24/01/01 00:00','20/01/01 00:00','1d')
    False
    '''
    if intervalString not in INTERVALS.keys():
        raise Exception(f'Error: interval is incorrect')

    startDate = normalize_date_to_datetime(startTime)
    endDate = normalize_date_to_datetime(endTime)
    if endDate - startDate > 1000*INTERVALS[intervalString]:
        maxStartDate = endDate - 1000*INTERVALS[intervalString]
        logger.error(f'startTime cannot be before {maxStartDate}')
        raise Exception('Error: Dates are too far apart.')
    return True

def get_klines(symbol='ETHUSDT', interval='1d',
               startTime=(datetime.datetime.now()-DAY_DELTA),
               endTime=datetime.datetime.now(), timeZone='+3:30', limit=1000):
    '''get_klines(symbol, interval, startTime, endTime, timeZone, limit)
    == klines Parameters ==
    Name 	Type 	Mandatory 	Description
    symbol 	STRING 	YES 	
    interval 	ENUM 	YES 	
    startTime 	LONG 	NO 	
    endTime 	LONG 	NO 	
    timeZone 	STRING 	NO 	Default: 0 (UTC)
    limit 	INT 	NO 	Default 500; max 1000.'''

    ensure_interval(interval)
    startTime = normalize_date_to_datetime(startTime)
    endTime = normalize_date_to_datetime(endTime)
    ensure_dates(startTime, endTime)

    parameters = dict(symbol=symbol, interval=interval,
                      startTime=datetime_to_milliseconds(startTime),
                      endTime=datetime_to_milliseconds(endTime), timeZone=timeZone,
                      limit=limit)
    req = requests.get(SITE+'/api/v3/klines', params=parameters)

    if req.status_code == 200:
        return eval(req.text)
    else:
        code, msg = json.loads(req.text).values()
        raise Exception(f'Error code {code}: {msg}')

PARSED_OUTPUT_COLUMNS = [('open time', datetime.datetime),
                         ('Open price', float),
                         ('High price', float),
                         ('Low price', float),
                         ('Close price', float),
                         ('Volume', float),
                         ('Close time', int),
                         ('Quote asset volume', float),
                         ('Number of trades', int),
                         ('Taker buy base asset volume', float),
                         ('Taker buy quote asset volume', float)]

def parse_kline(kline):
    '''KLine data Input:
    [Kline open time, Open price, High price, Low price, Close price,
    Volume, Kline Close time, Quote asset volume, Number of trades,
    Taker buy base asset volume, Taker buy quote asset volume, Unused
    field, ignore.]

    KLine data Output:
    [Open time, Open price, High price, Low price, Close price,
    Volume]
    '''

    open_time = milliseconds_to_datetime(kline[0])
    open_price = kline[1]
    high_price = kline[2]
    low_price = kline[3]
    close_price = kline[4]
    volume = kline[5]
    close_time = kline[6]
    quote_volume = kline[7]
    num_of_trades = kline[8]
    taker_buy_base_asset_volume = kline[9]
    taker_buy_quote_asset_volume = kline[10]

    return np.array([open_time, open_price, high_price, low_price,
                     close_price, volume, close_time, quote_volume,
                     num_of_trades, taker_buy_base_asset_volume,
                     taker_buy_quote_asset_volume])

def create_dataframe(dataList):
    '''Takes the data received from server and converts it into proper
    datatypes and returns DataFrame.

    Only the time needs to be converted.'''
    # Turn into np.array so I can use slices.
    dataArray = np.array(dataList)

    # Convert milliseconds from server into datetime objects
    open_time = pd.Series(data=[milliseconds_to_datetime(row[0]) for
                                row in dataList], dtype='datetime64[ns]')
    return pd.DataFrame(data=dataArray[:,1:6], index=open_time,
                        columns=['Open', 'High', 'Low', 'Close',
                                 'Volume'], dtype=float)

def main(symbol='ETHUSDT', interval='1d', startDate=datetime.datetime.now()-1000*INTERVALS['1d'] , endDate=datetime.datetime.now(), number_of_intervals=1000):
    ensure_interval(interval)
    ensure_symbol(symbol)
    startTime = endDate - INTERVALS[interval]*number_of_intervals
    dataList = get_klines(symbol, interval, startTime=startTime, endTime=endDate, timeZone='00:00', limit=number_of_intervals)
    return create_dataframe(dataList)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
