def volatility(df, number_of_days=None):
    if number_of_days and number_of_days < df.shape[0]:
        df = df.tail(number_of_days)
    close = df.Close.values
    daily_log_return = np.log(close[1:]/close[:-1])
    mean_log_return = np.mean(daily_log_return)
    log_minus_mean_squared = np.power(daily_log_return-mean_log_return, 2)
    return np.sqrt(1/(len(close)-1)*np.sum(log_minus_mean_squared))*100

def std_deviation(arr, trading_days=365):
    '''Calculates close-to-close standard deviation.

    Example calculation
    >>> arr = np.array([50.00, 49.66, 50.75, 51.20, 51.47, 50.74, 50.61, 50.03, 51.17, 52.40, 52.84, 51.75, 51.74, 52.25, 51.84, 52.92])
    >>> round(volatility(arr),3)
    1.393
    >>> round(volatility(arr, 256),3)*np.sqrt(256)
    22.288'''

    daily_log_return = np.log(arr[1:]/arr[:-1])
    mean_log_return = np.mean(daily_log_return)
    log_minus_mean_squared = np.power(daily_log_return-mean_log_return, 2)
    return np.sqrt(1/(len(arr)-1)*np.sum(log_minus_mean_squared))*100

def volatility_parkinson(df, number_of_days=None):
    '''Calculates close-to-close standard deviation.

    Example calculation
    >>> arr = np.array([50.00, 49.66, 50.75, 51.20, 51.47, 50.74, 50.61, 50.03, 51.17, 52.40, 52.84, 51.75, 51.74, 52.25, 51.84, 52.92])
    >>> round(volatility(arr),3)
    1.393
    >>> round(volatility(arr, 256),3)*np.sqrt(256)
    22.288'''

    if number_of_days and number_of_days < df.shape[0]:
        df = df.tail(number_of_days)
    high = df.High.values
    low = df.Low.values
    daily_log = np.log(high/low)
    mean_log = np.mean(daily_log)
    squared = np.power(daily_log-mean_log, 2)
    squared = np.power(daily_log, 2)
    return np.sqrt(1/(len(high))/(4*np.log(2))*np.sum(squared))*100
