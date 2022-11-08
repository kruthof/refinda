from pyfolio import timeseries


def get_summary(returns_vector,nan='remove'):
    '''
    Providing summary statistics for a given return vector

    @param returns_vector pandas series

    @return pandas series summary statistics
    '''

    if nan is 'remove':
        returns_vector = returns_vector[returns_vector.notna()]
    elif nan is 'zero':
        returns_vector = returns_vector.fillna(0)
    else:
        raise KeyError('missing value strategy not available')

    perf_func = timeseries.perf_stats
    return perf_func(returns_vector)