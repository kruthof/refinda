import pandas as pd
import numpy as np
from scipy.stats import norm


def significance_sharp(sharps1, sharps2,window):
    '''
    Conducts significance test of two series of sharp values
    where each entry has been calculated on a look-back period = window

    @param sharps1 series sharp value of series 1
    @param sharps2 series sharp value of series 2
    @param window int look-back window of previous sharp calculation

    @return dataframe  z-score and p-value
    '''

    z_score = zvalue_sharp(sharps1,sharps2,window)
    p_value = z_p_value(z_score)

    return pd.DataFrame({'z-score':z_score,'p-value one sided':p_value,'p-value two sided':p_value*2})

def zvalue_sharp(returns1, returns2,window):
    '''
    Function performs t-test according to Memmel (2003)
    and Jobson & Korkie (1981)

    @param returns1 series returns for candiate 1
    @param returns2 series returns for candidate 2

    @return float z-value
    '''

    mu_1 = np.mean(returns1)
    mu_2 = np.mean(returns2)

    sigma_1 = np.std(returns1)
    sigma_2=np.std(returns2)

    sigma_sq_1 = np.square(sigma_1)
    sigma_sq_2= np.square(sigma_2)
    cov = np.cov(returns1.iloc[:,0],returns2.iloc[:,0])[0][1]

    theta = 1/(len(returns1)-window) * (2*sigma_sq_1*sigma_sq_2 + 2 * sigma_1*sigma_2*cov + 0.5 * np.square(mu_1)*
                                        sigma_sq_1 + 0.5*np.square(mu_2)*sigma_sq_2 - (mu_1*mu_2 / (sigma_1*sigma_2) * np.square(cov)))

    z_value = (sigma_1*mu_1 - sigma_2*mu_2) / np.sqrt(theta)

    return z_value

def z_p_value(z):
    '''
    Calculates p-value of given z-score
    @param z float z-value

    @return float p-value
    '''

    return norm.sf(np.abs(z))

