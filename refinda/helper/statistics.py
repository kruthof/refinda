import pandas as pd
import numpy as np

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
    cov = np.cov(returns1,returns2)

    theta = 1/(len(returns1)-window) * (2*sigma_sq_1*sigma_sq_2 + 2 * sigma_1*sigma_2*cov + 0.5 * np.square(mu_1)*
                                        sigma_sq_1 + 0.5*np.square(mu_2)*sigma_sq_2 - (mu_1*mu_2 / (sigma_1*sigma_2) * np.square(cov)))

    z_value = (sigma_1*mu_1 - sigma_2*mu_2) / np.sqrt(theta)

    return z_value[0,1]