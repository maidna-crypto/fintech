import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

Data= 'All.csv' 
BTC = 'Dataset.csv'
Oil= 'Oil-1.csv'
Gold= 'Gold-1.csv'

df = pd.read_csv(Data)

correlation_df = df.corr()
print(correlation_df)

hm = sns.heatmap(df.corr(), annot = True)

hm.set(xlabel='\Currency', ylabel='Currency\t', title = "Correlation matrix of Currency data\n")

plt.show()

# Time lagged cross correlation
def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

df1 = pd.read_csv(BTC)
df2 = pd.read_csv(Oil)
df3 = pd.read_csv(Gold)

df1_series = df1.squeeze()
df2_series = df2.squeeze()
df3_series = df3.squeeze()

window = 10
# lags = np.arange(-(fs), (fs), 1)  # uncontrained
lags = np.arange(0, (200), 1)  # contrained
rs = np.nan_to_num([crosscorr(df1_series, df2_series, lag) for lag in lags])

print(
    "xcorr {}-{}".format(df1_series, df2_series), lags[np.argmax(rs)], np.max(rs))

rs = np.nan_to_num([crosscorr(df1_series, df3_series, lag) for lag in lags])

print(
    "xcorr {}-{}".format(df1_series, df3_series), lags[np.argmax(rs)], np.max(rs))

lags= np.arange(-200, (0), 1)  # contrained
rs = np.nan_to_num([crosscorr(df1_series, df2_series, lag) for lag in lags])

print(
    "xcorr {}-{}".format(df1_series, df2_series), lags[np.argmax(rs)], np.max(rs))

rs = np.nan_to_num([crosscorr(df1_series, df3_series, lag) for lag in lags])

print(
    "xcorr {}-{}".format(df1_series, df3_series), lags[np.argmax(rs)], np.max(rs))




