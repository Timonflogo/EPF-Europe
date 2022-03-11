#%% import libraries
import pandas as pd
import numpy as np
#%% load datasets Windows
DE = pd.read_csv('elspotprices-DE.csv').iloc[:,1:]
DK1 = pd.read_csv('elspotprices-DK1.csv').iloc[:,1:]
DK2 = pd.read_csv('elspotprices-DK2.csv').iloc[:,1:]
SE3 = pd.read_csv('elspotprices-SE3.csv').iloc[:,1:]
SE4 = pd.read_csv('elspotprices-SE4.csv').iloc[:,1:]
NO2 = pd.read_csv('elspotprices-NO2.csv').iloc[:,1:]

#%% Drop Duplicates in HourDK
DE = DE.drop_duplicates(subset=['HourDK'])
DK1 = DK1.drop_duplicates(subset=['HourDK'])
DK2 = DK2.drop_duplicates(subset=['HourDK'])
SE3 = SE3.drop_duplicates(subset=['HourDK'])
SE4 = SE4.drop_duplicates(subset=['HourDK'])
NO2 = NO2.drop_duplicates(subset=['HourDK'])

#%% pivot Data 
df = pd.concat([DE, DK1, DK2, SE3, SE4, NO2])

#%% Inspect for missing data
df.isna().sum()
# No missing data deteceted

#%% drop Time UTC column and SportPriceDKK
df = df[['HourDK', 'PriceArea', 'SpotPriceEUR']]

#%% Set Datetime 
df['HourDK'] = pd.to_datetime(df['HourDK'])

#%% Create new Index
df.index = np.arange(len(df))

#%% Pivot PriceArea
df = df.pivot(index='HourDK', columns=['PriceArea'], values=['SpotPriceEUR'])

#%% delete first two rows
df = df.iloc[2:]


#%% save data to CSV
# df.to_csv('Elprices.csv')

#%% Load Data
df = pd.read_csv('Elprices.csv')

#%% Rename columns
df = df.rename(columns={"Unnamed: 0": "HourDK","SpotPriceEUR": "DE","SpotPriceEUR.1": "DK1","SpotPriceEUR.2": "DK2","SpotPriceEUR.3": "NO2","SpotPriceEUR.4": "SE3","SpotPriceEUR.5": "SE4"})

#%% Delete first two rows
df = df.iloc[2:]

#%% set index
df = df.set_index('HourDK')

#%% save data to CSV
df.to_csv('Elprices.csv')

#%% Load dataset
df = pd.read_csv('Elprices.csv', index_col='HourDK', parse_dates=True)
