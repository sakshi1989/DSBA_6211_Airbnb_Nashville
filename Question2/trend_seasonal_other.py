# -*- coding: utf-8 -*-
"""airbnb.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1F9XBOPPU0fcfOkSpLkfbEhsa9Ic_aWtA
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from shapely.geometry import Point #location
# %pylab inline
# %matplotlib inline

from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from mpl_toolkits.mplot3d import Axes3D

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()
from statsmodels.tsa.seasonal import seasonal_decompose
from time import time
import seaborn as sns
sns.set(style="whitegrid")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope


import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = np.random.seed(0)

summary_listings = pd.read_csv('summary-listings.csv')

summary_listings.info()

summary_listings.describe()

summary_listings.head()

summary_listings.isnull().sum()/len(summary_listings)

summary_listings['price'].describe()

summary_listings['price'].hist()

summary_listings['neighbourhood'].unique()

summary_listings['neighbourhood'].hist(bins =100, range=(0, 35), figsize=(40,16))

summary_listings['neighbourhood'].value_counts()

summary_listings['neighbourhood'].value_counts().index[0]

summary_listings.groupby('neighbourhood')['price'].describe()

import folium
from folium.plugins import HeatMap
m=folium.Map([40.7128,-74.0060],zoom_start=11)
HeatMap(summary_listings[['latitude','longitude']],radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)
display(m)

# Display the location of the 300 most expensive listings with price and room type
import branca
import branca.colormap as cmp
Long=-86.5
Lat=36
private_room_data = summary_listings.sort_values(by=['price'], ascending=False).head(300)
prd = private_room_data
linear = cmp.LinearColormap(
    ['white', 'black'],
    vmin=np.log(prd['price'].min()), vmax=np.log(prd['price'].max())
)
#'Private room', 'Entire home/apt', 'Shared room', 'Hotel room'
color_dict = {
    'Private room': 'blue', 'Entire home/apt': 'red', 'Shared room': 'green', 'Hotel room':'purple'
    
}

mapdf1=folium.Map([Lat,Long],zoom_start=10)

for lat, lon, price, room_type, name in zip(prd.latitude, prd.longitude, prd.price, prd.room_type, prd.name):
    folium.Marker(location=[lat,lon], icon=folium.Icon(color=color_dict[room_type], icon_color=linear(np.log(price)), icon='home'), popup="%s\t$%d"%(name, price)).add_to(mapdf1)
mapdf1

summary_listings['room_type'].unique()

#See what types of houses are most popular
summary_listings['room_type'].hist(bins = 10, range=(0, 10), figsize=(20,16))

summary_listings.room_type.describe()

summary_listings.groupby('room_type')['price'].describe()

summary_listings[['host_id','name']].groupby('host_id').count().sort_values(by='name', ascending=False)

summary_listings['minimum_nights'].describe()

summary_listings['minimum_nights'].hist(bins = 100, range=(0, 50), figsize=(20,16))

reviews = pd.read_csv('reviews.csv')
reviews.info()

reviews.head()

reviews['date'] = pd.to_datetime(reviews['date'],format='%Y/%m/%d')

reviews['weekday'] = reviews['date'].dt.weekday
reviews['month'] = reviews['date'].dt.month
reviews['year'] = reviews['date'].dt.year
reviews['day'] = reviews['date'].dt.day

n_reviews_year = reviews.groupby('year').size();
sns.barplot(x = n_reviews_year.index, y = n_reviews_year.values);
plt.show();

n_reviews_month = reviews.groupby('month').size();
sns.barplot(x = n_reviews_month.index, y = n_reviews_month.values);
plt.show();

n_reviews_week = reviews.groupby('weekday').size();
sns.barplot(x = n_reviews_week.index, y = n_reviews_week.values);
plt.show();

year_month_reviews =reviews.groupby(['year', 'month']).size().unstack('month').fillna(0);
# Plot (month-comment) line chart by month
fig, ax = plt.subplots(figsize=(20,10));
for index in year_month_reviews.index:
    series = year_month_reviews.loc[index];
    sns.lineplot(x = series.index, y = series.values, ax = ax);
ax.legend(labels = year_month_reviews.index);
ax.grid();
# Display all months on the horizontal axis
_ = ax.set_xticks(list(range(1,13)))
plt.show();

reviews.info()

ser=reviews.groupby(['year', 'month']).agg(n_reviews=("id", lambda x: x.nunique())).unstack('month').fillna(0)

ser

ser1 = ser.copy()

ser1.index

ser2 = ser1.stack()  # stack的返回对象df1是一个二级索引Series对象
ser3 = ser2.reset_index() # 通过reset_index函数将Series对象的二级索引均转化为DataFrame对象的列值
ser3.columns = ['year','month','n_reviews']

ser3

ser4 = ser3.copy(deep=True)

ser4['date'] = ser4['year'].map(str)+"-"+ser4['month'].map(str)
ser4['date'] = pd.to_datetime(ser4['date'])

ser4.info()

n_reviews_ts = pd.Series(ser4.n_reviews.values,
                         index=ser4.date,
                         name='n_reviews')

ax = n_reviews_ts.plot(figsize=(40,8))
ax.set_xlabel('Time')
ax.set_ylabel('NO of reviews')
plt.show()

ser5 = ser4.drop(columns=['year', 'month'])

ser5.info()

ser5.index = pd.period_range(start='2009-01', end='2022-12', freq='M')

ser5.index = ser5.index.to_timestamp()

from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates

decomposition = seasonal_decompose(ser5['n_reviews'], model='additive') 
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)

plt.rc('figure',figsize=(12,6))
plt.rc('font',size=15)

fig, ax = plt.subplots()
x = decomposition.resid.index
y = decomposition.resid.values
ax.plot_date(x, y, color='black',linestyle='--')

ax.annotate('Anomaly', (mdates.date2num(x[135]), y[135]), xytext=(30, 20), 
           textcoords='offset points', color='red',arrowprops=dict(facecolor='red',arrowstyle='fancy'))

fig.autofmt_xdate()
plt.show()

ser5

ser6 = ser5.copy()

ser6

ser6.to_csv(r'/content/month_review.csv')

calendar = pd.read_csv('calendar.csv')

calendar.head()

#Remove the $ sign from the price
calendar['price'] = calendar['price'].replace(r"[$,]","",regex=True).astype(np.float32)
calendar['adjusted_price'] = calendar['adjusted_price'].replace(r"[$,]","",regex = True).astype(np.float32)
calendar.head()

calendar['date'] = pd.to_datetime(calendar['date'],format='%Y/%m/%d')

#Add week and month
calendar['weekday'] = calendar['date'].dt.weekday
calendar['month'] = calendar['date'].dt.month
calendar['year'] = calendar['date'].dt.year

#Plot a histogram by grouping prices by month and averaging them:
month_price = calendar.groupby('month')['price'].mean()
sns.barplot(month_price.index,month_price.values)

weekday_price = calendar.groupby('weekday')['price'].mean()
sns.barplot(weekday_price.index, weekday_price.values)

summary_listings.info()

summary_listings1 = summary_listings.copy()

summary_listings1.rename(columns={'id': 'mergeid'}, inplace=True)

reviews1 = reviews.copy()

reviews1.rename(columns={'listing_id': 'mergeid'}, inplace=True)

result = pd.merge(summary_listings1, reviews1, how='left', on=['mergeid'])

result

result['neighbourhood'].hist(bins =100, range=(0, 35), figsize=(40,16))