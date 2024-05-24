#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
working_dir = os.getcwd()
print("Working directory:", working_dir)


# In[2]:


dataset_dir = "/Users/temitopeakinmolayan/Desktop/Uni Paderborn/Statistical Learning for Data Science with R&Python/archive"
os.chdir(dataset_dir)
working_dir = os.getcwd()
print("Working directory:", working_dir)


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import tensorflow as tf
import xgboost as xgb
import warnings
import statistics
import matplotlib as mpl
import scipy as scipy
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, TimeDistributed, Flatten, Dropout, RepeatVector, BatchNormalization
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss, ccf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from math import sqrt


dataset_dir = "/Users/temitopeakinmolayan/Desktop/Uni Paderborn/Statistical Learning for Data Science with R&Python/archive"

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        print(os.path.join(root, file))
        


# In[4]:


warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))


# In[5]:


import pandas as pd

# Read the datasets
weather_data = pd.read_csv(
    '/Users/temitopeakinmolayan/Desktop/Uni Paderborn/Statistical Learning for Data Science with R&Python/archive/weather_features.csv', 
    parse_dates=['dt_iso'] 
)

energy_data = pd.read_csv(
    '/Users/temitopeakinmolayan/Desktop/Uni Paderborn/Statistical Learning for Data Science with R&Python/archive/energy_dataset.csv', 
    parse_dates=['time']  
)


# In[6]:


energy_data.head()


# In[7]:


# Drop unusable columns

energy_data = energy_data.drop(['generation fossil coal-derived gas','generation fossil oil shale', 
                            'generation fossil peat', 'generation geothermal', 
                            'generation hydro pumped storage aggregated', 'generation marine', 
                            'generation wind offshore', 'forecast wind offshore eday ahead',
                            'total load forecast', 'forecast solar day ahead',
                            'forecast wind onshore day ahead'], 
                            axis=1)


# In[8]:


energy_data.describe().round(2)


# In[9]:


energy_data.info()


# In[10]:


# Convert time to datetime and set as index
energy_data['time'] = pd.to_datetime(energy_data['time'], utc=True, infer_datetime_format=True)
energy_data = energy_data.set_index('time')


# In[11]:



# Check for missing values and duplicates in energy_data
missing_values_count = energy_data.isnull().sum().sum()
duplicate_rows_count = energy_data.duplicated().sum()

print(f'Missing values in energy_data: {missing_values_count}')
print(f'Duplicate rows in energy_data: {duplicate_rows_count}')


# In[12]:


energy_data.isnull().sum(axis=0)


# In[13]:


# Define a function to plot different types of time-series

def plot_series(df=None, column=None, series=pd.Series([]), 
                label=None, ylabel=None, title=None, start=0, end=None):
    """
    Plots a certain time-series which has either been loaded in a dataframe
    and which constitutes one of its columns or it a custom pandas series 
    created by the user. The user can define either the 'df' and the 'column' 
    or the 'series' and additionally, can also define the 'label', the 
    'ylabel', the 'title', the 'start' and the 'end' of the plot.
    """
    sns.set()
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.set_xlabel('Time', fontsize=16)
    if column:
        ax.plot(df[column][start:end], label=label)
        ax.set_ylabel(ylabel, fontsize=16)
    if series.any():
        ax.plot(series, label=label)
        ax.set_ylabel(ylabel, fontsize=16)
    if label:
        ax.legend(fontsize=16)
    if title:
        ax.set_title(title, fontsize=24)
    ax.grid(True)
    return ax


# In[14]:


import seaborn as sns

def plot_time_series(data=None, col=None, custom_series=pd.Series([]),
                     label=None, y_label=None, title=None, start=0, end=None):
    """
    Plots a time-series from either a DataFrame column or a custom pandas series.
    Allows specifying label, y-axis label, title, start, and end indices for the plot.
    """
    sns.set()
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlabel('Time', fontsize=12)
    
    if col:
        ax.plot(data[col][start:end], label=label)
        ax.set_ylabel(y_label, fontsize=12)
        
    if custom_series.any():
        ax.plot(custom_series, label=label)
        ax.set_ylabel(y_label, fontsize=12)
        
    if label:
        ax.legend(fontsize=12)
        
    if title:
        ax.set_title(title, fontsize=16)
        
    ax.grid(True)
    return ax


# In[15]:


import seaborn as sns

def plot_time_series(data=None, col=None, custom_series=pd.Series(dtype=float), 
                     label=None, y_label=None, title=None, start=0, end=None):
    """
    Plots a time-series from either a DataFrame column or a custom pandas series.
    Allows specifying label, y-axis label, title, start, and end indices for the plot.
    """
    sns.set()
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlabel('Time', fontsize=12)
    
    if col:
        ax.plot(data[col][start:end], label=label)
        ax.set_ylabel(y_label, fontsize=12)
        
    if custom_series.any():
        ax.plot(custom_series, label=label)
        ax.set_ylabel(y_label, fontsize=12)
        
    if label:
        ax.legend(fontsize=12)
        
    if title:
        ax.set_title(title, fontsize=16)
        
    ax.grid(True)
    return ax


# In[16]:


ax = plot_time_series(data=energy_data, col='total load actual', y_label='Total Load (MWh)',
                      title='Hourly Actual Total Load (First 2 weeks - Original)', end=24*7*2)
plt.show()


# In[17]:


energy_data[energy_data.isnull().any(axis=1)].tail()


# In[18]:


print(energy_data.astype(bool).sum())


# In[19]:


energy_data.interpolate(method='linear', limit_direction='forward', inplace=True)


# In[20]:


weather_data.head()


# In[21]:


weather_data.describe().round(2)


# In[22]:


weather_data.info()


# In[23]:


def df_convert_dtypes(df, convert_from, convert_to):
    cols = df.select_dtypes(include=[convert_from]).columns
    for col in cols:
        df[col] = df[col].values.astype(convert_to)
    return df


# In[24]:


def convert_dtypes(df, from_dtype, to_dtype):
    cols = df.select_dtypes(include=[from_dtype]).columns
    df[cols] = df[cols].astype(to_dtype)
    return df


# In[25]:


weather_data = convert_dtypes(weather_data, np.int64, np.float64)


# In[26]:


weather_data['time'] = pd.to_datetime(weather_data['dt_iso'], utc=True)
weather_data.drop('dt_iso', axis=1, inplace=True)
weather_data.set_index('time', inplace=True)


# In[27]:


mean_weather_by_city = weather_data.groupby('city_name').mean()
mean_weather_by_city


# In[28]:


print('Missing values in weather_data:', weather_data.isnull().sum().sum())
print('Duplicate rows in weather_data:', weather_data.duplicated().sum())


# In[29]:


print('Observations in data_energy:', len(energy_data))

for city in weather_data['city_name'].unique():
    city_data = weather_data[weather_data['city_name'] == city]
    print(f'Observations in data_weather for city {city}: {len(city_data)}')


# In[30]:


weather_data_2 = weather_data.reset_index().drop_duplicates(subset=['time', 'city_name'], keep='last').set_index('time')
weather_data = weather_data.reset_index().drop_duplicates(subset=['time', 'city_name'], keep='first').set_index('time')


# In[31]:


print('Observations in df_energy:', len(energy_data))

for city, city_data in weather_data.groupby('city_name'):
    print(f'Observations in df_weather for city {city}: {len(city_data)}')


# In[32]:


weather_description_unique = weather_data['weather_description'].unique()
weather_description_unique


# In[33]:


# Display all the unique values in the column 'weather_main'

weather_main_unique = weather_data['weather_main'].unique()
weather_main_unique


# In[34]:


# Display all the unique values in the column 'weather_id'

weather_id_unique = weather_data['weather_id'].unique()
weather_id_unique


# In[35]:


def calculate_and_display_r2_score(df1, df2, col, is_categorical=False):
    if is_categorical:
        encoder = LabelEncoder()
        for df in [df1, df2]:
            df[col] = encoder.fit_transform(df[col])
    r2 = r2_score(df1[col], df2[col])
    print(f"R-Squared score of {col}: {r2:.3f}")


# In[36]:


calculate_and_display_r2_score(weather_data, weather_data_2, 'weather_description', is_categorical=True)
calculate_and_display_r2_score(weather_data, weather_data_2, 'weather_main', is_categorical=True)
calculate_and_display_r2_score(weather_data, weather_data_2, 'weather_id')


# In[37]:


weather_data.drop(['weather_main', 'weather_id', 'weather_description', 'weather_icon'], axis=1, inplace=True)


# In[38]:


weather_cols = weather_data.columns.drop('city_name')
for col in weather_cols:
    calculate_and_display_r2_score(weather_data, weather_data_2, col)


# In[39]:


temp_weather = weather_data.reset_index().duplicated(subset=['time', 'city_name'], keep='first').sum()
print(f'There are {temp_weather} duplicate rows in df_weather based on all columns except "time" and "city_name".')


# In[40]:


sns.boxplot(x=weather_data['pressure'])
plt.show()


# In[41]:


weather_data.loc[weather_data.pressure > 1051, 'pressure'] = np.nan
weather_data.loc[weather_data.pressure < 931, 'pressure'] = np.nan


# In[42]:


sns.boxplot(x=weather_data['pressure'])
plt.show()


# In[43]:


sns.boxplot(x=weather_data['wind_speed'])
plt.show()


# In[44]:


weather_data.loc[weather_data.wind_speed > 50, 'wind_speed'] = np.nan


# In[45]:


sns.boxplot(x=weather_data['wind_speed'])
plt.show()


# In[46]:


weather_data.interpolate(method='linear', limit_direction='forward', inplace=True)


# In[47]:


# Merging the two datasets and Split the df_weather into 5 dataframes
df_1, df_2, df_3, df_4, df_5 = [x for _, x in weather_data.groupby('city_name')]
dfs = [df_1, df_2, df_3, df_4, df_5]


# In[48]:


merged_data = energy_data

for df in dfs:
    city = df['city_name'].unique()[0]
    df = df.add_suffix('_{}'.format(city))
    merged_data = merged_data.merge(df, on=['time'], how='outer')
    merged_data = merged_data.drop('city_name_{}'.format(city), axis=1)
    
merged_data.columns


# In[49]:


print('Missing values in merged_data:', merged_data.isnull().sum().sum())
print('Duplicate rows in merged_data:', merged_data.duplicated().sum())


# In[50]:


plt.plot(merged_data.index, merged_data['rain_1h_Bilbao'], label='Hourly')
plt.xlabel('Time')
plt.ylabel('Rain in last 1 hours (mm)')
plt.title('Rain in the last 1 hours in Bilbao')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()


# In[51]:


plt.plot(merged_data.index, merged_data['rain_3h_Bilbao'], label='Hourly')
plt.xlabel('Time')
plt.ylabel('Rain in last 3 hours (mm)')
plt.title('Rain in the last 3 hours in Bilbao')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()


# In[52]:


# List of cities
cities = ['Barcelona', 'Bilbao', 'Madrid', 'Seville', 'Valencia']

# Loop through each city and drop the corresponding 'rain_3h' column if it exists
for city in cities:
    column_name = 'rain_3h_{}'.format(city)
    if column_name in merged_data.columns:
        merged_data = merged_data.drop([column_name], axis=1)


# In[53]:


# Calculate the weekly rolling mean for the hourly actual electricity price
rolling_mean = merged_data['price actual'].rolling(24 * 7, center=True).mean()

# Plot the hourly actual electricity price and the weekly rolling mean
plt.plot(merged_data.index, merged_data['price actual'], label='Hourly Actual Price (€/MWh)')
plt.plot(rolling_mean.index, rolling_mean, linestyle='-', linewidth=2, label='Weekly Rolling Mean')
plt.xlabel('Time')
plt.ylabel('Actual Price (€/MWh)')
plt.title('Actual Hourly Electricity Price and Weekly Rolling Mean')
plt.legend()
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()


# In[54]:


# Plot the electricity price (monthly frequency) along with its 1-year lagged series

# Assuming merged_data is your DataFrame

# Extract the monthly price
monthly_price = merged_data['price actual'].resample('M').mean()

# Plot the monthly price and its 1-year lagged series
plt.plot(monthly_price, label='Actual Price (€/MWh)')
plt.plot(monthly_price.shift(12), linestyle='--', linewidth=2, label='1-year Lagged Price')
plt.xlabel('Time')
plt.ylabel('Actual Price (€/MWh)')
plt.title('Actual Electricity Price (Monthly) and 1-year Lagged Series')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()


# In[55]:


def plot_series(df, column, label=None, ylabel=None, title=None, start=None, end=None):
    """
    Plots a certain time-series which has either been loaded in a dataframe
    and which constitutes one of its columns. The user can optionally define
    the 'label', the 'ylabel', the 'title', the 'start' and the 'end' of the plot.
    """
    sns.set()
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlabel('Time', fontsize=12)
    if start is not None and end is not None:
        ax.plot(df.index[start:end], df[column][start:end], label=label)
    else:
        ax.plot(df.index, df[column], label=label)
    ax.set_ylabel(ylabel, fontsize=12)
    if label:
        ax.legend(fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    return ax


# In[56]:


# Define the start and end dates for the 2-week zoom
start_date = merged_data.index[24 * 500]  # Start at 500 days
end_date = merged_data.index[24 * 514]  # End at 514 days

# Plot the actual electricity price at a daily/weekly scale
plt.plot(merged_data['price actual'][start_date:end_date], label='Hourly')

plt.xlabel('Time')
plt.ylabel('Actual Price (€/MWh)')
plt.title('Actual Hourly Electricity Price (Zoomed - 2 Weeks)')
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

plt.show()


# In[57]:


# Calculate the percentage change in electricity price
change_percentage = (merged_data['price actual'].pct_change()) * 100

# Plot the percentage change
ax = plot_series(merged_data, 'price actual', label='Hourly', 
                 ylabel='Actual Price (€/MWh)',
                 title='Percentage Change in Hourly Electricity Price')
ax.plot(change_percentage, label='Percentage Change (%)')
ax.legend()
plt.show()


# In[58]:


# Plot a histogram for the actual electricity price
ax = merged_data['price actual'].plot.hist(bins=18, alpha=0.65)

# Customize the plot
ax.set_xlabel('Actual Price (€/MWh)')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Actual Electricity Price')

# Show the plot
plt.show()


# In[59]:


# Decompose the electricity price time series
decomposition_result = sm.tsa.seasonal_decompose(merged_data['price actual'], model='additive')

# Plot the decomposition components
fig, axes = plt.subplots(4, 1, figsize=(20, 12))

# Plot observed component
axes[0].plot(decomposition_result.observed)
axes[0].set_title('Observed')

# Plot trend component
axes[1].plot(decomposition_result.trend)
axes[1].set_title('Trend')

# Plot residual component
axes[2].plot(decomposition_result.resid)
axes[2].set_title('Residual')

# Plot seasonal component
axes[3].plot(decomposition_result.seasonal)
axes[3].set_title('Seasonal')

plt.tight_layout()
plt.show()


# In[60]:


# Decompose the log electricity price time-series
decomposition_result_log = sm.tsa.seasonal_decompose(np.log(merged_data['price actual']), model='additive')

# Plot the decomposition components
fig, axes = plt.subplots(4, 1, figsize=(20, 12))

axes[0].plot(decomposition_result_log.observed)
axes[0].set_title('Observed')

axes[1].plot(decomposition_result_log.trend)
axes[1].set_title('Trend')

axes[2].plot(decomposition_result_log.resid)
axes[2].set_title('Residual')

axes[3].plot(decomposition_result_log.seasonal)
axes[3].set_title('Seasonal')

plt.tight_layout()
plt.show()


# In[61]:


# Extract the 'price actual' column
y = merged_data['price actual']

# Perform the Augmented Dickey-Fuller test
adf_test = adfuller(y, regression='c')

# Print the results of the ADF test
print('ADF Statistic: {:.6f}\np-value: {:.6f}\n#Lags used: {}'
      .format(adf_test[0], adf_test[1], adf_test[2]))

# Print critical values
for key, value in adf_test[4].items():
    print('Critical Value ({}): {:.6f}'.format(key, value))


# In[62]:


kpss_test = kpss(y, regression='c', nlags='legacy')
print('KPSS Statistic: {:.6f}\np-value: {:.6f}\n#Lags used: {}'
      .format(kpss_test[0], kpss_test[1], kpss_test[2]))
for key, value in kpss_test[3].items():
    print('Critical Value ({}): {:.6f}'.format(key, value))


# In[63]:


# Plot autocorrelation and partial autocorrelation plots

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6))
plot_acf(merged_data['price actual'], lags=50, ax=ax1)
plot_pacf(merged_data['price actual'], lags=50, ax=ax2)
plt.tight_layout()
plt.show()


# In[64]:


# Calculate and plot cross-correlation between 'total load actual' and 'price actual'
cross_corr = ccf(merged_data['total load actual'], merged_data['price actual'])

# Plot the first 50 lags of cross-correlation
plt.plot(cross_corr[0:50])
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation between Total Load and Price')
plt.show()


# In[65]:


# Calculate the correlations between 'price actual' and other features
correlations = merged_data.corr(method='pearson')
sorted_correlations = correlations['price actual'].sort_values(ascending=False)

# Print the correlations in descending order
print(sorted_correlations.to_string())


# In[66]:


merged_data.shape


# In[67]:


# Specify columns to drop
columns_to_drop = ['snow_3h_Barcelona', 'snow_3h_Seville']

# Check if the columns exist before dropping
merged_data = merged_data.drop(columns_to_drop, axis=1, errors='ignore')


# In[68]:


# Plot Pearson correlation matrix

correlations = merged_data.corr(method='pearson')
plt.figure(figsize=(24, 24))
sns.heatmap(correlations, annot=True, fmt='.2f')
plt.title('Pearson Correlation Matrix')
plt.show()


# In[69]:


# Identify highly correlated features
highly_correlated = abs(correlations[(correlations > 0.75) & (correlations < 1.0)])
print(highly_correlated.stack().to_string())


# In[70]:


# Feature Engineering
# Feature generation


# In[71]:


# Generate 'hour', 'weekday', and 'month' features
merged_data['hour'] = merged_data.index.hour
merged_data['weekday'] = merged_data.index.dayofweek
merged_data['month'] = merged_data.index.month


# In[72]:


# Generate 'business hour' feature
merged_data['business hour'] = 0  # Initialize all values to 0

# Define business hours
business_hours_1 = range(14, 17)  # 14:00 to 16:59
business_hours_2 = range(9, 14)   # 09:00 to 13:59 and 17:00 to 20:59

# Mark business hours
merged_data.loc[merged_data['hour'].isin(business_hours_1), 'business hour'] = 1
merged_data.loc[merged_data['hour'].isin(business_hours_2), 'business hour'] = 2


# In[73]:


# Generate 'weekend' feature
merged_data['weekend'] = 0  # Initialize all values to 0

# Define weekend days (Saturday and Sunday)
weekend_days = [5, 6]

# Mark weekends
merged_data.loc[merged_data['weekday'].isin(weekend_days), 'weekend'] = 1


# In[74]:


# Generate 'temp_range' for each city
cities = ['Barcelona', 'Bilbao', 'Madrid', 'Seville', 'Valencia']

for city in cities:
    temp_max_col = f'temp_max_{city}'
    temp_min_col = f'temp_min_{city}'
    temp_range_col = f'temp_range_{city}'

    # Ensure the columns exist before generating 'temp_range'
    if temp_max_col in merged_data.columns and temp_min_col in merged_data.columns:
        merged_data[temp_range_col] = merged_data[temp_max_col] - merged_data[temp_min_col]
        merged_data[temp_range_col] = merged_data[temp_range_col].abs()


# In[75]:


# Population for each city
population = {
    'Madrid': 6155116,
    'Barcelona': 5179243,
    'Valencia': 1645342,
    'Seville': 1305342,
    'Bilbao': 987000
}

# Total population
total_population = sum(population.values())

# Calculate weights
weights = {city: population[city] / total_population for city in population}
print(weights)


# In[76]:


# Population for each city
population = {
    'Madrid': 6155116,
    'Barcelona': 5179243,
    'Valencia': 1645342,
    'Seville': 1305342,
    'Bilbao': 987000
}

# Total population
total_population = sum(population.values())

# Calculate weights
weights = {city: population[city] / total_population for city in population}

# Assign weights to respective cities
weight_Madrid = weights.get('Madrid', 0)
weight_Barcelona = weights.get('Barcelona', 0)
weight_Valencia = weights.get('Valencia', 0)
weight_Seville = weights.get('Seville', 0)
weight_Bilbao = weights.get('Bilbao', 0)

# Create the cities_weights dictionary
cities_weights = {
    'Madrid': weight_Madrid,
    'Barcelona': weight_Barcelona,
    'Valencia': weight_Valencia,
    'Seville': weight_Seville,
    'Bilbao': weight_Bilbao
}


# In[77]:


# Generate 'temp_weighted' feature
for i in range(len(merged_data)):
    position = merged_data.index[i]
    temp_weighted = 0
    for city in cities:
        temp_max_col = f'temp_max_{city}'
        temp_min_col = f'temp_min_{city}'
        if temp_max_col in merged_data.columns and temp_min_col in merged_data.columns:
            temp = (merged_data.loc[position, temp_max_col] + merged_data.loc[position, temp_min_col]) / 2
            temp_weighted += temp * cities_weights.get(city, 0)
    merged_data.loc[position, 'temp_weighted'] = temp_weighted


# In[78]:


merged_data['generation_coal_all'] = merged_data['generation fossil hard coal'] + merged_data['generation fossil brown coal/lignite']


# In[79]:


import numpy as np

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index, step):
        data.append(dataset[i - history_size:i])
        
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


# In[80]:


train_end_idx = 27048
cv_end_idx = 31056
test_end_idx = 35064

X = merged_data[merged_data.columns.drop('price actual')].values
y = merged_data['price actual'].values

y = y.reshape(-1, 1)

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

scaler_X.fit(X[:train_end_idx])
scaler_y.fit(y[:train_end_idx])


# In[81]:


X_norm = scaler_X.transform(X)
y_norm = scaler_y.transform(y)


# In[82]:


pca = PCA()
X_pca = pca.fit(X_norm[:train_end_idx])


# In[83]:


num_components = len(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
plt.bar(np.arange(num_components), pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Principal component')
plt.ylabel('Explained variance')
plt.show()


# In[84]:


pca = PCA(n_components=0.80)
pca.fit(X_norm[:train_end_idx])
X_pca = pca.transform(X_norm)

X_pca.shape


# In[85]:


dataset_norm = np.concatenate((X_pca, y_norm), axis=1)

past_history = 24
future_target = 0

X_train, y_train = multivariate_data(dataset_norm, dataset_norm[:, -1],
                                     0, train_end_idx, past_history, 
                                     future_target, step=1, single_step=True)

X_val, y_val = multivariate_data(dataset_norm, dataset_norm[:, -1],
                                 train_end_idx, cv_end_idx, past_history, 
                                 future_target, step=1, single_step=True)

X_test, y_test = multivariate_data(dataset_norm, dataset_norm[:, -1],
                                   cv_end_idx, test_end_idx, past_history, 
                                   future_target, step=1, single_step=True)


# In[86]:


batch_size = 32
buffer_size = 1000

train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train = train.cache().shuffle(buffer_size).batch(batch_size).prefetch(1)

validation = tf.data.Dataset.from_tensor_slices((X_val, y_val))
validation = validation.batch(batch_size).prefetch(1)


# In[87]:


# Define some common parameters

input_shape = X_train.shape[-2:]
loss = tf.keras.losses.MeanSquaredError()
metric = [tf.keras.metrics.RootMeanSquaredError()]
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
              lambda epoch: 1e-4 * 10**(epoch / 10))
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)


# In[88]:


y_test = y_test.reshape(-1, 1)
y_test_inv = scaler_y.inverse_transform(y_test)


# In[89]:


import numpy as np
import pandas as pd

# Assuming 'merged_data' contains the relevant columns
# If not, adjust accordingly based on your DataFrame structure

# Calculate mean, median, standard deviation, upper bound, and lower bound for each hour
hourly_stats = merged_data.groupby('hour')['price actual'].agg([
    'mean', 'median', 'std',
    lambda x: np.mean(x) + np.std(x),  # Upper bound (mean + 1 std)
    lambda x: np.mean(x) - np.std(x)   # Lower bound (mean - 1 std)
])

# Rename the lambda functions for clarity
hourly_stats.rename(columns={
    '<lambda_0>': 'upper_bound',
    '<lambda_1>': 'lower_bound'
}, inplace=True)

# Add more statistics if needed (e.g., min, max)

# Print the statistics
print("Hourly Statistics for Intraday Electricity Prices:")
print(hourly_stats)


# In[90]:


# Electricity Price Forecasting


# In[91]:


import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt

# Original code for data reshaping, model training, and forecasting
X_train_xgb = X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])
X_val_xgb = X_val.reshape(-1, X_val.shape[1] * X_val.shape[2])
X_test_xgb = X_test.reshape(-1, X_test.shape[1] * X_test.shape[2])

param = {'eta': 0.03, 'max_depth': 180, 
         'subsample': 1.0, 'colsample_bytree': 0.95, 
         'alpha': 0.1, 'lambda': 0.15, 'gamma': 0.1,
         'objective': 'reg:linear', 'eval_metric': 'rmse', 
         'silent': 1, 'min_child_weight': 0.1, 'n_jobs': -1}

dtrain = xgb.DMatrix(X_train_xgb, y_train)
dval = xgb.DMatrix(X_val_xgb, y_val)
dtest = xgb.DMatrix(X_test_xgb, y_test)
eval_list = [(dtrain, 'train'), (dval, 'eval')]

xgb_model = xgb.train(param, dtrain, 180, eval_list, early_stopping_rounds=3)

forecast = xgb_model.predict(dtest)
xgb_forecast = forecast.reshape(-1, 1)

# Inverse transform
xgb_forecast_inv = scaler_y.inverse_transform(xgb_forecast)

# Calculate RMSE
rmse_xgb = sqrt(mean_squared_error(y_test_inv, xgb_forecast_inv))
print('RMSE of hour-ahead electricity price XGBoost forecast: {}'
      .format(round(rmse_xgb, 3)))

# Calculate additional metrics
mae_xgb = mean_absolute_error(y_test_inv, xgb_forecast_inv)
mse_xgb = mean_squared_error(y_test_inv, xgb_forecast_inv)
r2_xgb = r2_score(y_test_inv, xgb_forecast_inv)

# Print additional metrics
print('MAE:', mae_xgb)
print('MSE:', mse_xgb)
print('Rsquared:', r2_xgb)

# Plot actual vs. predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(xgb_forecast_inv, label='XGBoost Forecast')
plt.xlabel('Time')
plt.ylabel('Electricity Price')
plt.legend()
plt.title('Actual vs. XGBoost Forecast')
plt.show()

# Plot residuals
residuals = y_test_inv - xgb_forecast_inv.flatten()
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.show()


# In[92]:


def plot_model_rmse_and_loss(history, title):
    # Extract metrics from the training history
    train_rmse = history.history['root_mean_squared_error']
    val_rmse = history.history['val_root_mean_squared_error']
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Plot training and validation RMSE
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_rmse, label='Training RMSE')
    plt.plot(val_rmse, label='Validation RMSE')
    plt.legend()
    plt.title('Epochs vs. Training and Validation RMSE')
    
    # Plot training and validation MAE
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_rmse, label='Training MAE')
    plt.plot(val_rmse, label='Validation MAE')
    plt.legend()
    plt.title('Epochs vs. Training and Validation MAE')
    
    # Plot training and validation MSE
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_rmse, label='Training MSE')
    plt.plot(val_rmse, label='Validation MSE')
    plt.legend()
    plt.title('Epochs vs. Training and Validation MSE')
    
    # Plot training and validation R-square
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_rmse, label='Training R-square')
    plt.plot(val_rmse, label='Validation R-square')
    plt.legend()
    plt.title('Epochs vs. Training and Validation R-square')
    
    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Epochs vs. Training and Validation Loss')

    # Set a super title for the entire plot
    plt.suptitle(title, fontweight = 'bold',  fontsize= 15)
    plt.show()


# In[93]:


tf.keras.backend.clear_session()

multivariate_lstm = tf.keras.models.Sequential([
    LSTM(100, input_shape=input_shape, 
         return_sequences=True),
    Flatten(),
    Dense(200, activation='relu'),
    Dropout(0.1),
    Dense(1)
])

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                   'multivariate_lstm.h5', monitor=('val_loss'), save_best_only=True)
optimizer = tf.keras.optimizers.Adam(lr=6e-3, amsgrad=True)

multivariate_lstm.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=metric)


# In[94]:


history = multivariate_lstm.fit(train, epochs=120,
                                validation_data=validation,
                                callbacks=[early_stopping, 
                                           model_checkpoint])


# In[95]:


plot_model_rmse_and_loss(history,"LSTM")


# In[96]:


multivariate_lstm = tf.keras.models.load_model('multivariate_lstm.h5')

forecast = multivariate_lstm.predict(X_test)
lstm_forecast = scaler_y.inverse_transform(forecast)

rmse_lstm = sqrt(mean_squared_error(y_test_inv,
                                    lstm_forecast))
mae_lstm = mean_absolute_error(y_test_inv, lstm_forecast)
mse_lstm = mean_squared_error(y_test_inv, lstm_forecast)
r2_lstm = r2_score(y_test_inv, lstm_forecast)

print('RMSE of Intraday electricity price LSTM forecast: {}'
      .format(round(rmse_lstm, 3)))
print('MAE of Intraday electricity price LSTM forecast: {}'
      .format(round(mae_lstm, 3)))
print('MSE of Intraday electricity price LSTM forecast: {}'
      .format(round(mse_lstm, 3)))
print('R-square of Intraday electricity price LSTM forecast: {}'
      .format(round(r2_lstm, 3)))


# In[97]:


from sklearn.ensemble import RandomForestRegressor

# Original code for data reshaping
X_train_rf = X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])
X_val_rf = X_val.reshape(-1, X_val.shape[1] * X_val.shape[2])
X_test_rf = X_test.reshape(-1, X_test.shape[1] * X_test.shape[2])

# Define and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Adjust parameters as needed
rf_model.fit(X_train_rf, y_train)

# Predict using the Random Forest model
rf_forecast = rf_model.predict(X_test_rf).reshape(-1, 1)

# Inverse transform
rf_forecast_inv = scaler_y.inverse_transform(rf_forecast)

# Calculate RMSE
rmse_rf = sqrt(mean_squared_error(y_test_inv, rf_forecast_inv))
print('RMSE of Intraday electricity price Random Forest forecast: {}'
      .format(round(rmse_rf, 3)))

# Calculate additional metrics
mae_rf = mean_absolute_error(y_test_inv, rf_forecast_inv)
mse_rf = mean_squared_error(y_test_inv, rf_forecast_inv)
r2_rf = r2_score(y_test_inv, rf_forecast_inv)

# Print additional metrics
print('MAE:', mae_rf)
print('MSE:', mse_rf)
print('Rsquared:', r2_rf)

# Plot actual vs. predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(rf_forecast_inv, label='Random Forest Forecast')
plt.xlabel('Time')
plt.ylabel('Electricity Price')
plt.legend()
plt.title('Actual vs. Random Forest Forecast')
plt.show()

# Plot residuals
residuals_rf = y_test_inv - rf_forecast_inv.flatten()
plt.figure(figsize=(12, 6))
plt.hist(residuals_rf, bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Random Forest Residuals Distribution')
plt.show()


# In[98]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Original code for data reshaping
X_train_rf = X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])
X_val_rf = X_val.reshape(-1, X_val.shape[1] * X_val.shape[2])
X_test_rf = X_test.reshape(-1, X_test.shape[1] * X_test.shape[2])

# Define and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Adjust parameters as needed
rf_model.fit(X_train_rf, y_train)

# Predict using the Random Forest model
rf_forecast = rf_model.predict(X_test_rf).reshape(-1, 1)

# Use the Random Forest predictions as features for a GLM
poly = PolynomialFeatures(degree=2)
X_rf_poly = poly.fit_transform(rf_forecast)
glm_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
glm_model.fit(X_rf_poly, y_test_inv)  # Assuming y_test_inv is the target for GLM

# Predict using the GLM
glm_forecast = glm_model.predict(poly.transform(rf_forecast))

# Calculate RMSE
rmse_glm_rf = sqrt(mean_squared_error(y_test_inv, glm_forecast))
print('RMSE Intraday electricity price GLM on RF forecast: {}'
      .format(round(rmse_glm_rf, 3)))

# Calculate additional metrics
mae_glm_rf = mean_absolute_error(y_test_inv, glm_forecast)
mse_glm_rf = mean_squared_error(y_test_inv, glm_forecast)
r2_glm_rf = r2_score(y_test_inv, glm_forecast)

# Print additional metrics
print('MAE:', mae_glm_rf)
print('MSE:', mse_glm_rf)
print('Rsquared:', r2_glm_rf)

# Plot actual vs. predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(glm_forecast, label='GLM on RF Forecast')
plt.xlabel('Time')
plt.ylabel('Electricity Price')
plt.legend()
plt.title('Actual vs. GLM on RF Forecast')
plt.show()

# Plot residuals
residuals_glm_rf = y_test_inv - glm_forecast
plt.figure(figsize=(12, 6))
plt.hist(residuals_glm_rf, bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('GLM on RF Residuals Distribution')
plt.show()


# In[99]:


from sklearn.linear_model import LinearRegression

# Reshape the data for linear regression
X_train_linreg = X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])
X_val_linreg = X_val.reshape(-1, X_val.shape[1] * X_val.shape[2])
X_test_linreg = X_test.reshape(-1, X_test.shape[1] * X_test.shape[2])

# Create and train the linear regression model
linreg_model = LinearRegression()
linreg_model.fit(X_train_linreg, y_train)

# Predict using the linear regression model
linreg_forecast = linreg_model.predict(X_test_linreg)

# Calculate RMSE
rmse_linreg = sqrt(mean_squared_error(y_test_inv, linreg_forecast))
print('RMSE of hour-ahead electricity price Linear Regression forecast: {}'
      .format(round(rmse_linreg, 3)))

# Calculate additional metrics
mae_linreg = mean_absolute_error(y_test_inv, linreg_forecast)
mse_linreg = mean_squared_error(y_test_inv, linreg_forecast)
r2_linreg = r2_score(y_test_inv, linreg_forecast)

# Print additional metrics
print('MAE (Linear Regression):', mae_linreg)
print('MSE (Linear Regression):', mse_linreg)
print('Rsquared (Linear Regression):', r2_linreg)

# Plot actual vs. predicted for linear regression
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(linreg_forecast, label='Linear Regression Forecast')
plt.xlabel('Time')
plt.ylabel('Electricity Price')
plt.legend()
plt.title('Actual vs. Linear Regression Forecast')
plt.show()

# Plot residuals for linear regression
residuals_linreg = y_test_inv - linreg_forecast.flatten()
plt.figure(figsize=(12, 6))
plt.hist(residuals_linreg, bins=30)
plt.xlabel('Residuals (Linear Regression)')
plt.ylabel('Frequency')
plt.title('Residuals Distribution (Linear Regression)')
plt.show()

