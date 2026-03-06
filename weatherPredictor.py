# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 22:36:34 2024

@author: brian
"""

import requests
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


api_key = '113f0017e013c404ba5475181a0af295'
city = 'Berlin'
url = f'http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric'

response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"API call failed with status code {response.status_code}. Check your API key and city name.")
data = response.json()


# Extract relevant data
weather_data = []
for entry in data['list']:
    weather_entry = {
        'datetime': datetime.datetime.fromtimestamp(entry['dt']),
        'temperature': entry['main']['temp'],
        'humidity': entry['main']['humidity'],
        'weather_description': entry['weather'][0]['description']
    }
    weather_data.append(weather_entry)

# Convert to DataFrame
df = pd.DataFrame(weather_data)
print(df.head())


df['day_of_year'] = df['datetime'].apply(lambda x: x.timetuple().tm_yday)
df['hour'] = df['datetime'].dt.hour

# Drop unnecessary columns for model training
df = df.drop(['datetime', 'weather_description'], axis=1)
print(df.head())

# Define features (X) and target (y)
X = df[['day_of_year', 'hour', 'humidity']]
y = df['temperature']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Temperature')
plt.show()