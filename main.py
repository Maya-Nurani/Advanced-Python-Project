# This repository for final project
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Student 1 - Laly Datsyuk
# Student 2 - Maya Nurani

# Part A ex. 1 - Reading file content to data frame
try:
    flights_clustering_df = pd.read_csv('flights.csv')
except:
    print("Failed to read the file")
    flights_clustering_df = []  # In case the file is not read

# Data Understanding
print("There are", flights_clustering_df.shape[0], "rows and", flights_clustering_df.shape[1], "columns in this file.")
print("Columns names are: ", list(flights_clustering_df.columns))
print(flights_clustering_df.describe())

# unique source and destination flights
print(flights_clustering_df['Source'].unique())
print(flights_clustering_df['Destination'].unique())
print(flights_clustering_df.groupby(['Source', 'Destination']).size().reset_index().rename(columns={0: 'count_unique'}))

# Range of price
min_price = flights_clustering_df['price'].min()
max_price = flights_clustering_df['price'].max()

### Data Preparation ###

# Removing null rows
if (flights_clustering_df.isnull().values.any()):
    print("Print columns that contain NaN values", flights_clustering_df.columns[flights_clustering_df.isnull().any()].tolist())
    flights_clustering_df = flights_clustering_df.dropna()
else:
    print('There is no empty values in this dataframe')

# duplicate df for clustering and classification df
flights_class_df = flights_clustering_df.copy()

# Edit clustering data
flights_clustering_df.drop(flights_clustering_df.columns.difference(['Airline','Source','Destination','Price']), 1, inplace=True)
print(flights_clustering_df.head())


# Data one hot encoding
flights_clustering_df = pd.get_dummies(flights_clustering_df, columns=['Airline','Source','Destination'])
# TODO: check if need price column or not

scaler = MinMaxScaler()
normalize_data = pd.DataFrame(scaler.fit_transform(flights_clustering_df), columns=flights_clustering_df.columns)


#print for verifications:
print("Columns names are: ", list(flights_clustering_df.columns))
print(flights_clustering_df.head())