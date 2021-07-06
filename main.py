import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Student 1 - Laly Datsyuk
# Student 2 - Maya Nurani

# Part A ex. 1 - Reading file content to data frame
try:
    flights_df = pd.read_csv('flights.csv')
except:
    print("Failed to read the file")
    flights_df = []  # In case the file is not read

# Removing null rows
if (flights_df.isnull().values.any()):
    print("Print columns that contain NaN values",
          flights_df.columns[flights_df.isnull().any()].tolist())
    flights_df = flights_df.dropna()
else:
    print('There is no empty values in this dataframe')

# Save copy for clustering df
flights_clustering_df = flights_df.copy()

### Data Understanding ###
print("There are", flights_df.shape[0], "rows and", flights_df.shape[1], "columns in this file.")
print("Columns names are: ", list(flights_df.columns))
print(flights_df.describe())

months_dict = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
               7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}


def change_date_format(datestr):
    date_lst = datestr.split('/')
    month = int(date_lst[1])
    month = months_dict[month]
    return month


# Adding a column to our df which contains the date month
flights_df['Date Month'] = flights_df['Date'].apply(lambda y: change_date_format(y))
print(flights_df)

# Avg ticket price per month
flights_df.groupby(['Date Month']).mean().plot(kind='bar', color='red', rot=45)
plt.title('Average ticket price per month')
plt.grid()
plt.show()

# Range of price
min_price = flights_df['Price'].min()
max_price = flights_df['Price'].max()

# TODO: לאחד חברות
# Avg ticket price per company
flights_df.groupby(['Airline']).mean().plot(kind='bar', color='orange', rot=45)
plt.title('Average ticket price per company')
plt.grid()
plt.show()


# Unique Routes
def count_routes(routes_str):
    routes_amount = str(routes_str).split(',')
    return (len(routes_amount))


flights_df['Routes Amount'] = flights_df['Route'].apply(lambda z: count_routes(z))
print(flights_df[['Routes Amount', 'Route']])
flights_df.groupby(['Routes Amount']).mean().plot(kind='bar', color='green')
plt.title('Average ticket price per route amount')
plt.show()

# Unique source and destination flights
unique_flights_df = flights_df.groupby(['Source', 'Destination']).size().reset_index().rename(
    columns={0: 'count_unique'})
unique_labels = []


# A function that creates string labels from the unique flights
def unique_flights_labels():
    strlabel = unique_flights_df['Source'].values + ' To ' + unique_flights_df['Destination'].values
    unique_labels.append(strlabel)
    return


# unique flights pie chart
unique_count = np.asarray(unique_flights_df['count_unique'])
unique_flights_labels()
plt.pie(unique_count, labels=unique_labels[0], autopct='%1.2f%%')
plt.title('Unique flights pie chart')
plt.show()


# Changing duration time format
def time_format(str):
    str = str.replace('h', '')
    str = str.replace('m', '')
    time_lst = str.split()
    time_lst[0] = int(time_lst[0]) * 60
    if len(time_lst) > 1:
        total_time = int(time_lst[0]) + int(time_lst[1])
    else:
        total_time = int(time_lst[0])
    return total_time


flights_df['Duration Minutes'] = flights_df['Duration'].apply(lambda x: time_format(x))
print(flights_df[['Duration', 'Duration Minutes']])

### Data Preparation ###

# duplicate df for clustering and classification df
flights_class_df = flights_clustering_df.copy()

# Edit clustering data
flights_clustering_df.drop(flights_clustering_df.columns.difference(['Airline', 'Source', 'Destination', 'Price']), 1,
                           inplace=True)
print(flights_clustering_df.head())

# Data one hot encoding
flights_clustering_df = pd.get_dummies(flights_clustering_df, columns=['Airline', 'Source', 'Destination'])
# TODO: check if need price column or not

scaler = MinMaxScaler()
normalize_data = pd.DataFrame(scaler.fit_transform(flights_clustering_df), columns=flights_clustering_df.columns)

# print for verifications:
print("Columns names are: ", list(flights_clustering_df.columns))
print(flights_clustering_df.head())

# Classification df
unique_airline = flights_class_df['Airline'].unique()
airline_dict = dict(zip(unique_airline, range(1, len(unique_airline) + 1)))
flights_class_df['Airline'].replace(airline_dict, inplace=True)

# Data one hot encoding
flights_class_df = pd.get_dummies(flights_class_df,
                                  columns=['Source', 'Destination', 'Route', 'Duration', 'Total_Stops'])


def convertPrice(price):
    if price < 7000:
        price_rank = 1
    elif price >= 7000 and price <= 14000:
        price_rank = 2
    elif price > 14000:
        price_rank = 3
    return price_rank


flights_class_df['Price'] = flights_df['Price'].apply((lambda price: convertPrice(price)))

# Change dates to number of month
flights_class_df['Date'] = flights_class_df['Date'].apply(lambda date: change_date_format(date))

# Remove time columns
flights_class_df.drop(columns=['Dep_Time', 'Arrival_Time', 'Duration'])

print(flights_class_df.head())


