import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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

### Data Understanding and Analysis ###
print("There are", flights_df.shape[0], "rows and", flights_df.shape[1], "columns in this file.")
print("Columns names are: ", list(flights_df.columns))
print(flights_df.describe())

months_dict = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
               7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}

print(flights_df['Airline'].unique()) # a check for duplicated airlines


# Amount of flights per airline
flights_df['Airline'].value_counts().plot(kind='bar', rot=45, color='purple')
plt.grid()
plt.title('Number of flights per airline')
plt.show()

duplicated_companies = {'Vistara Premium economy': 'Vistara',
                        'Jet Airways Business': 'Jet Airways', 'Multiple carriers Premium economy': 'Multiple carriers'}


def change_date_format(datestr, expected_format):
    if expected_format == str:
        date_lst = datestr.split('/')
        month = int(date_lst[1])
        month = months_dict[month]

    elif expected_format == int:
        date_lst = datestr.split('/')
        month = int(date_lst[1])
    return month


# a function that combine duplicated airlines
def combine_duplicates(airline):
    if airline in duplicated_companies:
        airline = duplicated_companies[airline]
    return airline


flights_df['Airline'] = flights_df['Airline'].apply(lambda d: combine_duplicates(d))


# Avg and maximum ticket price per airline - how does the airline affect the price?
fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = plt.twinx(ax)
flights_df.groupby(['Airline']).max().plot(kind='bar', color='blue', rot=45, ax=ax, position=1,
                                           width=0.3, label='Maximum Price')
flights_df.groupby(['Airline']).mean().plot(kind='bar', color='orange', rot=45, ax=ax2, position=0,
                                            width=0.3, label='Average Price')
plt.title('Average ticket price compared to the maximum price per airline')
ax.legend('M', loc='upper left')
ax2.legend('A', loc='upper right')
plt.grid()
plt.show()

# Adding a column to our df which contains the date month
flights_df['Date Month'] = flights_df['Date'].apply(lambda date: change_date_format(date, str))
print(flights_df)

# Average ticket price per month - checking for month affect on ticker price
flights_df.groupby(['Date Month']).mean().plot(kind='bar', color='red', rot=45)
plt.title('Average ticket price per month')
plt.grid()
plt.show()


# Does the amount of routes affects the ticket price?
def count_routes(routes_str):
    routes_amount = str(routes_str).split(',')
    return len(routes_amount)


flights_df['Routes Amount'] = flights_df['Route'].apply(lambda z: count_routes(z))
print(flights_df[['Routes Amount', 'Route']])
flights_df.groupby(['Routes Amount']).mean().plot(kind='bar', color='green')
plt.title('Average ticket price per route amount')
plt.show()


def count_stops(total_stops):
    if total_stops == 'non-stop':
         stops = 0
    else:
        stops = total_stops[0]
    return int(stops) + 2


# Compare route <> stops and check if they give us the same data:
print(flights_df['Total_Stops'].unique())
flights_df['Stops Amount'] = flights_df['Total_Stops'].apply(lambda stops: count_stops(stops))

print(flights_df[['Stops Amount', 'Total_Stops']])

print("Is Route and Total_Stops columns are indicate the same data? ", flights_df['Routes Amount'].equals(flights_df['Stops Amount']))


# Unique source and destination flights
unique_flights_df = flights_df.groupby(['Source', 'Destination']).size().reset_index().rename(
    columns={0: 'count_unique'})
unique_labels = []


# A function that creates string labels from the unique flights - for the pie graph
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

# Checking the distribution of different duration flights
flights_df['Duration Minutes'].hist(color="brown")
plt.title('Distribution of flights duration')
plt.show()

### Data Preparation ###

# duplicate df for clustering and classification df
flights_class_df = flights_clustering_df.copy()

# Edit clustering data
flights_clustering_df.drop(flights_clustering_df.columns.difference(['Airline', 'Source', 'Destination', 'Price']), 1,
                           inplace=True)
print(flights_clustering_df.head())

# Data one hot encoding
flights_clustering_df = pd.get_dummies(flights_clustering_df, columns=['Airline', 'Source', 'Destination'])

scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(flights_clustering_df), columns=flights_clustering_df.columns)

# print for verifications:
print("(Clustering df) Columns names are: ", list(flights_clustering_df.columns))
print(flights_clustering_df.head())


# Classification df
unique_airline = flights_class_df['Airline'].unique()
airline_dict = dict(zip(unique_airline, range(1, len(unique_airline) + 1)))
flights_class_df['Airline'].replace(airline_dict, inplace=True)

# Data one hot encoding
flights_class_df = pd.get_dummies(flights_class_df,
                                  columns=['Source', 'Destination', 'Total_Stops', 'Additional_Info'])


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
flights_class_df['Date'] = flights_class_df['Date'].apply(lambda date: change_date_format(date, int))

# Remove time columns
flights_class_df = flights_class_df.drop(columns=['Dep_Time', 'Arrival_Time', 'Duration', 'Route'])

print(flights_class_df.head())
class_columns = list(flights_class_df.columns)
print("(class df) Columns names are: ", class_columns, "and number of columns: ", len(class_columns))
# print(flights_class_df['Route'].nunique())

# Create Training and Test Sets \ Split the data into training/testing sets
# X, y = make_blobs(n_samples=flights_class_df.shape[0], centers=4, random_state=0, cluster_std=1.0)
# plt.scatter(X[:, 0], X[:, 1], c=y)

X = flights_class_df.drop('Price', axis=1)
y = flights_class_df['Price']
feature_scaler = StandardScaler()
# X_norm = feature_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print("Train:", X_train.shape, "Test: ", X_test.shape)

# Random Forest
forest = RandomForestClassifier(n_estimators=25)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

# for part 5
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

print(forest.feature_importances_)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    # - "Feature": flights_class_df.coloumns, class_columns
    "Importance": forest.feature_importances_
})

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#tree = DecisionTreeClassifier()
#tree.fit(X_train, y_train)

### Clustering ###

normalized_dropped = normalized_data.drop(columns=['Price'])

def run_kmeans(df):
    sum_squared = []
    K = range(2, 11)
    for i in K:
        kmeans = KMeans(n_clusters= i, init="k-means++")
        kmeans.fit(df)
        sum_squared.append(kmeans.inertia_)
    return pd.DataFrame(
        {
            "K": K,
            "SSE": sum_squared,
        }
    )


measures = run_kmeans(normalized_dropped)
measures.set_index("K", inplace=True)
measures["SSE"].plot()
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(normalized_dropped)
flights_clustering_df['Cluster'] = kmeans.labels_

clusters = flights_clustering_df.groupby('Cluster')
print(clusters.describe())
print(clusters.max())
print(clusters.min())

sns.pairplot(flights_clustering_df[['Price', 'Cluster']], hue="Cluster", markers=["o", "s", "D"])
plt.show()
print(importance_df.sort_values(by=['Importance']))

# Training: (Decision Tree)
# tree = DecisionTreeClassifier()
# tree.fit(X_train, y_train)
# y_pred = tree.predict(X_test)
