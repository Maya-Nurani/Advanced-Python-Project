# This repository for final project
import pandas as pd

# Student 1 - Laly Datsyuk
# Student 2 - Maya Nurani

# Part A ex. 1 - Reading file content to data frame
try:
    flights_df = pd.read_csv('flights.csv')
except:
    print("Failed to read the file")
    flights_df = []  # In case the file is not read

