
import pandas as pd

# Function to read the data file 
def read_data(file_path, **kwargs):
    raw_data = pd.read_csv(file_path  ,**kwargs)
    return raw_data


# Function to drop columns from data
def inspection(dataframe):
    print("Types of the variables we are working with:")
    print(dataframe.dtypes)

    print("Total Samples with missing values:")
    print(dataframe.isnull().any(axis=1).sum())

    print("Total Missing Values per Variable")
    print(dataframe.isnull().sum())


# Function to remove null values
def null_values(df):
    df = df.dropna()
    return df

