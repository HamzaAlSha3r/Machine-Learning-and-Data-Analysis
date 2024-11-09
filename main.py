# Hamza Al Sha3r --> 1211162 --> section : 3
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from geopy.geocoders import Nominatim
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import folium
import geopandas as gpd
import re
pd.options.display.float_format = '{:.2f}'.format
# 1. Documenting missing values
def Document_Missing_Values(DataSet):
    Missing_Counts = DataSet.isnull().sum()
    # To take just True Value grater Than zero
    Missing_Counts = Missing_Counts[Missing_Counts > 0]
    missing_percentage = (Missing_Counts / len(DataSet)) * 100
    Report_Missing = pd.DataFrame({'Missing Values': Missing_Counts, 'Percentage': missing_percentage})
    print("Missing Values Report:\n", Report_Missing)

# 2. Handling missing values with various strategies
def Missing_Value_Strategies(DataSet):
    # apply multiple strategies (e.g., mean/median imputation, dropping rows)
    print("Missing values before applying strategies:")
    print(DataSet.isnull().sum())

    # Drop rows with missing values
    Dropped_Rows = DataSet.dropna()
    print("\nAfter dropping rows:\n", Dropped_Rows.isnull().sum())

    # Fill missing values with column means (numeric columns only)
    mean_imputed = DataSet.copy()
    Numeric_columns = mean_imputed.select_dtypes(include=[np.number]).columns
    mean_imputed[Numeric_columns] = mean_imputed[Numeric_columns].fillna(mean_imputed[Numeric_columns].mean())
    print("\nAfter mean imputation:\n", mean_imputed.isnull().sum())

# 3. Feature encoding for categorical attributes
def Feature_Encoding(DataSet, columns):
    print(f"The Feature Encoding for {columns}")
    # this condition to if the Attribute is string convert as List
    if isinstance(columns, str):
        columns = [columns]

    # Create an instance of the OneHotEncoder and Fit and transform the encoder on the specified columns
    ENC = OneHotEncoder(handle_unknown='ignore')
    ENC.fit(DataSet[columns])
    DataSet_encoded = ENC.transform(DataSet[columns]).toarray()

    # here we create a copy of the original DataFrame to store encoded data then Create a list of column names for the encoded data
    DataSet_ohenc = DataSet.copy()
    encoded_columns = [f'{col}_{val}' for col, values in zip(columns, ENC.categories_) for val in values]
    Encoded_Data = pd.DataFrame(DataSet_encoded, columns=encoded_columns, index=DataSet.index)
    DataSet_ohenc = pd.concat([DataSet_ohenc, Encoded_Data], axis=1)
    DataSet_ohenc.drop(columns, axis=1, inplace=True)
    print(DataSet_ohenc)

# 4. Normalizing numerical features using Min-Max Scaling
def Normalization (DataSet):
    print(" Before Min-Max Normalized Dataset:\n")
    print(DataSet.describe())
    numerical_columns = DataSet.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    # Fit and transform for the numerical columns
    DataSet[numerical_columns] = scaler.fit_transform(DataSet[numerical_columns])
    print("\nAfter Min-Max Normalized Dataset:\n")
    print(DataSet.describe())

# 5. Descriptive statistics for numerical features
def Descriptive_Statistics(DataSet):
    # Calculate mean, median, and standard deviation for each numerical column
    mean_values = DataSet.mean(numeric_only=True)
    median_values = DataSet.median(numeric_only=True)
    std_dev_values = DataSet.std(numeric_only=True)
    print("Mean values:\n", mean_values)
    print("\nMedian values:\n", median_values)
    print("\nStandard Deviation values:\n", std_dev_values)

# 6. Spatial Distribution
def Get_longitude_latitude(DataSet):
    # Adjust the regular expression to capture coordinates from the format "POINT (-longitude latitude)"
    Coordinates = DataSet['Vehicle Location'].str.extract(r'POINT \(([-+]?\d*\.\d+)\s([-+]?\d*\.\d+)\)')
    DataSet['longitude'] = Coordinates[0].astype(float)
    DataSet['latitude'] = Coordinates[1].astype(float)
    return DataSet


def Spatial_Distribution_Map(DataSet, shapefile_path, EvType_col='Electric Vehicle Type'):
    # Get unique values in the 'Electric Vehicle Type' column
    # unique values should be: ['Plug-in Hybrid Electric Vehicle (PHEV)' 'Battery Electric Vehicle (BEV)']
    DataSet = Get_longitude_latitude(DataSet)

    # Create a GeoDataFrame with valid geometry for plotting and load the shapefile for the background map
    gdf = gpd.GeoDataFrame(DataSet, geometry=gpd.points_from_xy(DataSet.longitude, DataSet.latitude))
    world = gpd.read_file(shapefile_path)

    if 'geometry' in world.columns:
        fig, ax = plt.subplots(figsize=(15, 10))
        world.plot(ax=ax, color='lightgrey', edgecolor='black')
        color_map = {
            'Plug-in Hybrid Electric Vehicle (PHEV)': 'red',
            'Battery Electric Vehicle (BEV)': 'blue'
        }
        for ev_type, color in color_map.items():
            ev_data = gdf[gdf[EvType_col] == ev_type]
            ev_data.plot(ax=ax, color=color, markersize=5, label=ev_type)
        plt.title('Spatial Distribution of Electric Vehicles by Type - 1211162')
        plt.legend(title="EV Type")
        plt.show()
    else:
        print("Error: Shapefile lacks 'geometry' column.")

# 7. Model popularity visualization for top models
def Model_Popularity(DataSet, TOP_N=20):
    #'Model' for top 20 car
    model_counts = DataSet['Model'].value_counts().head(TOP_N)
    model_counts.plot(kind='bar', figsize=(12, 6), color='red')
    plt.title('Popularity of Top EV Models - 1211162')
    plt.xlabel('EV Model')
    plt.ylabel('Number of Vehicles')
    plt.xticks(rotation=45, ha='right')
    plt.show()

# 8. Correlation matrix visualization for numerical features
def Correlations(DataSet):
    numeric_features = DataSet.select_dtypes(include='number')
    # Compute the correlation matrix
    correlation_matrix = numeric_features.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numeric Features - 1211162')
    plt.show()

# 9. General exploratory visualizations
def Data_Exploration_Visualizations (DataSet):
    numerical_columns = DataSet.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(14, 8))
    # Histogram of numerical columns
    print("Generating histograms for numerical features...")
    DataSet[numerical_columns].hist(bins=30, figsize=(16, 12), color='red', edgecolor='black')
    plt.suptitle('Histograms of Numerical Features - 1211162')
    plt.show()

    # Scatterplot to explore the relationship between two numerical variables as example we use Base MSRP and Model Year
    print("Generating scatterplot...")
    if 'Base MSRP' in numerical_columns and 'Model Year' in numerical_columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=DataSet['Model Year'], y=DataSet['Base MSRP'], hue=DataSet['Electric Vehicle Type'])
        plt.title('ScatterPlot of Base MSRP vs Model Year - 1211162')
        plt.xlabel('Model Year')
        plt.ylabel('Base MSRP')
        plt.show()

    # Boxplot to explore the distribution of numerical features grouped by a categorical variable
    print("Generating boxplot...")
    if 'Electric Vehicle Type' in DataSet.columns and 'Base MSRP' in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Electric Vehicle Type', y='Base MSRP', data=DataSet)
        plt.title('Boxplot of Base MSRP by Electric Vehicle Type - 1211162')
        plt.show()

# 10. Comparative analysis of EV distributions across cities and counties
def Comparative_Visualization(DataSet):
    # Group by 'City' and 'County' to get the count of EVs in each location we take top 10 city and counties
    city_counts = DataSet['City'].value_counts().head(10)
    county_counts = DataSet['County'].value_counts().head(10)
    # Plotting EV distribution across cities
    plt.figure(figsize=(12, 6))
    city_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of EVs Across Top 10 Cities - 1211162')
    plt.xlabel('City')
    plt.ylabel('Number of EVs')
    plt.xticks(rotation=45)
    plt.show()
    # Plotting EV distribution across counties
    plt.figure(figsize=(12, 6))
    county_counts.plot(kind='bar', color='salmon', edgecolor='black')
    plt.title('Distribution of EVs Across Top 10 Counties - 1211162')
    plt.xlabel('County')
    plt.ylabel('Number of EVs')
    plt.xticks(rotation=45)
    plt.show()

# 11. Temporal analysis of EV adoption and model popularity
def Temporal_Analysis(DataSet):
    DataSet['Model Year'] = DataSet['Model Year'].astype(int)
    # EV Adoption Trend from start time for now
    print("Generating EV Adoption Trend Over Time...")
    yearly_counts = DataSet['Model Year'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o', color='r')
    plt.title('EV Adoption Trend Over Time - 1211162')
    plt.xlabel('Model Year')
    plt.ylabel('Number of EVs')
    plt.xticks(rotation=45)
    plt.show()
    # Model Popularity from start time for now on Top 5 Models of car
    print("Generating Model Popularity Over Time...")
    top_models = DataSet['Model'].value_counts().head(5).index
    top_models_data = DataSet[DataSet['Model'].isin(top_models)]
    # take top models data in DataFrame by two columns: 'Model Year' and 'Model'.
    model_yearly_counts = top_models_data.groupby(['Model Year', 'Model']).size().unstack().fillna(0)
    plt.figure(figsize=(12, 6))
    model_yearly_counts.plot(marker='o', ax=plt.gca())
    plt.title('Top 5 EV Model Popularity Over Time - 1211162')
    plt.xlabel('Model Year')
    plt.ylabel('Number of Vehicles')
    plt.xticks(rotation=45)
    plt.legend(title='EV Model')
    plt.show()

def menu():
    print("Choose an option:")
    print("1. Document Missing Values")
    print("2. Missing Value Strategies")
    print("3. Feature Encoding")
    print("4. Normalize Numerical Features")
    print("5. Descriptive Statistics")
    print("6. Spatial Distribution Map")
    print("7. Model Popularity")
    print("8. Correlations")
    print("9. Data Exploration Visualizations")
    print("10. Comparative Visualization")
    print("11. Temporal Analysis")
    print("0. Exit")

# Call The DataSet
DataSet = pd.read_csv("C:\\Users\\hamza\\Downloads\\Electric_Vehicle_Population_Data.csv")
while True:
    menu()
    choice = input("Enter your choice: ")

    if choice == '1':
        Document_Missing_Values(DataSet)
    elif choice == '2':
        Missing_Value_Strategies(DataSet)
    elif choice == '3':
        column = input("Enter the column for feature encoding: ")
        Feature_Encoding(DataSet, column)
    elif choice == '4':
        Normalization(DataSet)
    elif choice == '5':
        Descriptive_Statistics(DataSet)
    elif choice == '6':
        Spatial_Distribution_Map(DataSet,"C:\\Users\\hamza\\Downloads\\ne_110m_admin_0_countries\\ne_110m_admin_0_countries.shp")
    elif choice == '7':
        Model_Popularity(DataSet)
    elif choice == '8':
        Correlations(DataSet)
    elif choice == '9':
        Data_Exploration_Visualizations(DataSet)
    elif choice == '10':
        Comparative_Visualization(DataSet)
    elif choice == '11':
        Temporal_Analysis(DataSet)
    elif choice == '0':
        print("Exiting program..Done")
        break
    else:
        print("Invalid choice, please try again.")




