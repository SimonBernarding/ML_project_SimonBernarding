# Import of basic packages
import numpy as np
import pandas as pd
import operator
from joblib import dump, load
import sys
import warnings

warnings.filterwarnings('ignore')

# Import of chart packages
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt

# Import for normal distribution
import pylab
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import shapiro

# Import of machine learning metric packages
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix, mean_squared_error, r2_score, accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, fbeta_score
from sklearn import metrics
from scipy.stats import randint, uniform, loguniform

# Import of preprossesor packages
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelBinarizer, PolynomialFeatures

# Import of machine learning packages
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, RandomForestClassifier, RandomForestRegressor, VotingClassifier, StackingRegressor, StackingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Set random seed 
RSEED = 0

# Load csv
df = pd.read_csv('data/train.csv')
df.head(3)

# Clean column names
df.columns = df.columns.str.replace(' ','_')
df.columns = df.columns.str.lower()

# Create feature airplane_model and producer
df['airplane_model'] = df['ac'].str[3:6]
replacement_dict = {
    '31A': 'Airbus',
    '31B': 'Airbus',
    '320': 'Airbus',
    '321': 'Airbus',
    '32A': 'Airbus',
    '332': 'Airbus',
    '343': 'Airbus',
    '345': 'Airbus',
    '733': 'Boeing',
    '734': 'Boeing',
    '736': 'Boeing',
    'AT7': 'ATR',
    'CR9': 'Bombardier'
}
df['producer'] = df['airplane_model']

# Replace values in the 'purpose' column
df['producer'] = df['producer'].replace(replacement_dict)

# Create feature airline_1 and airline_2
df['airline_1'] = df['fltid'].str[0:2]
df['airline_2'] = df['ac'].str[0:2]

# Create time features
# Add columns with weekdays, yyyy, mm, dd, hh:mm:ss

y = '_year'
m = '_month'
wd = '_wd'
M = '_min'

### std ###

date = 'std'

idx = df.columns.get_loc(date)

df[date] = pd.to_datetime(df[date], format='%Y-%m-%d %H:%M:%S')
df.insert(loc=idx+1, column=date+y, value=df[date].dt.strftime('%Y')) # year yyyy
df.insert(loc=idx+2, column=date+m, value=df[date].dt.strftime('%#m')) # month m
df.insert(loc=idx+3, column=date+wd, value=df[date].dt.strftime('%w')) # weekday wd
h = df[date].dt.strftime('%#H').astype(int) # hours
minutes = df[date].dt.strftime('%#M').astype(int) # minutes
# calcualte time in just minutes
t = 60*h + minutes
df.insert(loc=idx+4, column=date+M, value=t) # minutes

### sta ###

date = 'sta'

idx = df.columns.get_loc(date)

df[date] = pd.to_datetime(df[date], format='%Y-%m-%d %H.%M.%S')
df.insert(loc=idx+1, column=date+y, value=df[date].dt.strftime('%Y')) # year yyyy
df.insert(loc=idx+2, column=date+m, value=df[date].dt.strftime('%#m')) # month m
df.insert(loc=idx+3, column=date+wd, value=df[date].dt.strftime('%w')) # weekday wd
h = df[date].dt.strftime('%#H').astype(int) # hours
minutes = df[date].dt.strftime('%#M').astype(int)
# calcualte time in just minutes
t = 60*h + minutes
df.insert(loc=idx+4, column=date+M, value=t) # minutes

### datop ###

date = 'datop'

idx = df.columns.get_loc(date)

df[date] = pd.to_datetime(df[date], format='%Y-%m-%d')
df.insert(loc=idx+1, column=date+y, value=df[date].dt.strftime('%Y')) # year yyyy
df.insert(loc=idx+2, column=date+m, value=df[date].dt.strftime('%#m')) # month m
df.insert(loc=idx+3, column=date+wd, value=df[date].dt.strftime('%w')) # weekday wd

# convert new columns as integers
list = ['std_year', 'std_month', 'std_wd', 'sta_year', 'sta_month', 'sta_wd', 'datop_year', 'datop_month', 'datop_wd', 'target']

for date in list:
    df[date] = df[date].astype(int)

# change weekday numbers to EU where day 1 = Monday
list = ['std_wd', 'sta_wd', 'datop_wd']

for date in list:
    df[date][df[date] == 0] = 7 # Sunday

# Load geo data
# Load csv
df_airports = pd.read_csv('data/airports.csv')
df_airports.columns = ['id', 'name', 'city', 'country', 'short', 'rubbish_6', 'latitude', 'longitude', 'rubbish_1', 'rubbish_2', 'rubbish_3', 'rubbish_4', 'type', 'rubbish_5']
df_airports = df_airports.drop(['id', 'name', 'rubbish_1', 'rubbish_2', 'rubbish_3', 'rubbish_4', 'rubbish_5', 'rubbish_6', 'type'], axis=1)
df_airports = df_airports.dropna(subset=['short'])

# Merge datasets
df = df.merge(df_airports, left_on='depstn', right_on='short', how='left', suffixes=('', '_dep'))

# Merge based on arrival station
df = df.merge(df_airports, left_on='arrstn', right_on='short', how='left', suffixes=('', '_arr'))

# Rename columns for clarity
df = df.rename(columns={
    'city': 'city_dep',
    'country': 'country_dep',
    'latitude': 'latitude_dep',
    'longitude': 'longitude_dep'
})
df = df.drop_duplicates(subset='id', keep='first')

# Feature engeneering
df = df.drop(['datop_year', 'datop_month', 'datop_wd', 'datop', 'fltid', 'std', 'sta', 'ac', 'short', 'short_arr', 'city_dep', 'country_dep', 'city_arr', 'country_arr', 'airline_2', 'producer'], axis=1)

# List of columns to encode
columns_to_encode = ['depstn', 'status', 'arrstn', 'airline_1', 'airplane_model'] # reduced by aggressive feature drop

# Create a copy of the original dataframe
df_encoded = df.copy()

# Encode each column separately
for column in columns_to_encode:
    lb = LabelBinarizer()
    encoded = lb.fit_transform(df[column])
    
    # If binary classification, create a single column
    if len(lb.classes_) == 2:
        df_encoded[f'{column}_encoded'] = encoded
    else:
        # For multiclass, create multiple columns
        encoded_df = pd.DataFrame(encoded, columns=[f'{column}_{cls}' for cls in lb.classes_], index=df.index)
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

df_encoded = df_encoded.drop(column, axis=1)

# Now, combine the non-encoded columns from df with the encoded columns from df_encoded
df = pd.concat([df, df_encoded], axis=1)

df = df.drop(['depstn', 'arrstn', 'status', 'airline_1', 'airplane_model'], axis=1) # reduced by aggressive feature drop
duplicate_columns = df.columns[df.columns.duplicated()]
df = df.loc[:, ~df.columns.duplicated()]

#Target engeneering
# Convert target into certain category intervals

def target_interval(row):
    if row['target'] == 0:
        return 1
    elif 0 < row['target'] <= 30:
        return 2
    elif 30 < row['target'] <= 60:
        return 3
    elif 60 < row['target'] <= 120:
        return 4
    elif 120 < row['target'] <= 240:
        return 5   
    else:
        return 6  
    
df['target_cat'] = df.apply(target_interval, axis=1)

# Standardization

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform only the specified columns
columns_to_standardize = ['std_year', 'std_month', 'std_wd', 'std_min', 'sta_year', 'sta_month', 'sta_wd', 'sta_min', 'latitude_dep', 'longitude_dep', 'latitude_arr', 'longitude_arr']  # Replace with your actual column names
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
df = df.drop_duplicates(subset=['id'], keep='first')

# Define features and target variable (target)
X = df.drop(['target', 'id', 'target_cat'], axis=1)
y = df['target_cat']

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RSEED)

# Train model


# Save the model
dump(model, 'models/model.joblib')
