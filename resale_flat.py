import pandas as pd
import numpy as np
import pickle
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def exploratory_analysis(df):
    numeric_cols = ['floor_area_sqm', 'resale_price', 'lease_commence_date']

    st.write("### Data Distribution Plots")
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    plt.tight_layout()
    st.pyplot()

    st.write("### Correlation analysis")
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    st.pyplot()

    st.write("### Categorical Variables Analysis")
    categorical_cols = ['town','flat_type', 'storey_range', 'flat_model']
    plt.figure(figsize=(15, 20))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(4, 2, i)
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'Countplot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot()

    st.write("### Relationship between features and target variable")
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='floor_area_sqm', y='resale_price',hue='flat_type', palette='Set1', alpha=0.7)
    plt.title('Floor Area vs Resale Price')
    plt.xlabel('Floor Area (sqm)')
    plt.ylabel('Resale Price')
    st.pyplot()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='lease_commence_date', y='resale_price', alpha=0.7)
    plt.title('Lease commence Date vs Resale Price')
    plt.xlabel('Lease commence Date')
    plt.ylabel('Resale Price')
    st.pyplot()



def data_preprocessing(df,load_label_encoders=True,load_scaler=True, filename='label_encoders.pkl',scaler_filename='scaler.pkl'):

    df['remaining_lease'] = df['remaining_lease'].fillna('UNKNOWN')
    df['month'] = pd.to_datetime(df['month'])
    df['year'] = df['month'].dt.year
    df['month'] = df['month'].dt.month
    df = df.drop(columns = ['flat_type'])
    df['remaining_lease'] = df['remaining_lease'].astype(str)

    a = df.isnull().sum().sum()
    if a!=0:
        df['town']=df['town'].fillna('TAMPINES')
        df['block']=df['block'].fillna('2')
        df['street_name']=df['street_name'].fillna('YISHUN RING RD')
        df['storey_range']=df['storey_range'].fillna('04 TO 06')
        df['flat_model']=df['flat_model'].fillna('Model A')
        df['floor_area_sqm']=df['floor_area_sqm'].fillna(75.0)
        df['lease_commence_date'] = df['lease_commence_date'].fillna(1976)
        df['month'] = df['month'].fillna(6)
        df['year'] = df['year'].fillna(2010)
    
    categorical_columns = ['town','block', 'street_name', 'storey_range', 'flat_model','remaining_lease']

    

    # Initialize LabelEncoder
    label_encoders = {}
    
    if load_label_encoders:
        # Load the saved label encoders
        with open(filename, 'rb') as f:
            label_encoders = pickle.load(f)
    # Iterate over each categorical column
    for col in categorical_columns:
        # Initialize LabelEncoder for each column
        if col in label_encoders:
            # Transform using loaded LabelEncoders
            df[col] = df[col].map(lambda s: label_encoders[col].transform([s])[0] if s in label_encoders[col].classes_ else -1)
        else:
            # Initialize and fit LabelEncoder for new data
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])
    if not load_label_encoders:
        with open(filename, 'wb') as f:
            pickle.dump(label_encoders, f)

    df_encoded = df.copy()
    

    scaler = StandardScaler()
    if load_scaler:
        # Load the saved scaler
        with open(scaler_filename, 'rb') as f:
            scaler = pickle.load(f)
        df_encoded['floor_area_sqm'] = scaler.transform(df_encoded[['floor_area_sqm']])
    else:
        df_encoded['floor_area_sqm'] = scaler.fit_transform(df_encoded[['floor_area_sqm']])
        # Save the scaler
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scaler, f)
    return(df_encoded,label_encoders,scaler)

def plot_floor_area(df_encoded):
    st.write("## Plot of Floor Area")
    sns.set(style="whitegrid")

    # Create a figure with two subplots: one for the boxplot and one for the histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot
    sns.boxplot(y=df_encoded['floor_area_sqm'], ax=axes[0])
    axes[0].set_title('Boxplot of Floor Area (sqm) with outiers')
    axes[0].set_xlabel('Floor Area (sqm)')

    # Histogram
    sns.histplot(df_encoded['floor_area_sqm'], bins=10, kde=True, ax=axes[1])
    axes[1].set_title('Histogram of Floor Area (sqm) with outliers')
    axes[1].set_xlabel('Floor Area (sqm)')
    axes[1].set_ylabel('Frequency')

    # Show the plots
    plt.tight_layout()
    st.pyplot()

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df_encoded['floor_area_sqm'].quantile(0.25)
    Q3 = df_encoded['floor_area_sqm'].quantile(0.75)

    # Calculate the IQR
    IQR = Q3 - Q1

    # Determine outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    df_filtered = df_encoded[(df_encoded['floor_area_sqm'] >= lower_bound) & (df_encoded['floor_area_sqm'] <= upper_bound)]

    sns.set(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot after removing outliers
    sns.boxplot(y=df_filtered['floor_area_sqm'], ax=axes[0])
    axes[0].set_title('Boxplot of Floor Area (sqm) - After Outlier Removal')
    axes[0].set_ylabel('Floor Area (sqm)')

    # Histogram after removing outliers
    sns.histplot(df_filtered['floor_area_sqm'], bins=10, kde=True, ax=axes[1])
    axes[1].set_title('Histogram of Floor Area (sqm) - After Outlier Removal')
    axes[1].set_xlabel('Floor Area (sqm)')
    axes[1].set_ylabel('Frequency')

    # Show the plots
    plt.tight_layout()
    st.pyplot()
    return(df_filtered)

def plot_town(df_filtered):
    st.write("## Plot of Town")
    sns.set_style("whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Box plot
    sns.boxplot(y='town', data=df_filtered, ax=axes[0])
    axes[0].set_title('Box Plot of Town')
    axes[0].set_xlabel('Town')
    axes[0].set_ylabel('Frequency')

    # Histogram
    sns.histplot(df_filtered['town'], ax=axes[1], kde=True)
    axes[1].set_title('Histogram of Town')
    axes[1].set_xlabel('Town')
    axes[1].set_ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()

    # Show plots
    st.pyplot()

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df_filtered['town'].quantile(0.25)
    Q3 = df_filtered['town'].quantile(0.75)

    # Calculate the IQR
    IQR = Q3 - Q1

    # Determine outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    df_filtered = df_filtered[(df_filtered['town'] >= lower_bound) & (df_filtered['town'] <= upper_bound)]

    sns.set_style("whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Box plot
    sns.boxplot(y='town', data=df_filtered, ax=axes[0])
    axes[0].set_title('Box Plot of Town after removing outliers')
    axes[0].set_xlabel('Town')
    axes[0].set_ylabel('Frequency')

    # Histogram
    sns.histplot(df_filtered['town'], ax=axes[1], kde=True)
    axes[1].set_title('Histogram of Town after removing outliers')
    axes[1].set_xlabel('Town')
    axes[1].set_ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()

    # Show plots
    st.pyplot()
    return(df_filtered)


def plot_block(df_filtered):
    st.write("## Plot of Block")
    sns.set_style("whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Box plot
    sns.boxplot(x='block', data=df_filtered, ax=axes[0])
    axes[0].set_title('Box Plot of Block')
    axes[0].set_xlabel('Block')
    axes[0].set_ylabel('Frequency')

    # Histogram
    sns.histplot(df_filtered['block'], ax=axes[1], kde=True)
    axes[1].set_title('Histogram of Block')
    axes[1].set_xlabel('Block')
    axes[1].set_ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()

    # Show plots
    st.pyplot()

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df_filtered['block'].quantile(0.25)
    Q3 = df_filtered['block'].quantile(0.75)

    # Calculate the IQR
    IQR = Q3 - Q1

    # Determine outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    df_filtered = df_filtered[(df_filtered['block'] >= lower_bound) & (df_filtered['block'] <= upper_bound)]

    sns.set_style("whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Box plot
    sns.boxplot(x='block', data=df_filtered, ax=axes[0])
    axes[0].set_title('Box Plot of Block after removing outliers')
    axes[0].set_xlabel('Block')
    axes[0].set_ylabel('Frequency')

    # Histogram
    sns.histplot(df_filtered['block'], ax=axes[1], kde=True)
    axes[1].set_title('Histogram of Block after removing outliers')
    axes[1].set_xlabel('Block')
    axes[1].set_ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()

    # Show plots
    st.pyplot()
    return(df_filtered)

def plot_street_name(df_filtered):
    st.write("## Plot of Street Name")
    sns.set_style("whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Box plot
    sns.boxplot(x='street_name', data=df_filtered, ax=axes[0])
    axes[0].set_title('Box Plot of Street Name')
    axes[0].set_xlabel('Street Name')
    axes[0].set_ylabel('Frequency')

    # Histogram
    sns.histplot(df_filtered['street_name'], ax=axes[1], kde=True)
    axes[1].set_title('Histogram of Street Name')
    axes[1].set_xlabel('Street Name')
    axes[1].set_ylabel('Frequency')

    # Rotate x-axis labels for better readability
    axes[0].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='x', rotation=90)

    # Adjust layout
    plt.tight_layout()

    # Show plots
    st.pyplot()
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df_filtered['street_name'].quantile(0.25)
    Q3 = df_filtered['street_name'].quantile(0.75)

    # Calculate the IQR
    IQR = Q3 - Q1

    # Determine outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    df_filtered = df_filtered[(df_filtered['street_name'] >= lower_bound) & (df_filtered['street_name'] <= upper_bound)]

    sns.set_style("whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Box plot
    sns.boxplot(x='street_name', data=df_filtered, ax=axes[0])
    axes[0].set_title('Box Plot of Street Name after removing outliers')
    axes[0].set_xlabel('Street Name')
    axes[0].set_ylabel('Frequency')

    # Histogram
    sns.histplot(df_filtered['street_name'], ax=axes[1], kde=True)
    axes[1].set_title('Histogram of Street Name after removing outliers')
    axes[1].set_xlabel('Street Name')
    axes[1].set_ylabel('Frequency')

    # Rotate x-axis labels for better readability
    axes[0].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='x', rotation=90)

    # Adjust layout
    plt.tight_layout()

    # Show plots
    st.pyplot()
    return(df_filtered)

def plot_storey_range(df_filtered):
    st.write("## Plot of Storey Range")
    sns.set_style("whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Box plot
    sns.boxplot(x='storey_range', data=df_filtered, ax=axes[0])
    axes[0].set_title('Box Plot of Storey Range')
    axes[0].set_xlabel('Storey Range')
    axes[0].set_ylabel('Frequency')

    # Histogram
    sns.histplot(df_filtered['storey_range'], ax=axes[1], kde=True)
    axes[1].set_title('Histogram of Storey Range')
    axes[1].set_xlabel('Storey Range')
    axes[1].set_ylabel('Frequency')

    # Rotate x-axis labels for better readability
    axes[0].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='x', rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Show plots
    st.pyplot()

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df_filtered['storey_range'].quantile(0.25)
    Q3 = df_filtered['storey_range'].quantile(0.75)

    # Calculate the IQR
    IQR = Q3 - Q1

    # Determine outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    df_filtered = df_filtered[(df_filtered['storey_range'] >= lower_bound) & (df_filtered['storey_range'] <= upper_bound)]
    sns.set_style("whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Box plot
    sns.boxplot(x='storey_range', data=df_filtered, ax=axes[0])
    axes[0].set_title('Box Plot of Storey Range')
    axes[0].set_xlabel('Storey Range')
    axes[0].set_ylabel('Frequency')

    # Histogram
    sns.histplot(df_filtered['storey_range'], ax=axes[1], kde=True)
    axes[1].set_title('Histogram of Storey Range')
    axes[1].set_xlabel('Storey Range')
    axes[1].set_ylabel('Frequency')

    # Rotate x-axis labels for better readability
    axes[0].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='x', rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Show plots
    st.pyplot()
    return(df_filtered)

def plot_flat_model(df_filtered):
    st.write("## Plot of Flat Model")
    sns.set_style("whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Box plot
    sns.boxplot(x='flat_model', data=df_filtered, ax=axes[0])
    axes[0].set_title('Box Plot of Flat Model')
    axes[0].set_xlabel('Flat Model')
    axes[0].set_ylabel('Frequency')

    # Histogram
    sns.histplot(df_filtered['flat_model'], ax=axes[1], kde=True)
    axes[1].set_title('Histogram of Flat Model')
    axes[1].set_xlabel('Flat Model')
    axes[1].set_ylabel('Frequency')

    # Rotate x-axis labels for better readability
    axes[0].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='x', rotation=90)

    # Adjust layout
    plt.tight_layout()

    # Show plots
    st.pyplot()
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df_filtered['flat_model'].quantile(0.25)
    Q3 = df_filtered['flat_model'].quantile(0.75)

    # Calculate the IQR
    IQR = Q3 - Q1

    # Determine outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    df_filtered = df_filtered[(df_filtered['flat_model'] >= lower_bound) & (df_filtered['flat_model'] <= upper_bound)]

    sns.set_style("whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Box plot
    sns.boxplot(x='flat_model', data=df_filtered, ax=axes[0])
    axes[0].set_title('Box Plot of Flat Model after removing outliers')
    axes[0].set_xlabel('Flat Model')
    axes[0].set_ylabel('Frequency')

    # Histogram
    sns.histplot(df_filtered['flat_model'], ax=axes[1], kde=True)
    axes[1].set_title('Histogram of Flat Model after removing outliers')
    axes[1].set_xlabel('Flat Model')
    axes[1].set_ylabel('Frequency')

    # Rotate x-axis labels for better readability
    axes[0].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='x', rotation=90)

    # Adjust layout
    plt.tight_layout()

    # Show plots
    st.pyplot()
    return(df_filtered)

def plot_remaining_lease(df_filtered):
    st.write("## Plot of Remaining Lease")
    sns.set_style("whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Box plot
    sns.boxplot(x='remaining_lease', data=df_filtered, ax=axes[0])
    axes[0].set_title('Box Plot of Remaining Lease')
    axes[0].set_xlabel('Remaining Lease')
    axes[0].set_ylabel('Frequency')

    # Histogram
    sns.histplot(df_filtered['remaining_lease'], ax=axes[1], kde=True)
    axes[1].set_title('Histogram of Remaining Lease')
    axes[1].set_xlabel('Remaining Lease')
    axes[1].set_ylabel('Frequency')

    # Rotate x-axis labels for better readability
    axes[0].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='x', rotation=90)

    # Adjust layout
    plt.tight_layout()

    # Show plots
    st.pyplot()
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df_filtered['remaining_lease'].quantile(0.25)
    Q3 = df_filtered['remaining_lease'].quantile(0.75)

    # Calculate the IQR
    IQR = Q3 - Q1

    # Determine outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    df_filtered = df_filtered[(df_filtered['remaining_lease'] >= lower_bound) & (df_filtered['remaining_lease'] <= upper_bound)]
    sns.set_style("whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Box plot
    sns.boxplot(x='remaining_lease', data=df_filtered, ax=axes[0])
    axes[0].set_title('Box Plot of Remaining Lease after removing outliers')
    axes[0].set_xlabel('Remaining Lease')
    axes[0].set_ylabel('Frequency')

    # Histogram
    sns.histplot(df_filtered['remaining_lease'], ax=axes[1], kde=True)
    axes[1].set_title('Histogram of Remaining Lease after removing outliers')
    axes[1].set_xlabel('Remaining Lease')
    axes[1].set_ylabel('Frequency')

    # Rotate x-axis labels for better readability
    axes[0].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='x', rotation=90)

    # Adjust layout
    plt.tight_layout()

    # Show plots
    st.pyplot()
    return(df_filtered)

def train_test_split1(df_filtered,s_option,ts):
   
    X = df_filtered.drop(columns=[s_option])  # Features
    y = df_filtered[s_option]  # Target variable

    # Splitting the data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=ts, random_state=42)

    st.write("## TRAIN_TEST_SPLIT")

    plt.figure(figsize=(10, 5))
    sns.histplot(y_train, color='blue', label='Train', kde=True)
    sns.histplot(y_val, color='orange', label='Test', kde=True)
    plt.title('Distribution of Target Variable (Resale Price)')
    plt.xlabel('Resale Price')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot()

    select_1 = st.selectbox('Select the Feature :',X.columns.unique())
    feature_to_visualize = select_1
    plt.figure(figsize=(10, 5))
    sns.histplot(X_train[feature_to_visualize], color='blue', label='Train', kde=True)
    sns.histplot(X_val[feature_to_visualize], color='orange', label='Test', kde=True)
    plt.title(f'Distribution of {feature_to_visualize}')
    plt.xlabel(feature_to_visualize)
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot()

    st.write("### Training and Testing Data Overview")
    st.write("#### X_train")
    st.table(X_train.head())  
    st.write("#### y_train")
    st.table(y_train.head())  
    st.write("#### X_test")
    st.table(X_val.head())  
    st.write("#### y_test")
    st.table(y_val.head())  

def xgb_regression(df_filtered,s_option,ts):
    
    
    X = df_filtered.drop(columns=[s_option])  # Features
    y = df_filtered[s_option]  # Target variable

    # Splitting the data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=ts, random_state=42)

    # Convert the data into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Define parameters for XGBoost
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    # Define early stopping criteria
    early_stopping_rounds = 10

    # Train the model with early stopping
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dval, 'validation')], early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    # Save the trained model
    model.save_model('xgb_model.model')

    # Predict on the validation set
    y_pred = model.predict(dval)

    # Evaluate the model
    st.write("## OUTPUT OF XGB_REGRESSION")
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    st.write("RMSE on validation set:", rmse)

    r2 = r2_score(y_val, y_pred)
    st.write("R^2 Score:", r2)

    results_df = pd.DataFrame({'Actual': y_val, 'Predicted': y_pred},index = None)

    # Display the DataFrame as a table
    st.write("Actual vs Predicted Resale Price:")
    st.write(results_df)

    plt.figure(figsize=(10, 5))
    plt.scatter(y_val, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')
    plt.title('Actual vs Predicted Resale Price')
    plt.xlabel('Actual Resale Price')
    plt.ylabel('Predicted Resale Price')
    plt.grid(True)
    st.pyplot()

def user_input():
    new_data = {'month': [],
                'floor_area_sqm':[],
                'lease_commence_date':[],
                'town':[],
                'flat_type':[],
                'block':[],
                'street_name':[],
                'storey_range':[],
                'flat_model':[],
                'remaining_lease':[]
                }
    mon = st.text_input("Enter the month in format YYYY/MM :")
    new_data['month'].append(mon)
    fas = st.number_input("Enter the floor area in sqm :")
    new_data['floor_area_sqm'].append(fas)
    lcd = st.number_input("Enter the lease commence date (YYYY) :")
    new_data['lease_commence_date'].append(lcd)
    town = st.text_input("Enter the town :")
    new_data['town'].append(town)
    ft = st.text_input("Enter the flat type :")
    new_data['flat_type'].append(ft)
    bloc = st.text_input("Enter the block :")
    new_data['block'].append(bloc)
    sn = st.text_input("Enter the street name : ")
    new_data['street_name'].append(sn)
    sr = st.text_input("Enter the storey range : ")
    new_data['storey_range'].append(sr)
    fm = st.text_input("Enter the flat model :")
    new_data['flat_model'].append(fm)
    rl = st.text_input("Enter the remaining lease : ")
    new_data['remaining_lease'].append(rl)

    Df = pd.DataFrame(new_data)
    return(Df)



def train_test_split2(df_filtered,s_option,ts):
    X = df_filtered.drop(columns=[s_option])  # Features
    y = df_filtered[s_option]  # Target variable

    st.write("## TRAIN TEST SPLIT")
    st.write("### PLOT OF TARGET VARIABLE")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

    # Plot the distribution of target variable (Resale Price)
    plt.figure(figsize=(10, 5))
    sns.histplot(y_train, color='blue', label='Train', kde=True)
    sns.histplot(y_test, color='orange', label='Test', kde=True)
    plt.title('Distribution of Target Variable (Resale Price)')
    plt.xlabel('Resale Price')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot()

    st.write("### PLOT OF FEATURES")

    select_1 = st.selectbox('Select the Feature :',X.columns.unique())
    feature_to_visualize = select_1
    plt.figure(figsize=(10, 5))
    sns.histplot(X_train[feature_to_visualize], color='blue', label='Train', kde=True)
    sns.histplot(X_test[feature_to_visualize], color='orange', label='Test', kde=True)
    plt.title(f'Distribution of {feature_to_visualize}')
    plt.xlabel(feature_to_visualize)
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot()

    st.write("### Training and Testing Data Overview")
    st.write("#### X_train")
    st.table(X_train.head())  
    st.write("#### y_train")
    st.table(y_train.head())  
    st.write("#### X_test")
    st.table(X_test.head())  
    st.write("#### y_test")
    st.table(y_test.head())  

def decisiontree_regression(df_filtered,s_option,ts):
    # Separate features and target variable
    X = df_filtered.drop(columns=[s_option])  # Features
    y = df_filtered[s_option]  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

    # Initialize the decision tree regressor
    decision_tree = DecisionTreeRegressor(random_state=42)

    # Fit the model to the training data
    decision_tree.fit(X_train, y_train)

    # Predict on the testing set
    y_pred = decision_tree.predict(X_test)

    # Evaluate the model
    st.write("## OUTPUT OF DECISION_TREE")
    mse = mean_squared_error(y_test, y_pred)
    st.write("Mean Squared Error:", mse)
    rmse = np.sqrt(mse)

    st.write("Root Mean Squared Error:", rmse)

    r2 = r2_score(y_test, y_pred)
    st.write("R^2 Score:", r2)

    joblib.dump(decision_tree, 'decision_tree_model.pkl')

    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred},index = None)

    # Display the DataFrame as a table
    st.write("Actual vs Predicted Resale Price:")
    st.write(results_df)

    # Plotting actual vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)

    # Display plot in Streamlit
    st.pyplot(plt)

    # Plotting distribution of residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, edgecolor='k')
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Display plot in Streamlit
    st.pyplot(plt)

   
# Streamlit part

st.title(" RESALE PRICE OF FLAT MODEL ")

# Define session state to store DataFrames
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {
        "df1": pd.DataFrame(),
        "df2": pd.DataFrame(),
        "df3": pd.DataFrame(),
        "df4": pd.DataFrame(),
        "df5": pd.DataFrame()
    }

with st.sidebar:
    select = option_menu("Main Menu", ["HOME","EXPLORATORY DATA ANALYSIS","DATA PREPROCESSING","MACHINE LEARNING","PREDICTION"])

if select == "HOME":

    st.subheader("Select the CSV files for training the dataset:")

    uploaded_file_1 = st.file_uploader("Upload CSV file1", type=["csv"])
    if uploaded_file_1 is not None:
        st.session_state.dataframes["df1"] = pd.read_csv(uploaded_file_1)
        st.write("Uploaded CSV file1:")
        st.write(st.session_state.dataframes["df1"])
        st.success("Upload completed successfully!")

    uploaded_file_2 = st.file_uploader("Upload CSV file2", type=["csv"])
    if uploaded_file_2 is not None:
        st.session_state.dataframes["df2"] = pd.read_csv(uploaded_file_2)
        st.write("Uploaded CSV file2:")
        st.write(st.session_state.dataframes["df2"])
        st.success("Upload completed successfully!")

    uploaded_file_3 = st.file_uploader("Upload CSV file3", type=["csv"])
    if uploaded_file_3 is not None:
        st.session_state.dataframes["df3"] = pd.read_csv(uploaded_file_3)
        st.write("Uploaded CSV file3:")
        st.write(st.session_state.dataframes["df3"])
        st.success("Upload completed successfully!")

    uploaded_file_4 = st.file_uploader("Upload CSV file4", type=["csv"])
    if uploaded_file_4 is not None:
        st.session_state.dataframes["df4"] = pd.read_csv(uploaded_file_4)
        st.write("Uploaded CSV file4:")
        st.write(st.session_state.dataframes["df4"])
        st.success("Upload completed successfully!")

    uploaded_file_5 = st.file_uploader("Upload CSV file5", type=["csv"])
    if uploaded_file_5 is not None:
        st.session_state.dataframes["df5"] = pd.read_csv(uploaded_file_5)
        st.write("Uploaded CSV file5:")
        st.write(st.session_state.dataframes["df5"])
        st.success("Upload completed successfully!")

elif select == "EXPLORATORY DATA ANALYSIS":
    df6 = pd.concat(
        [st.session_state.dataframes["df1"],
         st.session_state.dataframes["df2"],
         st.session_state.dataframes["df3"],
         st.session_state.dataframes["df4"],
         st.session_state.dataframes["df5"]],
        ignore_index=True
    )      
    st.set_option('deprecation.showPyplotGlobalUse', False)      
    exploratory_analysis(df6)

    

elif select == "DATA PREPROCESSING":
    st.subheader("Data Preprocessing Section")
    
    df = pd.concat(
        [st.session_state.dataframes["df1"],
         st.session_state.dataframes["df2"],
         st.session_state.dataframes["df3"],
         st.session_state.dataframes["df4"],
         st.session_state.dataframes["df5"]],
        ignore_index=True
    )      
        
    if not df.empty:
        st.write(df)
        data_encoded,label_encoders,scaler = data_preprocessing(df,load_label_encoders=False,load_scaler=False)
        st.write("## Preprocessed Data")
        buffer = StringIO()
        data_encoded.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        data_1 = plot_floor_area(data_encoded)
        data_2 = plot_town(data_1)
        data_4 = plot_block(data_2)
        data_5 = plot_street_name(data_4)
        data_6 = plot_storey_range(data_5)
        data_7 = plot_flat_model(data_6)
        data_8 = plot_remaining_lease(data_7)
        st.write("## DATA PREPROCESSING FINISHED !!")
        st.session_state.data_8 = data_8
    else:
        st.warning("Please upload the required CSV files first !!")


        
elif select == "MACHINE LEARNING":
    st.subheader("Machine Learning Section")
    if "data_8" in st.session_state:
        data_8 = st.session_state.data_8
        st.write(data_8)
        selected_option = st.selectbox('Select the target variable :',data_8.columns.unique())
        if selected_option == 'resale_price':           
            test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2, 0.05)
            option = st.radio("## Select the type of Machine learning", ("None", "XGBoost", "DecisionTreeRegressor"))
            if option == "XGBoost":
                train_test_split1(data_8,selected_option,test_size)
                xgb_regression(data_8,selected_option,test_size)
            elif option == "DecisionTreeRegressor":
                train_test_split2(data_8,selected_option,test_size)
                decisiontree_regression(data_8,selected_option,test_size)
        else:
            print("Please select a valid target variable from the available options !!")
    else:
        st.warning("Please run the Data Preprocessing step first.")
elif select == "PREDICTION":
    st.write("## Give user inputs for prediction")
    Df = user_input()
    ndf, _ , _ = data_preprocessing(Df, load_label_encoders=True,load_scaler=True)
    choose = st.radio("## Select the Model to predict :", ("None","DecisionTreeRegressor")) 
    
    if choose == "DecisionTreeRegressor":
        desired_order = ['month', 'town','block', 'street_name', 
                 'storey_range', 'floor_area_sqm', 'flat_model', 
                 'lease_commence_date', 'remaining_lease', 'year']
        ndf = ndf[desired_order]
        loaded_model = joblib.load('decision_tree_model.pkl')
        y_pred_loaded = loaded_model.predict(ndf)
        st.write("## The predicted value is given as :")
        # Display predictions
        st.markdown(f"**Predicted Resale Value:** `{y_pred_loaded}`", unsafe_allow_html=True)


    