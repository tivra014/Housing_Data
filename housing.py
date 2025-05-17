# -*- coding: utf-8 -*-
"""Housing.ipynb


Original file is located at
    https://colab.research.google.com/drive/1MMdCbDSkrRdudafGQ-ii8uw4sosz0Eqw
"""

from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Data loading
try:
    df = pd.read_csv('Housing.csv')
    display(df.head())
except FileNotFoundError:
    print("Error: 'Housing.csv' not found. Please make sure the file exists in the current directory.")
    df = None

if df is not None:
    # Data exploration
    display(df.describe())
    display(df.isnull().sum())
    display(df.info())

    numerical_features = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    for col in numerical_features:
        plt.figure(figsize=(8, 6))
        plt.hist(df[col], bins=10, edgecolor='black')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {col}')
        plt.show()

    categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
    for col in categorical_features:
        plt.figure(figsize=(8, 6))
        df[col].value_counts().plot(kind='bar')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {col}')
        plt.show()

    # Data preparation
    df_prepared = df.copy()
    binary_mapping = {'yes': 1, 'no': 0}
    for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
        df_prepared[col] = df_prepared[col].map(binary_mapping)
    df_prepared = pd.get_dummies(df_prepared, columns=['furnishingstatus'], drop_first=True)

    X = df_prepared.drop('price', axis=1)
    y = df_prepared['price']

    numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    display(X.head())
    display(y.head())

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    # Simple Linear Regression
    X_train_simple = X_train[['area']]
    X_test_simple = X_test[['area']]

    simple_model = LinearRegression()
    simple_model.fit(X_train_simple, y_train)
    y_pred_simple = simple_model.predict(X_test_simple)

    # Multiple Linear Regression
    multiple_model = LinearRegression()
    multiple_model.fit(X_train, y_train)
    y_pred_multiple = multiple_model.predict(X_test)

    # Model evaluation
    # Evaluate Simple Linear Regression
    r2_simple = r2_score(y_test, y_pred_simple)
    mse_simple = mean_squared_error(y_test, y_pred_simple)
    rmse_simple = np.sqrt(mse_simple)

    # Evaluate Multiple Linear Regression
    r2_multiple = r2_score(y_test, y_pred_multiple)
    mse_multiple = mean_squared_error(y_test, y_pred_multiple)
    rmse_multiple = np.sqrt(mse_multiple)

    print("Simple Linear Regression Metrics:")
    print(f"R-squared: {r2_simple}")
    print(f"MSE: {mse_simple}")
    print(f"RMSE: {rmse_simple}")

    print("\nMultiple Linear Regression Metrics:")
    print(f"R-squared: {r2_multiple}")
    print(f"MSE: {mse_multiple}")
    print(f"RMSE: {rmse_multiple}")