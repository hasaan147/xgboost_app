import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.datasets import load_iris

def main():
    st.title('XGBoost Model for Iris Dataset')

    # Load the dataset
    st.write("Loading Iris dataset...")
    df = load_iris()
    X = df.data
    y = df.target

    # Split the dataset
    st.write("Splitting the dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Parameters for GridSearchCV
    booster = st.selectbox('Booster', ['gbtree', 'gblinear', 'dart'])
    n_estimators = st.slider('Number of Estimators', 50, 200, 100, 50)
    eta = st.slider('Learning Rate (eta)', 0.01, 0.2, 0.1, 0.01)
    max_depth = st.slider('Max Depth', 3, 7, 5, 1)
    min_child_weight = st.slider('Min Child Weight', 1, 10, 1, 1)
    subsample = st.slider('Subsample', 0.8, 1.0, 1.0, 0.1)
    colsample_bytree = st.slider('Column Sample by Tree', 0.8, 1.0, 1.0, 0.1)
    gamma = st.slider('Gamma', 0.0, 0.2, 0.0, 0.1)
    lambda_ = st.slider('Lambda', 0, 10, 0, 1)  # Renamed to lambda_
    alpha = st.slider('Alpha', 0, 10, 0, 1)

    # Initialize and tune the XGBoost model
    st.write("Training and tuning the model...")
    xg_model = xgb.XGBClassifier(
        booster=booster,
        n_estimators=n_estimators,
        eta=eta,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        lambda_=lambda_,  # Renamed to lambda_
        alpha=alpha
    )

    # Perform Grid Search with Cross-Validation
    param_grid = {
        'booster': [booster],
        'n_estimators': [n_estimators],
        'eta': [eta],
        'max_depth': [max_depth],
        'min_child_weight': [min_child_weight],
        'subsample': [subsample],
        'colsample_bytree': [colsample_bytree],
        'gamma': [gamma],
        'lambda_': [lambda_],  # Renamed to lambda_
        'alpha': [alpha]
    }

    grid_search = GridSearchCV(xg_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters and score
    st.write("Best Parameters:", grid_search.best_params_)
    st.write("Best Score:", grid_search.best_score_)

    # Train the model with the best parameters
    best_xg_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_xg_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.write("Accuracy:", accuracy)
    st.write("Classification Report:\n", report)

if __name__ == "__main__":
    main()
