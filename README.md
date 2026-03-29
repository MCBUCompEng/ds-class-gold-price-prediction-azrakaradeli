[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Xx5gAEjB)
# Assignment: Predicting Gram Gold Prices in Turkish Lira (TL)
[MCBU Class Template Repository]

# 1. Background and Motivation

Gold has historically been a critical safe-haven asset and an indicator of broader economic trends, particularly in dynamic markets. 
Investors constantly analyze price trends to make informed decisions—often debating the merits of holding physical gold versus investing 
in managed gold funds or technology-driven portfolios. Accurate forecasting models are essential for navigating these investment vehicles.

In this assignment, you will take on the role of a data scientist. Your objective is to build a machine learning pipeline that predicts 
the future price of Gram Gold in Turkish Lira (TL) based on a decade of historical market data.

# 2. Project Structure

To simulate a professional development environment, you are required to adhere strictly to the following directory structure. 
Modularity is key to building maintainable machine learning applications.

Folder Structure:

<pre>
gold_prediction_project/
├── DataSet/
│   └── gram_gold_10yrs.csv       # The provided 10-year historical dataset
├── Model/
│   ├── __init__.py               # Initializes the Model directory as a Python package
│   └── functions.py              # Contains all your machine learning and utility functions
└── predict.py                    # The main execution script that imports from Model
</pre>

# 3. Component Details
# 3.1. The Dataset (DataSet/gram_gold_10yrs.csv)

This file contains the daily market data for Gram Gold (TL) over the last 10 years.

    Columns        : Date, Opening TL, Closing TL
    Target Variable: You will be predicting the Close price.

    Note:   You are expected to handle any missing values, datetime conversions, and standard 
            scaling as part of your preprocessing pipeline.

# 3.2. The Model Module (Model/functions.py)

This file should contain the core logic of your machine learning pipeline. Do not put execution code 
(like if __name__ == "__main__":) here. Instead, define robust, reusable functions or classes.

Required Functions/Methods:

    * load_and_preprocess_data(filepath)   : Reads the CSV, parses dates, handles null values, and extracts relevant features 
    * train_model(X_train, y_train)        : Accepts the training features and target variable, trains a regression model (Your own Regression Code-Not Library) 
    * evaluate_model(model, X_test, y_test): Calculates and returns performance metrics such as Mean Squared Error (MSE), and R-squared (R2).

# 3.3. The Execution Script (predict.py)

This is the entry point of your application. It must import the necessary functions from your Model package and execute the pipeline sequentially.

predict.py Workflow:

    - Import functions from Model.functions.
    - Load and preprocess the dataset from DataSet/gram_gold_10yrs.csv.
    - Split the data into training and testing sets (e.g., 80% train, 20% test, ensuring chronological order is maintained to prevent data leakage).
    - Train the predictive model.
    - Evaluate the model on the test set and print the performance metrics to the console.
    - Final Output: Print a prediction for the next day's closing price based on the most recent data point in the dataset.

# 4. Evaluation Criteria

Your submission will be graded on the following:

    * Code Architecture (2 pnt): Strict adherence to the provided folder structure and proper modularity.
    * Data Preparation (3 pnt): Correct handling of time-series data, feature engineering, and avoiding future-data leakage during the train/test split.
    * Model Implementation (3 pnt): Successful training and evaluation of an appropriate regression model.
    * Documentation & Cleanliness (2 pnt): Code readability, meaningful variable names, and clear comments explaining your logic.

Zip your entire gold_prediction_project directory (excluding any virtual environment folders like venv or .env) and upload it to the assignment portal before the deadline. Ensure that your predict.py script runs flawlessly from the root directory.
