# Food Demand Forecasting: Machine Learning Hackathon
Overview
This project tackles the Food Demand Forecasting problem for a meal delivery company with multiple fulfillment centers across various cities. The company is looking for a solution to predict the demand for meals over the next 10 weeks to help plan raw material procurement and staffing. The goal is to build a robust machine learning model that can accurately forecast meal orders based on historical data, product features, and fulfillment center information.

Problem Description
The client, a meal delivery service, wants to forecast the demand for meals (orders) for the next 10 weeks (Weeks 146-155). The data available includes:

Historical demand data (Weeks 1 to 145)
Meal features such as category, sub-category, price, and discounts
Fulfillment center information like center area and city
Objective
The objective is to predict the demand for the next 10 weeks for center-meal combinations, enabling the company to optimize procurement and staffing decisions.

Data Sources
The data for this problem consists of the following CSV files:

train.csv - Historical demand data for meals across different fulfillment centers.
test_QoiMO9B.csv - Test data with missing demand information for the upcoming weeks.
meal_info.csv - Details about the meals, including meal category, sub-category, price, and discount.
fulfilment_center_info.csv - Information about the fulfillment centers, including city and area details.
sample_submission_hSlSoT6.csv - Template for the submission file.
Features
Key Features:
num_orders: Target variable representing the number of orders for a center-meal combination.
meal_id: Unique identifier for each meal.
center_id: Unique identifier for each fulfillment center.
price_diff: Difference between the base price and checkout price.
cuisine: The cuisine type of the meal.
category: The category of the meal.
center_type: Type of the fulfillment center.
region_code, city_code: Geographical codes for the region and city of the fulfillment center.
Derived Features:
Log transformation of num_orders to handle skewed data.
Sine and Cosine transformations of the week to capture cyclical patterns.
Aggregated features like the mean, min, max, and standard deviation of orders across different groupings (e.g., by center_id, meal_id, city_code, etc.).
Feature Engineering
Several transformations and aggregations were applied to the raw data:

Price category: Classifying the price difference into categories such as "Discount", "Taxes", and "Additional Charges".
Marketing Feature: Combining promotional efforts into a single column.
Geographical Features: Analyzing demand based on center type, region, and city codes.
Temporal Features: Using sine and cosine transformations to model seasonality based on the week number.
Aggregated Features: Calculating statistics like mean, min, max, and standard deviation of orders for different groupings.
Exploratory Data Analysis (EDA)
EDA was performed to identify patterns, correlations, and trends in the data:

Distribution of target variable (num_orders) was analyzed using both raw and log-transformed values.
Correlation analysis was performed to check relationships between numerical features and the target variable.
Categorical features like cuisine, category, center_type were analyzed for their impact on meal demand.
Geographical and temporal patterns were analyzed to understand variations in demand across regions and over time.
Model Building
Data Preprocessing
Missing values were handled, and categorical variables were encoded.
Features such as center_id, meal_id, city_code, and region_code were dropped, as they were not significant for prediction.
Model Selection
Multiple models were tested:

Linear Regression
Random Forest Regressor
Gradient Boosting Regressor
Model Evaluation
K-Fold Cross-Validation was used for model evaluation.
The Root Mean Squared Log Error (RMSLE) metric was used to measure model performance.
Models were trained on features like num_orders, price_diff, cuisine, center_type, and other engineered features.
Final Submission
The final submission was generated using the Linear Regression model. The predicted values were transformed back to their original scale using the exponentiation of log-transformed values.

Results
The following models were evaluated based on the Root Mean Squared Log Error (RMSLE):

Linear Regression: A baseline model achieved a RMSLE of 106.
Random Forest Regressor: A more complex model achieved a RMSLE of 53.66.
Gradient Boosting Regressor: Other models were also evaluated but focused on simplicity for final submission.
The final submission file includes predictions of the number of orders for the test dataset (Weeks 146-155).


Installation
To run this project locally, make sure you have the required dependencies:

bash
Copy code
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels
Usage
Download the data files:

train.csv
test_QoiMO9B.csv
meal_info.csv
fulfilment_center_info.csv
sample_submission_hSlSoT6.csv
Place them in the appropriate directory and run the notebook Food Demand Forecasting.ipynb.

The final submission will be saved in CSV format, ready to be submitted for evaluation.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Genpact Machine Learning Hackathon for providing the dataset and problem statement.
Scikit-learn, Pandas, Matplotlib, Seaborn for their powerful libraries in data science and machine learning.
