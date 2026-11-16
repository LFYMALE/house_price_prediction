# House Price Prediction

## Problem Description

Estimating the market value of a house is a central task in real estate. A price depends on many interacting factors: the size of the living space, the number of rooms, the age of the building, the size of the lot, the presence of a garage, and the quality of the surrounding neighborhood. Manual appraisals are slow and subjective, and buyers and sellers often have only a rough idea of what a property is worth.

This project builds a **supervised machine learning model that predicts the price of a house** from its physical characteristics and the quality of its neighborhood. The target variable, `House_Price`, is a continuous monetary value.

Given details such as square footage, number of bedrooms and bathrooms, year of construction, lot size, garage size, and neighborhood quality, the model returns an estimated price. This turns the question "How much is this house worth?" into a fast, data-driven answer.

## Business Context & Use Cases

### Target Audience

| Stakeholder | Need |
|---|---|
| **Real estate agencies** | Quickly estimate a fair listing price and advise clients |
| **Online property platforms** | Display automated price estimates and flag over- or under-priced listings |
| **Home buyers** | Check whether an asking price is reasonable before making an offer |
| **Home sellers and investors** | Understand which features drive value and where renovation pays off |

### How the Solution Will Be Used

1. **Automated price estimation (agencies and platforms).**
   When a property is listed, the model produces an instant baseline price from its characteristics. Agents can use it as a starting point for valuation, and platforms can compare it to the asking price to highlight listings that look under- or over-priced.

2. **Offer evaluation (buyers).**
   Buyers can enter the characteristics of a house they are considering and compare the predicted price with the asking price. This supports more informed negotiation and reduces the risk of overpaying.

3. **Value-driver analysis (sellers and investors).**
   By varying one feature at a time (for example, adding a garage space or a bathroom), users can see how much each feature is expected to change the price. This helps prioritize improvements and evaluate investment opportunities.

### Why It Matters

- **For businesses:** Faster and more consistent valuations reduce manual work and improve the accuracy of listings.
- **For buyers and sellers:** Greater price transparency leads to better-informed decisions.
- **For both:** A data-driven model replaces rough rules of thumb with estimates learned from historical property data.

## Dataset Overview

The project uses the **Home Value Insights** dataset from Kaggle ([add the link here]), a regression dataset designed for beginners. It contains **1,000 rows**, each representing one house.

### Columns

| Column | Type | Description |
|---|---|---|
| `Square_Footage` | Numerical | Size of the house in square feet |
| `Num_Bedrooms` | Numerical (discrete) | Number of bedrooms |
| `Num_Bathrooms` | Numerical (discrete) | Number of bathrooms |
| `Year_Built` | Numerical (discrete) | Year the house was built |
| `Lot_Size` | Numerical | Size of the lot, in acres |
| `Garage_Size` | Numerical (discrete) | Number of cars that fit in the garage |
| `Neighborhood_Quality` | Ordinal | Neighborhood quality rating on a scale of 1 to 10 (10 = highest quality) |
| `House_Price` | Numerical (continuous) | **Target variable**: price of the house |

All input features are numerical or ordinal, so no categorical encoding is required. `Neighborhood_Quality` is kept as an ordinal numerical feature, though one-hot encoding could be tested during modeling.

## Machine Learning Formulation

- **Learning paradigm:** Supervised learning
- **Problem type:** Regression, since the target is a continuous monetary value
- **Input:** A feature vector with the 7 house characteristics listed above
- **Output:** A single predicted value, the estimated `House_Price`
- **Feature engineering (optional):** Derived variables such as the house age (`current year - Year_Built`) may be tested. Features computed from the target (such as price per square foot) are not used for training, to avoid target leakage. Transformations of the target (for example, a log transform) will be evaluated during the EDA if the price distribution is skewed
- **Evaluation metrics:** **RMSE** (penalizes large errors and is expressed in monetary units), **MAE** (average absolute error), and **R²** (proportion of variance explained)
- **Modeling approach:** Several regression models (for example, Linear Regression, Random Forest, Gradient Boosting) are compared using a train/validation/test split. Because the dataset is small (1,000 rows), cross-validation is also used to obtain more reliable metrics, along with hyperparameter tuning to select the best performer

## Expected Outcome

The final deliverable is a trained and validated regression model, exposed as a prediction service, that accepts the characteristics of a house and returns an estimated price. This gives agencies, platforms, buyers, and sellers a practical tool for **price estimation, offer evaluation, and value-driver analysis**.
