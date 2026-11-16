# House Price Prediction

## Problem Description

Estimating the market value of a house is a central task in real estate. A price depends on many interacting factors: the size of the living space, the number of rooms, the age of the building, the size of the lot, the presence of a garage, and the quality of the surrounding neighborhood. Manual appraisals are slow and subjective, and buyers and sellers often have only a rough idea of what a property is worth.

This project builds a **supervised machine learning model that predicts the price of a house** from its physical characteristics and the quality of its neighborhood. The target variable, `house_price`, is a continuous monetary value.

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

The project uses the **[Home Value Insights](https://www.kaggle.com/datasets/prokshitha/home-value-insights)** dataset from Kaggle, a regression dataset designed for beginners. It contains **1,000 rows**, each representing one house.

### Columns

Column names are standardized to lowercase during preprocessing.

| Column | Type | Description |
|---|---|---|
| `square_footage` | Numerical | Size of the house in square feet |
| `num_bedrooms` | Numerical (discrete) | Number of bedrooms |
| `num_bathrooms` | Numerical (discrete) | Number of bathrooms |
| `year_built` | Numerical (discrete) | Year the house was built |
| `lot_size` | Numerical | Size of the lot, in acres |
| `garage_size` | Numerical (discrete) | Number of cars that fit in the garage |
| `neighborhood_quality` | Ordinal | Neighborhood quality rating on a scale of 1 to 10 (10 = highest quality) |
| `house_price` | Numerical (continuous) | **Target variable**: price of the house |

All input features are numerical or ordinal, so no categorical encoding is required. `neighborhood_quality` is kept as an ordinal numerical feature.

## Machine Learning Formulation

- **Learning paradigm:** Supervised learning
- **Problem type:** Regression, since the target is a continuous monetary value
- **Input:** A feature vector with the 7 house characteristics listed above
- **Output:** A single predicted value, the estimated `house_price`
- **Feature engineering:** Derived variables (house age, total rooms, square footage per room, building-to-lot ratio) were tested during the EDA but did not improve validation RMSE, so only the 7 original features are used. Features computed from the target (such as price per square foot) are excluded to avoid target leakage. No log transform of the target was needed because its distribution is symmetric.
- **Evaluation metrics:** **RMSE** (penalizes large errors and is expressed in monetary units), **MAE** (average absolute error), and **R²** (proportion of variance explained)
- **Modeling approach:** Six regression algorithms (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting) were compared with a 60% / 20% / 20% train/validation/test split and 5-fold cross-validation, followed by hyperparameter tuning. Cross-validation was used because the dataset is small (1,000 rows) and a single split gives less reliable metrics.

## Expected Outcome

The final deliverable is a trained and validated regression model, exposed as a prediction service, that accepts the characteristics of a house and returns an estimated price. This gives agencies, platforms, buyers, and sellers a practical tool for **price estimation, offer evaluation, and value-driver analysis**.

## Exploratory Data Analysis (EDA) & Key Insights

An in-depth exploratory analysis was conducted in `notebook.ipynb` on the 1,000-row housing dataset:

* **Data Quality:** All 8 columns are numeric (`int64` / `float64`) with **no missing values**, **no duplicate rows**, and **no IQR outliers**.
* **Target Distribution (`house_price`):** Values span from **$111,626.85** to **$1,108,236.84** (median: **$628,267.29**). The distribution is symmetric (`skewness = -0.0640`), making a log transformation unnecessary.
* **Feature Importance & Correlations:** `square_footage` exhibits a near-perfect linear correlation with price (`r = 0.991`) and accounts for **98.6%** of Random Forest feature importance, followed by `year_built`, `lot_size`, and `num_bedrooms`. All VIF scores are ~1.01, confirming no multicollinearity.
* **Feature Selection:** Candidate derived features (`house_age`, `sqft_per_room`, `building_to_lot_ratio`, `total_rooms`) slightly worsened validation RMSE ($20,000.94 with derived features vs. $19,825.27 without), so only the **7 original features** were retained for model training.

## Model Training & Evaluation Results

Six regression algorithms were evaluated using a 60% / 20% / 20% train/validation/test split and 5-fold cross-validation (RMSE as primary metric). Models are sorted by 5-fold CV mean RMSE, from best to worst:

| Model | Validation RMSE ($) | Validation MAE ($) | Validation R² | 5-Fold CV Mean RMSE ($) |
| :--- | :---: | :---: | :---: | :---: |
| **Tuned Lasso (`alpha=100.0`)** | — | — | — | **9,752.39** |
| Lasso (`alpha=1.0`) | 9,343.42 | 7,277.00 | 0.999 | 9,757.42 |
| Linear Regression | 9,343.65 | 7,277.14 | 0.999 | 9,757.50 |
| Tuned Ridge (`alpha=0.001`) | — | — | — | 9,757.50 |
| Ridge (`alpha=1.0`) | 9,295.68 | 7,227.14 | 0.999 | 9,765.62 |
| Tuned Gradient Boosting | — | — | — | 14,147.90 |
| Gradient Boosting (default) | 15,968.70 | 12,073.39 | 0.996 | 16,027.76 |
| Random Forest (default) | 21,636.92 | 16,889.32 | 0.993 | 22,203.90 |
| Decision Tree (default) | 32,781.71 | 26,153.43 | 0.984 | 32,821.38 |

> **Note:** The "Tuned" models were evaluated through cross-validation only during hyperparameter search, so their validation metrics are not reported (—).

Linear Regression, Ridge, and Lasso perform almost identically (the differences are a few dollars), which is consistent with the strong linear relationship between the features and the price and the absence of multicollinearity. The final model was therefore chosen as the one with the lowest CV RMSE. Tree-based models perform clearly worse on this dataset.

* **Final Selected Model:** `StandardScaler` + `Lasso(alpha=100.0)`
* **Held-Out Test Set Metrics:** **RMSE = $10,082.26** | **MAE = $8,190.77** | **R² = 0.99842**

The test set was used **only once**, to evaluate the final selected model, and never for model selection or tuning.

## Deployment & Usage

The trained model is served as a REST API with a `/predict` endpoint. It can be run locally with Python or inside a Docker container.

### 1. Environment Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train and Export the Model

Run `train.py` to train the `StandardScaler` + `Lasso(alpha=100.0)` pipeline, evaluate it on the test split, and save the serialized model to `model.bin`:

```bash
python train.py
```

### 3. Run the Prediction Service

#### Option A: With Docker

Build the image and run the container on port `9696`:

```bash
docker build -t house-price-predictor .
docker run --rm -p 9696:9696 --name house-api house-price-predictor
```

#### Option B: Locally with Python

Start the prediction service directly:

```bash
python predict.py
```

### 4. Test the `/predict` Endpoint

Once the service is running, send a `POST` request with the house features, either with the provided test script:

```bash
python test_service.py
```

or with `curl`:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "square_footage": 2500,
    "num_bedrooms": 4,
    "num_bathrooms": 3,
    "year_built": 2015,
    "lot_size": 2.5,
    "garage_size": 2,
    "neighborhood_quality": 8
  }'
```

The API returns the predicted house price as a JSON response, for example:

```json
{
  "predicted_price": 745123.45
}
```

## Project Structure

```
.
├── data/
│   └── house_price.csv      # Raw dataset (Kaggle)
├── notebook.ipynb           # EDA, model comparison and hyperparameter tuning
├── train.py                 # Trains the final pipeline and saves model.bin
├── predict.py               # Web service exposing the /predict endpoint
├── test_service.py          # Example client that calls the API
├── model.bin                # Serialized model (StandardScaler + Lasso)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container definition for the service
└── README.md
```

## Limitations

- The dataset has only 1,000 rows and appears synthetic: `square_footage` alone explains almost all of the price variance (r = 0.991), and linear models outperform tree-based ones. Real-world performance on actual market data would likely be lower, as real prices also depend on location, condition, and market timing.
- The model has no geographic information beyond the `neighborhood_quality` rating.
- Predictions are only reliable within the range of the training data (roughly $110K to $1.1M).