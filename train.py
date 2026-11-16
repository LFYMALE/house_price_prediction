import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Constants
DATA_PATH = "data/house_price_regression_dataset.csv"
OUTPUT_MODEL_PATH = "model.bin"
RANDOM_STATE = 42
ALPHA = 100.0
TARGET = "house_price"

FEATURES = [
    "square_footage",
    "num_bedrooms",
    "num_bathrooms",
    "year_built",
    "lot_size",
    "garage_size",
    "neighborhood_quality",
]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error manually."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_data(path: str) -> pd.DataFrame:
    """Load dataset and standardize column names."""
    df = pd.read_csv(path)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def train() -> None:
    print(f"Loading dataset from '{DATA_PATH}'...")
    df = load_data(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET].values

    # Reproduce exact 80/20 full_train/test split from notebook
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"Training final pipeline: StandardScaler + Lasso(alpha={ALPHA})...")
    pipeline = make_pipeline(
        StandardScaler(),
        Lasso(alpha=ALPHA, max_iter=10000, random_state=RANDOM_STATE),
    )
    pipeline.fit(X_full_train, y_full_train)

    # Evaluate on held-out test set
    test_preds = pipeline.predict(X_test)
    test_rmse = rmse(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)

    print("-" * 50)
    print(f"Test RMSE : ${test_rmse:,.2f}")
    print(f"Test MAE  : ${test_mae:,.2f}")
    print(f"Test R2   : {test_r2:.5f}")
    print("-" * 50)

    # Save model artifact along with expected feature order
    artifact = {
        "model": pipeline,
        "features": FEATURES,
        "alpha": ALPHA,
    }
    with open(OUTPUT_MODEL_PATH, "wb") as f_out:
        pickle.dump(artifact, f_out)

    print(f"Model successfully exported to '{OUTPUT_MODEL_PATH}'.")


if __name__ == "__main__":
    train()