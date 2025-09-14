import numpy as np
import pandas as pd
from pathlib import Path
import kagglehub
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_iris
from ucimlrepo import fetch_ucirepo

# --- Settings ---
# Base directory for saving datasets
BASE_DIR = Path("dataset_classification_toy")
# Number of splits for cross-validation
N_SPLITS = 10
# Random seed for reproducibility
RANDOM_STATE = 42

# --- Dataset Loading and Preprocessing Functions ---

def get_breast_cancer_data():
    """Loads the Wisconsin Breast Cancer dataset."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    y = 1 - y  # Convert labels: malignant to Positive(1), benign to Negative(0)
    return X, y

def get_iris_binary_data():
    """Loads and processes the Iris dataset for binary classification."""
    data = load_iris()
    X, y = data.data, data.target
    mask = (y == 1) | (y == 2)
    X_binary, y_binary = X[mask], y[mask]
    y_binary = np.where(y_binary == 1, 0, 1)  # Convert class 1 to 0 and class 2 to 1
    return X_binary, y_binary

def get_pima_diabetes_data():
    """Loads the Pima Indians Diabetes dataset using kagglehub."""
    print("    > Downloading/Finding Pima Diabetes dataset from Kaggle...")
    # kagglehub.dataset_download returns the path to the EXTRACTED DIRECTORY
    dataset_dir = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
    
    # Construct the full path to the csv file within that directory
    csv_path = Path(dataset_dir) / 'diabetes.csv'
    
    # Read the CSV file directly
    df = pd.read_csv(csv_path)
            
    # Separate features (X) and target (y)
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    
    return X, y

def get_banknote_data():
    """Loads the Banknote Authentication dataset."""
    banknote = fetch_ucirepo(id=267)
    X = banknote.data.features.values
    y = banknote.data.targets.values.ravel()
    return X, y.astype(int)

def get_haberman_data():
    """Loads the Haberman's Survival dataset."""
    haberman = fetch_ucirepo(id=43)
    X = haberman.data.features.values
    y = haberman.data.targets.values.ravel()
    y = np.where(y == 1, 1, 0).astype(int)  # Convert 1 (survived) to 1, 2 (died) to 0
    return X, y

# --- Main Processing ---

def main():
    """Processes each dataset and saves it in the specified format."""
    datasets = {
        "BreastCancer": get_breast_cancer_data,
        "IrisBinary": get_iris_binary_data,
        "PimaDiabetes": get_pima_diabetes_data,
        "Banknote": get_banknote_data,
        "Haberman": get_haberman_data,
    }
    print(f"Saving datasets to '{BASE_DIR}'...")
    for name, load_func in datasets.items():
        print(f"\n--- Processing: {name} ---")
        X, y = load_func()
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"  - split_{i}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            save_dir = BASE_DIR / name / f"split_{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(X_train_scaled).to_csv(save_dir / "train_features.csv", index=False, header=False)
            pd.DataFrame(y_train).to_csv(save_dir / "train_target.csv", index=False, header=False)
            pd.DataFrame(X_test_scaled).to_csv(save_dir / "test_features.csv", index=False, header=False)
            pd.DataFrame(y_test).to_csv(save_dir / "test_target.csv", index=False, header=False)
    print("\nAll processing is complete.")

if __name__ == "__main__":
    main()