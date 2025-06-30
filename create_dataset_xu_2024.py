import os
import pandas as pd
import zipfile
import numpy as np
import requests
from io import StringIO, BytesIO

from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def download_and_save_datasets(output_dir='temp'):
    """
    Downloads, processes, and saves the datasets used in the
    "Sparse Variational Student-t Processes" paper.
    """
    # --- Configuration ---
    # Create a directory to store the datasets
    os.makedirs(output_dir, exist_ok=True)
    print(f"Datasets will be saved in the '{output_dir}/' directory.")

    # --- 1. Concrete Slump Test Data ---
    try:
        print("\n[1/8] Downloading Concrete Data...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data"
        # The file is comma-separated and has a header on the first line
        df = pd.read_csv(url)
        # Clean up column names by removing extra spaces
        df.columns = df.columns.str.strip()
        df.to_csv(os.path.join(output_dir, 'concrete_slump.csv'), index=False)
        print(" -> Success: Saved concrete_slump.csv")
    except Exception as e:
        print(f" -> Failed to download Concrete data. Error: {e}")

    # --- 2. Boston Housing Data ---
    try:
        print("\n[2/8] Downloading Boston Housing Data...")
        # The original UCI dataset is deprecated. We fetch a processed version from OpenML.
        boston = fetch_openml(name='boston', version=1, as_frame=True, parser='liac-arff')
        df_boston = boston.frame
        # The target column in this version is named 'MEDV'
        df_boston.to_csv(os.path.join(output_dir, 'boston_housing.csv'), index=False)
        print(" -> Success: Saved boston_housing.csv")
        print(" -> Note: The Boston Housing dataset has known ethical concerns.")
    except Exception as e:
        print(f" -> Failed to download Boston Housing data. Error: {e}")

    # --- 3. Kin8nm Data ---
    try:
        print("\n[3/8] Downloading Kin8nm Data...")
        # This dataset is available on OpenML
        kin8nm = fetch_openml(name='kin8nm', version=1, as_frame=True, parser='liac-arff')
        df_kin8nm = kin8nm.frame
        # The target column in this version is named 'y'
        df_kin8nm.to_csv(os.path.join(output_dir, 'kin8nm.csv'), index=False)
        print(" -> Success: Saved kin8nm.csv")
    except Exception as e:
        print(f" -> Failed to download Kin8nm data. Error: {e}")

    # --- 4. Yacht Hydrodynamics Data ---
    try:
        print("\n[4/8] Downloading Yacht Hydrodynamics Data...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
        # The data is space-separated and has no header
        column_names = [
            'longitudinal_pos', 'prismatic_coeff', 'length_displacement_ratio',
            'beam_draught_ratio', 'length_beam_ratio', 'froude_number',
            'residuary_resistance'
        ]
        df = pd.read_csv(url, delim_whitespace=True, header=None, names=column_names)
        df.to_csv(os.path.join(output_dir, 'yacht_hydrodynamics.csv'), index=False)
        print(" -> Success: Saved yacht_hydrodynamics.csv")
    except Exception as e:
        print(f" -> Failed to download Yacht data. Error: {e}")

    # --- 5. Energy Efficiency Data ---
    try:
        print("\n[5/8] Downloading Energy Efficiency Data...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
        # The data is in an Excel file
        df = pd.read_excel(url, engine='openpyxl')
        df.to_csv(os.path.join(output_dir, 'energy_efficiency.csv'), index=False)
        print(" -> Success: Saved energy_efficiency.csv")
    except Exception as e:
        print(f" -> Failed to download Energy Efficiency data. Error: {e}")

    # --- 6. Elevators Data ---
    try:
        print("\n[6/8] Downloading Elevators Data...")
        # This dataset is available on OpenML
        elevators = fetch_openml(name='elevators', version=1, as_frame=True, parser='liac-arff')
        df_elevators = elevators.frame
        df_elevators.to_csv(os.path.join(output_dir, 'elevators.csv'), index=False)
        print(" -> Success: Saved elevators.csv")
    except Exception as e:
        print(f" -> Failed to download Elevators data. Error: {e}")

    # --- 7. Protein Tertiary Structure Data ---
    try:
        print("\n[7/8] Downloading Protein Structure Data...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
        # The data is in a CSV file with a header
        df = pd.read_csv(url)
        df.to_csv(os.path.join(output_dir, 'protein_structure.csv'), index=False)
        print(" -> Success: Saved protein_structure.csv")
    except Exception as e:
        print(f" -> Failed to download Protein Structure data. Error: {e}")

    # --- 8. Taxi Trip Fare Data (Kaggle) ---
    print("\n[8/8] Instructions for Taxi Trip Fare Data:")
    print(" -> This dataset is from the 'New York City Taxi Fare Prediction' Kaggle competition.")
    print(" -> Due to its size, it must be downloaded using the Kaggle API.")
    print(" -> Instructions:")
    print("    1. Install the Kaggle library: pip install kaggle")
    print("    2. Go to your Kaggle account, 'Settings' page, and click 'Create New Token'.")
    print("    3. Place the downloaded 'kaggle.json' file in the required location (e.g., '~/.kaggle/').")
    
    # Check for Kaggle API setup and attempt download
    try:
        import kaggle
        print("\nAttempting to download Taxi Fare data via Kaggle API...")
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(
            'new-york-city-taxi-fare-prediction',
            path=output_dir,
            quiet=False
        )
        zip_path = os.path.join(output_dir, 'new-york-city-taxi-fare-prediction.zip')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(zip_path)
        print(f" -> Success: Taxi data downloaded and extracted to '{output_dir}/'")
    except Exception as e:
        print(f" -> Kaggle API download failed. Please ensure 'kaggle.json' is set up correctly or download manually. Error: {e}")

    print("\n\nAll tasks complete.")


def split_and_save_dataset(source_file_path, output_dir, dataset_name, target_column, n_splits=10):
    """
    Loads a single dataset, applies StandardScaler, performs k-fold splitting, 
    and saves the scaled train/test sets for each fold.
    """
    # --- 1. Create Output Directory for the specific dataset ---
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Directory for '{dataset_name}' is ready at '{dataset_dir}'")

    # --- 2. Load the Dataset ---
    print(f"--> Attempting to load data from: '{source_file_path}'")
    try:
        df = pd.read_csv(source_file_path)
        print(f"--> Successfully loaded with {len(df)} rows.")
    except FileNotFoundError:
        print(f"--> [ERROR] File Not Found. Please ensure '{source_file_path}' exists.")
        print(f"--> SKIPPING '{dataset_name}'.\n")
        return
    except Exception as e:
        print(f"--> [ERROR] Could not read the file. Reason: {e}")
        print(f"--> SKIPPING '{dataset_name}'.\n")
        return

    # --- 3. Clean and Separate Data ---
    original_columns = list(df.columns)
    df.columns = [str(col).strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '') for col in df.columns]
    standardized_target = target_column.strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')

    if standardized_target not in df.columns:
        print(f"--> [ERROR] Target column '{target_column}' (standardized to '{standardized_target}') was not found.")
        print(f"--> Original columns were: {original_columns}")
        print(f"--> Standardized columns are: {list(df.columns)}")
        print(f"--> SKIPPING '{dataset_name}'.\n")
        return

    # Select only numeric feature columns
    X_df = df.drop(columns=[standardized_target]).select_dtypes(include=np.number)
    y_s = df[standardized_target]
    
    # Re-align X and y after dropping non-numeric cols and before dropping NaNs
    df_clean = pd.concat([X_df, y_s], axis=1).dropna()
    
    X = df_clean[X_df.columns].values
    y = df_clean[standardized_target].values

    # --- 4. Set up, Perform K-Fold Cross-Validation, and Scale Data ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"--> Creating {n_splits} scaled splits for '{dataset_name}'...")
    fold_number = 0
    for train_index, test_index in kf.split(X):
        split_dir = os.path.join(dataset_dir, f'split_{fold_number}')
        os.makedirs(split_dir, exist_ok=True)

        # Get the unscaled data for the current fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize scalers
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        # Fit the scalers ONLY on the training data
        X_train_scaled = x_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))

        # Use the FITTED scalers to transform the test data
        X_test_scaled = x_scaler.transform(X_test)
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

        # Save the SCALED data
        np.savetxt(os.path.join(split_dir, 'train_features.csv'), X_train_scaled, delimiter=",")
        np.savetxt(os.path.join(split_dir, 'train_target.csv'), y_train_scaled, delimiter=",")
        np.savetxt(os.path.join(split_dir, 'test_features.csv'), X_test_scaled, delimiter=",")
        np.savetxt(os.path.join(split_dir, 'test_target.csv'), y_test_scaled, delimiter=",")
        
        fold_number += 1
    print(f"--> Finished saving {n_splits} splits for '{dataset_name}'.\n")


if __name__ == '__main__':
    # Directory to save the raw downloaded data
    source_dir = "source_data_xu_2024"
    
    # First, download all the raw datasets
    download_and_save_datasets(output_dir=source_dir)

    # Directory to save the final, split, and scaled datasets
    output_data_dir = 'dataset_xu_2024'

    # The paper used 5-fold cross-validation. Set n_splits to 5.
    num_splits = 5
    
    # Dictionary mapping dataset names to their filenames and original target columns.
    datasets_to_process = {
        'Concrete': ('concrete_slump.csv', 'SLUMP(cm)'),
        'Boston': ('boston_housing.csv', 'MEDV'),
        'Kin8nm': ('kin8nm.csv', 'y'),
        'Yacht': ('yacht_hydrodynamics.csv', 'residuary_resistance'),
        'Energy': ('energy_efficiency.csv', 'Y1'),
        'Elevators': ('elevators.csv', 'Goal'),
        'Protein': ('protein_structure.csv', 'RMSD'),
        'Taxi': ('train.csv', 'fare_amount') 
    }
    
    # --- Main Loop to process and split each dataset ---
    print("\n" + "="*50)
    print("STARTING DATA SPLITTING AND SCALING")
    print("="*50)
    for name, (filename, target) in datasets_to_process.items():
        print(f"\n--- PROCESSING DATASET: {name} ---")
        
        full_source_path = os.path.join(source_dir, filename)
        
        split_and_save_dataset(
            source_file_path=full_source_path,
            output_dir=output_data_dir,
            dataset_name=name,
            target_column=target,
            n_splits=num_splits
        )
