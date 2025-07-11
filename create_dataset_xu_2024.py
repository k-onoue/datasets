import os
import pandas as pd
import zipfile
import numpy as np
import requests
from io import StringIO, BytesIO

from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


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
    # try:
    #     print("\n[1/8] Downloading Concrete Data...")
    #     url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data"
    #     df = pd.read_csv(url)
    #     df.columns = df.columns.str.strip()
    #     df.to_csv(os.path.join(output_dir, 'concrete_slump.csv'), index=False)
    #     print(" -> Success: Saved concrete_slump.csv")
    # except Exception as e:
    #     print(f" -> Failed to download Concrete data. Error: {e}")

    try:
        # Concrete Compressive Strength

        # from ucimlrepo import fetch_ucirepo 
        
        # # fetch dataset 
        # concrete_compressive_strength = fetch_ucirepo(id=165) 
        
        # # data (as pandas dataframes) 
        # X = concrete_compressive_strength.data.features 
        # y = concrete_compressive_strength.data.targets 
        
        # # metadata 
        # print(concrete_compressive_strength.metadata) 
        
        # # variable information 
        # print(concrete_compressive_strength.variables) 
        print("\n[1/8] Downloading Concrete Data...")
        concrete = fetch_ucirepo(id=165)
        df_concrete = concrete.data
        df_concrete.to_csv(os.path.join(output_dir, 'concrete_compressive_strength.csv'), index=False)
        print(" -> Success: Saved concrete_compressive_strength.csv")
    except Exception as e:
        print(f" -> Failed to download Concrete data. Error: {e}")

    # --- 2. Boston Housing Data ---
    try:
        print("\n[2/8] Downloading Boston Housing Data...")
        boston = fetch_openml(name='boston', version=1, as_frame=True, parser='liac-arff')
        df_boston = boston.frame
        df_boston.to_csv(os.path.join(output_dir, 'boston_housing.csv'), index=False)
        print(" -> Success: Saved boston_housing.csv")
        print(" -> Note: The Boston Housing dataset has known ethical concerns.")
    except Exception as e:
        print(f" -> Failed to download Boston Housing data. Error: {e}")

    # --- 3. Kin8nm Data ---
    try:
        print("\n[3/8] Downloading Kin8nm Data...")
        kin8nm = fetch_openml(name='kin8nm', version=1, as_frame=True, parser='liac-arff')
        df_kin8nm = kin8nm.frame
        df_kin8nm.to_csv(os.path.join(output_dir, 'kin8nm.csv'), index=False)
        print(" -> Success: Saved kin8nm.csv")
    except Exception as e:
        print(f" -> Failed to download Kin8nm data. Error: {e}")

    # --- 4. Yacht Hydrodynamics Data ---
    try:
        print("\n[4/8] Downloading Yacht Hydrodynamics Data...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
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
        df = pd.read_excel(url, engine='openpyxl')
        df.to_csv(os.path.join(output_dir, 'energy_efficiency.csv'), index=False)
        print(" -> Success: Saved energy_efficiency.csv")
    except Exception as e:
        print(f" -> Failed to download Energy Efficiency data. Error: {e}")

    # --- 6. Elevators Data ---
    try:
        print("\n[6/8] Downloading Elevators Data...")
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
        df = pd.read_csv(url)
        df.to_csv(os.path.join(output_dir, 'protein_structure.csv'), index=False)
        print(" -> Success: Saved protein_structure.csv")
    except Exception as e:
        print(f" -> Failed to download Protein Structure data. Error: {e}")

    # # --- 8. Taxi Trip Fare Data (Kaggle) ---
    # print("\n[8/8] Instructions for Taxi Trip Fare Data:")
    # print(" -> This dataset is from the 'New York City Taxi Fare Prediction' Kaggle competition.")
    # print(" -> Due to its size, it must be downloaded using the Kaggle API.")
    # print(" -> Instructions:")
    # print("    1. Install the Kaggle library: pip install kaggle")
    # print("    2. Go to your Kaggle account, 'Settings' page, and click 'Create New Token'.")
    # print("    3. Place the downloaded 'kaggle.json' file in the required location (e.g., '~/.kaggle/').")
    
    # try:
    #     import kaggle
    #     print("\nAttempting to download Taxi Fare data via Kaggle API...")
    #     kaggle.api.authenticate()
    #     kaggle.api.competition_download_files(
    #         'new-york-city-taxi-fare-prediction',
    #         path=output_dir,
    #         quiet=False
    #     )
    #     zip_path = os.path.join(output_dir, 'new-york-city-taxi-fare-prediction.zip')
    #     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #         zip_ref.extractall(output_dir)
    #     os.remove(zip_path)
    #     print(f" -> Success: Taxi data downloaded and extracted to '{output_dir}/'")
    # except Exception as e:
    #     print(f" -> Kaggle API download failed. Please ensure 'kaggle.json' is set up correctly or download manually. Error: {e}")

    print("\n\nAll tasks complete.")


def split_and_save_dataset(X, y, output_dir, dataset_name, n_splits=10):
    """
    Takes numpy arrays X and y, applies StandardScaler within a k-fold loop, 
    and saves the scaled train/test sets.
    """
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"--> Creating {n_splits} scaled splits for '{dataset_name}'...")
    fold_number = 0
    for train_index, test_index in kf.split(X):
        split_dir = os.path.join(dataset_dir, f'split_{fold_number}')
        os.makedirs(split_dir, exist_ok=True)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_scaled = x_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))

        X_test_scaled = x_scaler.transform(X_test)
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

        np.savetxt(os.path.join(split_dir, 'train_features.csv'), X_train_scaled, delimiter=",")
        np.savetxt(os.path.join(split_dir, 'train_target.csv'), y_train_scaled, delimiter=",")
        np.savetxt(os.path.join(split_dir, 'test_features.csv'), X_test_scaled, delimiter=",")
        np.savetxt(os.path.join(split_dir, 'test_target.csv'), y_test_scaled, delimiter=",")
        
        fold_number += 1
    print(f"--> Finished saving {n_splits} splits for '{dataset_name}'.\n")


def process_dataset(source_file_path, output_dir, dataset_name, target_column, n_splits, create_outliers=False):
    """
    Loads, cleans, optionally adds outliers, and processes a dataset.
    """
    print(f"\n--- PROCESSING DATASET: {dataset_name}{'_Outliers' if create_outliers else ''} ---")
    
    try:
        df = pd.read_csv(source_file_path)
    except FileNotFoundError:
        print(f"--> [ERROR] File Not Found. Please ensure '{source_file_path}' exists.")
        return

    # Standardize column names
    df.columns = [str(col).strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '') for col in df.columns]
    standardized_target = target_column.strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')

    if standardized_target not in df.columns:
        print(f"--> [ERROR] Target column '{target_column}' not found. Skipping.")
        return

    # Select numeric features and specified target
    X_df = df.drop(columns=[standardized_target]).select_dtypes(include=np.number)
    y_s = df[standardized_target]
    
    df_clean = pd.concat([X_df, y_s], axis=1).dropna()
    
    X = df_clean[X_df.columns].values
    y = df_clean[standardized_target].values

    # --- Outlier Injection Logic ---
    if create_outliers:
        print("--> Injecting outliers...")
        # 1. Calculate std dev of the original target variable
        std_dev = np.std(y)
        
        # 2. Determine number of outliers (5% of the data)
        n_outliers = int(len(y) * 0.05)
        
        # 3. Randomly select indices to modify
        np.random.seed(42) # for reproducibility
        outlier_indices = np.random.choice(len(y), n_outliers, replace=False)
        
        # 4. Add 3 * std_dev to the target values at these indices
        y[outlier_indices] += 3 * std_dev
        dataset_name += '_Outliers' # Append suffix to the output directory name

    # --- Call the splitting and saving function ---
    split_and_save_dataset(X, y, output_dir, dataset_name, n_splits)


if __name__ == '__main__':
    source_dir = "source_data_xu_2024"
    output_data_dir = 'dataset_xu_2024'
    num_splits = 5
    
    print("\n" + "="*50)
    print("STEP 1: DOWNLOADING RAW DATASETS")
    print("="*50)
    download_and_save_datasets(output_dir=source_dir)

    datasets_to_process = {
        'Concrete': ('concrete_compressive_strength.csv', 'Concrete compressive strength'),
        'Boston': ('boston_housing.csv', 'MEDV'),
        'Kin8nm': ('kin8nm.csv', 'y'),
        'Yacht': ('yacht_hydrodynamics.csv', 'residuary_resistance'),
        'Energy': ('energy_efficiency.csv', 'Y1'),
        'Elevators': ('elevators.csv', 'Goal'),
        'Protein': ('protein_structure.csv', 'RMSD'),
        # 'Taxi': ('train.csv', 'fare_amount'), 
    }
    
    print("\n" + "="*50)
    print("STEP 2: PROCESSING, SPLITTING, AND SCALING DATASETS")
    print("="*50)
    
    for name, (filename, target) in datasets_to_process.items():
        full_source_path = os.path.join(source_dir, filename)
        process_dataset(full_source_path, output_data_dir, name, target, num_splits)

    # --- Create Outlier Datasets ---
    print("\n" + "="*50)
    print("STEP 3: CREATING OUTLIER DATASETS")
    print("="*50)
    
    # Process Concrete with outliers
    process_dataset(
        source_file_path=os.path.join(source_dir, 'concrete_slump.csv'),
        output_dir=output_data_dir,
        dataset_name='Concrete',
        target_column='SLUMP(cm)',
        n_splits=num_splits,
        create_outliers=True
    )
    
    # Process Kin8nm with outliers
    process_dataset(
        source_file_path=os.path.join(source_dir, 'kin8nm.csv'),
        output_dir=output_data_dir,
        dataset_name='Kin8nm',
        target_column='y',
        n_splits=num_splits,
        create_outliers=True
    )
