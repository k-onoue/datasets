import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import io

# --- Helper Functions for Specific Dataset Processing ---

def process_dat_file(file_path, dataset_name, columns, feature_cols, target_col, output_dir, n_splits):
    """
    Generic function to parse .dat files, apply StandardScaler, split them, and save them.
    """
    print("-" * 20)
    print(f"Processing DAT file for: {dataset_name}")
    
    save_path_base = os.path.join(output_dir, dataset_name)
    os.makedirs(save_path_base, exist_ok=True)

    try:
        with open(file_path, 'r') as f:
            lines = [line for line in f.readlines() if not line.strip().startswith("@")]
        
        data_as_string = "".join(lines)
        string_io = io.StringIO(data_as_string)
        df = pd.read_csv(string_io, header=None, sep='[,\s]+', engine='python')
        if len(df.columns) == len(columns):
            df.columns = columns
        else:
            print(f"--> [WARNING] Column mismatch for {dataset_name}. Skipping.")
            return

        X = df[feature_cols].values
        y = df[target_col].values

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            save_path = os.path.join(save_path_base, f"split_{fold}")
            os.makedirs(save_path, exist_ok=True)
            
            # Get data for the current fold
            train_features, test_features = X[train_index], X[test_index]
            train_target, test_target = y[train_index], y[test_index]

            # Initialize and apply scalers
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            # Fit on training data and transform both train and test data
            train_features_scaled = x_scaler.fit_transform(train_features)
            test_features_scaled = x_scaler.transform(test_features)
            
            train_target_scaled = y_scaler.fit_transform(train_target.reshape(-1, 1))
            test_target_scaled = y_scaler.transform(test_target.reshape(-1, 1))

            np.savetxt(os.path.join(save_path, "train_features.csv"), train_features_scaled, delimiter=",")
            np.savetxt(os.path.join(save_path, "train_target.csv"), train_target_scaled, delimiter=",")
            np.savetxt(os.path.join(save_path, "test_features.csv"), test_features_scaled, delimiter=",")
            np.savetxt(os.path.join(save_path, "test_target.csv"), test_target_scaled, delimiter=",")
        print(f"Successfully processed and saved {dataset_name}.\n")

    except FileNotFoundError:
        print(f"--> [ERROR] File Not Found: {file_path}. Please place it in the 'source_data' directory. Skipping.\n")
    except Exception as e:
        print(f"Failed to process {dataset_name}. Error: {e}\n")


class NealDatasetGenerator:
    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    def make_train_dataset(self, with_input_outliers=False):
        x = self.rng.rand(20, 1) * 10 - 5
        t = np.sin(x) + self.rng.randn(20, 1) * 0.1
        if with_input_outliers:
            x[0] = 20.0 
        return {'x': x, 't': t}
    def make_test_dataset(self):
        x = np.linspace(-7, 7, 100).reshape(-1, 1)
        t = np.sin(x)
        return {'x': x, 't': t}

def generate_neal_dataset(dataset_name, output_dir, n_splits, with_input_outliers=False):
    """
    Generates and saves scaled splits for the Neal synthetic dataset.
    """
    print("-" * 20)
    print(f"Generating synthetic dataset: {dataset_name}")
    save_path_base = os.path.join(output_dir, dataset_name)
    os.makedirs(save_path_base, exist_ok=True)

    for seed in range(n_splits):
        try:
            save_path = os.path.join(save_path_base, f"split_{seed}")
            os.makedirs(save_path, exist_ok=True)

            generator = NealDatasetGenerator(seed=seed)
            train_set = generator.make_train_dataset(with_input_outliers=with_input_outliers)
            test_set = generator.make_test_dataset()

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            
            # Fit on training data and transform both
            train_x_scaled = x_scaler.fit_transform(train_set['x'])
            test_x_scaled = x_scaler.transform(test_set['x'])
            
            train_y_scaled = y_scaler.fit_transform(train_set['t'])
            test_y_scaled = y_scaler.transform(test_set['t'])
            
            np.savetxt(os.path.join(save_path, "train_features.csv"), train_x_scaled, delimiter=",")
            np.savetxt(os.path.join(save_path, "train_target.csv"), train_y_scaled, delimiter=",")
            np.savetxt(os.path.join(save_path, "test_features.csv"), test_x_scaled, delimiter=",")
            np.savetxt(os.path.join(save_path, "test_target.csv"), test_y_scaled, delimiter=",")
        except Exception as e:
            print(f"Failed to generate split {seed} for {dataset_name}. Error: {e}")
    print(f"Successfully generated and saved {dataset_name}.\n")


def process_uci_repo_dataset(fetch_id, dataset_name, output_dir, n_splits, target_col_name=None, subset_logic=None):
    """
    Fetches a dataset from ucimlrepo, applies StandardScaler, splits it, and saves it.
    """
    print("-" * 20)
    print(f"Processing ucimlrepo dataset: {dataset_name}")

    save_path_base = os.path.join(output_dir, dataset_name)
    os.makedirs(save_path_base, exist_ok=True)
    
    try:
        from ucimlrepo import fetch_ucirepo
        repo_data = fetch_ucirepo(id=fetch_id)
        X_df = repo_data.data.features
        y_df = repo_data.data.targets

        X_df = X_df.select_dtypes(include=np.number)

        if target_col_name:
            y_df = y_df[[target_col_name]]

        feature_names = X_df.columns.tolist()
        target_names = y_df.columns.tolist()

        full_df = pd.concat([X_df, y_df], axis=1)
        full_df.dropna(inplace=True)
        
        X_np = full_df[feature_names].values
        y_np = full_df[target_names].values

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold, (train_index, test_index) in enumerate(kf.split(X_np)):
            train_features, test_features = X_np[train_index], X_np[test_index]
            train_target, test_target = y_np[train_index], y_np[test_index]

            if subset_logic:
                train_features, train_target, test_features, test_target = subset_logic(
                    train_features, train_target, test_features, test_target
                )
            
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            train_features_scaled = x_scaler.fit_transform(train_features)
            test_features_scaled = x_scaler.transform(test_features)
            
            train_target_scaled = y_scaler.fit_transform(train_target.reshape(-1, 1))
            test_target_scaled = y_scaler.transform(test_target.reshape(-1, 1))
            
            save_path = os.path.join(save_path_base, f"split_{fold}")
            os.makedirs(save_path, exist_ok=True)
            np.savetxt(os.path.join(save_path, "train_features.csv"), train_features_scaled, delimiter=",")
            np.savetxt(os.path.join(save_path, "train_target.csv"), train_target_scaled, delimiter=",")
            np.savetxt(os.path.join(save_path, "test_features.csv"), test_features_scaled, delimiter=",")
            np.savetxt(os.path.join(save_path, "test_target.csv"), test_target_scaled, delimiter=",")

        print(f"Successfully processed and saved {dataset_name}.\n")

    except Exception as e:
        print(f"Failed to process {dataset_name}. Error: {e}\n")


# --- Main Orchestration Script ---
if __name__ == '__main__':
    # --- Configuration ---
    num_splits = 10
    output_dir = 'dataset_tang_2017_v2'
    source_dat_dir = 'source_data_tang_2017'
    
    print("="*50)
    print("Starting Dataset Generation, Scaling, and Splitting")
    print(f"Output will be saved in: '{output_dir}'")
    print(f"Looking for .dat files in: '{source_dat_dir}'")
    print("="*50 + "\n")

    try:
        import ucimlrepo
    except ImportError:
        print("[FATAL ERROR] 'ucimlrepo' library is not installed.")
        print("Please run 'pip install ucimlrepo' before running this script.")
        exit()

    os.makedirs(source_dat_dir, exist_ok=True)

    # --- 1. Process Datasets from Local .dat files ---
    process_dat_file(
        file_path=os.path.join(source_dat_dir, 'diabetes.dat'),
        dataset_name='Diabetes',
        columns=['Age', 'Deficit', 'C_peptide'],
        feature_cols=['Age', 'Deficit'],
        target_col='C_peptide',
        output_dir=output_dir,
        n_splits=num_splits
    )
    process_dat_file(
        file_path=os.path.join(source_dat_dir, 'machineCPU.dat'),
        dataset_name='Machine_CPU',
        columns=["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP"],
        feature_cols=["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"],
        target_col="PRP",
        output_dir=output_dir,
        n_splits=num_splits
    )
    process_dat_file(
        file_path=os.path.join(source_dat_dir, 'ele-1.dat'),
        dataset_name='ELE',
        columns=["Inhabitants", "Distance", "Length"],
        feature_cols=["Inhabitants", "Distance"],
        target_col="Length",
        output_dir=output_dir,
        n_splits=num_splits
    )

    # --- 2. Generate Synthetic Datasets ---
    generate_neal_dataset('Neal', output_dir, num_splits, with_input_outliers=False)
    generate_neal_dataset('Neal_XOutlier', output_dir, num_splits, with_input_outliers=True)
    
    # --- 3. Process Datasets from ucimlrepo ---
    process_uci_repo_dataset(fetch_id=9, dataset_name='MPG', output_dir=output_dir, n_splits=num_splits)
    
    # MODIFICATION: Removed the subset_logic function definition
    
    # MODIFICATION: Removed subset_logic argument to use the full dataset
    process_uci_repo_dataset(fetch_id=165, dataset_name='Concrete', output_dir=output_dir, n_splits=num_splits)
    
    # MODIFICATION: Removed subset_logic argument to use the full dataset
    process_uci_repo_dataset(fetch_id=275, dataset_name='Bike', output_dir=output_dir, n_splits=num_splits, target_col_name='cnt')

    print("="*50)
    print("All processing tasks complete.")
    print("="*50)