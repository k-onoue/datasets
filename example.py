#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_csv.py
  - 再学習あり・モデル非保持の RMSE ベンチマーク
  - CSV で保存された複数データセット / 複数 split に対応
  - 行 = モデル名
    列 = (dataset, mean|std) の DataFrame を出力

必要パッケージ:
    pip install numpy pandas scikit-learn tqdm
"""

import gc
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm


# ------------------------------------------------------------
# 1. DatasetLoader : CSV スプリット版
# ------------------------------------------------------------
class DatasetLoader:
    """
    Expected directory tree
    └─ base_path/
       ├─ DatasetA/
       │   ├─ split_0/
       │   │   ├─ train_features.csv
       │   │   ├─ train_target.csv
       │   │   ├─ test_features.csv
       │   │   └─ test_target.csv
       │   ├─ split_1/
       │   └─ ...
       └─ DatasetB/
           └─ split_0/ ...
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    # ---------- public ----------
    def list_datasets(self) -> list[str]:
        return [
            name for name in os.listdir(self.base_path)
            if (self.base_path / name).is_dir()
        ]

    def list_splits(self, dataset_name: str) -> list[int]:
        ds_path = self.base_path / dataset_name
        splits = []
        for name in os.listdir(ds_path):
            if name.startswith("split_"):
                try:
                    idx = int(name.split("_", 1)[1])
                    splits.append(idx)
                except ValueError:
                    pass
        return sorted(splits)

    def load_split(self, dataset_name: str, split_idx: int):
        split_dir = self.base_path / dataset_name / f"split_{split_idx}"
        X_train = np.loadtxt(split_dir / "train_features.csv", delimiter=",")
        y_train = np.loadtxt(split_dir / "train_target.csv",  delimiter=",")
        X_test  = np.loadtxt(split_dir / "test_features.csv",  delimiter=",")
        y_test  = np.loadtxt(split_dir / "test_target.csv",   delimiter=",")
        return X_train, y_train, X_test, y_test


# ------------------------------------------------------------
# 2. Benchmark
# ------------------------------------------------------------
class Benchmark:
    """
    行 = モデル名
    列 = (dataset, mean|std) の RMSE DataFrame
    """

    def __init__(self, loader: DatasetLoader, model_factories: dict[str, callable]):
        self.loader = loader
        self.model_factories = model_factories
        # model -> dataset -> list[rmse]
        self._rmse_dict = defaultdict(lambda: defaultdict(list))

    # ------------------------
    def evaluate(self) -> pd.DataFrame:
        for dataset in tqdm(self.loader.list_datasets(), desc="Datasets"):
            for split in tqdm(self.loader.list_splits(dataset),
                              leave=False, desc=f"{dataset} splits"):
                X_tr, y_tr, X_te, y_te = self.loader.load_split(dataset, split)

                if X_tr.ndim == 1:
                    X_tr = X_tr.reshape(-1, 1)
                    X_te = X_te.reshape(-1, 1)

                y_tr = y_tr.reshape(-1, 1)
                y_te = y_te.reshape(-1, 1)

                for mname, mfactory in self.model_factories.items():
                    model = mfactory()
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_te)

                    rmse = root_mean_squared_error(y_te, pred)
                    self._rmse_dict[mname][dataset].append(rmse)

                    # メモリ節約：学習済みモデルを即破棄
                    del model
                    gc.collect()

        return self._to_dataframe()

    # ------------------------
    def _to_dataframe(self) -> pd.DataFrame:
        rows = {}
        for mname, ds_dict in self._rmse_dict.items():
            row = {}
            for dset, vals in ds_dict.items():
                row[(dset, "mean")] = np.mean(vals)
                row[(dset, "std")]  = np.std(vals, ddof=1)
            rows[mname] = row

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.columns = pd.MultiIndex.from_tuples(df.columns,
                                               names=["dataset", "metric"])
        return df.sort_index(axis=1)


# ------------------------------------------------------------
# 3. 実行サンプル
# ------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    # ― base path を自分の環境に合わせて変更してください ―
    BASE_PATH = "./datasets"

    # 使いたいモデルをここに追加
    model_factories = {
        "Linear": lambda: LinearRegression(),
        "RF-100": lambda: RandomForestRegressor(n_estimators=100, random_state=0),
        "GBR":    lambda: GradientBoostingRegressor(random_state=0),
    }

    loader = DatasetLoader(BASE_PATH)
    bm = Benchmark(loader, model_factories)

    result_df = bm.evaluate()

    print("\n=== Benchmark Summary ===")
    print(result_df)               # コンソール表示
    result_df.to_csv("benchmark_summary.csv")
    print("\n→ 'benchmark_summary.csv' に書き出しました")
