# Benchmark Datasets

## 1. Setup

```
pyenv local 3.11
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Create Datasets

### Tang, Qingtao, et al. "Student-t process regression with student-t likelihood." International Joint Conference on Artificial Intelligence 2017. Association for the Advancement of Artificial Intelligence (AAAI), 2017.

1. Install dat files for Tang 2017

From

https://sci2s.ugr.es/keel/category.php?cat=reg

for

- Diabetes
- ELE
- Machine_CPU

2. Execute the following command

```
python3 create_dataset_tang_2017.py
```

### Xu, Jian, and Delu Zeng. "Sparse variational student-t processes." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 14. 2024.

1. Enable kaggle api

2. Execute the following command

```
python3 create_dataset_xu_2024.py
```