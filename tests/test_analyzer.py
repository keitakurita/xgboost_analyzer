import pytest
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from xgboost_analyzer import XGBAnalyzer

@pytest.fixture(scope="module")
def xy():
    dataset_dict = fetch_california_housing()
    return dataset_dict["data"], dataset_dict["target"]

@pytest.fixture(scope="module")
def X(xy): return xy[0]

@pytest.fixture(scope="module")
def y(xy): return xy[1]

def test_xgb_regressor(X, y):
    model = xgb.XGBRegressor(n_estimators=10)
    model.fit(X, y)
    analyzer = XGBAnalyzer.from_model(model)
    paths = analyzer.get_paths(X[0])
    assert len(paths) == 10

def test_xgb_classifier(X, y):
    model = xgb.XGBClassifier(n_estimators=15)
    model.fit(X, y >= y.mean())
    analyzer = XGBAnalyzer.from_model(model)
    paths = analyzer.get_paths(X[0])
    assert len(paths) == 15

def test_pandas_df():
    df = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5],
         "b": [3, 4, 5, 1, 7],
         "c": [2, 0, 5, 1, 2],
         "y": [0, 1, 1, 0, 1]}
    )

    model = xgb.XGBClassifier(n_estimators=15)
    model.fit(df.drop("y", axis=1), df["y"])
    analyzer = XGBAnalyzer.from_model(model)
    paths = analyzer.get_paths(df.iloc[0])
    assert len(paths) == 15
