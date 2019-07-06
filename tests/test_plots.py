import pytest
import numpy as np
import xgboost as xgb

from xgboost_analyzer import *

@pytest.fixture
def path():
    return [
        (1, Split("foo", -3.5, True), 1.0),
        (2, Split("bar", 1.1, False), 1.0),
        (3, Split("foo", 2.3, False), 1.0),
        (4, Split(None, None, None), 1.0),
    ]

@pytest.fixture
def analyzer():
    model = xgb.XGBRegressor(n_estimators=5)
    model.fit(np.random.randn(100, 10),
              np.random.randn(100))
    return XGBAnalyzer.from_model(model)

def test_plot_path(path):
    """Simple smoke test"""
    plot_path(path)

def test_analyzer_plot_path(analyzer):
    path = analyzer.get_paths(np.random.randn(10))[0]
    plot_path(path)
