# XGBoost Analyzer
Provides a simple interface into the underlying decision trees behind an XGBoost model.

### Features
- XGBAnalyzer
We can analyze construct a python interface into the decision tree like this:

```python
import numpy as np
from xgboost import XGBRegressor
from xgboost_analyzer import XGBAnalyzer
from sklearn.datasets import fetch_california_housing

# use any xgboost compatible dataset
dataset_dict = fetch_california_housing()
X, y = dataset_dict["data"], dataset_dict["target"]
model = XGBRegressor.fit(X, y)

# build the analyzer
analyzer = XGBAnalyzer.from_model(model)
# returns the decision path for each inner tree
paths = analyzer.get_paths(X[0])
print(paths[0])
```

 You'll get something like this
 ```python
 [(0, 0 >= 4.574, -0.03752095627552327),
  (2, 0 >= 5.589, 0.12235774648100714),
  (6, 6 < 37.965, 0.17783592623847694),
  (13, None, 0.182738945)]
 ```

- Plotting
Currently supports plotting waterfall plots in matplotlib and plotly to visualize decision paths.
The contribution of each split is computed as the average of the predictions of each subtree weighted by the number of examples in each split (in other words, the value at each split is computed as the expected value of the prediction for any item that is allocated to that split)

![sample waterfall plot](images/waterfall_sample.png?raw=true)
