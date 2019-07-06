import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import plotly.offline as py
    import plotly.graph_objs as go
    PLOTLY_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn("Plotly is not available so plotly "
                  "waterfall plots will be unavailable")
    PLOTLY_AVAILABLE = False

def plot_path(path, title="Waterfall"):
    vals = [0] + [v for _, split, v in path]
    splits = ["init"] + [str(split) for _,split,v in path[:-1]]
    trans = pd.Series(vals[1:], index=splits)
    blank = trans.shift(1).fillna(vals[0])
    trans = trans - blank
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan
    plot = trans.plot(kind='bar', stacked=True,
                      bottom=blank,legend=None,
                      title=title)
    plot.plot(step.index, step.values,'k')
    plot.set_ylim(min(vals) - 1e-2, max(vals) + 1e-2)

if PLOTLY_AVAILABLE:
    def plot_path_plotly(pth, title="Waterfall Plotly"):
        vals = [v for _, split, v in pth]
        after = pd.Series(vals)
        before = after.shift(1).fillna(0)
        transitions = after - before

        trace = go.Waterfall(
            orientation = "v",
            measure = ["relative" for _ in pth],
            x =["init"] + [str(split) for _,split,v in pth[:-1]],
            textposition = "outside",
            y = transitions,
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        )

        layout = go.Layout(
            title=title,
            showlegend=True
        )

        py.iplot(go.Figure([trace], layout),
                filename = "basic_waterfall_chart")
