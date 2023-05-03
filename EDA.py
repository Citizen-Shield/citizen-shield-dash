# %%
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

# %%
df = pd.read_csv(
    "../multi-method-protective-behaviour/data/shield_gjames_21-09-20_prepped.csv",
    index_col=[0],
)

# %%
target = "intention_behavior_composite"
df[target] = (df[target] - 10) * -1
features_list = df.filter(
    regex="^automaticity|attitude|^norms|^risk|^effective"
).columns.tolist()
df = df[["demographic_age", "demographic_higher_education"] + features_list + [target]]
# %%
df.head()
# %%
# fig = px.box(df[[target]].melt(), y="variable", x="value", points="all")
# fig.show()
# %%
# df.to_csv(("data/shield_features_outcome.csv"))
# %%
px.box(
    df[features_list]
    .melt()
    .assign(**{"value": lambda x: x["value"].add(np.random.rand(x["value"].shape[0])*0.75)}),
    x="value",
    y="variable",
    # height=1000,
    # stripmode="overlay",
    points="all",
    boxmode="overlay",
    color_discrete_sequence=["rgba(184, 184, 184, 0.25)"],
)

# %%
def linear_regression_r2(df, features_list, target):
    X = df[features_list]
    y = df[target]
    reg = LinearRegression().fit(X, y)
    return reg.score(X, y)
# %%
df[features_list].apply(lambda x: linear_regression_r2(df, [x.name], target), axis=0)
# %%
linear_regression_r2(df, features_list=[features_list[0]], target=target)
# %%
[features_list[0]]
# %%
