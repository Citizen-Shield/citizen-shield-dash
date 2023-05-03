#%%
import pandas as pd
import plotly.express as px

# %%
df = pd.read_csv("../multi-method-protective-behaviour/data/shield_gjames_21-09-20_prepped.csv", index_col=[0])

# %%
target = "intention_behavior_composite"
df[target] = (df[target] - 10) * -1
features_list = df.filter(regex="^automaticity|attitude|^norms|^risk|^effective").columns.tolist()
df = (df[["demographic_age", "demographic_higher_education"] + features_list + [target]])
# %%
df.head()
# %%
fig = px.box(df[[target]].melt(), x="variable", y="value", points="all")
fig.show()
# %%
# df.to_csv(("data/shield_features_outcome.csv"))
# %%
