#  Dashboard specific packages
import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State
# import dash_core_components as dcc
from dash import dcc
from dash import html
# import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
# import dash_table

# Analyses/ utils packages
import pandas as pd
import numpy as np
import seaborn as sns
from jmspack.frequentist_statistics import potential_for_change_index
from jmspack.utils import JmsColors, flatten
from catboost import CatBoostRegressor, Pool

def naive_catboost_shap(df: pd.DataFrame,
                                 grouping_var: str,
                                 column_list: list,
                                 ):
    y = df[grouping_var]
    X = df[column_list]

    model = CatBoostRegressor(iterations=500,
                               depth=None,
                               learning_rate=1,
                               loss_function='RMSE',
                               verbose=False)

    # train the model
    _ = model.fit(X, y, cat_features=column_list)

    shap_values = model.get_feature_importance(Pool(X, label=y,cat_features=X.columns.tolist()), type="ShapValues")

    shap_values = shap_values[:,:-1]
    
    tmp_actual = (X
     .melt(value_name='actual_value')
    )

    tmp_shap = (pd.DataFrame(shap_values, columns=column_list)
     .melt(value_name='shap_value')
    )

    shap_actual_df = pd.concat([tmp_actual, tmp_shap[["shap_value"]]], axis=1)

    return shap_actual_df

app = dash.Dash()

server = app.server

app.layout = html.Div([
    html.H1("Multi-Method Dashboard"),
    html.H4("Please upload your dataset"),
dcc.Upload(
        id='upload-data',
        children=html.Div([
        'Drag and Drop or ',
        html.A('Select Files')
        ]),
        style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px'
         },
        # Allow multiple files to be uploaded
        multiple=True
),
dcc.Input(id="target", type="text", placeholder="Please Type In Target Variable", style={'marginRight':'10px'}),
html.Div(id='output-data-upload'),
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
        # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')),
                sep=","
                )
        elif 'xls' in filename:
        # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    target = "intention_behavior_composite"
    
    df[target] = (df[target] - 10) * -1
    features_list = df.filter(regex="^automaticity|attitude|^norms|^risk|^effective").columns.tolist()
    df = (df[["demographic_age", "demographic_higher_education"] + features_list + [target]])
    
    target_fig = px.box(df[[target]].melt(), x="variable", y="value", points="all")
    
    lm_fig = px.scatter(
        data_frame=df, x=target, y=features_list[0], opacity=0.65,
        color="demographic_age",
        trendline='ols', 
        # trendline_color_override='darkblue'
        )
    
    all_group_pci_df = potential_for_change_index(data=df.drop(["demographic_age", "demographic_higher_education"], axis=1),
                                           features_list=features_list,
                                            target=target,
                                            minimum_measure = 'min',
                                            centrality_measure = 'mean',
                                            maximum_measure = 'max',
                                            weight_measure = 'r-value',
                                            scale_data = True,
                                            pci_heatmap = False,)
    all_group_pci_df = all_group_pci_df.rename(columns={"PCI": "PCI_All"})
    
    # cmap = sns.diverging_palette(5, 250, as_cmap=True)
    
    # Sort values on absolute PCI
    
    pci_df=(all_group_pci_df
        .reindex(all_group_pci_df["PCI_All"].abs().sort_values(ascending=False).index)
     .sort_values(by="PCI_All", ascending=False)
     .round(3)
     .reset_index()
     .rename(columns={"index": "variable name"})
     )
    
    pci_fig = go.Figure(data=[go.Table(
        columnwidth = [400] + list(np.repeat(50, repeats=pci_df.shape[1]-1)),
    header=dict(values=list(pci_df.columns),
                fill_color=JmsColors.PURPLE,
                align='center',
                font=dict(color='white', size=12)),
    cells=dict(values=[pci_df["variable name"],
                        pci_df["PCI_All"], 
                       pci_df["min"], 
                       pci_df["mean"],
                       pci_df["max"],
                       pci_df["r-value"],
                       pci_df["p-value"],
                       ],
               fill_color=JmsColors.YELLOW,
               align='center'))
                ])
    
    display_length = 5
    shap_df = naive_catboost_shap(df = df,
                    grouping_var = target,
                    column_list = features_list,
                    plot_title="All",
                   max_display=display_length)
    var_order = shap_df.groupby("variable").var().sort_values(by = "shap_value", ascending = False).index.tolist()
    shap_fig = px.strip(data_frame=shap_df.assign(**{"actual_value": lambda d: d["actual_value"].astype(float)}), 
                    x="shap_value", 
                    y="variable",
                    color="actual_value",
                  category_orders={"variable":var_order,
                                   "actual_value": shap_df["actual_value"].sort_values().unique().tolist()},
                  color_discrete_sequence=px.colors.sequential.Plasma_r,
                  height=800
                    )

    dets_df = pd.DataFrame({"Potential_For_Change_Index": pci_df["variable name"].head(display_length).tolist(), 
                            "CatBoost_Shap": var_order[0:display_length]})
    dets_fig = go.Figure(data=[go.Table(
        columnwidth = [400, 400],
    header=dict(values=list(dets_df.columns),
                fill_color=JmsColors.PURPLE,
                align='center',
                font=dict(color='white', size=12)),
    cells=dict(values=[dets_df["Potential_For_Change_Index"],
                        dets_df["CatBoost_Shap"], 
                       ],
               fill_color=JmsColors.YELLOW,
               align='center'))
                ])
    
    return html.Div([
                    html.H1("Top Determinants"),
                    dcc.Graph(figure=dets_fig),
                    html.H1("Distribution of Target"),
                    dcc.Graph(figure=target_fig),
                    html.H1("Regression Plot Between Target and Example Feature"),
                    dcc.Graph(figure=lm_fig),
                    html.H1("Potential For Change Index"),
                    dcc.Graph(figure=pci_fig),
                    html.H1("Catboost Shapley Values"),
                    dcc.Graph(figure=shap_fig),
                    ])

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
            #    Input('target', 'value')
               ],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__ == '__main__':
    app.run_server(debug=True)