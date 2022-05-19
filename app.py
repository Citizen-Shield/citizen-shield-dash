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
import pandas as pd
import numpy as np
import seaborn as sns
from jmspack.frequentist_statistics import potential_for_change_index
from jmspack.utils import JmsColors, flatten

app = dash.Dash()

# app = dash.Dash(
#     __name__,
#     assets_external_path='https://raw.githubusercontent.com/jameshtwose/services.jms.rocks/main/static_cv/static_style.css'
# )
# app.scripts.config.serve_locally = True

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
    
    cmap = sns.diverging_palette(5, 250, as_cmap=True)
    # pci_df_html = (all_group_pci_df
    #     .reindex(all_group_pci_df["PCI_All"].abs().sort_values(ascending=False).index)
    # #  .sort_values(by="PCI_All", ascending=False)
    # #  .round(3)
    # .style.background_gradient(cmap, 
    #                             subset=['PCI_All'], 
    #                             axis=1, 
    #                             vmin=-0.15, 
    #                             vmax=0.25)
    # .to_html()
    # )
    
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

    
    return html.Div([
                    dcc.Graph(figure=target_fig),
                    dcc.Graph(figure=lm_fig),
                    dcc.Graph(figure=pci_fig),
                    ])

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
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