import pandas as pd
import shap
import streamlit as st
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import RepeatedKFold

# shap.initjs()


def run_catboost_var(data, feature_list, outcome, amount_splits, amount_repeats):
    X = data[feature_list]
    y = data[outcome]

    kfold = RepeatedKFold(n_splits=amount_splits, n_repeats=amount_repeats, random_state=42)
    fold_number = 1

    model = CatBoostRegressor(
        iterations=500, depth=None, learning_rate=1, loss_function="RMSE", verbose=False
    )
    all_shap_df = pd.DataFrame()
    all_gini_df = pd.DataFrame()
    # enumerate the splits and summarize the distributions
    for train_ix, test_ix in kfold.split(X):
        # select rows
        train_X, test_X = X.loc[train_ix, :], X.loc[test_ix, :]
        train_y, test_y = y.loc[train_ix], y.loc[test_ix]

        _ = model.fit(X=train_X, y=train_y, cat_features=X.columns.tolist())

        shap_values = model.get_feature_importance(
            Pool(test_X, label=test_y, cat_features=test_X.columns.tolist()),
            type="ShapValues",
        )
        shap_values = shap_values[:, :-1]
        shap_values_df = (
            pd.DataFrame(shap_values, columns=test_X.columns)
            .var()
            .to_frame(name="Shap_Values_Var")
            .T.assign(**{"fold_number": fold_number})
        )

        all_shap_df = pd.concat([all_shap_df, shap_values_df])
        
        fold_number += 1
        
    return all_shap_df
