import pandas as pd
import numpy as np
import datetime as dt
import os
import warnings

from models_wind import (CornishFisherQuantileRegressionAveraging, 
                    QuantileRegressionAveraging,
                    ConformalizedQuantileRegression,
                    AdaptiveConformalInference,
                    WidthAdaptiveConformalInference)

warnings.filterwarnings(action='ignore')

np.random.seed(123)

df = pd.read_csv("Data//df_wind.csv")
df['full_date'] = pd.to_datetime(df.full_date)

df['mean_pred'] = df[['pred1', 'pred2', 'pred3']].mean(axis=1)
df['std_pred'] = df[['pred1', 'pred2', 'pred3']].std(axis=1, ddof=0)

date_test = dt.datetime(2021, 1, 1)
idx_date_test = df[pd.to_datetime(df.date) == date_test].head(1).index.values[0]
idx_first_date_qr = idx_date_test - 6*30*24
m = 3
alphas = [0.2, 0.1, 0.05, 0.01]

for alpha in alphas:
    if not not os.path.isfile(f"Data//df_quantile_regression_wind_alpha_{alpha}.csv"):
        # Cornish-Fisher Quantile Regression Averaging
        preds_inf_cfqra, preds_sup_cfqra = CornishFisherQuantileRegressionAveraging(idx_first_date_qr, df, alpha, by_hour=False, save_coefs=True)
        df.loc[df.tail(len(preds_inf_cfqra)).index, 'preds_inf_cfqra'] = preds_inf_cfqra
        df.loc[df.tail(len(preds_sup_cfqra)).index, 'preds_sup_cfqra'] = preds_sup_cfqra

        # Quantile Regression Averaging
        preds_inf_qra, preds_sup_qra = QuantileRegressionAveraging(idx_first_date_qr, df, alpha, m=m, by_hour=False, save_coefs=True)
        df.loc[df.tail(len(preds_inf_qra)).index, 'preds_inf_qra'] = preds_inf_qra
        df.loc[df.tail(len(preds_sup_qra)).index, 'preds_sup_qra'] = preds_sup_qra

        # Save the dataframe (just in case)
        df.to_csv(f"Data//df_quantile_regression_wind_alpha_{alpha}.csv", index=False)
    else:
        df = pd.read_csv(f"Data//df_quantile_regression_wind_alpha_{alpha}.csv")
        # df = pd.read_csv(f"Data//df_final_wind_alpha_{alpha}.csv")
        df['full_date'] = pd.to_datetime(df.full_date)

    df['length_cfqra'] = df['preds_sup_cfqra'] - df['preds_inf_cfqra']
    df['length_qra'] = df['preds_sup_qra'] - df['preds_inf_qra']

    # Conformalized Quantile Regression over CFQRA
    preds_inf_cqr_cfqra, preds_sup_cqr_cfqra = ConformalizedQuantileRegression(idx_date_test, df, alpha, method='cfqra', by_hour=False)
    df.loc[df.tail(len(preds_inf_cqr_cfqra)).index, 'preds_inf_cqr_cfqra'] = preds_inf_cqr_cfqra
    df.loc[df.tail(len(preds_sup_cqr_cfqra)).index, 'preds_sup_cqr_cfqra'] = preds_sup_cqr_cfqra

    # Conformalized Quantile Regression over QRA
    preds_inf_cqr_qra, preds_sup_cqr_qra = ConformalizedQuantileRegression(idx_date_test, df, alpha, method='qra', by_hour=False)
    df.loc[df.tail(len(preds_inf_cqr_qra)).index, 'preds_inf_cqr_qra'] = preds_inf_cqr_qra
    df.loc[df.tail(len(preds_sup_cqr_qra)).index, 'preds_sup_cqr_qra'] = preds_sup_cqr_qra

    # Adaptive Conformal Inference over CFQRA
    preds_inf_aci_cfqra, preds_sup_aci_cfqra = AdaptiveConformalInference(idx_date_test, df, alpha, gamma=0.02, method='cfqra', by_hour=False)
    df.loc[df.tail(len(preds_inf_aci_cfqra)).index, 'preds_inf_aci_cfqra'] = preds_inf_aci_cfqra
    df.loc[df.tail(len(preds_sup_aci_cfqra)).index, 'preds_sup_aci_cfqra'] = preds_sup_aci_cfqra

    # Adaptive Conformal Inference over QRA
    preds_inf_aci_qra, preds_sup_aci_qra = AdaptiveConformalInference(idx_date_test, df, alpha, gamma=0.02, method='qra', by_hour=False)
    df.loc[df.tail(len(preds_inf_aci_qra)).index, 'preds_inf_aci_qra'] = preds_inf_aci_qra
    df.loc[df.tail(len(preds_sup_aci_qra)).index, 'preds_sup_aci_qra'] = preds_sup_aci_qra
        
    # Width Adaptive Conformal Inference over CFQRA
    preds_inf_waci_cfqra, preds_sup_waci_cfqra = WidthAdaptiveConformalInference(idx_date_test, df, alpha, gamma = 0.02, sigma=0.5, method='cfqra', by_hour=False)
    df.loc[df.tail(len(preds_inf_waci_cfqra)).index, 'preds_inf_waci_cfqra'] = preds_inf_waci_cfqra
    df.loc[df.tail(len(preds_sup_waci_cfqra)).index, 'preds_sup_waci_cfqra'] = preds_sup_waci_cfqra

    # Width Adaptive Conformal Inference over QRA
    preds_inf_waci_qra, preds_sup_waci_qra = WidthAdaptiveConformalInference(idx_date_test, df, alpha, gamma = 0.02, sigma=0.5, method='qra', by_hour=False)
    df.loc[df.tail(len(preds_inf_waci_qra)).index, 'preds_inf_waci_qra'] = preds_inf_waci_qra
    df.loc[df.tail(len(preds_sup_waci_qra)).index, 'preds_sup_waci_qra'] = preds_sup_waci_qra

    # Save dataframe
    df.to_csv(f"Data//df_final_wind_alpha_{alpha}.csv", index=False)











