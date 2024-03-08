import pandas as pd
import numpy as np
import datetime as dt
import os
import warnings

from models import (CornishFisherQuantileRegressionAveraging, 
                    QuantileRegressionAveraging,
                    ConformalizedQuantileRegression,
                    AdaptiveConformalInference,
                    WidthAdaptiveConformalInference,
                    WidthAdaptiveConformalInference_2)

warnings.filterwarnings(action='ignore')

np.random.seed(123)

df = pd.read_csv("Data//df.csv")
df['full_date'] = pd.to_datetime(df.full_date)

df['mean_pred'] = df[['pred1', 'pred2', 'pred3', 'pred4']].mean(axis=1)
df['std_pred'] = df[['pred1', 'pred2', 'pred3', 'pred4']].std(axis=1, ddof=0)

date_test = dt.datetime(2022, 10, 5)
idx_date_test = df[pd.to_datetime(df.date) == date_test].head(1).index.values[0]
idx_first_date_qr = idx_date_test - 6*30*24
m = 4
alphas = [0.01, 0.05, 0.1, 0.2]

for alpha in alphas:
    if not os.path.isfile(f"Data//df_quantile_regression_alpha_{alpha}.csv"):
        # Cornish-Fisher Quantile Regression Averaging
        preds_inf_cfqra, preds_sup_cfqra = CornishFisherQuantileRegressionAveraging(idx_first_date_qr, df, alpha, by_hour=False, save_coefs=True)
        df.loc[df.tail(len(preds_inf_cfqra)).index, 'preds_inf_cfqra'] = preds_inf_cfqra
        df.loc[df.tail(len(preds_sup_cfqra)).index, 'preds_sup_cfqra'] = preds_sup_cfqra

        # Cornish-Fisher Quantile Regression Averaging by hour
        preds_inf_cfqra_by_hour, preds_sup_cfqra_by_hour = CornishFisherQuantileRegressionAveraging(idx_first_date_qr, df, alpha, by_hour=True)
        df.loc[df.tail(len(preds_inf_cfqra_by_hour)).index, 'preds_inf_cfqra_by_hour'] = preds_inf_cfqra_by_hour
        df.loc[df.tail(len(preds_sup_cfqra_by_hour)).index, 'preds_sup_cfqra_by_hour'] = preds_sup_cfqra_by_hour

        # Quantile Regression Averaging
        preds_inf_qra, preds_sup_qra = QuantileRegressionAveraging(idx_first_date_qr, df, alpha, m=m, by_hour=False)
        df.loc[df.tail(len(preds_inf_qra)).index, 'preds_inf_qra'] = preds_inf_qra
        df.loc[df.tail(len(preds_sup_qra)).index, 'preds_sup_qra'] = preds_sup_qra

        # Quantile Regression Averaging by hour
        preds_inf_qra_by_hour, preds_sup_qra_by_hour = QuantileRegressionAveraging(idx_first_date_qr, df, alpha, m=m, by_hour=True)
        df.loc[df.tail(len(preds_inf_qra_by_hour)).index, 'preds_inf_qra_by_hour'] = preds_inf_qra_by_hour
        df.loc[df.tail(len(preds_sup_qra_by_hour)).index, 'preds_sup_qra_by_hour'] = preds_sup_qra_by_hour

        # Save the dataframe (just in case)
        df.to_csv(f"Data//df_quantile_regression_alpha_{alpha}.csv", index=False)
    else:
        # df = pd.read_csv(f"Data//df_quantile_regression_alpha_{alpha}.csv")
        df = pd.read_csv(f"Data//df_final_alpha_{alpha}.csv")
        df['full_date'] = pd.to_datetime(df.full_date)

    df['length_cfqra'] = df['preds_sup_cfqra'] - df['preds_inf_cfqra']
    df['length_cfqra_by_hour'] = df['preds_sup_cfqra_by_hour'] - df['preds_inf_cfqra_by_hour']
    df['length_qra'] = df['preds_sup_qra'] - df['preds_inf_qra']
    df['length_qra_by_hour'] = df['preds_sup_qra_by_hour'] - df['preds_inf_qra_by_hour']

    # Conformalized Quantile Regression over CFQRA
    preds_inf_cqr_cfqra, preds_sup_cqr_cfqra = ConformalizedQuantileRegression(idx_date_test, df, alpha, method='cfqra', by_hour=False)
    df.loc[df.tail(len(preds_inf_cqr_cfqra)).index, 'preds_inf_cqr_cfqra'] = preds_inf_cqr_cfqra
    df.loc[df.tail(len(preds_sup_cqr_cfqra)).index, 'preds_sup_cqr_cfqra'] = preds_sup_cqr_cfqra

    # Conformalized Quantile Regression over CFQRA by hour
    preds_inf_cqr_cfqra_by_hour, preds_sup_cqr_cfqra_by_hour = ConformalizedQuantileRegression(idx_date_test, df, alpha, method='cfqra', by_hour=True)
    df.loc[df.tail(len(preds_inf_cqr_cfqra_by_hour)).index, 'preds_inf_cqr_cfqra_by_hour'] = preds_inf_cqr_cfqra_by_hour
    df.loc[df.tail(len(preds_sup_cqr_cfqra_by_hour)).index, 'preds_sup_cqr_cfqra_by_hour'] = preds_sup_cqr_cfqra_by_hour

    # Conformalized Quantile Regression over QRA
    preds_inf_cqr_qra, preds_sup_cqr_qra = ConformalizedQuantileRegression(idx_date_test, df, alpha, method='qra', by_hour=False)
    df.loc[df.tail(len(preds_inf_cqr_qra)).index, 'preds_inf_cqr_qra'] = preds_inf_cqr_qra
    df.loc[df.tail(len(preds_sup_cqr_qra)).index, 'preds_sup_cqr_qra'] = preds_sup_cqr_qra

    # Conformalized Quantile Regression over QRA by hour
    preds_inf_cqr_qra_by_hour, preds_sup_cqr_qra_by_hour = ConformalizedQuantileRegression(idx_date_test, df, alpha, method='qra', by_hour=True)
    df.loc[df.tail(len(preds_inf_cqr_qra_by_hour)).index, 'preds_inf_cqr_qra_by_hour'] = preds_inf_cqr_qra_by_hour
    df.loc[df.tail(len(preds_sup_cqr_qra_by_hour)).index, 'preds_sup_cqr_qra_by_hour'] = preds_sup_cqr_qra_by_hour

    # Adaptive Conformal Inference over CFQRA
    preds_inf_aci_cfqra, preds_sup_aci_cfqra = AdaptiveConformalInference(idx_date_test, df, alpha, gamma=0.02, method='cfqra', by_hour=False, save_alphas=True)
    df.loc[df.tail(len(preds_inf_aci_cfqra)).index, 'preds_inf_aci_cfqra'] = preds_inf_aci_cfqra
    df.loc[df.tail(len(preds_sup_aci_cfqra)).index, 'preds_sup_aci_cfqra'] = preds_sup_aci_cfqra

    # Adaptive Conformal Inference over CFQRA by hour
    preds_inf_aci_cfqra_by_hour, preds_sup_aci_cfqra_by_hour = AdaptiveConformalInference(idx_date_test, df, alpha, gamma=0.02, method='cfqra', by_hour=True)
    df.loc[df.tail(len(preds_inf_aci_cfqra_by_hour)).index, 'preds_inf_aci_cfqra_by_hour'] = preds_inf_aci_cfqra_by_hour
    df.loc[df.tail(len(preds_sup_aci_cfqra_by_hour)).index, 'preds_sup_aci_cfqra_by_hour'] = preds_sup_aci_cfqra_by_hour

    # Adaptive Conformal Inference over QRA
    preds_inf_aci_qra, preds_sup_aci_qra = AdaptiveConformalInference(idx_date_test, df, alpha, gamma=0.02, method='qra', by_hour=False)
    df.loc[df.tail(len(preds_inf_aci_qra)).index, 'preds_inf_aci_qra'] = preds_inf_aci_qra
    df.loc[df.tail(len(preds_sup_aci_qra)).index, 'preds_sup_aci_qra'] = preds_sup_aci_qra

    # Adaptive Conformal Inference over QRA by hour
    preds_inf_aci_qra_by_hour, preds_sup_aci_qra_by_hour = AdaptiveConformalInference(idx_date_test, df, alpha, gamma=0.02, method='qra', by_hour=True)
    df.loc[df.tail(len(preds_inf_aci_qra_by_hour)).index, 'preds_inf_aci_qra_by_hour'] = preds_inf_aci_qra_by_hour
    df.loc[df.tail(len(preds_sup_aci_qra_by_hour)).index, 'preds_sup_aci_qra_by_hour'] = preds_sup_aci_qra_by_hour
        
    # Width Adaptive Conformal Inference over CFQRA
    preds_inf_waci_cfqra, preds_sup_waci_cfqra = WidthAdaptiveConformalInference(idx_date_test, df, alpha, gamma = 0.02, sigma=3, method='cfqra', by_hour=False, save_alphas=True)
    df.loc[df.tail(len(preds_inf_waci_cfqra)).index, 'preds_inf_waci_cfqra'] = preds_inf_waci_cfqra
    df.loc[df.tail(len(preds_sup_waci_cfqra)).index, 'preds_sup_waci_cfqra'] = preds_sup_waci_cfqra

    # Width Adaptive Conformal Inference over CFQRA by hour
    preds_inf_waci_cfqra_by_hour, preds_sup_waci_cfqra_by_hour = WidthAdaptiveConformalInference(idx_date_test, df, alpha, gamma = 0.02, sigma=3, method='cfqra', by_hour=True)
    df.loc[df.tail(len(preds_inf_waci_cfqra_by_hour)).index, 'preds_inf_waci_cfqra_by_hour'] = preds_inf_waci_cfqra_by_hour
    df.loc[df.tail(len(preds_sup_waci_cfqra_by_hour)).index, 'preds_sup_waci_cfqra_by_hour'] = preds_sup_waci_cfqra_by_hour

    # Width Adaptive Conformal Inference over QRA
    preds_inf_waci_qra, preds_sup_waci_qra = WidthAdaptiveConformalInference(idx_date_test, df, alpha, gamma = 0.02, sigma=3, method='qra', by_hour=False)
    df.loc[df.tail(len(preds_inf_waci_qra)).index, 'preds_inf_waci_qra'] = preds_inf_waci_qra
    df.loc[df.tail(len(preds_sup_waci_qra)).index, 'preds_sup_waci_qra'] = preds_sup_waci_qra

    # Width Adaptive Conformal Inference over QRA by hour
    preds_inf_waci_qra_by_hour, preds_sup_waci_qra_by_hour = WidthAdaptiveConformalInference(idx_date_test, df, alpha, gamma = 0.02, sigma=3, method='qra', by_hour=True)
    df.loc[df.tail(len(preds_inf_waci_qra_by_hour)).index, 'preds_inf_waci_qra_by_hour'] = preds_inf_waci_qra_by_hour
    df.loc[df.tail(len(preds_sup_waci_qra_by_hour)).index, 'preds_sup_waci_qra_by_hour'] = preds_sup_waci_qra_by_hour

    # # Width Adaptive Conformal Inference 2 over CFQRA
    # preds_inf_waci_2_cfqra, preds_sup_waci_2_cfqra = WidthAdaptiveConformalInference_2(idx_date_test, df, alpha, gamma=0.02, method='cfqra', by_hour=False, save_alphas=True, rate=1.02)
    # df.loc[df.tail(len(preds_inf_waci_2_cfqra)).index, 'preds_inf_waci_2_cfqra'] = preds_inf_waci_2_cfqra
    # df.loc[df.tail(len(preds_sup_waci_2_cfqra)).index, 'preds_sup_waci_2_cfqra'] = preds_sup_waci_2_cfqra

    # # Width Adaptive Conformal Inference 2 over CFQRA by hour
    # preds_inf_waci_2_cfqra_by_hour, preds_sup_waci_2_cfqra_by_hour = WidthAdaptiveConformalInference_2(idx_date_test, df, alpha, gamma=0.02, method='cfqra', by_hour=True, rate=1.02)
    # df.loc[df.tail(len(preds_inf_waci_2_cfqra_by_hour)).index, 'preds_inf_waci_2_cfqra_by_hour'] = preds_inf_waci_2_cfqra_by_hour
    # df.loc[df.tail(len(preds_sup_waci_2_cfqra_by_hour)).index, 'preds_sup_waci_2_cfqra_by_hour'] = preds_sup_waci_2_cfqra_by_hour

    # # Width Adaptive Conformal Inference 2 over QRA
    # preds_inf_waci_2_qra, preds_sup_waci_2_qra = WidthAdaptiveConformalInference_2(idx_date_test, df, alpha, gamma=0.02, method='qra', by_hour=False, rate=1.02)
    # df.loc[df.tail(len(preds_inf_waci_2_qra)).index, 'preds_inf_waci_2_qra'] = preds_inf_waci_2_qra
    # df.loc[df.tail(len(preds_sup_waci_2_qra)).index, 'preds_sup_waci_2_qra'] = preds_sup_waci_2_qra

    # # Width Adaptive Conformal Inference 2 over QRA by hour
    # preds_inf_waci_2_qra_by_hour, preds_sup_waci_2_qra_by_hour = WidthAdaptiveConformalInference_2(idx_date_test, df, alpha, gamma=0.02, method='qra', by_hour=True, rate=1.02)
    # df.loc[df.tail(len(preds_inf_waci_2_qra_by_hour)).index, 'preds_inf_waci_2_qra_by_hour'] = preds_inf_waci_2_qra_by_hour
    # df.loc[df.tail(len(preds_sup_waci_2_qra_by_hour)).index, 'preds_sup_waci_2_qra_by_hour'] = preds_sup_waci_2_qra_by_hour

    # Save dataframe
    df.to_csv(f"Data//df_final_alpha_{alpha}.csv", index=False)











