from metrics_wind import (winkler_score,
                     empirical_coverage,
                     average_length,
                     median_length,
                     coverage_by_interval_length_plot,
                     error_by_interval_length_plot,
                     interval_length_distribution_plot
                     )

import pandas as pd

alphas = [0.2, 0.1, 0.05, 0.01]

for alpha in alphas:
    df = pd.read_csv(f"Data//df_final_wind_alpha_{alpha}.csv")
    # df = pd.read_csv(f"Data//df_quantile_regression_wind_alpha_{alpha}.csv")
    df['full_date'] = pd.to_datetime(df.full_date)

    df = df.dropna()

    # models = ['cfqra', 'cfqra_by_hour', 'qra', 'qra_by_hour', 
    #           'cqr_cfqra', 'cqr_cfqra_by_hour', 'cqr_qra', 'cqr_qra_by_hour', 
    #           'ucqr_cfqra', 'ucqr_cfqra_by_hour', 'ucqr_qra', 'ucqr_qra_by_hour', 
    #           'aci_cfqra', 'aci_cfqra_by_hour', 'aci_qra', 'aci_qra_by_hour',
    #           'waci_cfqra', 'waci_cfqra_by_hour', 'waci_qra', 'waci_qra_by_hour']

    models_without_by_hour = ['cfqra', 'qra', 
            'cqr_cfqra', 'cqr_qra',
            'aci_cfqra', 'aci_qra',
            'waci_cfqra', 'waci_qra']

    models_without_by_hour = [
              'cfqra',
              'qra']


    df_results = pd.DataFrame(columns=['empirical_coverage', 'average_length', 'median_length', 'winkler_score'])

    for model in models_without_by_hour:
        df_results.loc[model] = [empirical_coverage(df['real'], df[f'preds_sup_{model}'], df[f'preds_inf_{model}']),
                                average_length(df[f'preds_sup_{model}'], df[f'preds_inf_{model}']),
                                median_length(df[f'preds_sup_{model}'], df[f'preds_inf_{model}']),
                                winkler_score(df['real'], df[f'preds_sup_{model}'], df[f'preds_inf_{model}'], alpha)]

    df_results.to_csv(f"Results//df_wind_results_alpha_{alpha}.csv", index=True)

    # models_plot = ['cfqra', 'qra', 
    #         'cqr_cfqra',
    #         'aci_cfqra',
    #         'waci_cfqra']
    
    models_plot = ['aci_qra', 'waci_qra',
            'aci_cfqra',
            'waci_cfqra']

    # models_plot = ['cfqra', 'qra']

    for quantile_step in [0.02, 0.05, 0.1]:
        coverage_by_interval_length_plot(df, models_plot, alpha, quantile_step=quantile_step, bar_width = 0.1)
    error_by_interval_length_plot(df, models_plot, quantile_step=0.1, alpha=alpha, bar_width = 0.1)
    interval_length_distribution_plot(df, models_plot, quantile_step=0.02, alpha=alpha)