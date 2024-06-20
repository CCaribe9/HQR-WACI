from metrics import (winkler_score,
                     empirical_coverage,
                     average_length,
                     median_length,
                     coverage_by_interval_length_plot,
                     error_by_interval_length_plot,
                     interval_length_distribution_plot,
                     coverage_by_hour_plot)

import pandas as pd

alphas = [0.2, 0.1, 0.05, 0.01]
# alphas=[0.1]

for alpha in alphas:
    df = pd.read_csv(f"Data//df_final_alpha_{alpha}.csv")
    df['full_date'] = pd.to_datetime(df.full_date)

    df = df.dropna()

    models = ['hqr', 'hqr_by_hour', 'qra', 'qra_by_hour', 
              'cqr_hqr', 'cqr_hqr_by_hour', 'cqr_qra', 'cqr_qra_by_hour', 
              'ucqr_hqr', 'ucqr_hqr_by_hour', 'ucqr_qra', 'ucqr_qra_by_hour', 
              'aci_hqr', 'aci_hqr_by_hour', 'aci_qra', 'aci_qra_by_hour',
              'waci_hqr', 'waci_hqr_by_hour', 'waci_qra', 'waci_qra_by_hour']

    models_by_hour = ['hqr_by_hour', 'qra_by_hour',
            'cqr_hqr_by_hour', 'cqr_qra_by_hour',
            'aci_hqr_by_hour', 'aci_qra_by_hour',
            'waci_hqr_by_hour', 'waci_qra_by_hour']

    models_without_by_hour = ['hqr', 'qra', 
            'cqr_hqr', 'cqr_qra', 
            'aci_hqr', 'aci_qra',
            'waci_hqr', 'waci_qra']

    # models_without_by_hour = ['hqr', 'aci_hqr',
    #         'waci_hqr', 'waci_2_hqr']

#     # models_without_by_hour = [
#     #           'aci_hqr',
#     #           'waci_hqr']


    df_results = pd.DataFrame(columns=['empirical_coverage', 'average_length', 'median_length', 'winkler_score'])

    for model in models_without_by_hour:
        df_results.loc[model] = [empirical_coverage(df['real'], df[f'preds_sup_{model}'], df[f'preds_inf_{model}']),
                                average_length(df[f'preds_sup_{model}'], df[f'preds_inf_{model}']),
                                median_length(df[f'preds_sup_{model}'], df[f'preds_inf_{model}']),
                                winkler_score(df['real'], df[f'preds_sup_{model}'], df[f'preds_inf_{model}'], alpha)]

    df_results.to_csv(f"Results//df_results_alpha_{alpha}.csv", index=True)

    models_plot = ['hqr', 'qra', 
            'cqr_hqr',
            'aci_hqr',
            'waci_hqr'
            ]
    
    # models_plot = ['hqr',
    #         'aci_hqr',
    #         'waci_hqr'
    #         ]
    
    

    for quantile_step in [0.05, 0.1]:
    # for quantile_step in [0.05]:
        coverage_by_interval_length_plot(df, models_plot, alpha, quantile_step=quantile_step, bar_width = 0.1)
    error_by_interval_length_plot(df, models_plot, quantile_step=0.1, alpha=alpha, bar_width = 0.1)
    interval_length_distribution_plot(df, models_plot, quantile_step=0.02, alpha=alpha)
    coverage_by_hour_plot(df, models_plot, alpha, bar_width = 0.1)
