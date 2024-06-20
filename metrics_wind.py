import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

color_by_method_dict = {'aci_qra': 'C0', 'aci_hqr': 'C1', 'waci_qra': 'C2', 'waci_hqr': 'C3'}
# color_by_method_dict = {'hqr': 'C0', 'qra': 'C1'}
label_by_method_dict = {'aci_qra': 'ACI-QRA', 'aci_hqr': 'ACI-HQR', 'waci_qra': 'WACI-QRA', 'waci_hqr': 'WACI-HQR'}
# label_by_method_dict = {'hqr': 'HQR', 'qra': 'QRA'}


def winkler_score(y_true, upper_pred, lower_pred, alpha):
    lower_diff = lower_pred - y_true
    upper_diff = y_true - upper_pred
    interval_length = upper_pred - lower_pred

    lower_score = interval_length + (2 / alpha) * np.maximum(lower_diff, 0)
    upper_score = interval_length + (2 / alpha) * np.maximum(upper_diff, 0)

    score = np.where(y_true < lower_pred, lower_score, np.where(y_true > upper_pred, upper_score, interval_length))

    return np.mean(score)

def empirical_coverage(y_true, upper_pred, lower_pred):
    return np.where((y_true >= lower_pred) & (y_true <= upper_pred), True, False).mean()*100

def average_length(upper_pred, lower_pred):
    return np.mean(upper_pred - lower_pred)

def median_length(upper_pred, lower_pred):
    return np.median(upper_pred - lower_pred)

def coverage_by_interval_length_plot(df, models, alpha, quantile_step, bar_width = 0.1):
    df_coverage_by_interval_length = pd.DataFrame(columns=['mean_abs_deviation', 'max_abs_deviation'])
    obj_coverage = (1-alpha)*100

    for model in models:
        df['length_' + model] = df[f'preds_sup_{model}'] - df[f'preds_inf_{model}']
    dict_lengths = {model: [np.quantile(df[f'length_{model}'], i) for i in np.arange(0, 1.01, quantile_step)] for model in models}
    dict_coverages = {model: [] for model in models}
    labels = []
    for i_quantile in range(len(dict_lengths[models[0]])-1):
        labels.append(str((round(np.arange(0, 1.01, quantile_step)[i_quantile], 2), round(np.arange(0, 1.01, quantile_step)[i_quantile+1], 2))))
        for model in models:
            length_min_model, length_max_model = dict_lengths[model][i_quantile], dict_lengths[model][i_quantile+1]
            df_aux_model = df[(df[f'length_{model}'] >= length_min_model) & (df[f'length_{model}'] <= length_max_model)]
            dict_coverages[model].append(empirical_coverage(df_aux_model['real'], df_aux_model[f'preds_sup_{model}'], df_aux_model[f'preds_inf_{model}']))
    
    for model in models:
        df_coverage_by_interval_length.loc[model] = [np.mean(np.abs(np.array(dict_coverages[model]) - obj_coverage)), np.max(np.abs(np.array(dict_coverages[model]) - obj_coverage))]
    df_coverage_by_interval_length.to_csv(f"Results//df_wind_coverage_by_interval_length_alpha_{alpha}_quantile_step_{quantile_step}.csv", index=True)


    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    bar_heights = list(dict_coverages.values())
    n = len(models)

    x = np.arange(1, len(labels) + 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=90)
    ax.set_yticks(np.arange(10, 101, 5))
    ax.tick_params(axis='y', labelsize=12)
    for i in range(n):
        ax.plot(x, bar_heights[i], '--o', label=label_by_method_dict[models[i]], color=color_by_method_dict[models[i]])
    ax.grid()
    ax.legend(prop={'size': 14})
    ax.axhline(np.round(obj_coverage), color='k', lw=2)
    ax.set_title("Coverage by interval length", fontsize=16)
    ax.set_xlabel("Quantile (IW)", fontsize=14)
    ax.set_ylabel("Empirical Coverage", fontsize=14)
    if obj_coverage > 98:
        ax.set_ylim(obj_coverage-6.5, 100)
    elif obj_coverage > 94:
        ax.set_ylim(obj_coverage-6.5, 100)
    elif obj_coverage > 89:
        ax.set_ylim(obj_coverage-6.5, 100)
    else:
        ax.set_ylim(obj_coverage-8, 100)
    plt.tight_layout()
    plt.savefig(f"Results//wind_coverage_by_interval_length_alpha_{alpha}_quantile_step_{quantile_step}.pdf")

def error_by_interval_length_plot(df, models, quantile_step, alpha, bar_width = 0.1):
    df['error_abs'] = (df['real'] - df['mean_pred']).abs()
    for model in models:
        df['length_' + model] = df[f'preds_sup_{model}'] - df[f'preds_inf_{model}']
    dict_lengths = {model: [np.quantile(df[f'length_{model}'], i) for i in np.arange(0, 1.01, quantile_step)] for model in models}
    dict_abs_error = {model: [] for model in models}
    labels = []
    for i_quantile in range(len(dict_lengths[models[0]])-1):
        labels.append(str((round(np.arange(0, 1.01, quantile_step)[i_quantile], 2), round(np.arange(0, 1.01, quantile_step)[i_quantile+1], 2))))
        for model in models:
            length_min_model, length_max_model = dict_lengths[model][i_quantile], dict_lengths[model][i_quantile+1]
            df_aux_model = df[(df[f'length_{model}'] >= length_min_model) & (df[f'length_{model}'] <= length_max_model)]
            dict_abs_error[model].append(df_aux_model.error_abs.mean())

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_heights = list(dict_abs_error.values())
    n = len(models)

    x = np.arange(1, len(labels) + 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=90)
    ax.tick_params(axis='y', labelsize=12)
    for i in range(n):
        ax.plot(x, bar_heights[i], '--o', label=label_by_method_dict[models[i]], color=color_by_method_dict[models[i]])
    ax.grid()
    ax.legend(loc='lower right', prop={'size': 14})
    ax.set_title("MAE by interval length", fontsize=16)
    ax.set_xlabel("Quantile (IW)", fontsize=14)
    ax.set_ylabel("MAE", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"Results//wind_error_by_interval_length_alpha_{alpha}.pdf")

def error_by_interval_length_plot_2(df, models, quantile_step, alpha, bar_width = 0.1):
    df['error_abs'] = (df['real'] - df['mean_pred']).abs()
    for model in models:
        df['length_' + model] = df[f'preds_sup_{model}'] - df[f'preds_inf_{model}']
    df = df[df.length_cfqra.notnull()]
    df_1 = df.iloc[:2000]
    df_2 = df.iloc[4000:6000]
    dict_lengths_1 = {model: [np.quantile(df_1[f'length_{model}'], i) for i in np.arange(0, 1.01, quantile_step)] for model in models}
    dict_lengths_2 = {model: [np.quantile(df_2[f'length_{model}'], i) for i in np.arange(0, 1.01, quantile_step)] for model in models}
    dict_abs_error_1 = {model: [] for model in models}
    dict_abs_error_2 = {model: [] for model in models}
    labels = []
    for i_quantile in range(len(dict_lengths_1[models[0]])-1):
        labels.append(str((round(np.arange(0, 1.01, quantile_step)[i_quantile], 2), round(np.arange(0, 1.01, quantile_step)[i_quantile+1], 2))))
        for model in models:
            length_min_model, length_max_model = dict_lengths_1[model][i_quantile], dict_lengths_1[model][i_quantile+1]
            df_aux_model = df_1[(df_1[f'length_{model}'] >= length_min_model) & (df_1[f'length_{model}'] <= length_max_model)]
            dict_abs_error_1[model].append(df_aux_model.error_abs.mean())
    
    for i_quantile in range(len(dict_lengths_2[models[0]])-1):
        for model in models:
            length_min_model, length_max_model = dict_lengths_2[model][i_quantile], dict_lengths_2[model][i_quantile+1]
            df_aux_model = df_2[(df_2[f'length_{model}'] >= length_min_model) & (df_2[f'length_{model}'] <= length_max_model)]
            dict_abs_error_2[model].append(df_aux_model.error_abs.mean())

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    bar_heights_1 = list(dict_abs_error_1.values())
    bar_heights_2 = list(dict_abs_error_2.values())
    n = len(models)

    x = np.arange(1, len(labels) + 1)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=12, rotation=90)
    ax[0].tick_params(axis='y', labelsize=12)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=12, rotation=90)
    ax[1].tick_params(axis='y', labelsize=12)
    for i in range(n):
        ax[0].plot(x, bar_heights_1[i], '--o', label=label_by_method_dict[models[i]], color=color_by_method_dict[models[i]])
        ax[1].plot(x, bar_heights_2[i], '--o', label=label_by_method_dict[models[i]], color=color_by_method_dict[models[i]])
    ax[0].grid()
    ax[0].legend(loc='lower right', prop={'size': 14})
    ax[0].set_title("$\\lambda_2(\\alpha)$ significant", fontsize=16)
    ax[0].set_xlabel("Quantile (IW)", fontsize=14)
    ax[0].set_ylabel("MAE", fontsize=14)
    ax[1].grid()
    ax[1].legend(loc='lower right', prop={'size': 14})
    ax[1].set_title("$\\lambda_2(\\alpha)$ not significant", fontsize=16)
    ax[1].set_xlabel("Quantile (IW)", fontsize=14)
    ax[1].set_ylabel("MAE", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"Results//wind_error_by_interval_length_alpha_{alpha}_significance.pdf")

def interval_length_distribution_plot(df, models, quantile_step, alpha):
    for model in models:
        df['length_' + model] = df[f'preds_sup_{model}'] - df[f'preds_inf_{model}']
    dict_lengths = {model: [np.quantile(df[f'length_{model}'], i) for i in np.arange(quantile_step, 1-quantile_step+0.0001, quantile_step)] for model in models}

    fig, ax = plt.subplots(figsize=(12, 6))
    n = len(models)
    for model in models:
        ax.plot(np.arange(quantile_step, 1-quantile_step+0.0001, quantile_step), dict_lengths[model], "-o", label=label_by_method_dict[model], color=color_by_method_dict[model])
    ax.grid()
    ax.legend(loc='lower right', prop={'size': 14})
    ax.set_title("Interval length distribution", fontsize=16)
    ax.set_xlabel("Quantile", fontsize=14)
    ax.set_ylabel("Interval length", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"Results//wind_interval_length_distribution_alpha_{alpha}.pdf")

def coverage_by_hour_plot(df, models, alpha, bar_width = 0.1):
    df_coverage_by_hour = pd.DataFrame(columns=['mean_abs_deviation', 'max_abs_deviation', 'std_empirical_coverage'])
    obj_coverage = (1-alpha)*100

    for model in models:
        df['length_' + model] = df[f'preds_sup_{model}'] - df[f'preds_inf_{model}']
    dict_coverages = {model: [] for model in models}
    labels = []
    for hour in range(1, 25):
        labels.append(str(hour))
        for model in models:
            df_aux_model = df[df.hour == hour]
            dict_coverages[model].append(empirical_coverage(df_aux_model['real'], df_aux_model[f'preds_sup_{model}'], df_aux_model[f'preds_inf_{model}']))
    
    for model in models:
        df_coverage_by_hour.loc[model] = [np.mean(np.abs(np.array(dict_coverages[model]) - obj_coverage)), np.max(np.abs(np.array(dict_coverages[model]) - obj_coverage)), np.std(dict_coverages[model], ddof=0)]
    df_coverage_by_hour.to_csv(f"Results//df_wind_coverage_by_hour_alpha_{alpha}.csv", index=True)


    fig, ax = plt.subplots(figsize=(12, 6))
    bar_heights = list(dict_coverages.values())
    n = len(models)

    x = np.arange(1, len(labels) + 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    if obj_coverage > 98:
        ax.set_ylim(obj_coverage-3, 100)
    elif obj_coverage > 94:
        ax.set_ylim(obj_coverage-5, 100)
    elif obj_coverage > 89:
        ax.set_ylim(obj_coverage-6, 100)
    else:
        ax.set_ylim(obj_coverage-12, 100)
    for i in range(n):
        ax.plot(x, bar_heights[i], '--o', label=label_by_method_dict[models[i]], color=color_by_method_dict[models[i]])
    ax.grid()
    ax.legend(prop={'size': 14})
    ax.axhline(np.round(obj_coverage), color='k', lw=2)
    ax.set_title("Coverage by hour", fontsize=16)
    ax.set_xlabel("Hour", fontsize=14)
    ax.set_ylabel("Empirical Coverage", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"Results//wind_coverage_by_hour_alpha_{alpha}.pdf")


