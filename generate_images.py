import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from os.path import dirname, abspath
import datetime as dt
import pickle
import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.values())

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

def generate_expected_error_3d_plot():
    d = dirname(abspath(__file__))
    # Define the parameters for the three normal distributions
    mu1 = np.array([-1., 2.])
    sigma1 = np.array([[1., .5], [.5, 1.]])

    mu2 = np.array([2., -1.])
    sigma2 = np.array([[1.5, .5], [.5, 1.5]])

    mu3 = np.array([-2., -2.])
    sigma3 = np.array([[0.5, .2], [.2, 0.5]])

    # Generate the grid
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x, y)
    # Generate the normal distributions
    rv1 = multivariate_normal(mu1, sigma1)
    rv2 = multivariate_normal(mu2, sigma2)
    rv3 = multivariate_normal(mu3, sigma3)
    z = np.zeros_like(x)
    for i in range(len(x)):
        for j in range(len(y)):
            pos = np.array([x[i, j], y[i, j]])
            z[i, j] = rv1.pdf(pos) + rv2.pdf(pos) + rv3.pdf(pos)
    z = z / np.sum(z)
    pos = np.dstack((x, y))


    # Create the 3D plot of the distributions
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(x, y, z, cmap='viridis')
    ax1.set_title("Distribution of explanatory features", fontsize=14, fontname='serif')

    # Create the complementary distribution
    z2 = 1 - z
    z2 = (z2 - np.min(z2)) / (np.max(z2) - np.min(z2))

    # Create the 3D plot of the complementary distribution
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x, y, z2, cmap='viridis')
    ax2.set_title("Expected error of a point forecaster by region", fontsize=14, fontname='serif')

    # Set the z axis ticks and labels
    ax2.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax2.set_zticklabels(["  Min Error", "   20\% Max Error", "  40\% Max Error", "  60\% Max Error", "  80\% Max Error", "  Max Error"], fontsize=10)
    # plt.show()
    plt.savefig(d + "\\Images\\expected_error_3d_plot.pdf")

def generate_day_ahead_market_figure(test_date=dt.datetime(2022, 10, 5)):
    d = dirname(abspath(__file__))
    df = pd.read_csv(d + "\\Data\\df.csv")
    df['full_date'] = pd.to_datetime(df['full_date'])

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(df['full_date'], df['real'])
    ax.plot(df.loc[df[df.full_date > test_date].index - 6*30*24, 'full_date'], df.loc[df[df.full_date > test_date].index - 6*30*24, 'real'])
    ax.plot(df.loc[df[df.full_date > test_date].index, 'full_date'], df.loc[df[df.full_date > test_date].index, 'real'])
    ax.axvline(x=test_date, color='k', linestyle='--', lw=2)
    ax.axvline(x=test_date - dt.timedelta(days=6*30), color='k', linestyle='--', lw=2)
    ax.grid()
    ax.set_title("Spanish Day-Ahead market price", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (€/MWh)", fontsize=12)

    # Add text and arrow
    ax.text(test_date - dt.timedelta(days=15), 650, "Quantile Regression predictions", ha='right', va='top', rotation=0, fontsize=11)
    ax.annotate("", xy=(test_date - dt.timedelta(days=5), 610), xytext=(test_date - dt.timedelta(days=6*30), 610),
                arrowprops=dict(arrowstyle="->"))
    
    ax.text(test_date + dt.timedelta(days=6*30) - dt.timedelta(days=15), 650, "Conformal prediction methods", ha='right', va='top', rotation=0, fontsize=11)
    ax.annotate("", xy=(test_date + dt.timedelta(days=6*30), 610), xytext=(test_date, 610),
                arrowprops=dict(arrowstyle="->"))

    plt.savefig(d + "\\Images\\day_ahead_market_figure.pdf")
    plt.tight_layout()
    plt.show()
    
def generate_day_ahead_market_predictions():
    d = dirname(abspath(__file__))
    df = pd.read_csv(d + "\\Data\\df.csv")
    df['full_date'] = pd.to_datetime(df['full_date'])

    df_aux = df[(pd.to_datetime(df.date) >= dt.datetime(2022, 3, 5)) & (pd.to_datetime(df.date) < dt.datetime(2022, 3, 5) + dt.timedelta(days=7))]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(df_aux['full_date'], df_aux['real'])
    ax.plot(df_aux['full_date'], df_aux['pred1'], '--', label='Model 1')
    ax.plot(df_aux['full_date'], df_aux['pred2'], '--', label='Model 2')
    ax.plot(df_aux['full_date'], df_aux['pred3'], '--', label='Model 3')
    ax.plot(df_aux['full_date'], df_aux['pred4'], '--', label='Model 4')
    for date in df_aux.date.unique():
        ax.axvline(x=pd.to_datetime(date), color='k', linestyle='--', lw=1)
    ax.grid()
    ax.set_title("Spanish Day-Ahead market price forecasts", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (€/MWh)", fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.savefig(d + "\\Images\\day_ahead_market_predictions.pdf")
    # plt.show()


def generate_progression_alphas(hour=13):
    d = dirname(abspath(__file__))
    with open('dict_list_alphas_aci.pkl', 'rb') as file:
        list_alphas_aci = pickle.load(file)

    with open('dict_list_alphas_waci.pkl', 'rb') as file:
        list_alphas_waci = pickle.load(file)

    list_alphas_aci = list_alphas_aci[hour]
    list_alphas_waci = list_alphas_waci[hour]

    for i in range(len(list_alphas_aci)):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.axhline(0.2, xmin= 0, xmax=500, lw=1, color='k', linestyle='--', label='Objective miscoverage', alpha=0.4)
        ax.plot(np.arange(0, 500, 0.1), list_alphas_waci[i], color=colors[0], label='WACI', lw=1)
        ax.axhline(list_alphas_aci[i], xmin= 0, xmax=500, lw=1, color=colors[1], label='ACI')
        ax.set_xlabel('Interval length', fontsize=14)
        ax.set_ylabel("$\\alpha_t$", fontsize=14)
        ax.grid()
        ax.set_ylim(0.1, 0.3)
        ax.set_xlim(0, 100)
        ax.legend(prop={'size': 14})
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        plt.tight_layout()
        plt.savefig(d + f"\\Images\\alpha_progression\\alpha_progression_hour{hour}_{i}.pdf")

def generate_wind_forecasts():
    d = dirname(abspath(__file__))
    df = pd.read_csv(d + "\\Data\\df_wind.csv")
    df['full_date'] = pd.to_datetime(df['full_date'])

    df_aux = df[(pd.to_datetime(df.date) >= dt.datetime(2021, 3, 5)) & (pd.to_datetime(df.date) < dt.datetime(2021, 3, 5) + dt.timedelta(days=7))]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(df_aux['full_date'], df_aux['real'])
    ax.plot(df_aux['full_date'], df_aux['pred1'], '--', label='Model 1')
    ax.plot(df_aux['full_date'], df_aux['pred2'], '--', label='Model 2')
    ax.plot(df_aux['full_date'], df_aux['pred3'], '--', label='Model 3')
    for date in df_aux.date.unique():
        ax.axvline(x=pd.to_datetime(date), color='k', linestyle='--', lw=1)
    ax.grid()
    ax.set_title("Wind power forecasts", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("GWh", fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.savefig(d + "\\Images\\wind_energy_forecasts.pdf")
    plt.show()

def generate_wind_power_figure(test_date=dt.datetime(2021, 1, 1)):
    d = dirname(abspath(__file__))
    df = pd.read_csv(d + "\\Data\\df_wind.csv")
    df['full_date'] = pd.to_datetime(df['full_date'])

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(df['full_date'], df['real'])
    ax.plot(df.loc[df[df.full_date > test_date].index - 6*30*24, 'full_date'], df.loc[df[df.full_date > test_date].index - 6*30*24, 'real'])
    ax.plot(df.loc[df[df.full_date > test_date].index, 'full_date'], df.loc[df[df.full_date > test_date].index, 'real'])
    ax.axvline(x=test_date, color='k', linestyle='--', lw=2)
    ax.axvline(x=test_date - dt.timedelta(days=6*30), color='k', linestyle='--', lw=2)
    ax.grid()
    ax.set_title("Wind Power", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("GWh", fontsize=12)

    # Add text and arrow
    ax.text(test_date - dt.timedelta(days=15), 19, "Quantile Regression predictions", ha='right', va='top', rotation=0, fontsize=8)
    ax.annotate("", xy=(test_date - dt.timedelta(days=5), 18.25), xytext=(test_date - dt.timedelta(days=6*30), 18.25),
                arrowprops=dict(arrowstyle="->"))
    
    ax.text(test_date + dt.timedelta(days=6*30) - dt.timedelta(days=15), 19, "Conformal prediction methods", ha='right', va='top', rotation=0, fontsize=8)
    ax.annotate("", xy=(test_date + dt.timedelta(days=6*30), 18.25), xytext=(test_date, 18.25),
                arrowprops=dict(arrowstyle="->"))

    plt.tight_layout()
    plt.savefig(d + "\\Images\\wind_power_figure.pdf")
    # plt.show()
    
def generate_coefs_variation():
    d = dirname(abspath(__file__))
    with open('dict_coefs_cfqra_alpha_0.01.pkl', 'rb') as file:
        dict_coefs_cfqra_001 = pickle.load(file)
    with open('dict_coefs_cfqra_alpha_0.05.pkl', 'rb') as file:
        dict_coefs_cfqra_005 = pickle.load(file)
    with open('dict_coefs_cfqra_alpha_0.1.pkl', 'rb') as file:
        dict_coefs_cfqra_01 = pickle.load(file)
    with open('dict_coefs_cfqra_alpha_0.2.pkl', 'rb') as file:
        dict_coefs_cfqra_02 = pickle.load(file)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    # ax.plot(dict_coefs_cfqra_001['inf'], label='$\\alpha=0.01$', color=colors[0])
    # ax.plot(dict_coefs_cfqra_001['sup'], '--', color=colors[0])
    ax.plot(dict_coefs_cfqra_005['inf'], label='$\\alpha=0.05$', color=colors[0])
    ax.plot(dict_coefs_cfqra_005['sup'], '--', color=colors[0])
    ax.plot(dict_coefs_cfqra_01['inf'], label='$\\alpha=0.10$', color=colors[1])
    ax.plot(dict_coefs_cfqra_01['sup'], '--', color=colors[1])
    ax.plot(dict_coefs_cfqra_02['inf'], label='$\\alpha=0.20$', color=colors[2])
    ax.plot(dict_coefs_cfqra_02['sup'], '--', color=colors[2])

    ax.set_title("$\\hat{\\lambda}_{2}(\\frac{\\alpha}{2})$ and $\\hat{\\lambda}_{2}(1-\\frac{\\alpha}{2})$ over time", fontsize=16)
    ax.set_xlabel("T+1", fontsize=12)
    ax.set_ylabel("Coefficient value", fontsize=12)
    ax.legend()
    plt.tight_layout()
    ax.grid()
    # plt.show()
    plt.savefig(d+"\\Images\\coefs_variation.pdf")

def generate_coefs_models_qra():
    d = dirname(abspath(__file__))
    with open('dict_coefs_qra_alpha_0.01.pkl', 'rb') as file:
        dict_coefs_cfqra_001 = pickle.load(file)
    with open('dict_coefs_qra_alpha_0.05.pkl', 'rb') as file:
        dict_coefs_cfqra_005 = pickle.load(file)
    with open('dict_coefs_qra_alpha_0.1.pkl', 'rb') as file:
        dict_coefs_cfqra_01 = pickle.load(file)
    with open('dict_coefs_qra_alpha_0.2.pkl', 'rb') as file:
        dict_coefs_cfqra_02 = pickle.load(file)
    
    fig, ax = plt.subplots(4, 1, figsize=(10, 16))
    ax[0].plot(dict_coefs_cfqra_001['inf']['pred1'], label='Model 1', color=colors[0])
    # ax[0].plot(dict_coefs_cfqra_001['sup']['pred1'], '--', color=colors[0])
    ax[0].plot(dict_coefs_cfqra_001['inf']['pred2'], label='Model 2', color=colors[1])
    # ax[0].plot(dict_coefs_cfqra_001['sup']['pred2'], '--', color=colors[1])
    ax[0].plot(dict_coefs_cfqra_001['inf']['pred3'], label='Model 3', color=colors[2])
    # ax[0].plot(dict_coefs_cfqra_001['sup']['pred3'], '--', color=colors[2])
    ax[0].set_title("$\\alpha = 0.01$", fontsize=16)
    ax[0].set_xlabel("T+1", fontsize=12)
    ax[0].set_ylabel("Coefficient value", fontsize=12)
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(dict_coefs_cfqra_005['inf']['pred1'], label='Model 1', color=colors[0])
    # ax[1].plot(dict_coefs_cfqra_005['sup']['pred1'], '--', color=colors[0])
    ax[1].plot(dict_coefs_cfqra_005['inf']['pred2'], label='Model 2', color=colors[1])
    # ax[1].plot(dict_coefs_cfqra_005['sup']['pred2'], '--', color=colors[1])
    ax[1].plot(dict_coefs_cfqra_005['inf']['pred3'], label='Model 3', color=colors[2])
    # ax[1].plot(dict_coefs_cfqra_005['sup']['pred3'], '--', color=colors[2])
    ax[1].set_title("$\\alpha = 0.05$", fontsize=16)
    ax[1].set_xlabel("T+1", fontsize=12)
    ax[1].set_ylabel("Coefficient value", fontsize=12)
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(dict_coefs_cfqra_01['inf']['pred1'], label='Model 1', color=colors[0])
    # ax[2].plot(dict_coefs_cfqra_01['sup']['pred1'], '--', color=colors[0])
    ax[2].plot(dict_coefs_cfqra_01['inf']['pred2'], label='Model 2', color=colors[1])
    # ax[2].plot(dict_coefs_cfqra_01['sup']['pred2'], '--', color=colors[1])
    ax[2].plot(dict_coefs_cfqra_01['inf']['pred3'], label='Model 3', color=colors[2])
    # ax[2].plot(dict_coefs_cfqra_01['sup']['pred3'], '--', color=colors[2])
    ax[2].set_title("$\\alpha = 0.10$", fontsize=16)
    ax[2].set_xlabel("T+1", fontsize=12)
    ax[2].set_ylabel("Coefficient value", fontsize=12)
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(dict_coefs_cfqra_02['inf']['pred1'], label='Model 1', color=colors[0])
    # ax[3].plot(dict_coefs_cfqra_02['sup']['pred1'], '--', color=colors[0])
    ax[3].plot(dict_coefs_cfqra_02['inf']['pred2'], label='Model 2', color=colors[1])
    # ax[3].plot(dict_coefs_cfqra_02['sup']['pred2'], '--', color=colors[1])
    ax[3].plot(dict_coefs_cfqra_02['inf']['pred3'], label='Model 3', color=colors[2])
    # ax[3].plot(dict_coefs_cfqra_02['sup']['pred3'], '--', color=colors[3])
    ax[3].set_title("$\\alpha = 0.20$", fontsize=16)
    ax[3].set_xlabel("T+1", fontsize=12)
    ax[3].set_ylabel("Coefficient value", fontsize=12)
    ax[3].legend()
    ax[3].grid()

    plt.tight_layout()
    # plt.show()
    plt.savefig(d+"\\Images\\coefs_variation_qra.pdf")

def generate_coefs_variation_wind():
    d = dirname(abspath(__file__))
    # with open('dict_coefs_cfqra_wind_alpha_0.01.pkl', 'rb') as file:
    #     dict_coefs_cfqra_001 = pickle.load(file)
    with open('dict_coefs_cfqra_wind_alpha_0.05.pkl', 'rb') as file:
        dict_coefs_cfqra_005 = pickle.load(file)
    with open('dict_coefs_cfqra_wind_alpha_0.1.pkl', 'rb') as file:
        dict_coefs_cfqra_01 = pickle.load(file)
    with open('dict_coefs_cfqra_wind_alpha_0.2.pkl', 'rb') as file:
        dict_coefs_cfqra_02 = pickle.load(file)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    # ax.plot(dict_coefs_cfqra_001['inf'], label='$\\alpha=0.01$', color=colors[0])
    # ax.plot(dict_coefs_cfqra_001['sup'], '--', color=colors[0])
    ax.plot(dict_coefs_cfqra_005['inf'], label='$\\alpha=0.05$', color=colors[0])
    ax.plot(dict_coefs_cfqra_005['sup'], '--', color=colors[0])
    ax.plot(dict_coefs_cfqra_01['inf'], label='$\\alpha=0.10$', color=colors[1])
    ax.plot(dict_coefs_cfqra_01['sup'], '--', color=colors[1])
    ax.plot(dict_coefs_cfqra_02['inf'], label='$\\alpha=0.20$', color=colors[2])
    ax.plot(dict_coefs_cfqra_02['sup'], '--', color=colors[2])

    ax.set_title("$\\hat{\\lambda}_{2}(\\frac{\\alpha}{2})$ and $\\hat{\\lambda}_{2}(1-\\frac{\\alpha}{2})$ over time", fontsize=16)
    ax.set_xlabel("T+1", fontsize=12)
    ax.set_ylabel("Coefficient value", fontsize=12)
    ax.legend()
    plt.tight_layout()
    ax.grid()
    # plt.show()
    plt.savefig(d+"\\Images\\coefs_variation_wind.pdf")

# def generate_progression_alphas_2(hour=13):
#     d = dirname(abspath(__file__))
#     with open('dict_list_alphas_aci.pkl', 'rb') as file:
#         list_alphas_aci = pickle.load(file)

#     with open('dict_list_alphas_waci.pkl', 'rb') as file:
#         list_alphas_waci = pickle.load(file)
    
#     with open('dict_list_alphas_waci_2.pkl', 'rb') as file:
#         list_alphas_waci_2 = pickle.load(file)

#     list_alphas_aci = list_alphas_aci[hour]
#     list_alphas_waci = list_alphas_waci[hour]
#     list_alphas_waci_2 = list_alphas_waci_2[hour]

#     for i in range(len(list_alphas_aci)):
#         fig, ax = plt.subplots(1, 1, figsize=(10, 4))
#         ax.axhline(0.2, xmin= 0, xmax=500, lw=1, color='k', linestyle='--', label='Objective miscoverage', alpha=0.4)
#         ax.plot(np.arange(0, 500, 0.1), list_alphas_waci[i], color=colors[0], label='WACI', lw=1)
#         ax.plot(np.arange(0, 500, 0.1), list_alphas_waci_2[i], color=colors[1], label='WACI 2', lw=1)
#         ax.axhline(list_alphas_aci[i], xmin= 0, xmax=500, lw=1, color=colors[2], label='ACI')
#         ax.set_xlabel('Interval length', fontsize=14)
#         ax.set_ylabel("$\\alpha_t$", fontsize=14)
#         ax.grid()
#         ax.set_ylim(0.1, 0.3)
#         ax.set_xlim(0, 100)
#         ax.legend(prop={'size': 14})
#         ax.tick_params(axis='x', labelsize=12)
#         ax.tick_params(axis='y', labelsize=12)
#         plt.tight_layout()
#         plt.savefig(d + f"\\Images\\alpha_progression_2\\alpha_progression_hour{hour}_{i}.pdf")



# generate_expected_error_3d_plot()
# generate_day_ahead_market_figure()
# generate_day_ahead_market_predictions()
# generate_progression_alphas(hour=13)
# generate_wind_forecasts()
# generate_wind_power_figure()
# generate_coefs_variation()
# generate_coefs_models_qra()
# generate_coefs_variation_wind()
# generate_progression_alphas_2(hour=13)