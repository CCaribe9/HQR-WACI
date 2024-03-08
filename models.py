import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
import numpy as np
from copy import copy
import pickle

def CornishFisherQuantileRegressionAveraging(first_idx, df, alpha, by_hour, save_coefs=False):
    """
    Perform Cornish-Fisher Quantile Regression Averaging.

    Args:
        first_idx (int): The index of the first observation.
        df (pandas.DataFrame): The input dataframe containing the data.
        alpha (float): The objective miscoverage rate.
        by_hour (bool): Flag indicating whether to perform the calculation by hour.
        save_coefs (bool, optional): Flag indicating whether to save the coefficients. Defaults to False.

    Returns:
        tuple: A tuple containing two lists: preds_inf and preds_sup.
            - preds_inf (list): The lower quantile predictions.
            - preds_sup (list): The upper quantile predictions.
    """
    preds_inf = []
    preds_sup = []
    alpha_inf = alpha/2
    alpha_sup = 1 - alpha_inf
    if save_coefs:
        dict_coefs = {'sup': [], 'inf': []}
    for obs in tqdm(range(first_idx, len(df)), desc=f'Cornish-Fisher Quantile Regression Averaging. By hours = {by_hour}'):
        if by_hour:
            df_cal = df[(df.index < obs) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date'])) & (df.hour == df.loc[obs, 'hour'])][['real', 'mean_pred', 'std_pred']]
            if len(df_cal) > 6*30:
                df_cal = df_cal.tail(6*30)
        else:   
            df_cal = df[(df.index < obs) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date']))][['real', 'mean_pred', 'std_pred']]
            if len(df_cal) > 6*30*24:
                df_cal = df_cal.tail(6*30*24)
        
        df_test = pd.DataFrame(df.loc[obs][['mean_pred', 'std_pred']]).T
        df_test['const'] = 1.0

        X_train = sm.add_constant(df_cal[['mean_pred', 'std_pred']])
        y_train = df_cal['real']

        X_test = df_test[['const', 'mean_pred', 'std_pred']]

        model = sm.QuantReg(y_train, X_train)
        res = model.fit(q=alpha_inf)
        preds_inf.append(res.predict(X_test).values[0])
        if save_coefs:
            dict_coefs['inf'].append(res.params['std_pred'])

        model = sm.QuantReg(y_train, X_train)
        res = model.fit(q=alpha_sup)
        preds_sup.append(res.predict(X_test).values[0])
        if save_coefs:
            dict_coefs['sup'].append(res.params['std_pred'])
        
    if save_coefs:
        with open(f'dict_coefs_cfqra_alpha_{alpha}.pkl', 'wb') as f:
            pickle.dump(dict_coefs, f)

    return preds_inf, preds_sup

def QuantileRegressionAveraging(first_idx, df, alpha, by_hour, m):
    """
    Perform Quantile Regression Averaging.

    Args:
        first_idx (int): The index of the first observation.
        df (pandas.DataFrame): The input dataframe containing the data.
        alpha (float): The objective miscoverage rate.
        by_hour (bool): Flag indicating whether to perform the calculation by hour.
        m (int): The number of predictors.

    Returns:
        tuple: A tuple containing two lists: preds_inf and preds_sup.
            - preds_inf (list): The lower quantile predictions.
            - preds_sup (list): The upper quantile predictions.
    """
    preds_inf = []
    preds_sup = []
    alpha_inf = alpha/2
    alpha_sup = 1 - alpha_inf
    for obs in tqdm(range(first_idx, len(df)), desc=f'Quantile Regression Averaging. By hours = {by_hour}'):
        if by_hour:
            df_cal = df[(df.index < obs) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date'])) & (df.hour == df.loc[obs, 'hour'])][['real'] + [f'pred{i}' for i in range(1, m+1)]]
            if len(df_cal) > 6*30:
                df_cal = df_cal.tail(6*30)
        else:   
            df_cal = df[(df.index < obs) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date']))][['real'] + [f'pred{i}' for i in range(1, m+1)]]
            if len(df_cal) > 6*30*24:
                df_cal = df_cal.tail(6*30*24)
        
        df_test = pd.DataFrame(df.loc[obs][[f'pred{i}' for i in range(1, m+1)]]).T
        df_test['const'] = 1.0

        X_train = sm.add_constant(df_cal[[f'pred{i}' for i in range(1, m+1)]])
        y_train = df_cal['real']

        X_test = df_test[['const'] + [f'pred{i}' for i in range(1, m+1)]]

        model = sm.QuantReg(y_train, X_train)
        res = model.fit(q=alpha_inf)
        preds_inf.append(res.predict(X_test).values[0])

        model = sm.QuantReg(y_train, X_train)
        res = model.fit(q=alpha_sup)
        preds_sup.append(res.predict(X_test).values[0])

    return preds_inf, preds_sup

def ConformalizedQuantileRegression(first_idx, df, alpha, method, by_hour):
    """
    Perform Conformalized Quantile Regression.

    Args:
        first_idx (int): The index of the first observation.
        df (pandas.DataFrame): The input dataframe.
        alpha (float): The objective miscoverage rate.
        method (str): The method used for quantile regression.
        by_hour (bool): Whether the previous quantile regression was done by hour.

    Returns:
        tuple: A tuple containing two lists: preds_cqr_inf and preds_cqr_sup.
            - preds_cqr_inf (list): The conformalized lower quantile predictions.
            - preds_cqr_sup (list): The conformalized upper quantile predictions.

    """
    if by_hour:
        add_string = method + '_by_hour'
    else:
        add_string = method
    preds_cqr_inf, preds_cqr_sup = [], []

    for obs in tqdm(range(first_idx, len(df)), desc=f'Conformalized Quantile Regression. Method = {method}. By hours = {by_hour}'):
        df_cal = df[(df.index < obs) & (~df[f'length_{add_string}'].isna()) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date'])) & (df.hour == df.loc[obs, 'hour'])].tail(500)
        
        n=len(df_cal)
        
        cal_scores = np.maximum(df_cal.real - df_cal[f'preds_sup_{add_string}'], df_cal[f'preds_inf_{add_string}'] - df_cal.real)
        quantile = np.min([1, np.max([0, (np.ceil(n+1)*(1-alpha))/n])])
        qhat = np.quantile(cal_scores.to_numpy(), quantile)
        
        preds_cqr_inf.append(df.loc[obs, f'preds_inf_{add_string}'] - qhat)
        preds_cqr_sup.append(df.loc[obs, f'preds_sup_{add_string}'] + qhat)
    return preds_cqr_inf, preds_cqr_sup

def AdaptiveConformalInference(first_idx, df, alpha, gamma, method, by_hour, save_alphas=False):
    """
    Perform adaptive conformal inference.

    Args:
        first_idx (int): The index of the first observation.
        df (pandas.DataFrame): The input dataframe.
        alpha (float): The objective miscoverage rate.
        gamma (float): The adaptation rate.
        method (str): The method used for quantile regression.
        by_hour (bool): Whether the previous quantile regression was done by hour.
        save_alphas (bool, optional): Flag indicating whether to save the alpha values. Defaults to False.

    Returns:
        tuple: A tuple containing two lists: preds_cqr_inf and preds_cqr_sup.
            - preds_aci_inf (list): The adaptive conformalized lower quantile predictions.
            - preds_aci_sup (list): The adaptive conformalized upper quantile predictions.
    """
    preds_aci_inf, preds_aci_sup = [], []
    alpha_obj = alpha
    # list_alpha_hora_13 = []
    
    if by_hour:
        add_string = method + '_by_hour'
    else:
        add_string = method
    
    alpha_by_hour = {hour: alpha_obj for hour in range(1, 25)}
    if save_alphas:
        dict_list_alphas = {hour: [alpha_obj] for hour in range(1, 25)}

    for obs in tqdm(range(first_idx, len(df)), desc=f'Adaptive Conformal Inference. Method = {method}. By hours = {by_hour}'):
        hour = df.loc[obs, 'hour']
        df_cal = df[(df.index < obs) & (~df[f'length_{add_string}'].isna()) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date'])) & (df.hour == hour)].tail(500)
        df_test = df.loc[obs]

        alpha = alpha_by_hour[hour]

        # if hour == 13:
        #     list_alpha_hora_13.append(alpha)

        n = len(df_cal)

        cal_scores = np.maximum(df_cal.real - df_cal[f'preds_sup_{add_string}'], df_cal[f'preds_inf_{add_string}'] - df_cal.real)
        quantile = np.min([1, np.max([0, (np.ceil(n+1)*(1-alpha))/n])])
        qhat = np.quantile(cal_scores.to_numpy(), quantile)
        
        pred_aci_inf = df_test[f'preds_inf_{add_string}'] - qhat
        pred_aci_sup = df_test[f'preds_sup_{add_string}'] + qhat
        preds_aci_inf.append(pred_aci_inf)
        preds_aci_sup.append(pred_aci_sup)

        err = 1 - float(df_test.real >= pred_aci_inf and df_test.real <= pred_aci_sup)
        if alpha + gamma*(alpha_obj - err) < 0.99 and alpha + gamma*(alpha_obj - err) > 0.01: # We do this so we don't get empty or infinite intervals
            alpha = alpha + gamma*(alpha_obj - err)
        
        alpha_by_hour[hour] = alpha
        if save_alphas:
            dict_list_alphas[hour].append(alpha)

    if save_alphas:
        with open(f'dict_list_alphas_aci.pkl', 'wb') as f:
            pickle.dump(dict_list_alphas, f)

    return preds_aci_inf, preds_aci_sup

def WidthAdaptiveConformalInference(first_idx, df, alpha, gamma, sigma, method, by_hour, save_alphas=False):
    """
    Perform width-adaptive conformal inference.

    Args:
        first_idx (int): The index of the first observation.
        df (pandas.DataFrame): The input dataframe.
        alpha (float): The objective miscoverage rate.
        gamma (float): The adaptation rate.
        sigma (float): The impact of the gaussian kernel.
        method (str): The method used for quantile regression.
        by_hour (bool): Whether the previous quantile regression was done by hour.
        save_alphas (bool, optional): Flag indicating whether to save the alpha values. Defaults to False.

    Returns:
        tuple: A tuple containing two lists: preds_cqr_inf and preds_cqr_sup.
            - preds_waci_inf (list): The width-adaptive conformalized lower quantile predictions.
            - preds_waci_sup (list): The width-adaptive conformalized upper quantile predictions.
    """
    preds_waci_inf, preds_waci_sup = [], []
    alpha_obj = alpha
    if by_hour:
        add_string = method + '_by_hour'
    else:
        add_string = method
    
    lengths_by_hour = {hour: np.arange(0, 500, 0.1) for hour in range(1, 25)}
    alphas_by_hour = {hour: np.array([alpha_obj] * len(lengths_by_hour[hour])) for hour in range(1, 25)}
    if save_alphas:
        dict_list_alphas = {hour: [np.array([alpha_obj] * len(lengths_by_hour[hour]))] for hour in range(1, 25)}
    
    for obs in tqdm(range(first_idx, len(df)), desc=f'Width-Adaptive Conformal Inference. Method = {method}. By hours = {by_hour}'):
        hour = df.loc[obs, 'hour']
        df_cal = df[(df.index < obs) & (~df[f'length_{add_string}'].isna()) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date'])) & (df.hour == hour)].tail(500)
        df_test = df.loc[obs]

        lengths = lengths_by_hour[hour]
        alphas = alphas_by_hour[hour]

        length_obs = df_test[f'length_{add_string}']
        closest_iw = min(lengths, key=lambda x:abs(x-length_obs))
        idx_closest_iw = list(lengths).index(closest_iw)
        alpha = alphas[idx_closest_iw]

        n = len(df_cal)

        cal_scores = np.maximum(df_cal.real - df_cal[f'preds_sup_{add_string}'], df_cal[f'preds_inf_{add_string}'] - df_cal.real)
        quantile = np.min([1, np.max([0, (np.ceil(n+1)*(1-alpha))/n])])
        qhat = np.quantile(cal_scores.to_numpy(), quantile)
        pred_waci_inf = df_test[f'preds_inf_{add_string}'] - qhat
        pred_waci_sup = df_test[f'preds_sup_{add_string}'] + qhat
        preds_waci_inf.append(pred_waci_inf)
        preds_waci_sup.append(pred_waci_sup)

        err = 1 - float(df_test.real >= pred_waci_inf and df_test.real <= pred_waci_sup)
        
        distances = np.abs(np.array(lengths) - length_obs)
        unnormalized_weights = np.exp(-distances**2/(2*sigma**2))
        max_weight = np.max(unnormalized_weights)
        weights = unnormalized_weights / max_weight
        

        old_alphas = copy(alphas)
        alphas = alphas + gamma * weights * (alpha_obj - err)
        alphas = np.where(np.logical_and(alphas < 0.99, alphas > 0.01), alphas, old_alphas)

        alphas_by_hour[hour] = alphas

        if save_alphas:
            dict_list_alphas[hour].append(alphas)

    if save_alphas:
        with open(f'dict_list_alphas_waci.pkl', 'wb') as f:
            pickle.dump(dict_list_alphas, f)

    return preds_waci_inf, preds_waci_sup

def WidthAdaptiveConformalInference_2(first_idx, df, alpha, gamma, rate, method, by_hour, save_alphas=False):
    """
    Perform width-adaptive conformal inference with the second weighting scheme.

    Args:
        first_idx (int): The index of the first observation.
        df (pandas.DataFrame): The input dataframe.
        alpha (float): The objective miscoverage rate.
        gamma (float): The adaptation rate.
        sigma (float): The impact of the gaussian kernel.
        method (str): The method used for quantile regression.
        by_hour (bool): Whether the previous quantile regression was done by hour.
        save_alphas (bool, optional): Flag indicating whether to save the alpha values. Defaults to False.

    Returns:
        tuple: A tuple containing two lists: preds_cqr_inf and preds_cqr_sup.
            - preds_waci_inf (list): The width-adaptive conformalized lower quantile predictions.
            - preds_waci_sup (list): The width-adaptive conformalized upper quantile predictions.
    """
    preds_waci_inf, preds_waci_sup = [], []
    alpha_obj = alpha
    if by_hour:
        add_string = method + '_by_hour'
    else:
        add_string = method
    
    lengths_by_hour = {hour: np.arange(0, 500, 0.1) for hour in range(1, 25)}
    alphas_by_hour = {hour: np.array([alpha_obj] * len(lengths_by_hour[hour])) for hour in range(1, 25)}
    if save_alphas:
        dict_list_alphas = {hour: [np.array([alpha_obj] * len(lengths_by_hour[hour]))] for hour in range(1, 25)}
    
    for obs in tqdm(range(first_idx, len(df)), desc=f'Width-Adaptive Conformal Inference 2 . Method = {method}. By hours = {by_hour}'):
        hour = df.loc[obs, 'hour']
        df_cal = df[(df.index < obs) & (~df[f'length_{add_string}'].isna()) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date'])) & (df.hour == hour)].tail(500)
        df_test = df.loc[obs]

        lengths = lengths_by_hour[hour]
        alphas = alphas_by_hour[hour]

        length_obs = df_test[f'length_{add_string}']
        closest_iw = min(lengths, key=lambda x:abs(x-length_obs))
        idx_closest_iw = list(lengths).index(closest_iw)
        alpha = alphas[idx_closest_iw]

        n = len(df_cal)

        cal_scores = np.maximum(df_cal.real - df_cal[f'preds_sup_{add_string}'], df_cal[f'preds_inf_{add_string}'] - df_cal.real)
        quantile = np.min([1, np.max([0, (np.ceil(n+1)*(1-alpha))/n])])
        qhat = np.quantile(cal_scores.to_numpy(), quantile)
        pred_waci_inf = df_test[f'preds_inf_{add_string}'] - qhat
        pred_waci_sup = df_test[f'preds_sup_{add_string}'] + qhat
        preds_waci_inf.append(pred_waci_inf)
        preds_waci_sup.append(pred_waci_sup)

        err = 1 - float(df_test.real >= pred_waci_inf and df_test.real <= pred_waci_sup)
        
        distances = np.abs(np.array(lengths) - length_obs)
        min_distance_position = np.argmin(distances)
        position_distances = np.abs(np.arange(len(distances)) - min_distance_position)
        weights = 1 / (rate ** position_distances)
        weights[weights < 0] = 0
        weights[np.isinf(weights)] = 0
                

        old_alphas = copy(alphas)
        alphas = alphas + gamma * weights * (alpha_obj - err)
        alphas = np.where(np.logical_and(alphas < 0.99, alphas > 0.01), alphas, old_alphas)

        alphas_by_hour[hour] = alphas

        if save_alphas:
            dict_list_alphas[hour].append(alphas)

    if save_alphas:
        with open(f'dict_list_alphas_waci_2.pkl', 'wb') as f:
            pickle.dump(dict_list_alphas, f)

    return preds_waci_inf, preds_waci_sup





