import numpy as np

def MAE(real_values, pred_values):
    return np.mean(np.abs(real_values - pred_values))

def MSE(real_values, pred_values):
    return np.mean((real_values - pred_values) ** 2)

def RMSE(real_values, pred_values):
    return np.sqrt(MSE(real_values, pred_values))

def MAPE(real_values, pred_values):
    return np.mean(np.abs((real_values - pred_values) / real_values)) * 100

def SMAPE(real_values, pred_values):
    return np.mean(np.abs(real_values - pred_values) / (np.abs(real_values) + np.abs(pred_values) / 2)) * 100

def R2(real_values, pred_values):
    mean = np.mean(real_values)
    return 1 - (np.sum((pred_values - real_values) ** 2) / np.sum((mean - real_values) ** 2))