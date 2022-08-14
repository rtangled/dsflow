from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import numpy as np
import pandas as pd

from linear_reg import LinearReg
from stats_linreg import StatsModReg


if __name__ == "__main__":
    df = pd.read_csv("../data/raw/AirQualityUCI.csv")
    df = df.dropna()
    x_cols = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)']
    y_col = ['C6H6(GT)']
    reg = LinearReg(df, x_cols, y_col, fit_intercept=True)
    # print(reg.model_train().coef_)
    reg.get_betas()
    stats = reg.get_stats(reg.model_train())
    print(stats["coeff_summary"])
    print("R-squared:", stats["rsq"])
    print("Adj. R-squared:", stats["adj_rsq"])
    print("Durbin-Watson:", stats["dw"])

    sm_reg = StatsModReg(df, x_cols, y_col, fit_intercept=True)
    sm_model = sm_reg.model_train()
    # print(sm_reg.eval_model(sm_model))
    print(sm_model.summary())
    # print(sm_model.fvalue, sm_model.f_pvalue)


