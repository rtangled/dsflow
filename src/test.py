from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import numpy as np
import pandas as pd

from linear_reg import LinearReg
from stats_linreg import StatsModReg

# numpy syntax testing
# a = np.array([10, 20, 30, 40, 50])
# b = np.array([5, 10, 15, 20, 25])
# a = a.reshape((1, 5))
# c = (a-b)**2
# print(a.shape)
# print(np.sum(c))
# print(np.mean(a))
# print(np.c_[ a, np.ones(1)] )
# print(np.append(a, np.ones(1).reshape((1,1)), axis=1))
# print(np.ones(1))

if __name__ == "__main__":
    df = pd.read_csv("../data/raw/AirQualityUCI.csv")
    df = df.dropna()
    x_cols = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)']
    y_col = ['C6H6(GT)']
    reg = LinearReg(df, x_cols, y_col, fit_intercept=True)
    print(reg.model_train().coef_)
    reg.get_betas()
    reg.get_stats(reg.model_train())

    sm_reg = StatsModReg(df, x_cols, y_col)
    sm_model = sm_reg.model_train()
    print(sm_reg.eval_model(sm_model))
    print(sm_model.summary())
    # print(sm_model.fvalue, sm_model.f_pvalue)