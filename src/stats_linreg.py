import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score


class StatsModReg:

    def __init__(self, data, x_cols, y_col, fit_intercept=True):
        self.data = data
        self.x_train = data[x_cols]
        if fit_intercept:
            self.x_train = sm.add_constant(data[x_cols])
        self.y_train = data[y_col]

    def model_train(self):
        reg_model = sm.OLS(self.y_train, self.x_train).fit()
        return reg_model

    def eval_model(self, reg_model):
        y_pred = reg_model.predict(self.x_train)
        r2sq = r2_score(self.y_train, y_pred)
        return r2sq
