from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import numpy as np
import pandas as pd


class LinearReg:

    def __init__(self, data, x_cols, y_col, fit_intercept=True):
        self.data = data
        self.x_cols = x_cols
        self.x_train = data[x_cols].values
        self.y_train = data[y_col].values
        self.fit_intercept = fit_intercept
        self.df_m = len(x_cols)
        if fit_intercept:
            self.df_r = self.x_train.shape[0] - self.df_m - 1
        else:
            self.df_r = self.x_train.shape[0] - self.df_m

    def model_train(self):
        lin_reg = LinearRegression(fit_intercept=self.fit_intercept)
        lin_reg.fit(self.x_train, self.y_train)
        return lin_reg

    def get_vif(self):
        if self.fit_intercept:
            vif_list = [0]
        else:
            vif_list = []
        for x_col in self.x_cols:
            reg = LinearRegression(fit_intercept=True)
            y = self.data[x_col].values
            x_col_names = [i for i in self.x_cols if i != x_col]
            x_vif = self.data[x_col_names].values
            reg.fit(x_vif, y)
            rsq = reg.score(x_vif, y)
            vif = np.round(1 / (1 - rsq), 1)
            vif_list.append(vif)
        vif_out = np.array(vif_list)
        vif_out = vif_out.reshape((vif_out.shape[0], 1))
        return (vif_out)

    def get_betas(self):
        x = self.x_train
        if self.fit_intercept:
            x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
        y = self.y_train
        xt = x.transpose()
        xt_x = np.matmul(xt, x)
        inv_xt_x = np.linalg.inv(xt_x)
        xt_y = np.matmul(xt, y)
        beta = np.matmul(inv_xt_x, xt_y)
        return beta

    def get_stats(self, reg_model):
        x = self.x_train
        y = self.y_train
        # no. of regressors
        p = len(self.x_cols)
        # no. of examples (data points)
        n = x.shape[0]
        # calculate predicted values
        print(x.shape)
        y_pred = reg_model.predict(x)

        # calculate r-square and adj r-square scores
        rsq = r2_score(y, y_pred)
        adj_rsq = 1 - (((1-rsq) * (n-1)) / (n-p-1))

        # calculate various errors
        # calculate total sum of square (TSS)
        SST = np.sum((y-np.mean(y))**2)
        # calculate regression sum of squares
        SSR = np.sum((y_pred-np.mean(y))**2)
        # calculate residual sum of squares
        SSE = np.sum((y - y_pred)**2)

        # calculate f-statistic
        MSR = SSR / p
        MSE = SSE / (n-p-1)
        fstat = MSR / MSE
        pvalue = 1 - stats.f.cdf(fstat, self.df_m, self.df_r)
        # print(fstat, pvalue)

        # calculate t-stat, p-value, std-err, confidence interval
        if self.fit_intercept:
            x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
        var_beta = MSE * (np.linalg.inv(np.dot(x.T, x)).diagonal())
        std_dev = np.sqrt(var_beta)
        std_dev = std_dev.reshape((std_dev.shape[0], 1))
        params = self.get_betas()
        t_value = params / std_dev
        t_value = t_value.reshape((t_value.shape[0], 1))

        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (n-p))) for i in t_value]

        std_dev = np.round(std_dev, 3)
        t_value = np.round(t_value, 3)
        p_values = np.round(p_values, 3)

        # get VIF values
        vif = self.get_vif()
        stats_arr = np.column_stack((params, std_dev, t_value, p_values, vif))
        stats_df = pd.DataFrame(stats_arr)
        stats_df.columns = ['coeff', 'std-err', 't-value', 'p-value', 'vif']
        if self.fit_intercept:
            stats_df.index = ["Intercept"] + self.x_cols
        else:
            stats_df.index = self.x_cols
        return stats_df

