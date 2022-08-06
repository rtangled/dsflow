from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class ModelTrain:

    def __init__(self, data, x_cols, y_col):
        self.data = data
        self.x_cols = x_cols
        self.y_col = y_col

    def split_data(self):
        x = self.data[self.x_cols].values
        y = self.data[self.y_col].values
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
        return X_train, X_test, y_train, y_test

    def model_train(self, X_train, y_train):
        regr = LinearRegression()
        regr.fit(X_train, y_train)
        return regr

    def model_eval(self, reg_model, x_test, y_test):
        return reg_model.score(x_test, y_test)
