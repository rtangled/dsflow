# This is a sample Python script.
import pandas as pd
from data_process import DataProcess
from feature_eng import FeatureEng
from training import ModelTrain
import logging

logging.basicConfig(filename='logs/file.log', level=logging.INFO, format= '%(asctime)s:%(levelname)s:%(name)s:%('
                                                                          'message)s')


def ml_process(path, x_cols, y_col):
    df = pd.read_csv(path)

    # Data processing
    logging.info('Data Processing Started')
    dproc = DataProcess(df)
    df_noNA = dproc.data_process()
    logging.info("cleaned data shape: %s", df_noNA.shape)
    logging.info('Data Processing Finished')

    # feature engineering
    logging.info('feature engineering Started')
    dfeat = FeatureEng(df_noNA)
    df_final = dfeat.feat_generate()
    logging.info('feature engineering Finished')

    # train test split
    mod = ModelTrain(df_final, x_cols, y_col)
    X_train, X_test, y_train, y_test = mod.split_data()

    # model training and evaluation
    logging.info('model training Started')
    reg_model = mod.model_train(X_train, y_train)
    print("Model Accuracy:", mod.model_eval(reg_model, X_test, y_test))
    logging.info('model training Finished')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = "data/raw/AirQualityUCI.csv"
    x_cols = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'NO_feat']
    y_col = ['C6H6(GT)']

    # run ml training pipeline
    ml_process(path, x_cols, y_col)
