import logging


class DataProcess:

    def __init__(self, data):
        self.data = data

    def data_process(self):
        return self.data.dropna()
