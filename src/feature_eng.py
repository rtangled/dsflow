class FeatureEng:

    def __init__(self, data):
        self.data = data

    def feat_generate(self):
        tdf = self.data.copy()
        tdf['NO_feat'] = tdf['NOx(GT)'] + tdf['NO2(GT)']
        return tdf

