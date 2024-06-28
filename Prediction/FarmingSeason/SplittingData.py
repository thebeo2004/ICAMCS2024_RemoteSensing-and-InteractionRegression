import pandas as pd

def splitting_data():

    weather_df = pd.read_csv('C:\Application\Crop-Yield-Prediction\Data\Farming Season\Weather(Farming Season).csv')
    yield_df = pd.read_csv('C:\Application\Crop-Yield-Prediction\Data\Farming Season\RandomCropYield (Farming Season).csv')

    X_train = weather_df.loc[weather_df['Year'] < 2017].drop(columns=['Year', 'Province'])
    y_train = yield_df.loc[yield_df['Year'] < 2017].drop(columns=['Year', 'Province'])

    test_set = []

    for year in range(2017, 2021):
        X_test = weather_df.loc[weather_df['Year'] == year].drop(columns=['Year', 'Province'])
        y_test = yield_df.loc[yield_df['Year'] == year].drop(columns=['Year', 'Province'])
        test_set.append((X_test, y_test))

    return X_train, y_train, test_set
