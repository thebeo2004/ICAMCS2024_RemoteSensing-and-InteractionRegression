import pandas as pd
from pathlib import Path

def splitting_data():
    
    dir_path = Path(__file__).parent.parent.parent.absolute()
    weather_path = dir_path / 'Data/Farming Season/Weather(Farming Season).csv'
    yield_path = dir_path / 'Data/Farming Season/RandomCropYield (Farming Season).csv'

    weather_df = pd.read_csv(weather_path)
    yield_df = pd.read_csv(yield_path)

    X_train = weather_df.loc[weather_df['Year'] < 2017].drop(columns=['Year', 'Province'])
    y_train = yield_df.loc[yield_df['Year'] < 2017].drop(columns=['Year', 'Province'])

    test_set = []

    for year in range(2017, 2021):
        X_test = weather_df.loc[weather_df['Year'] == year].drop(columns=['Year', 'Province'])
        y_test = yield_df.loc[yield_df['Year'] == year].drop(columns=['Year', 'Province'])
        test_set.append((X_test, y_test))

    return X_train, y_train, test_set
