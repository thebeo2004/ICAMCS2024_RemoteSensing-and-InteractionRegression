import pandas as pd 
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit

FOLD_NUM = 3

def splitting_data():
    
    dir_path = Path(__file__).parent.parent.absolute()
    weather_path = dir_path / 'Data/Weather.csv'
    yield_path = dir_path / 'Data/RandomCropYield.csv'
    
    weather_df = pd.read_csv(weather_path)
    yield_df = pd.read_csv(yield_path)
    
    X = weather_df.drop(columns=['Year', 'Province'])
    y = yield_df.drop(columns=['Year', 'Province'])
    
    folds = []
    
    tscv = TimeSeriesSplit(n_splits=FOLD_NUM, test_size=7)
    
    for (train_index, test_index) in (tscv.split(y)):
        folds.append({
            'X_train': X.iloc[train_index],
            'y_train': y.iloc[train_index],
            'X_test': X.iloc[test_index],
            'y_test': y.iloc[test_index]
        })
    
    return folds

def split_train_test(fold):
    return fold['X_train'], fold['X_test'], fold['y_train'], fold['y_test']

