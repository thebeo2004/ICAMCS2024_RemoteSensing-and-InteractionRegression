import pandas as pd 
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

FOLD_NUM = 3

def splitting_data(is_scaled=False):
    
    dir_path = Path(__file__).parent.parent.absolute()
    weather_path = dir_path / 'Data/Weather.csv'
    yield_path = dir_path / 'Data/RandomCropYield2.csv'
    
    weather_df = pd.read_csv(weather_path)
    yield_df = pd.read_csv(yield_path)
    
    X = weather_df.drop(columns=['Year', 'Province'])
    y = yield_df.drop(columns=['Year', 'Province'])
    
    if (is_scaled):
        X_scaled = MinMaxScaler().fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
    
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