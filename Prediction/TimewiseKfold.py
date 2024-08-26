import pandas as pd 
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

FOLD_NUM = 3
PROVINCES_NUM = 24

def splitting_data(is_scaled=False, is_NDVI=False):
    
    dir_path = Path(__file__).parent.parent.absolute()
    weather_path = dir_path / 'Data/Weather.csv'
    yield_path = dir_path / 'Data/RandomCropYield(Regression).csv'
    ndvi_path = dir_path / 'Data/NDVI.csv'
    
    weather_df = pd.read_csv(weather_path)
    ndvi_df = pd.read_csv(ndvi_path)
    yield_df = pd.read_csv(yield_path)
    
    #Removing data of Ben Tre and Tra Vinh in 2020
    remove_index = list(weather_df[(weather_df['Year'] == 2020) & (weather_df['Province'].isin(['Ben_Tre']))].index)
    # print(remove_index)
    
    if (is_NDVI == False):
        X = weather_df.drop(columns=['Year', 'Province'])
    else:
        X = ndvi_df.drop(columns=['Year', 'Province'])
        
    y = yield_df.drop(columns=['Year', 'Province'])
    
    if (is_scaled):
        X_scaled = MinMaxScaler().fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
    
    folds = []
    
    tscv = TimeSeriesSplit(n_splits=FOLD_NUM, test_size=PROVINCES_NUM)
    
    
    for (train_index, test_index) in (tscv.split(y)):
        
        test_index = list(test_index)
        if (test_index.count(remove_index[0]) > 0):
            test_index.remove(remove_index[0])
        # if (test_index.count(remove_index[1]) > 0):
        #     test_index.remove(remove_index[1])
        
        folds.append({
            'X_train': X.iloc[train_index],
            'y_train': y.iloc[train_index],
            'X_test': X.iloc[test_index],
            'y_test': y.iloc[test_index]
        })
        # print(type(train_index))
        # print(train_index, test_index)
    
    return folds

def split_train_test(fold):
    return fold['X_train'], fold['X_test'], fold['y_train'], fold['y_test']

def split_train_validation(X, y):
    X_train = X[:(len(X) - PROVINCES_NUM)]
    y_train = y[:(len(X) - PROVINCES_NUM)]
    
    X_validation = X[-PROVINCES_NUM:]
    y_validation = y[-PROVINCES_NUM:]
    
    return X_train, X_validation, y_train, y_validation
