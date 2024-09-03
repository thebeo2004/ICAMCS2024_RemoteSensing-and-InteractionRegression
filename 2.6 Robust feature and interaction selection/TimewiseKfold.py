import pandas as pd 
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import numpy as np

FOLD_NUM = 3
MAXIMUM_YEAR = 2020

RedRiverDelta = ['Ha_Noi', 'Hai_Duong', 'Hung_Yen', 'Nam_Dinh', 'Ninh_Binh', 'Thai_Binh', 'Ha_Nam']
MekongRiverDelta = [ 'Long_An', 'Tien_Giang', 'Ben_Tre', 'Tra_Vinh', 'Dong_Thap', 'An_Giang', 'Kien_Giang', 'Can_Tho', 'Soc_Trang', 'Bac_Lieu', 'Ca_Mau']
NorthSouthCentralCoast = ['Thanh_Hoa', 'Nghe_An', 'Ha_Tinh', 'Quang_Binh', 'Quang_Tri', 'Hue']

def remove_province(df, province):
    return df.drop(df[df['Province'] == province].index)

def remove_year(df, year):
    return df.drop(df[df['Year'] == year].index)

def remove(df, province, year):
    return df.drop(df[(df['Province'] == province) & (df['Year'] == year)].index)

def filtering(weather_df, ndvi_df, yield_df, region):
    df1 = weather_df.loc[weather_df['Province'].isin(region)]
    df2 = ndvi_df.loc[ndvi_df['Province'].isin(region)]
    df3 = yield_df.loc[yield_df['Province'].isin(region)]
    
    return df1, df2, df3
    
def splitting_data(is_scaled=False, is_NDVI=False, region=np.nan):
    
    #region = 0: Red River Delta
    #region = 1: North Central & South Central Coast
    #region = 2: Mekong River Delta
    
    if (region != region):
        print('Entering region number')
        return False
    
    if (region != 0 and region != 1 and region != 2):
        print('Invalid region number')
        return False
    
    dir_path = Path(__file__).parent.parent.absolute()
    weather_path = dir_path / 'Data/Weather.csv'
    yield_path = dir_path / 'Data/RandomCropYield(Regression).csv'
    ndvi_path = dir_path / 'Data/NDVI.csv'
    
    weather_df = pd.read_csv(weather_path)
    ndvi_df = pd.read_csv(ndvi_path)
    yield_df = pd.read_csv(yield_path)
    
    PROVINCES_NUM = 24
    
    if (region == 0):
        region = RedRiverDelta
        PROVINCES_NUM = 7
    elif(region == 1):
        region = NorthSouthCentralCoast
        PROVINCES_NUM = 6
    else:
        region = MekongRiverDelta
        PROVINCES_NUM = 11
    
    weather_df, ndvi_df, yield_df = filtering(weather_df=weather_df, ndvi_df=ndvi_df, yield_df=yield_df, region=region)
    
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
    
    for (i, (train_index, test_index)) in enumerate(tscv.split(y)):
        
        test_index = list(test_index)
        
        folds.append({
            'X_train': X.iloc[train_index],
            'y_train': y.iloc[train_index],
            'X_test': X.iloc[test_index],
            'y_test': y.iloc[test_index]
        })
    
    return folds, PROVINCES_NUM

def split_train_test(fold):
    return fold['X_train'], fold['X_test'], fold['y_train'], fold['y_test']

def split_train_validation(X, y, PROVINCES_NUM):
    X_train = X[:(len(X) - 1 * PROVINCES_NUM)]
    y_train = y[:(len(X) - 1 * PROVINCES_NUM)]
    
    X_validation = X[-1 * PROVINCES_NUM:]
    y_validation = y[-1 * PROVINCES_NUM:]
    
    return X_train, X_validation, y_train, y_validation
