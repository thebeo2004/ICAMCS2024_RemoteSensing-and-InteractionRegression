from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score
import numpy as np 
import pandas as pd

def flatten(y_actual, y_pred):
     return np.array(y_actual).flatten(), np.array(y_pred).flatten()

def relative_absolute_error(y_actual, y_pred):

    y_actual, y_pred = flatten(y_actual=y_actual, y_pred=y_pred)

    absolute_error = np.sum(np.abs(y_actual - y_pred))
    denominator = np.sum(np.abs(y_actual - np.mean(y_actual)))

    if (denominator == 0):
         print("Warning at RAE")

    return absolute_error/denominator

def relative_squared_error(y_actual, y_pred):

    y_actual, y_pred = flatten(y_actual=y_actual, y_pred=y_pred)

    squared_error = np.sum((y_actual - y_pred) ** 2)
    denominator = np.sum((y_actual - np.mean(y_actual)) ** 2)

    if (denominator == 0):
         print("Warning at RSE")

    return squared_error/denominator

def relative_root_mean_squared_error(y_actual, y_pred):
        
        RMSE = mean_squared_error(y_true=y_actual, y_pred=y_pred) ** 1/2
        
        y_actual, y_pred = flatten(y_actual=y_actual, y_pred=y_pred)
        denominator = np.mean(y_actual)

        if (denominator == 0):
             print("Warning at RRMSE")

        return RMSE/denominator
   
def r2(y_actual, y_pred, avg_y_train):
     
     y_actual, y_pred = flatten(y_actual=y_actual, y_pred=y_pred)
     
     rss = np.sum((y_actual - y_pred) ** 2)
     tss = np.sum((y_actual - avg_y_train) ** 2)

     r2 = 1 - rss/tss
     
     return r2

def metrics_calculation(y_actual, y_pred, y_train):
     
     avg_y_train = y_train['RandomCropYield'].mean()
     
     RMSE = root_mean_squared_error(y_actual, y_pred)
     RRMSE = relative_root_mean_squared_error(y_actual=y_actual, y_pred=y_pred) * 100
     R2_Score = r2(y_actual=y_actual, y_pred=y_pred, avg_y_train=avg_y_train)
     MSE = mean_squared_error(y_true=y_actual, y_pred=y_pred)
     MAE = mean_absolute_error(y_true=y_actual, y_pred=y_pred)
     RAE = relative_absolute_error(y_actual=y_actual, y_pred=y_pred)
     RSE = relative_squared_error(y_actual=y_actual, y_pred=y_pred)

     metrics = [RMSE, RRMSE, R2_Score, MSE, MAE, RAE, RSE]

     return metrics
     
def metrics_table():
     
     table = pd.DataFrame()

     columns = ['Year', 'RMSE', 'RRMSE (in %)', 'R2_Score', 'MSE', 'MAE', 'RAE', 'RSE']

     for i, col in enumerate(columns):
         table.insert(i, col, None)
           
     return table


    