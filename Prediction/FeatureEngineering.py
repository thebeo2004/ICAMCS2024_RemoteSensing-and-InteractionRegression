from TimewiseKfold import *
from Metrics import *
from InteractionDetection import *
import numpy as np
import time

scaled_folds = splitting_data(is_scaled=True)
features = ['Eva_I', 'Eva_II', 'Eva_III', 'Eva_IV', 'Eva_V', 'Eva_VI', 'Eva_VII', 'Eva_VIII', 'Eva_IX', 'Eva_X', 'Eva_XI', 'Eva_XII', 'Prcp_I', 'Prcp_II', 'Prcp_III', 'Prcp_IV', 'Prcp_V', 'Prcp_VI', 'Prcp_VII', 'Prcp_VIII', 'Prcp_IX', 'Prcp_X', 'Prcp_XI', 'Prcp_XII', 'Prcp_max_I', 'Prcp_max_II', 'Prcp_max_III', 'Prcp_max_IV', 'Prcp_max_V', 'Prcp_max_VI', 'Prcp_max_VII', 'Prcp_max_VIII', 'Prcp_max_IX', 'Prcp_max_X', 'Prcp_max_XI', 'Prcp_max_XII', 'Rdays_I', 'Rdays_II', 'Rdays_III', 'Rdays_IV', 'Rdays_V', 'Rdays_VI', 'Rdays_VII', 'Rdays_VIII', 'Rdays_IX', 'Rdays_X', 'Rdays_XI', 'Rdays_XII', 'Tavg_I', 'Tavg_II', 'Tavg_III', 'Tavg_IV', 'Tavg_V', 'Tavg_VI', 'Tavg_VII', 'Tavg_VIII', 'Tavg_IX', 'Tavg_X', 'Tavg_XI', 'Tavg_XII', 'Tmax_avg_I', 'Tmax_avg_II', 'Tmax_avg_III', 'Tmax_avg_IV', 'Tmax_avg_V', 'Tmax_avg_VI', 'Tmax_avg_VII', 'Tmax_avg_VIII', 'Tmax_avg_IX', 'Tmax_avg_X', 'Tmax_avg_XI', 'Tmax_avg_XII', 'Tmin_avg_I', 'Tmin_avg_II', 'Tmin_avg_III', 'Tmin_avg_IV', 'Tmin_avg_V', 'Tmin_avg_VI', 'Tmin_avg_VII', 'Tmin_avg_VIII', 'Tmin_avg_IX', 'Tmin_avg_X', 'Tmin_avg_XI', 'Tmin_avg_XII', 'Tmax_I', 'Tmax_II', 'Tmax_III', 'Tmax_IV', 'Tmax_V', 'Tmax_VI', 'Tmax_VII', 'Tmax_VIII', 'Tmax_IX', 'Tmax_X', 'Tmax_XI', 'Tmax_XII', 'Tmin_I', 'Tmin_II', 'Tmin_III', 'Tmin_IV', 'Tmin_V', 'Tmin_VI', 'Tmin_VII', 'Tmin_VIII', 'Tmin_IX', 'Tmin_X', 'Tmin_XI', 'Tmin_XII', 'Havg_I', 'Havg_II', 'Havg_III', 'Havg_IV', 'Havg_V', 'Havg_VI', 'Havg_VII', 'Havg_VIII', 'Havg_IX', 'Havg_X', 'Havg_XI', 'Havg_XII', 'Hmin_I', 'Hmin_II', 'Hmin_III', 'Hmin_IV', 'Hmin_V', 'Hmin_VI', 'Hmin_VII', 'Hmin_VIII', 'Hmin_IX', 'Hmin_X', 'Hmin_XI', 'Hmin_XII', 'Srad_I', 'Srad_II', 'Srad_III', 'Srad_IV', 'Srad_V', 'Srad_VI', 'Srad_VII', 'Srad_VIII', 'Srad_IX', 'Srad_X', 'Srad_XI', 'Srad_XII']

selected_features = ['Eva_I', 'Eva_IV', 'Eva_X', 'Eva_XII', 'Prcp_I', 'Prcp_II', 'Prcp_III', 'Prcp_V', 'Prcp_VII', 'Prcp_VIII', 'Prcp_IX', 'Prcp_XI', 'Prcp_XII', 'Prcp_max_VI', 'Prcp_max_VIII', 'Prcp_max_X', 'Prcp_max_XI', 'Rdays_IX', 'Tmin_avg_II', 'Havg_XI', 'Hmin_II', 'Hmin_III', 'Hmin_IV', 'Hmin_VI', 'Hmin_VII', 'Hmin_X', 'Hmin_XII', 'Srad_IX', 'Tmax_avg_XII', 'Tmin_avg_IV', 'Tmax_VI', 'Havg_V', 'Srad_XI']

#After 2st round
selected_features += ['Eva_VI', 'Prcp_VI', 'Prcp_max_IV', 'Prcp_max_V', 'Tavg_XI', 'Tmin_V']

#After 3nd round
selected_features += ['Eva_II', 'Eva_VIII', 'Prcp_IV', 'Tavg_VII', 'Tmax_avg_VII', 'Tmax_IX', 'Srad_XII']
selected_features.remove('Prcp_max_XI')
selected_features.remove('Hmin_III')

#After 4th round
selected_features += ['Eva_III', 'Rdays_X', 'Tavg_I', 'Tmax_avg_I', 'Srad_VIII']
selected_features.remove('Eva_I')
selected_features.remove('Eva_X')
selected_features.remove('Prcp_I')
selected_features.remove('Hmin_II')

#After 5t round
selected_features += ['Eva_I', 'Eva_XI', 'Prcp_I', 'Prcp_max_IX', 'Rdays_VI', 'Tavg_II', 'Tmin_avg_VII', 'Tmin_XII', 'Havg_VIII']
selected_features.remove('Tmin_avg_II')
selected_features.remove('Hmin_X')
selected_features.remove('Tavg_VII')

#After 6th round, removing round is in progress
selected_features += ['Tmax_IV', 'Hmin_VIII']

start_time = time.time()

alphas_res = [
    # np.array([0. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    #      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 0.5, 0.5, 0.5, 0.5, 0.5,
    #      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    #      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    # np.array([1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    #      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    #      0.5, 0.5, 0.5, 0.5, 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    #      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    # np.array([1., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
] 
deltas_res = [
    # np.array([0., 1., 0., 0., 0., 0.]),
    # np.array([0., 0., 0., 0., 1., 0.]),
    # np.array([0, 0, 0, 0, 1, 0])
    # np.array([0, 0, 1, 0, 0, 0]),
    # np.array([0, 0, 0, 0, 1, 0]),
    # np.array([0, 0, 1, 0, 0, 0]),
    # np.array([0, 1, 0, 0, 0, 0])
]

heuristic_interaction_detection(scaled_folds=scaled_folds, selected_features=selected_features, K=6, alphas_res=alphas_res, deltas_res=deltas_res, M=2)

print(time.time() - start_time)