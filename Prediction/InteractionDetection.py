import numpy as np 
from sklearn import datasets
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import root_mean_squared_error
from TimewiseKfold import split_train_test, split_train_validation

def initialize(X):
    
    ans = []
    
    for i in X:
        ans.append(list(i))
    
    return ans

def Step0(attribute_num, K):
    
    M = 1
    
    alphas_res = np.zeros((1, attribute_num)) + 0.5
    
    deltas_res = np.zeros((1, K))
    
    return alphas_res, deltas_res, M

def check_neighborhood(alpha):
    if (np.abs(alpha - 0.5).sum() <= 1):
        return True
    return False

def valid_elements(x):
    if (x == 0.5):
        return 0, 1
    elif (x == 1):
        return 0, 0.5 
    return 0.5, 1

def d_hop_neighborood(alpha_set, alpha, d, start):
    
    if (d == 0):
        if (check_neighborhood(alpha)):
            alpha_set.append(alpha)
        return 
    
    if (len(alpha) - start < d):
        return 
    
    for i in range(start, len(alpha)):
        a, b = valid_elements(alpha[i])
        
        new_alpha1 = np.array(alpha) 
        new_alpha2 = np.array(alpha)
        
        new_alpha1[i] = a 
        new_alpha2[i] = b
        
        d_hop_neighborood(alpha_set, new_alpha1, d - 1, i + 1)
        d_hop_neighborood(alpha_set, new_alpha2, d - 1, i + 1)

def alpha_generation(attribute_num):
    alpha_set = []
    
    alpha = np.zeros(attribute_num) + 0.5
    
    for d in range(0, 3):
        d_hop_neighborood(alpha_set=alpha_set, alpha=alpha, d=d, start=0)
    
    return alpha_set

def delta_generation(K):

    delta_set = []

    for i in range(0, K):

        delta = np.zeros(K)
        
        delta[i] = 1

        delta_set.append(delta)
    
    return delta_set

sigma_f = 0.5
sigma_l = 0.5
theta = 0.5

def linear(r):
    return r ** 2

def exponential(coef, r):
    #coef = 1 -> Exponential kernel
    #coef = 2 -> Squared exponential kernel
    return (sigma_f ** 2) * np.exp(-1 * (1 / coef) *  (r ** coef) / (sigma_l ** 2))

def matern(coef, r):
    #coef = 3 -> matern 3/2
    #coef = 5 -> matern 5/2
    if (coef == 3):
        return (sigma_f ** 2) * (1 + np.sqrt(3) * r / sigma_l) * np.exp(-1 * np.sqrt(3) * r / sigma_l)
    
    return (sigma_f ** 2) * (1 + np.sqrt(5) * r / sigma_l + np.sqrt(5) * (r ** 2) / (sigma_l ** 2)) * np.exp(-1 * np.sqrt(5) * r / sigma_l)

def quadratic(r):
    return (sigma_f ** 2) * (1 + (r ** 2) / (2 * theta * (sigma_l ** 2)))

def kernel_functions(r):
    
    return np.array([linear(r), exponential(1, r), exponential(2, r), matern(3, r), matern(5, r), quadratic(r)])

def interaction_type_detection(alpha):
    
    interaction_type = []
    
    for i in range(0, len(alpha)):
        if (alpha[i] != 0.5):
            interaction_type.append(i)
                
    if (len(interaction_type) > 2):
        print('There is an exeception that an alpha has more than 2 elements not equal to 0.5')
    
    return interaction_type

def interaction_contribution(sample, interaction_type, alpha, delta):
    
    r = 0

    if (len(interaction_type) == 2):
        j = interaction_type[0]
        l = interaction_type[1]
        r = (2 * alpha[j] - 1) * (sample[j] + alpha[j] - 1) + (2 * alpha[l] - 1) * (-sample[l] - alpha[l] + 1)
    else:
        j = interaction_type[0]
        r = (2 * alpha[j] - 1) * (sample[j] + alpha[j] - 1)
    
    return np.dot(delta, kernel_functions(r))

def adding_interaction(X, alphas=[], deltas=[], interaction_type=None):
    
    X_transformed = initialize(X)
    
    for i, alpha in enumerate(alphas):
        
        if (np.abs(alpha - 0.5).sum() == 0):
            continue
        
        if (interaction_type is None):
            interaction_type = interaction_type_detection(alpha=alpha)
        
        for sample in X_transformed:
            sample.append(interaction_contribution(sample=sample, interaction_type=interaction_type, alpha=alpha, delta=deltas[i]))
        
        #Reset value
        interaction_type = None
         
    return X_transformed

def transform_data(scaled_folds, selected_features, alphas=[], deltas=[]):
    
    transformed_folds = []
    
    for fold in scaled_folds:
        X_train, X_test, y_train, y_test = split_train_test(fold)
        X_train_reduced = X_train[selected_features].to_numpy()
        X_test_reduced = X_test[selected_features].to_numpy()
        
        X_train_transformed = adding_interaction(X=X_train_reduced, alphas=alphas, deltas=deltas)
        X_test_transformed = adding_interaction(X=X_test_reduced, alphas=alphas, deltas=deltas)
        
        transformed_folds.append({
            'X_train': X_train_transformed,
            'y_train': y_train,
            'X_test': X_test_transformed,
            'y_test': y_test
        })
    
    return transformed_folds
            
def RMSE(X_train, y_train, X_validation, y_validation):
    
    linear_model = LinearRegression().fit(X_train, y_train)
    
    RMSE_train = root_mean_squared_error(y_true=y_train, y_pred=linear_model.predict(X_train))
    RMSE_validation = root_mean_squared_error(y_true=y_validation, y_pred=linear_model.predict(X_validation))
    
    return RMSE_train, RMSE_validation

def RMSE_FOLDS(transformed_folds, alphas=[], deltas=[], interaction_type=None):
    
    RMSE_train_avg = 0
    RMSE_validation_avg = 0
    
    for fold in transformed_folds:
        X_train, X_test, y_train, y_test = split_train_test(fold)
        X_train, X_validation, y_train, y_validation = split_train_validation(X=X_train, y=y_train)
        # X_train, X_validation, y_train, y_validation = split_train_test(fold)
        
        X_train_added = adding_interaction(X=X_train, alphas=alphas, deltas=deltas, interaction_type=interaction_type)
        X_validation_added = adding_interaction(X=X_validation, alphas=alphas, deltas=deltas, interaction_type=interaction_type)
        
        RMSE_train_cur, RMSE_validation_cur = RMSE(X_train=X_train_added, y_train=y_train, X_validation=X_validation_added, y_validation=y_validation)
        
        RMSE_train_avg += RMSE_train_cur
        RMSE_validation_avg += RMSE_validation_cur 
        
    return RMSE_train_avg/len(transformed_folds), RMSE_validation_avg/len(transformed_folds)
        
def Evaluate(transformed_folds, alpha_set, delta_set, interaction_types):
    
    alpha_d = []
    delta_d = []
    
    RMSE_validation = 0
    RMSE_min = 999999999
    
    # transformed_folds = transform_data(scaled_folds=scaled_folds, selected_features=selected_features, alphas=alphas_res, deltas=deltas_res)
    
    for i, alpha in enumerate(alpha_set):
        for delta in delta_set:
            RMSE_train_cur, RMSE_validation_cur = RMSE_FOLDS(transformed_folds=transformed_folds, alphas=[alpha], deltas=[delta], interaction_type=interaction_types[i])
            
            if (RMSE_train_cur < RMSE_min):
                RMSE_min = RMSE_train_cur
                RMSE_validation = RMSE_validation_cur
                alpha_d = alpha 
                delta_d = delta
                # print(alpha_d, delta_d, RMSE_train_cur, RMSE_validation_cur)
    
    return alpha_d, delta_d, RMSE_validation

        
#folds: It has got 3 items, each item consists of 4 parts: X_train, X_test, y_train and y_test
#selected_features: a set of common features among 3 set of selected features which are determined from each fold by Elastic Net
#K: number of choosen kernel functions
def heuristic_interaction_detection(scaled_folds, selected_features, K):
    
    attribute_num = len(selected_features)
    
    # if (M == -1):
    alphas_res = []
    deltas_res = []
    M = 1
    # else:
    #     M += 1

    alpha_set = alpha_generation(attribute_num=attribute_num)
    
    interaction_types = []
    
    for alpha in alpha_set:
        interaction_types.append(interaction_type_detection(alpha=alpha))
    
    delta_set = delta_generation(K=K)
    
    while(True):
        
        # print("Round:", M)

        transformed_folds = transform_data(scaled_folds=scaled_folds, selected_features=selected_features, alphas=alphas_res, deltas=deltas_res)
        
        alpha_d, delta_d, RMSE_validatioin = Evaluate(transformed_folds=transformed_folds, alpha_set=alpha_set, delta_set=delta_set, interaction_types=interaction_types)
    
        if (RMSE_FOLDS(transformed_folds=transformed_folds)[1] > RMSE_validatioin):
            alphas_res.append(alpha_d)
            deltas_res.append(delta_d)
            M = M + 1
            # if (M == 9):
            #     break
            # break
        else:
            break
    
    
    transformed_folds = transform_data(scaled_folds=scaled_folds, selected_features=selected_features, alphas=alphas_res, deltas=deltas_res)
    rmse_train, rmse_validation = RMSE_FOLDS(transformed_folds=transformed_folds, alphas=alphas_res, deltas=deltas_res)
    
    return alphas_res, deltas_res, rmse_train, rmse_validation