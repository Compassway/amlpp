from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import SGDRegressor, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

import numpy as np

##############################################################################
def _lightgbm_regressor(trial):
    return {'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.75,0.8,0.85]),
    'learning_rate': trial.suggest_categorical('learning_rate', [0.005,0.006,0.008,0.01,0.015,0.02,0.03]),
    'subsample': trial.suggest_categorical('subsample', [0.6,0.65,0.7,0.75,0.8,0.85]),
    'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
    'n_estimators': trial.suggest_int('n_estimators', 10000, 10000),
    'max_depth': trial.suggest_categorical('max_depth', [-1,10,20]),
    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
    'random_state': trial.suggest_int('random_state', 42, 42),
    'metric': trial.suggest_categorical('metric', ['rmse']),
    'num_leaves' : trial.suggest_int('num_leaves', 5, 300),
    'cat_smooth' : trial.suggest_int('cat_smooth', 1, 100)}

def _random_forest_regressor(trial):
    return {'max_features': trial.suggest_categorical('max_features', ["auto", "sqrt", "log2"]),
    'max_depth': trial.suggest_categorical('max_depth', np.append( np.arange(1,10,1), [None])),
    'min_samples_split': trial.suggest_int('min_samples_split', 2, 21),
    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    'n_estimators': trial.suggest_int('n_estimators', 100, 2000, 200),
    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 21),
    'random_state': trial.suggest_int('random_state', 42, 42),
    'verbose': trial.suggest_categorical('verbose', [0])}

def _decision_tree_regressor(trial):
    return {'min_samples_split': trial.suggest_int('min_samples_split', 2, 21),
    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 21),
    'random_state': trial.suggest_int('random_state', 42, 42),
    'max_depth': trial.suggest_int('max_depth', 1, 11),}

def _sgd_regressor(trial):
    return {'loss': trial.suggest_categorical('loss', ['squared_loss', 'huber', 'epsilon_insensitive']), 
        'power_t': trial.suggest_categorical('power_t', [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', ['invscaling', 'constant']),
        'l1_ratio': trial.suggest_categorical('l1_ratio', [0.25, 0.0, 1.0, 0.75, 0.5]),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'penalty': trial.suggest_categorical ('penalty', ['elasticnet']),
        'alpha': trial.suggest_categorical('alpha', [0.0, 0.01, 0.001]),
        'eta0': trial.suggest_categorical('eta0', [0.1, 1.0, 0.01]),
        'random_state': trial.suggest_int('random_state', 42, 42)}
        
# def _xgb_regressor(trial):
#     return {'learning_rate': trial.suggest_categorical('learning_rate', [1e-3, 1e-2, 1e-1, 0.5, 1.]),
#         'subsample': trial.suggest_categorical('subsample', np.arange(0.05, 1.01, 0.05)),
#         'objective': trial.suggest_categorical('objective', ['reg:squarederror']),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 21),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 600, 100), 
#         'verbosity': trial.suggest_categorical('verbosity', [0]),
#         'max_depth': trial.suggest_int('max_depth', 1, 11)}
        
def _xgb_regressor(trial):
    return {'learning_rate': trial.suggest_categorical('learning_rate', [1e-3, 1e-2, 1e-1, 0.3,  0.5, 1.]),
        'subsample': trial.suggest_categorical('subsample', np.arange(0.05, 1.01, 0.05)),
        'objective': trial.suggest_categorical('objective', ['reg:squarederror']),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 21),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, 100), 
        'verbosity': trial.suggest_categorical('verbosity', [0]),
        'max_depth': trial.suggest_int('max_depth', 1, 11),
        'alpha': trial.suggest_categorical('alpha', [1,10]),
        'booster': trial.suggest_categorical('booster', ['gbtree']),}
        # 'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinearили', 'dart'])}

def _ridge_cv(trial):
    return {'alphas': trial.suggest_categorical('alphas', [1e-3, 1e-2, 1e-1, 1]),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'normalize': trial.suggest_categorical('min_child_weight', [True, False])}

def _extra_trees_regressor(trial):
    return {'max_features': trial.suggest_categorical('max_features',  np.arange(0.05, 1.01, 0.05)),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 21),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, 100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 21),
        'random_state': trial.suggest_int('random_state', 42, 42)}

def _gradient_boosting_regressor(trial):
    return {'learning_rate' : trial.suggest_categorical('learning_rate', [1e-3, 1e-2, 1e-1, 0.5, 1.]),
        'max_features' : trial.suggest_categorical('max_features', np.arange(0.05, 1.01, 0.05)),
        'min_samples_split' : trial.suggest_categorical('min_samples_split', range(2, 21)),
        'subsample' : trial.suggest_categorical('subsample', np.arange(0.05, 1.01, 0.05)),
        'alpha' : trial.suggest_categorical('alpha', [0.75, 0.8, 0.85, 0.9, 0.95, 0.99 ]),
        'min_samples_leaf' : trial.suggest_categorical('min_samples_leaf', range(1, 21)),
        'loss' : trial.suggest_categorical('loss', ["ls", "lad", "huber", "quantile"]),
        'max_depth' : trial.suggest_categorical('max_depth', range(1, 11)),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, 100),
        'random_state': trial.suggest_int('random_state', 42, 42)}

def _ada_boost_regressor(trial):
    return {'n_estimators' : trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500, 600]),
        'learning_rate' : trial.suggest_categorical('learning_rate', [1e-3, 1e-2, 1e-1, 0.5, 1.]),
        'loss' : trial.suggest_categorical('loss', ["linear", "square", "exponential"]),
        'random_state': trial.suggest_int('random_state', 42, 42)}

def _kneighbors_regressor(trial):
    return {'n_neighbors' : trial.suggest_categorical('n_neighbors', range(1, 101)),
        'weights' : trial.suggest_categorical('weights', ["uniform", "distance"]),
        'p' : trial.suggest_categorical('p', [1, 2])}
    
def _linear_svr(trial):
    return {'C' : trial.suggest_categorical('C', [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]),
        'epsilon' : trial.suggest_categorical('epsilon', [1e-4, 1e-3, 1e-2, 1e-1, 1.]),
        'loss' : trial.suggest_categorical('loss', ["squared_epsilon_insensitive"]),
        'tol' : trial.suggest_categorical('tol', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        'dual' : trial.suggest_categorical('dual', [True, False]),
        'random_state': trial.suggest_int('random_state', 42, 42)}

sklearn_models = {
    RandomForestRegressor: _random_forest_regressor,
    LGBMRegressor:_lightgbm_regressor, 
    XGBRegressor: _xgb_regressor,
    DecisionTreeRegressor: _decision_tree_regressor,
    ExtraTreesRegressor: _extra_trees_regressor,
    GradientBoostingRegressor: _gradient_boosting_regressor,
    AdaBoostRegressor: _ada_boost_regressor,
    SGDRegressor: _sgd_regressor,
    RidgeCV: _ridge_cv,
    KNeighborsRegressor: _kneighbors_regressor,
    LinearSVR: _linear_svr
    }










