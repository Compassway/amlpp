from sklearn.metrics import r2_score

from lightgbm import LGBMRegressor

from typing import List

import pandas as pd

class LGBOptimizer(object):
    def __init__(self, X_train:pd.DataFrame, Y_train:pd.DataFrame, 
                       X_test:pd.DataFrame = None, Y_test:pd.DataFrame = None,
                       categorical_columns:List[str] = []
                       ):

        self.X_train, self.Y_train = X_train, Y_train
        self.X_test, self.Y_test = X_test, Y_test
        if categorical_columns != []:
            self.all_columns = list(X_train.columns)
            self.categorical_columns = [col for col in categorical_columns if col in self.all_columns]
        else:
            self.all_columns = None
            self.categorical_columns = None

    def __call__(self, trial):
        params = {
        'metric': 'rmse', 
        'random_state':42,
        'n_estimators': 10000,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.75,0.8,0.85]),
        'subsample': trial.suggest_categorical('subsample', [0.6,0.65,0.7,0.75,0.8,0.85]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005,0.006,0.008,0.01,0.015,0.02,0.03]),
        'max_depth': trial.suggest_categorical('max_depth', [-1,10,20]),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth' : trial.suggest_int('cat_smooth', 1, 100)
        }
        params_columns = {}
        model = LGBMRegressor(**params)

        if self.all_columns != None and self.categorical_columns != None:
            params_columns = {"feature_name":self.all_columns, "categorical_feature":self.categorical_columns}

        model.fit(self.X_train, self.Y_train, eval_set = [(self.X_test, self.Y_test)], verbose = False, 
                early_stopping_rounds=300, **params_columns)

        return r2_score(self.Y_test, model.predict(self.X_test))