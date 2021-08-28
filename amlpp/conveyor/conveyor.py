from sklearn.inspection import permutation_importance

from typing import List, Callable

from datetime import datetime

import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
import numpy as np
import warnings
import optuna
import pickle
import tqdm 
import shap
import time
import gc

from ..fit_model import *
from ..additional import *

optuna.logging.set_verbosity(optuna.logging.WARNING)
##############################################################################
class Conveyor:
    """Conveyor consisting of blocks that carry processing of
    data passing inside the conveyor, and ending with a regressor
    Parameters
    ----------
    estimator : object = None
        Regressor that performs the prediction task
    * blocks : Transformes
        Transformers that carry out data processing
    """
    def __init__(self, *blocks, estimator:object  = None):
        self.blocks = list(blocks)
        self.estimator = estimator
        warnings.filterwarnings('ignore')
        
    def __repr__(self):
        _repr = self.__class__.__name__ + "= (\n"
        indent = " " * (len(_repr) - 1)
        for block in self.blocks:
            _repr += f"{indent}{repr(block)}, \n"
        _repr += f"{indent}estimator = {repr(self.estimator)}\n{indent} )"
        return _repr

    ##############################################################################
    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        """ Function that is responsible for filling the model with data and training the model
        Parameters
        ----------
        X : pd.DataFrame
            Input data, features (regressors)
        Y : pd.DataFrame or pd.Series
            Input data, targets
        """
        _, __ = self.fit_transform(X, Y, estimator = True)

    def fit_transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series, estimator:bool = False):
        """ Function that is responsible for filling the model with data and training the model, and returning the transformed
        Parameters
        ----------
        X : pd.DataFrame
            Input data, features (regressors)
        Y : pd.DataFrame or pd.Series
            Input data, targets
        estimator : bool
            fit estimator or not

        Returns
        ----------
        transformed data : list or pd.DataFrame
            Transformed data
        """
        X_, Y_  = (X.copy(), Y.copy())

        pbar = ProgressBar(len(self.blocks) + int(estimator))
        for block in self.blocks:
            pbar.set_postfix('transform', block.__class__.__name__)
            X_, Y_ = self._transform(block.fit(X_, Y_), X_, Y_)
            pbar.update()

        if estimator:
            pbar.set_postfix('transform', self.estimator.__class__.__name__)
            self.estimator.fit(X_, Y_)
            pbar.update()
        return X_, Y_

    ##############################################################################
    def transform(self,
                        X:pd.DataFrame,
                        Y:pd.DataFrame or pd.Series = pd.DataFrame()):
        """ Сonducting input data through transformers
        Parameters
        ----------
        X : pd.DataFrame
            Input data, features (regressors)
        Y : pd.DataFrame or pd.Series
            Input data, targets
        Returns
        ----------
        transformed data : list or pd.DataFrame
            Transformed data
        """
        X_, Y_  = (X.copy(), Y.copy())
        for block in self.blocks:
            X_, Y_ = self._transform(block, X_, Y_)
        return X_, Y_

    def _transform(self, 
                        block:Callable,
                        X:pd.DataFrame,
                        Y:pd.DataFrame or pd.Series = pd.DataFrame()):
        """ Using a transformer
        Parameters
        ----------
        block : Callable
            Transformer
        X : pd.DataFrame
            Input data, features (regressors)
        Y : pd.DataFrame or pd.Series
            Input data, targets
        Returns
        ----------
        transformed data : list or pd.DataFrame
            Transformed data
        """
        X = block.transform(X)
        if not Y.empty and 'target_transform' in dir(block):
            Y = block.target_transform(Y)
        return X, Y
        
    ##############################################################################
    def predict(self, X:pd.DataFrame):
        """ Getting the result
        Parameters
        ----------
        X : pd.DataFrame
            Input data, features (regressors)
        Returns
        ----------
        output : list
            prediction
        """
        return self.estimator.predict(self.transform(X.copy())[0])

    ##############################################################################
    def score(self,
                X:pd.DataFrame,
                Y:pd.DataFrame or pd.Series,
                sklearn_function:List[str] = ['r2_score','roc_auc_score', 'accuracy_score', 'explained_variance_score'],
                precision_function:List[Callable] = []):
        """ Function of obtaining an estimate on test data
        Parameters
        ----------
        X : pd.DataFrame
            Input data, features (regressors)
        Y : pd.DataFrame or pd.Series
            Input data, targets
        sklearn_function : List [str] = ['r2_score','roc_auc_score', 'accuracy_score', 'explained_variance_score']
            sklearn function to evaluate
        precision_function : List [Callable] = []
            custom evaluation functions
        """
        X_, Y_ = self.transform(X.copy(), Y.copy())
        result = self.estimator.predict(X_)

        score = ""
        for func in sklearn_function:
            score += self._get_score(eval(func), Y_, result)
        for func in precision_function:
            score += self._get_score(func, Y_, result)

        print(score)
    
    def _get_score(self, func:Callable, y:List[float], result:List[float]) -> str:
        try:
            return f"function - {func.__name__} = {func(y, result)}\n"
        except Exception as e:
            return f"function - {func.__name__} = ERROR: {e}\n"
        
    ##############################################################################
    def feature_importances(self,
                            X:pd.DataFrame, Y:pd.DataFrame or pd.Series, 
                            show:str = ['sklearn', 'eli5', "lgbm"],
                            save:bool = True,
                            name_plot:str = ""): 
        """Plotting feature importances
         Parameters
         ----------
         X : pd.DataFrame
             Input data, features (regressors)
         Y : pd.DataFrame or pd.Series
             Input data, targets
         show : str = 'all' variants = ['sklearn', 'shap', 'lgbm', 'eli5'] or ['all']
             Type of graph shown
         save : bool = True
             Save graphs as images
         name_plot: str = ""
             Chart name
         """
        X_, Y_ = self.transform(X.copy(), Y.copy())
        
        name_plot = name_plot if name_plot != "" else datetime.now().strftime("%Y-%m-%d_%M")

        if ("all" in show  or "lgbm" in show) and self.estimator.__class__.__name__ == "LGBMRegressor":
            lgb.plot_importance(self.estimator, figsize=(20, 10))
            if save:
                plt.savefig(f'{name_plot}_lgb.jpeg')
            plt.show()

        if 'all' in show  or 'shap' in show:
            try:
                explainer = shap.Explainer(self.estimator)
                shap_values = explainer(X_)
                shap.plots.bar(shap_values[0], show = False)
                if save:
                    plt.savefig(f'{name_plot}_shap.jpeg', dpi = 150,  pad_inches=0)
                plt.show()
            except Exception as e:
                print('shap plot - ERROR: ', e)

        if "all" in show  or "sklearn" in show:
            try:
                result = permutation_importance(self.estimator, X_, Y_, n_repeats=2, random_state=42)
                index = X_.columns if type(X_) == pd.DataFrame else X.columns
                forest_importances = pd.Series(result.importances_mean, index=index)
                fig, ax = plt.subplots(figsize=(20, 10))
                forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
                fig.tight_layout()
                if save:
                    plt.savefig(f'{name_plot}_sklearn.jpeg')
                plt.show()
            except Exception as e:
                print('Sklearn plot - ERROR: ', e)
            
    ##############################################################################
    def fit_model(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series, type_model:str,
                    verbose:bool = True,
                    optuna_params:dict = {}, optimize_params:dict = {},
                    categorical_columns:List[str] = []):
        """ Model selection
        Parameters
        ----------
        X : pd.DataFrame
            Input data, features (regressors)
        Y : pd.DataFrame or pd.Series
            Input data, targets
        optuna_params : dict = {"n_trials": 100, "n_jobs": -1, 'show_progress_bar': False}
            Optuna optimizer parameters for select sklearn and lgbm model
        categorical_columns : List [str] = []
            Category column names for lgbm optuna optimizer
        """
        optuna_params = {"n_trials":100,  "n_jobs" :-1, 'show_progress_bar':False, **optuna_params}
        best_model = {"model":object, "params":{}, "best_value":0 }
        X_, Y_ = self.fit_transform(X, Y)

        # Дополнительные параметры для lgbm модели
        params_columns = {'verbose':False}
        if type(X_) == pd.DataFrame:
            params_columns["feature_name"] =  list(X_.columns)
            params_columns['categorical_feature'] = [col for col in categorical_columns if col in params_columns['feature_name']]

        try:
            models = models_regression if type_model == 'regression' else models_classification

            pb = ProgressFitModel(optuna_params['n_trials'] * len(models), best_model['best_value'])
            for model in models:
                pb.set_postfix('model', model.__name__)
                study = optuna.create_study(direction="maximize")

                if model.__name__ == 'LGBMRegressor':
                    add_params = params_columns
                else:
                    add_params = {}
                    
                study.optimize(Optimizer(X_, Y_, model, models[model], add_params, pb, **optimize_params
                              ), callbacks=[lambda study, trial: gc.collect()], **optuna_params)

                currrent_model = {"model":model, "params":study.best_params, "best_value":study.best_value}
                self._update_fit_model_log(currrent_model)
                if study.best_value > best_model['best_value']:
                    best_model = currrent_model
                    print(self._repr_dict_model(best_model))

        except Exception as e:
            print(e)
            if str(e) == "Too small sample for cross validation!":
                raise e
            print(e)

        model = best_model['model'](**best_model['params']).fit(X_, Y_)
        print(self._repr_dict_model(best_model))

        self.estimator = model
        print("*"*100, f'\nBest model = {self.estimator}')
        with open("model_" + datetime.now().strftime("%Y_%m_%d_m%M"), 'wb') as save_file:
            pickle.dump(self, save_file)

    ##############################################################################

    # def _direction_study(self, func:str):
    #     minimize = ['max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error',
    #                 'neg_root_mean_squared_error', 'neg_mean_squared_log_error', 
    #                 'neg_median_absolute_error', 'neg_mean_poisson_deviance', 
    #                 'neg_mean_gamma_deviance', 'neg_mean_absolute_percentage_error', 
    #                 'neg_brier_score']

    #     return  'minimize' if func in minimize else 'maximize'        
    def _update_fit_model_log(self, model:dict):
        with open('fit_model_log.txt', 'a+') as file:
            text = self._repr_dict_model(model) + "\n"
            text += "*"*100 + "\n"
            file.write(text)

    def _repr_dict_model(self, model:dict) -> str:
        params = str(model['params'])[1:-1]
        params = params.replace(':', " =").replace("'", "")
        return f"{model['model'].__name__}({params})\nbest_value = {model['best_value']}"