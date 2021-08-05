from re import S
from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

import sys

from typing import List, Callable

import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from tpot import TPOTRegressor

# from ..additional import LGBOptimizer
from ..fit_model import *

from datetime import datetime
import lightgbm as lgb
import pandas as pd
import warnings

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import pickle
import shap
import time
import gc

import tqdm 

##############################################################################
class Conveyor:
    """ Подобие sklearn.Pipeline, адаптированный под простоту и добавленный функционал

    Parameters
    ----------
    *block : object
        Объекты классов, что будут использоваться при обработке, и моделирование

    """

    ##############################################################################

    def __init__(self, *blocks, estimator:object  = None, **params):
        self.blocks = list(blocks)
        self.estimator = estimator
        # self.iter = 0
        warnings.filterwarnings('ignore')
        
    def __repr__(self):
        _repr = self.__class__.__name__ + "= (\n"
        indent = " " * (len(_repr) - 1)
        for block in self.blocks:
            _repr += f"{indent}{repr(block)}, \n"
        _repr += f"{indent}estimator = {repr(self.estimator)}"
        _repr += f"\n{indent} )"
        return _repr

    # def __next__(self):
    #     if self.iter < len(self.blocks):
    #         self.iter +=1 
    #         return self.block[iter]
    #     else:
    #         self.iter = 0
    #         return StopIteration

    # def __getitem__(self, key):
    #     if isinstance(key, slice):
    #         return self.__class__(self.blocks[key])
    #     else:
    #         return self.blocks[key]
    ##############################################################################

    # @lead_time
    def fit(self, X:pd.DataFrame,
                  Y:pd.DataFrame or pd.Series,
                  feature_importances:str = False):
        self._fit(X, Y, estimator = True)
        if feature_importances:
            self.feature_importances(X, Y, transform = False)

    # @lead_time
    def fit_transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        return self._fit(X, Y)

    def _fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series, estimator:bool = False):
        X_, Y_  = (X.copy(), Y.copy())

        pbar = tqdm.tqdm(self.blocks)
        for block in pbar:
            pbar.set_postfix({'transform': block.__class__.__name__})
            block.fit(X_, Y_)
            X_, Y_ = self._transform(block, X_, Y_)

        pbar.set_postfix({'transform': self.estimator.__class__.__name__})
        if estimator:
            self.estimator.fit(X_, Y_)
        pbar.close()
        return X_, Y_
    ##############################################################################

    # @lead_time
    def transform(self,
                        X:pd.DataFrame,
                        Y:pd.DataFrame or pd.Series = pd.DataFrame()):
        X_, Y_  = (X.copy(), Y.copy())
        for block in self.blocks:
            X_, Y_ = self._transform(block, X_, Y_)
        return X_, Y_

    def _transform(self, 
                        block:Callable,
                        X:pd.DataFrame,
                        Y:pd.DataFrame or pd.Series = pd.DataFrame()):
        X = block.transform(X)
        if not Y.empty and 'target_transform' in dir(block):
            Y = block.target_transform(Y)
        return X, Y

    ##############################################################################

    # @lead_time
    def predict(self, X:pd.DataFrame):
        return self.estimator.predict(self.transform(X.copy())[0])

    ##############################################################################
    # @lead_time
    def score(self,
                X:pd.DataFrame,
                Y:pd.DataFrame or pd.Series,
                sklearn_function:List[str] = ['r2_score','roc_auc_score', 'accuracy_score', 'explained_variance_score'],
                precision_function:List[Callable] = [],
                _return:bool = False):
        """
        X:pd.DataFrame,
        Y:pd.DataFrame or pd.Series,
        sklearn_function:List[str] = ['roc_auc_score', 'r2_score', 'accuracy_score', 'explained_variance_score'],
        precision_function:List[Callable] = []
        """
        X_, Y_ = self.transform(X.copy(), Y.copy())
        result = self.estimator.predict(X_)

        score = ""
        for func in sklearn_function:
            score += self._get_score(eval(func), Y_, result)
        for func in precision_function:
            score += self._get_score(func, Y_, result)

        if _return:
            return score, result, Y_
        else:
            print(score)
    
    def _get_score(func:Callable, y:List[float], result:List[float]) -> str:
        try:
            return f"function - {func.__name__} = {func(y, result)}\n"
        except Exception as e:
            return f"function - {func.__name__} = ERROR: {e}\n"

        
    def feature_importances(self,
                            X:pd.DataFrame,
                            Y:pd.DataFrame or pd.Series, 
                            show:str = 'all', # all, sklearn, shap
                            save:bool = True,
                            name_plot:str = "",
                            transform = True): 
                            
        if transform:
            X_, Y_ = self.transform(X.copy(), Y.copy())

        if show == 'all' or show == 'shap':
            try:
                explainer = shap.Explainer(self.estimator)
                shap_values = explainer(X_)
                shap.plots.bar(shap_values[0], show = False)
                if save:
                    name_plot = name_plot if name_plot != "" else datetime.now().strftime("%Y-%m-%d_%M")
                    plt.savefig('{}_shap.jpeg'.format(name_plot), dpi = 150,  pad_inches=0)
                plt.show()
            except Exception as e:
                print('shap plot - ERROR: ', e)

        if show == "all" or show == "sklearn":
            try:
                result = permutation_importance(self.estimator, X_, Y_, n_repeats=2, random_state=42)
                index = X_.columns if type(X_) == pd.DataFrame else X.columns
                forest_importances = pd.Series(result.importances_mean, index=index)
                fig, ax = plt.subplots(figsize=(20, 10))
                forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                if save:
                    name_plot = name_plot if name_plot != "" else datetime.now().strftime("%Y-%m-%d_%M")
                    plt.savefig('{}_sklearn.jpeg'.format(name_plot))
                plt.show()
            except Exception as e:
                print('Sklearn plot - ERROR: ', e)

        if self.estimator.__class__.__name__ == "LGBMRegressor":
            lgb.plot_importance(self.estimator, figsize=(20, 10))
            plt.savefig('{}_lgb.jpeg'.format(name_plot))
            plt.show()
            
    ##############################################################################
    def fit_model(self, 
                    X:pd.DataFrame, Y:pd.DataFrame or pd.Series,
                    X_test:pd.DataFrame = None, Y_test:pd.DataFrame = None,
                    rating_func:str = 'r2_score', # sklearn metrics
                    optuna_params:dict = {}, 
                    categorical_columns:List[str] = []
                    ):

        rating_func = eval(rating_func)
        X_train, Y_train = self.fit_transform(X, Y)

        # if type(X_test) != type(None) and type(Y_test) != type(None):
        if not X_test is None and not Y_test is None:
            X_test, Y_test = self.transform(X_test, Y_test)
        else:
            X_test, X_test, Y_test, Y_test = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 42)
        #######################################################################
        # print("*"*100, '\n', 'start fit lgb model !!!!')
        # lgb_model, result = self.fit_lgbm_model(X_train, Y_train, X_test, Y_test, 
        #                                         categorical_columns = categorical_columns, 
        #                                         rating_func = rating_func,
        #                                         params = optuna_params)
        # lgb_score = rating_func(Y_test, result)
        # #######################################################################
        # #######################################################################
        # print("*"*100, '\n', 'start fit tpot model !!!!')
        tpot_model, result = self.fit_sklearn_model(X_train, Y_train, X_test, Y_test, rating_func, optuna_params)
        tpot_score = rating_func(Y_test, result)
        print("*"*100)
        print(tpot_model, f"\n{rating_func.__name__} = {tpot_score}")
        # #######################################################################
        # #######################################################################

        # if tpot_score > lgb_score:
        #     for step in tpot_model.steps[:-1]:
        #         self.blocks.append(step)
        #     self.estimator = tpot_model[-1]
        # else:
        #     self.estimator = lgb_model

        # print("*"*100, f'\nBest model = {self.estimator}')

        # with open("model_" + datetime.now().strftime("%Y_%m_%d_m%M"), 'wb') as save_file:
        #     pickle.dump(self, save_file)
    ########################################################################################
    def fit_sklearn_model(self, X:pd.DataFrame, Y:pd.DataFrame, 
                            X_test:pd.DataFrame, Y_test:pd.DataFrame,
                            rating_func:Callable = r2_score,
                            params:dict = {}):
        params = {**{"n_trials":100,  "n_jobs" :-1, 'show_progress_bar':False}, **params}
        
        best_model = {"model":object, "params":{}, "best_value":0}
        # try:
            # pbq = tqdm.tqdm(sklearn_models)
            # for model in tqdm.tqdm(sklearn_models):
        pb = bar(params['n_trials'] * len(sklearn_models))
        for model in sklearn_models:
            pb.set_postfix(model.__name__)
            study = optuna.create_study(direction='maximize')
            study.optimize(SklearnOptimizer(X, Y, X_test, Y_test, rating_func, params['n_trials'],
                                        model, sklearn_models[model], pb = pb)
                , callbacks=[lambda study, trial: gc.collect()], **params)
            currrent_model = {"model":model, "params":study.best_params, "best_value":study.best_value}

            # self._repr_dict_model(currrent_model)
            if study.best_value > best_model['best_value']:
                best_model = currrent_model.copy()                    
        # except:
        #     pass

        model = best_model['model'](**best_model['params']).fit(X, Y)
        result = model.predict(X_test)
        return model, result
    ########################################################################################

    def fit_lgbm_model(self, X:pd.DataFrame, Y:pd.DataFrame, 
                            X_test:pd.DataFrame, Y_test:pd.DataFrame, 
                            categorical_columns:List[str] = None,
                            rating_func:Callable = r2_score,
                            params:dict = {}):
        params = {**{"n_trials":100,  "n_jobs" :-1, 'show_progress_bar':False}, **params}

        params_columns = {"feature_name": list(X.columns)}
        params_columns['categorical_feature'] = [col for col in categorical_columns if col in params_columns['feature_name']]

        study = optuna.create_study(direction='maximize')
        study.optimize(
                        LGBMOptimizer(X, Y, 
                                    X_test, Y_test, 
                                    rating_func, params['n_trials'], 
                                    lightgbm_models[0], lightgbm_models[1],  params_columns)
                                    ,
                       callbacks=[lambda study, trial: gc.collect()], **params)
        model = lightgbm_models[0](**study.best_params)
        model.fit(X, Y, eval_set = [(X_test, Y_test)], verbose = False, **params_columns,  early_stopping_rounds = 300)
        result = model.predict(X_test)

        best_model = {"model":lightgbm_models[0], "params":study.best_params, "best_value":study.best_value}
        self._repr_dict_model(best_model)
        return model, result

    def _repr_dict_model(self, model:dict) -> str:
        params = str(model['params'])[1:-1]
        params = params.replace(':', " =").replace("'", "")
        print(f"{model['model'].__name__}({params})\nbest_value = {model['best_value']}")


class bar():
    def __init__(self, n):
        self.pbar = tqdm.tqdm(total = n)
        self.postfix = ""
    def set_postfix(self, postfix:str):
        self.postfix = postfix
        
    def up(self):
        self.pbar.set_postfix({'model':self.postfix}, refresh=True)
        self.pbar.update(1)

        #         pbar = tqdm.tqdm(self.blocks)
        # for block in pbar:
        #     pbar.set_postfix({'transform': block.__class__.__name__})
        #     block.fit(X_, Y_)
        #     X_, Y_ = self._transform(block, X_, Y_)

        # pbar.set_postfix({'transform': self.estimator.__class__.__name__})
        # if estimator:
        #     self.estimator.fit(X_, Y_)
        # pbar.close()
        # return X_, Y_