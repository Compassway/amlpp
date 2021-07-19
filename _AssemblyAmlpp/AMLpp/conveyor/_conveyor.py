from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

import sys
sys.path.insert(0, "C:\\Users\\analytic6\\Desktop\\Work Space Analitic 6 (Asir)\\AMLpp\\_AssemblyAmlpp\\AMLpp")
sys.path.insert(0,'C:\\Users\\analytic6\\Desktop\\Work Space Analitic 6 (Asir)')
sys.path.insert(0,'C:\\Users\\User\\Desktop\\work')

from typing import List, Callable

import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from tpot import TPOTRegressor

from additional import LGBOptimizer

from datetime import datetime
import lightgbm as lgb
import pandas as pd
import warnings
import optuna
import pickle
import shap
import time

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
        self._fit(X, Y)
        if feature_importances:
            self.feature_importances(X, Y, transform = False)

    # @lead_time
    def fit_transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        return self._fit(X, Y)

    def _fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        X_, Y_  = (X.copy(), Y.copy())

        pbar = tqdm.tqdm(self.blocks)
        for block in pbar:
            pbar.set_postfix({'transform': block.__class__.__name__})
            block.fit(X_, Y_)
            X_, Y_ = self._transform(block, X_, Y_)

        pbar.set_postfix({'transform': self.estimator.__class__.__name__})
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
                sklearn_function:List[str] = ['roc_auc_score', 'r2_score', 'accuracy_score'],
                precision_function:List[Callable] = [],
                _return:bool = False):
        """
        X:pd.DataFrame,
        Y:pd.DataFrame or pd.Series,
        sklearn_function:List[str] = ['roc_auc_score', 'r2_score', 'accuracy_score'],
        precision_function:List[Callable] = []
        """
        X_, Y_ = self.transform(X.copy(), Y.copy())
        result = self.estimator.predict(X_)
        score = ""
        for func in sklearn_function:
            try:
                exec('from sklearn.metrics import ' + func)
                score += "function - {} = {}\n".format(func, eval("{}(Y_, result)".format(func)))
            except Exception as e:
                score += "function - {} = ERROR: {}\n".format(func, e)
        for func in precision_function:
            try:
                score = "function - {} = {}\n".format(func.__name__, func(Y_, result))
            except Exception as e:
                score = "function - {} = ERROR: {}\n".format(func.__name__, e)

        if _return:
            return score, result, Y_
        else:
            print(score)
    # @lead_time
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
    # @lead_time
    def fit_model(self, 
                    X:pd.DataFrame, Y:pd.DataFrame or pd.Series,
                    X_test:pd.DataFrame = None, Y_test:pd.DataFrame = None,
                    type_model:str = 'regressor',
                    export_model:str = "default",
                    compare_model:bool = True,
                    tpot_params:dict  = {"generations":5, "population_size":50, "n_jobs":-1},
                    lgb_params:dict = {"n_trials":20,  "n_jobs" :-1}, categorical_columns:List[str] = []
                    ):


        X_train, Y_train = self.fit_transform(X, Y)

        if type(X_test) != type(None) and type(Y_test) != type(None):
            X_test, Y_test = self.transform(X_test, Y_test)
        else:
            X_test, Y_test = X_train, Y_train
            
        print('start fit tpot model !!!!')
        tpot_model, result = self.fit_model_tpot(X_train, Y_train, X_test, params = tpot_params)
        tpot_score = r2_score(Y_test, result)
        print(tpot_model,"\n",f"r2_score = {tpot_score}")
        time.sleep(1)
        print('start fit lgb model !!!!')
        lgb_model, result = self.fit_model_lgb(X_train, Y_train, X_test, Y_test, 
                                                categorical_columns = categorical_columns, params = lgb_params)
        lgb_score = r2_score(Y_test, result)
        print(lgb_model,"\n",f"r2_score = {lgb_score}")

        if tpot_score > lgb_score:
            for step in tpot_model.steps[:-1]:
                self.blocks.append(step)
            self.estimator = tpot_model[-1]
            print("BEST Conveyor: TPOT")
        else:
            self.estimator = lgb_model
            print("BEST Conveyor: LGB")
            
        print(self)
        if export_model != "":
            if export_model == "default":
                export_model = "model_" + datetime.now().strftime("%Y_%m_%d_m%M")
            with open(export_model, 'wb') as save_file:
                pickle.dump(self, save_file)

    def fit_model_tpot(self, X:pd.DataFrame, Y:pd.DataFrame, X_test:pd.DataFrame, params:dict = {}):
        tpot = TPOTRegressor(**params, random_state=42).fit(X, Y)
        make_pipe, import_libs = tpot.export('', get_pipeline=True)
        exec(import_libs)
        model = eval(make_pipe)
        model = model if (type(model) == Pipeline) else make_pipeline(model)
        return model.fit(X, Y), model.predict(X_test) if type(X_test) == pd.DataFrame else False

    def fit_model_lgb(self, X:pd.DataFrame, Y:pd.DataFrame, 
                            X_test:pd.DataFrame, Y_test:pd.DataFrame, 
                            categorical_columns:List[str] = None, params:dict = {}):
        study = optuna.create_study(direction='maximize')
        study.optimize(LGBOptimizer(X, Y, X_test, Y_test, categorical_columns = categorical_columns),
                                        **params, show_progress_bar = False)
        model = LGBMRegressor(**study.best_params, random_state=42).fit(X, Y)
        result = model.predict(X_test) if type(X_test) == pd.DataFrame else False 
        return model, result