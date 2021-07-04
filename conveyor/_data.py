from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline, make_pipeline

from AMLpp.additional.secondary_functions import lead_time

from typing import List, Callable

import matplotlib.pyplot as plt

from tpot import TPOTRegressor

import pandas as pd
import shap

##############################################################################
class Conveyor:
    """ Подобие sklearn.Pipeline, адаптированный под простоту и добавленный функционал

    Parameters
    ----------
    *block : object
        Объекты классов, что будут использоваться при обработке, и моделирование

    """

    ##############################################################################

    def __init__(self, *blocks, **params):
        self.blocks = list(blocks)
        
    def __repr__(self):
        text_repr = "STRUCTURE MODEL:\n"
        for iter in range(len(self.blocks)):
            text_repr += "{}. {} \n".format(iter,  repr(self.blocks[iter]))
        return text_repr

    ##############################################################################

    # @lead_time
    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series, feature_importances:str = False):
        self._fit(X, Y)
        if feature_importances:
            self.feature_importances(X, Y)

    # @lead_time
    def fit_transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        X_, Y_  = (X.copy(), Y.copy())
        for block in self.blocks:
            block.fit(X_, Y_)
            X_, Y_ = self._transform(block, X_, Y_)
        return X_, Y_

    def _fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        X_, Y_  = (X.copy(), Y.copy())
        for block in self.blocks[:-1]:
            block.fit(X_, Y_)
            X_, Y_ = self._transform(block, X_, Y_)
        self.blocks[-1].fit(X_, Y_)
        return X_, Y_
    ##############################################################################

    # @lead_time
    def transform(self,
                        X:pd.DataFrame,
                        Y:pd.DataFrame or pd.Series = pd.DataFrame()):
        X_, Y_  = (X.copy(), Y.copy())
        for block in self.blocks[:-1]:
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
        return self.blocks[-1].predict(self.transform(X.copy())[0])

    ##############################################################################
    # @lead_time
    def score(self,
                X:pd.DataFrame,
                Y:pd.DataFrame or pd.Series,
                sklearn_function:List[str] = ['roc_auc_score', 'r2_score', 'accuracy_score'],
                precision_function:List[Callable] = []):
        """
        X:pd.DataFrame,
        Y:pd.DataFrame or pd.Series,
        sklearn_function:List[str] = ['roc_auc_score', 'r2_score', 'accuracy_score'],
        precision_function:List[Callable] = []
        """
        X_, Y_ = self.transform(X.copy(), Y.copy())
        result = self.blocks[-1].predict(X_)

        for func in sklearn_function:
            try:
                exec('from sklearn.metrics import ' + func)
                print("function - {} = ".
                        format(func), eval("{}(result, Y_)".format(func)))
            except Exception as e:
                print("function - {} = ERROR: {}".format(func, e))
        for func in precision_function:
            try:
                print("function - {} = ".
                        format(func.__name__), func(result, Y_))
            except Exception as e:
                print("function - {} = ERROR: {}".format(func.__name__, e))
    # @lead_time
    def feature_importances(self,
                            X:pd.DataFrame,
                            Y:pd.DataFrame or pd.Series, 
                            show:str = 'all'): # all, sklearn, shap
                            
        X_, Y_ = self.transform(X.copy(), Y.copy())
        estimator = self.blocks[-1][-1] if type(self.blocks[-1]) == Pipeline else self.blocks[-1]

        if show == 'all' or show == 'shap':
            try:
                explainer = shap.Explainer(estimator)
                shap_values = explainer(X_)
                shap.plots.bar(shap_values[0])
            except Exception as e:
                print('shap plot - ERROR: ', e)

        if show == "all" or show == "sklearn":
            try:
                result = permutation_importance(estimator, X_, Y_, n_repeats=2, random_state=42)
                index = X_.columns if type(X_) == pd.DataFrame else X.columns
                forest_importances = pd.Series(result.importances_mean, index=index)
                fig, ax = plt.subplots(figsize=(20, 10))
                forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
                ax.set_title("Feature importances using permutation on full model")
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                plt.show()
            except Exception as e:
                print('Sklearn plot - ERROR: ', e)
    ##############################################################################
    # @lead_time
    def fit_model(self, 
                    X:pd.DataFrame, Y:pd.DataFrame or pd.Series,
                    type_model:str = 'regressor', estimator:bool = False,
                    generations:int = 5, population_size:int = 50, n_jobs:int = -1):

        tpot = TPOTRegressor(generations=1, population_size=20, n_jobs = -1, random_state=42)
        X_, Y_ = self.fit_transform(X, Y) if not estimator else self._fit(X, Y)
        tpot.fit(X_, Y_)
        make_pipe, import_libs = tpot.export('', get_pipeline=True)

        exec(import_libs)
        tpot_model = eval(make_pipe)
        tpot_model = tpot_model if (type(tpot_model) == Pipeline) else make_pipeline(tpot_model)

        if estimator:
            del self.blocks[-1]
        
        for step in tpot_model:
            self.blocks.append(step)
            self.blocks[-1].fit(X_, Y_)
            if step != tpot_model[-1]:
                X_, Y_ = self._transform(self.blocks[-1], X_, Y_)
            
        self.blocks[-1].fit(X_, Y_)
        print(self.blocks)
    ##############################################################################
    # @lead_time
    def export(self):
        pass