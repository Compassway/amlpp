from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from typing import List, Callable
import matplotlib.pyplot as plt
from tpot import TPOTRegressor
import pandas as pd
import numpy as np
import pymorphy2
import shelve
import time
import shap
import os 
############################################################################################################

def most_frequency(x:List[List[int]]):
    x_ = [i[0] for i in x]
    return np.argmax(np.bincount(x_))

def lead_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result =  func(*args, **kwargs)
        print('lead time {} = {:.3f}'.format(func.__name__, time.time() - start_time))
        return result 
    return wrapper
def shelve_save(save_data:object, key:str, path:str = 'model_property/data'):
    if not os.path.exists("/".join(path.split('/'))):
        os.makedirs("/".join(path.split('/')))
    with shelve.open(path) as save:
        save[key] = save_data

def shelve_load(key:str, path:str = 'model_property/data'):
    with shelve.open(path) as load:
        return load[key]
############################################################################################################
from sklearn.metrics import roc_auc_score
class Conveyor:
    """ Подобие sklearn.Pipeline, адаптированный под простоту и добавленный функционал

    Parameters
    ----------
    *block : object
        Объекты классов, что будут использоваться при обработке, и моделирование

    """
    ################################################################
    def __init__(self, *blocks, **params):
        self.blocks = list(blocks)
    
    @lead_time
    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        X_, Y_  = (X.copy(), Y.copy())
        for block in self.blocks[:-1]:
            block.fit(X_, Y_)
            X_, Y_ = self._transform(block, X_, Y_)
        self.blocks[-1].fit(X_, Y_)
        return X_, Y_

    @lead_time
    def fit_transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        X_, Y_  = (X.copy(), Y.copy())
        for block in self.blocks:
            block.fit(X_, Y_)
            X_, Y_ = self._transform(block, X_, Y_)
        return X_, Y_
    ################################################################
    @lead_time
    def transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series = pd.DataFrame()):
        X_, Y_  = (X.copy(), Y.copy())
        for block in self.blocks[:-1]:
            X_, Y_ = self._transform(block, X_, Y_)
        return X_, Y_

    def _transform(self, block, X:pd.DataFrame, Y:pd.DataFrame or pd.Series = pd.DataFrame()):
        X = block.transform(X)
        if not Y.empty and 'target_transform' in dir(block):
            Y = block.target_transform(Y)
        return X, Y

    ################################################################

    @lead_time
    def predict(self, X:pd.DataFrame):
        return self.blocks[-1].predict(self.transform(X.copy())[0])

    ################################################################
    @lead_time
    def score(self,
                X:pd.DataFrame,
                Y:pd.DataFrame or pd.Series,
                sklearn_function:List[str] = ['roc_auc_score', 'r2_score', 'accuracy_score'],
                precision_function:List[Callable] = []):

        X_, Y_ = self.transform(X.copy(), Y.copy())
        result = self.blocks[-1].predict(X_)

        for func in sklearn_function:
            try:
                exec('from sklearn.metrics import ' + func)
                print("function - {} = ".format(func), eval("{}(result, Y_)".format(func)))
            except Exception as e:
                print("function - {} = ERROR: {}".format(func, e))
        for func in precision_function:
            try:
                print("function - {} = ".format(func.__name__), func(result, Y_))
            except Exception as e:
                print("function - {} = ERROR: {}".format(func.__name__, e))
    @lead_time
    def feature_importances(self,
                            X:pd.DataFrame,
                            Y:pd.DataFrame or pd.Series, show:str = 'all'): # all, sklearn, shap
                            
        X_, Y_ = self.transform(X.copy(), Y.copy())
        estimator = self.blocks[-1][-1] if type(self.blocks[-1]) == Pipeline else self.blocks[-1]

        if show == 'all' or show == 'shap':
            explainer = shap.Explainer(estimator)
            shap_values = explainer(X_)
            shap.plots.bar(shap_values[0])

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
    ################################################################
    @lead_time
    def fit_model(self, 
                    X:pd.DataFrame, Y:pd.DataFrame or pd.Series,
                    type_model:str = 'regressor', estimator:bool = False,
                    generations:int = 5, population_size:int = 50, n_jobs:int = -1):

        tpot = TPOTRegressor(generations=1, population_size=20, n_jobs = -1, random_state=42)
        X_, Y_ = self.fit_transform(X, Y) if not estimator else self.fit(X, Y)
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
    ################################################################
    @lead_time
    def export(self):
        pass

class CategoricalEncoder():
    """ Класс кодирования категориальных данных, с заполнение пропусков на некоторое значение определенное сратегией

    Parameters
    ----------
    columns : List[str]
        Названия столбцов, которые будут подвегнуты обработке

    straegy : str
        Строка указывающая на используемую стратегию заполнения пропусков
    
    fill_value : float or str
        Заполнитель, которым будут заполняться пропущенные значения,
        при использование стратегии const
        
    """
    encoder = {}

    def __init__(self, columns:List[str], strategy:str='mean', fill_value:float or str = np.nan): # strategy in mean, median, most_frequency, const, iterative inputer?
        self.columns = columns
        self.fill_value = {'mean':np.mean, 'median':np.median, 'most_freq':most_frequency, 'const':(lambda x:fill_value)}
        self.fill_value = self.fill_value[strategy]

    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        for column in self.columns:
            self.encoder[column] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
            X_fit = pd.DataFrame(X[column].loc[~X[column].isnull()])
            self.encoder[column].fit(X_fit)
            X_transform = self.encoder[column].transform(pd.DataFrame(X_fit))
            self.encoder[column].unknown_value = self.fill_value(X_transform)
        shelve_save(self.encoder, 'CategoricalEncoder')
        return self
    def transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series = None):
        self.encoder = shelve_load('CategoricalEncoder')
        for column in self.columns:
            X[column] = self.encoder[column].transform(pd.DataFrame(X[column].fillna('NAN')))
        return X

# class Imputer():
class user_transform():
    def __init__(self):
        self.data = 1

    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        return self

    def transform(self, X:pd.DataFrame):
        return X
    
    def target_transform(self, Y:pd.DataFrame or pd.Series):
        Y['target'] = [Y.loc[i, 'target']*-1 for i in range(len(Y['target']))]
        return Y
