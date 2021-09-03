from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from amlpp.conveyor import Conveyor

from datetime import datetime
from typing import List

import pandas as pd
import pickle
import os
##############################################################################
class Experimenter():
    """ The class for working with the structure of experiments in the project
    Parameters
    ----------
    experiment: str
        Experiment name
    """
    def __init__(self, experiment:str):
        self.experiment = experiment
        self.path_experiment = "experiments/" + experiment
        if not os.path.exists(self.path_experiment):
            os.makedirs(self.path_experiment)
            self.model = None
        else:
            self.model = self._load_model()
            print("load model successful!" if self.model else "model not found!")

    def create_experiment(self, model:Conveyor, description_model:str, description_trainset:str):
        """ Creation of an experiment
        Parameters
        ----------
        model: Conveyor
            Trained model
        description_model: str
            Description of the experiment, model, projects and other significant details
        description_trainset: str
            Name or path to training set
        """
        with open(self.path_experiment + "/model", 'wb') as file:
            pickle.dump(model, file)

        self.model = model

        description = description_model 
        description += f"\nTrainset: {description_trainset}"
        description += f"\nModel:\n{repr(self.model)}"

        self.add_description(self.path_experiment, description, 'w+')
        
    def make_experiment(self, 
                            X:pd.DataFrame, Y:pd.DataFrame,
                            expr_description:str = "", expr_name:str = "",
                            X_features:List[str] = [],
                            feature_importances:bool = True):
        """Carrying out an experiment on a test dataset
        Parameters
        ----------
        X_test : pd.DataFrame = None
            Test dataset, features (regressors)
        Y_test : pd.DataFrame = None
            Test dataset, targets
        description : str = ""
            Description of a specific test
        testset_name : str = ""
            Test dataset name, or description
        X_test_features : List [str] = None
            features from the test dataset that will be included in the result set
        feature_importances : bool = True
            Display charts or not
        scoring : bool = True
            save and print scoring table or not
        """
        if self.model:
            date = datetime.now().strftime("%d-%m-%y %H-%M-%S")
            path_current_experiment = f"{self.path_experiment}/{date} - {expr_name}"
            os.makedirs(path_current_experiment)
            

            x_, y_ = self.model.transform(X, Y)
            predict = self.model.estimator.predict(x_)

            score = ""
            for metr in (r2_score,  mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score):
                try:
                    score += f"function - {metr.__name__} = {metr(y_, predict)}\n"
                except Exception as e:
                    score += f"function - {metr.__name__} = ERROR: {e}\n"

            expr_name = f"({self.experiment}) {expr_name}"

            expr_description = f"{expr_description}\nScore:\n{score}"
            self.add_description(path_current_experiment, expr_description, 'w+')
            print(expr_description)

            result_data = X[X_features] if X_features else pd.DataFrame()
            result_data['target'] = y_
            result_data['result'] = predict
            result_data.to_excel(path_current_experiment + f"/{expr_name}.xlsx")
            
            if feature_importances:
                plot_path = path_current_experiment + f"/{expr_name}"
                self.model.feature_importances(X, X, save = True, name_plot = plot_path)

    def add_description(self, path:str, description:str, mod:str = "a"):
        """ add description
        Parameters
        ----------
        add_description: str = ""
            Description of a specific experiment
        mod: str = ""
            mod for working with files
        """
        with open(path + "/desc.txt", mod, encoding="utf-8") as file:
            file.write(description)

    def _load_model(self) -> Conveyor:
        """ Loading the model
        Returns
        ----------
        model: Conveyor or None
            Loaded work
        """
        path_model = self.path_experiment + "/model"
        if os.path.exists(path_model):
             with open(path_model, 'rb') as file:
                    return pickle.load(file)
        else:
            return None