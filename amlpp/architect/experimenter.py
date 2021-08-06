from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from amlpp.conveyor import Conveyor

from datetime import datetime
from typing import List

import pandas as pd
import pickle
import os

##############################################################################
class Experimenter():

    def __init__(self, experiment:str):
        self.experiment = experiment
        self.path_experiment = "experiments/" + experiment
        if not os.path.exists(self.path_experiment):
            os.makedirs(self.path_experiment)
            self.model = None
        else:
            self.model = self._load_model()
            print("load model successful!" if self.model else "model not found!")

    def create_experiment(self, model:Conveyor, 
                          description_model:str, description_trainset:str):

        with open(self.path_experiment + "/model", 'wb') as file:
            pickle.dump(model, file)

        self.model = model

        description = description_model 
        description += f"\ntrainset = {description_trainset}"
        description += f"\n{repr(self.model)}"

        self.add_description(description, 'w')
        
    def make_experiment(self, 
                            X_test:pd.DataFrame, Y_test:pd.DataFrame,
                            description:str = "",
                            testset_name:str = "",
                            X_test_features:List[str] = None,
                            feature_importances:bool = True):
        if self.model:
            x_, y_ = self.transform(X_test, Y_test)
            res = self.estimator.predict(x_)
            score = ""
            for metr in (r2_score, roc_auc_score, accuracy_score):
                try:
                    score += f"function - {metr.__name__} = {metr(y_, res)}\n"
                except Exception as e:
                    score += f"function - {metr.__name__} = ERROR: {e}\n"

            testset_name = f"({self.experiment})" + testset_name

            description =  '\n' +"*"*60 + datetime.now()
            description += "\n" + description
            description += "\ntestset = " + testset_name
            description += "\nScore: " + score
            
            self.add_description(description)
            print(description)

            result_data = X_test[X_test_features] if X_test_features else pd.DataFrame()
            result_data['target'] = y_
            result_data['result'] = res
            result_data.to_excel(self.path_experiment + f"/{testset_name}.xlsx")
            
            if feature_importances:
                plot_path = self.path_experiment + f"/{testset_name}.jpeg"
                self.model.feature_importances(x_, y_, save = True, name_plot = plot_path, transform = False)
        else:
            print("You need to start to the experiment !")
            print("Connect to existing experimnet or create experiment !")

    def add_description(self, add_description:str, mod:str = "a"):
        with open(self.path_experiment + "/desc.txt", mod, encoding="utf-8") as file:
            file.write(add_description)

    def _load_model(self) -> Conveyor:
        path_model = self.path_experiment + "/model"
        if os.path.exists(path_model):
             with open(path_model, 'rb') as file:
                    return pickle.load(file)
        else:
            return None