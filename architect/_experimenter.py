import sys
sys.path.insert(0,'C:\\Users\\analytic6\\Desktop\\Work Space Analitic 6 (Asir)')
sys.path.insert(0,'C:\\Users\\User\\Desktop\\work')

from AMLpp.conveyor import Conveyor

from typing import List
import pandas as pd
import pickle 
import os

class Experimenter():

    def __init__(self, experiment:str):
        self.path_experiment = "experiments/" + experiment
        if not os.path.exists(self.path_experiment):
            os.makedirs(self.path_experiment)
            self.model = None
        else:
            self.model = self._load_model()
            print("load model successful!" if self.model else "model not found!")

    def create_experiment(self, 
                            model:Conveyor, 
                            description:str,
                            trainset:str, 
                            X_test:pd.DataFrame = None, 
                            Y_test:pd.DataFrame = None,
                            testset_name:str = "",
                            feature_importances:bool = True,
                            X_test_features:List[str] = None):

        with open(self.path_experiment + "/model", 'wb') as file:
            pickle.dump(model, file)
        self.model = model

        self.add_description(description + "\ntrainset = {}".format(trainset), 'w')
        if X_test:
            self.make_experiment(X_test, Y_test, testset_name,
                        feature_importances = feature_importances, X_test_features = X_test_features)

        
    def make_experiment(self, 
                            X_test:pd.DataFrame,
                            Y_test:pd.DataFrame = None, 
                            testset_name:str = "", 
                            add_description:str = "", 
                            feature_importances:bool = True,
                            X_test_features:List[str] = None):
        if self.model:
            score, pred, Y = self.model.score(X_test, Y_test, _return = True) 
            description = "\n*"*60
            description += "\ntestset = " + testset_name
            description += "\n" + score
            description += add_description
            self.add_description(description)
            print(description)

            result_data = X_test[X_test_features] if X_test_features else pd.DataFrame()
            result_data['target'] = Y
            result_data['result'] = pred
            result_data.to_excel(self.path_experiment + "/{}.xlsx".format(testset_name))
            
            if feature_importances:
                pass
        else:
            print("You need to start to the experiment !")
            print("Connect to existing experimnet or create experiment !")

    def add_description(self, add_description:str, mod:str = "a"):
        with open(self.path_experiment + "/desc.txt", mod) as file:
            file.write(add_description)

    def _load_model(self) -> Conveyor:
        path_model = self.path_experiment + "/model"
        if os.path.exists(path_model):
             with open(path_model, 'rb') as file:
                    return pickle.load(file)
        else:
            return None