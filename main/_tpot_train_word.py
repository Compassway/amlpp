import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from tpot import TPOTRegressor

X_train = pd.read_excel('X_train_word.xlsx',engine='openpyxl')
#X_test = pd.read_excel('X_test_vec.xlsx',engine='openpyxl')
Y_train = pd.read_excel('Y_train_word.xlsx',engine='openpyxl')['overdue_days']
#Y_test = pd.read_excel('Y_test.xlsx',engine='openpyxl')['overdue_days']
del X_train['Unnamed: 0']
X_train = X_train.values
#X_test = X_test.values
Y_train = Y_train.values
print(X_train)
print(Y_train)
#Y_test = Y_test.values


tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42, n_jobs = -1)
tpot.fit(X_train, Y_train)
tpot.export('C:\\Users\\analytic6\\Desktop\\analytic6\\tpot_main_regressor_pipeline.py')
#print(tpot.score(X_test, Y_test))
#Y_pred = tpot.predict(X_test)
#print(accuracy_score(Y_test,Y_pred))


input('exit')