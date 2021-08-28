import numpy as np

# increased negative impact
def ini_score(y_true, y_pred):
    score = 0
    for y, p in zip(y_true, y_pred):
        if y == 0:
            score += abs(y - p)
        else:
            score += (y - p)**2
    return 1 - score / len(y_true)

def default_score(y_true, y_pred):
    y_true['pred'] = y_pred
    score = 0
    for pred in np.arange(0, 1, 0.1):
        status_5 = len(y_true[(y_true['pred'] >= pred)  & (y_true['status_id'] == 5)]) 
        status_6 = len(y_true[(y_true['pred'] >= pred)  & (y_true['status_id'] == 6)])
        if status_5 + status_6 > 0:
            score += (status_6 / (status_5 + status_6)) * pred
    return 1 - score