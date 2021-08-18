# increased negative impact
def ini_score(y_true, y_pred):
    score = 0
    for y, p in zip(y_true, y_pred):
        if y == 0:
            score += abs(y - p)
        else:
            score += (y - p)**2
    return 1 - score / len(y_true)
