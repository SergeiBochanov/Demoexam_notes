def check_predictions(y_real, y_pred):
    TN = TP = FN = FP = 0
    for real, pred in zip(y_real, y_pred):
        if real == pred:
            if real == 1: TP += 1
            else: TN += 1
        else:
            if real == 1: FN += 1
            else: FP += 1
    return TN, TP, FN, FP
    
def accuracy(y_real, y_pred):
    TN, TP, FN, FP = check_predictions(y_real, y_pred)
    return (TP+TN)/(TP+FP+TN+FN)

def precision(y_real, y_pred):
    TN, TP, FN, FP = check_predictions(y_real, y_pred)
    return TP/(TP+FP)

def recall(y_real, y_pred):
    TN, TP, FN, FP = check_predictions(y_real, y_pred)
    return TP/(TP+FN)

def f1(y_real, y_pred):
    p = precision(y_real, y_pred)
    r = recall(y_real, y_pred)
    return (2*p*r)/(p+r)

def confusion_matrix(y_real, y_pred):
    TN, TP, FN, FP = check_predictions(y_real, y_pred)
    return [[TN, FP], [FN, TP]]
