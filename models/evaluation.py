from sklearn.metrics import confusion_matrix,f1_score,classification_report,auc,roc_curve,precision_recall_curve
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

def multi_label_metrics(true_label,pred_label):
    n_classes = 25
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_label[:, i], pred_label[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(true_label.ravel(), pred_label.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # print("==============================================")
    # print("AUROC:")
    # for i in range(n_classes):
    #     print("Phenotype {0}\t{1:0.2f}".format(i, roc_auc[i]))
    
    # print("Macro AUC: {0:0.2f}, Micro AUC: {1:0.2f}".format(roc_auc["macro"] ,roc_auc["micro"]))
    phen_auc = []
    for i in range(n_classes):
        phen_auc.append(roc_auc[i])

    macro_auc = roc_auc["macro"]
    micro_auc = roc_auc['micro']
    return macro_auc,micro_auc

def print_classification_metrics(true_label,pred_label):
    cm = confusion_matrix(true_label, pred_label.round())
    print("Confusion Matrix: ",cm)
    fpr_keras, tpr_keras, _ = roc_curve(true_label, pred_label,pos_label=1)
    auc_keras = auc(fpr_keras, tpr_keras)
    print("Area Under Curve:",auc_keras)
    (precisions, recalls, _) = precision_recall_curve(true_label, pred_label,pos_label=1)
    auprc = auc(recalls, precisions)
    # minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    print('Area under Precision Recall:',auprc)
    # print(minpse)
    TN, FP, FN, TP = confusion_matrix(true_label, pred_label.round()).ravel()
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    print(" Specificity(TNR): {:.2f}\n Sensitivity(TPR): {:.2f}\n PPV: {:.2f}\n NPV: {:.2f}".format(TNR,TPR,PPV,NPV))

def regression_metrics(true_label,pred_label,ts=None):
    errors = []
    true_stay = []
    pred_stay = []
    if ts is None:
        for i,(a,b) in enumerate(zip(np.squeeze(true_label), np.squeeze(pred_label))):
            l = np.squeeze(a).argmin()#nrows_ts[i]
            true_stay += list(a[:l])
            pred_stay += list(b[:l])
            e =  b[:l] - a[:l]
            errors += list(e)
    else:
        for i,(a,b) in enumerate(zip(np.squeeze(true_label), np.squeeze(pred_label))):
            l = ts[i]#nrows_ts[i]
            true_stay += list(a[:l])
            pred_stay += list(b[:l])
            e =  b[:l] - a[:l]
            errors += list(e)
    r2 = r2_score(true_stay, pred_stay)
    mse = mean_squared_error(true_stay, pred_stay)
    mae = mean_absolute_error(true_stay, pred_stay)
    print("R2 : {0:f}, MSE : {1:f}, MAE:{2:f}".format(r2,mse,mae))
    return r2,mse,mae