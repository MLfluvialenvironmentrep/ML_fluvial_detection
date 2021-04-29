''' IMPORTANTE: RICORDARSI CHE IL DATASET ESTERNO DI TESTING E' ANNOTATO PER OTTENERE I PARAMETRI DI CLASSIFICAZIONE - idea spiegata da Manuele Bicego,
    Università di Verona, nel corso "Riconoscimento e recupero dell’informazione per bioinformatica - Classificazione: validazione" pp. 1-9, vedi link
    https://www.di.univr.it/documenti/OccorrenzaIns/matdid/matdid753206.pdf '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy import interp
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from scikitplot.metrics import plot_confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize

### Caricamento dataset per training e addestramento del modello
df_train_data = pd.read_csv("path of training dataset in .txt format", sep=';', error_bad_lines=False, low_memory=False).dropna()
df_train_data = df_train_data.dropna()
feature_columns = ['R_int','G_int','B_int']  # or ['RE_int','NIR_int']
features = df_train_data[feature_columns]
labels = df_train_data['Class']
#print(labels.value_counts())

### Porzionamento se si hanno labels sbilanciate - altrimenti non usarlo
portion = df_train_data.groupby('Class', sort=False).head(330000)   # the .head() value is related of teh minumum number of points obtained with label.value_counts()
features = portion[feature_columns]
labels = portion['Class']

### Traduzione in matrice semplice e mescolamento (shuffling) dei dataset
x_sparse = coo_matrix(features)
features, x_sparse, labels = shuffle(features, x_sparse, labels, random_state=0)

### Addestramento modello (gli iperparametri e i loro valori sono definiti dallo script di CV e GridSearchCV)
model = RandomForestClassifier(n_estimators=50, criterion='gini', max_features='auto', min_samples_leaf=10, min_samples_split=10, random_state=None)
t1 = time.time()
model.fit(features, labels)
t2 = time.time()
print ('\n\n### %s ### \nTraining Time: %s s' % (RandomForestClassifier, round(t2-t1,5)))
y_pred_labels = model.predict(features)
accuracy = round(accuracy_score(labels, y_pred_labels) * 100,2)
print ("\tAccuracy on Train: %s" % (accuracy))

### Classificazione su 2° dataset vergine, parametri di accuratezza, matrice di confusione, ROC-AUC e Precision-Recall curves
df_testing = pd.read_csv("path of the classified testing dataset from 1b script, .txt format", sep=';', error_bad_lines=False, low_memory=False)
X_test = df_testing.dropna().drop(columns=['FID', 'pointid', 'X', 'Y', 'Class'])
y_test = df_testing['Class']
y_pred=model.predict(X_test)
y_score=model.predict_proba(X_test)

''' Definizione per il report di classificazione propedeutica
    al calcolo dei parametri accuratezza su dataset esterno '''

def class_report(y_true, y_pred, y_score=None, average='micro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum() 
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int), 
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(), 
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(), 
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"], 
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df


report_with_auc = class_report(
    y_true=y_test, 
    y_pred=y_pred, 
    y_score=y_score)

print(report_with_auc)
print('Matrice di confusione dataset testing esterno:',confusion_matrix(y_true, y_pred))
plot_confusion_matrix(y_true, y_pred)
plt.show()

###################################################################################################################################################################
###################################################################################################################################################################

### Preparazione dataset testing per analisi ROC_AUC e Precision_Recall curves: Binarize labels (normalizzazione dati in numeri scala 0-1)
y_test = label_binarize(y_test, classes=[11,22,33])
n_classes = 3

### ROC-AUC curve per ogni classe
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i],
                                  y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (area = %0.2f)' % roc_auc[i])

plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.legend(loc="lower right")
plt.title("ROC curve for Water (11), Vegetation (22), Ground/Gravel bars classes (33)")
plt.show()

### Precision-Recall curve per ogni classe
precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
    
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="best")
plt.title("Precision-Recall curve for Water (11), Vegetation (22), Ground/Gravel bars classes (33)")
plt.show()
