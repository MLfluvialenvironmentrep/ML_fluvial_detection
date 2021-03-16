''' Script valido con numeri INTEGER esportati da GIS appositamente -> molto più fruibile e senza la necessità di normalizzare i dati tramite moduli appositi,
    come LabelEncoder, OneHotEncoder(), etc. '''

''' Preso spunto da ottimo lavoro di tesi Vercellesi (università di Bologna) per idea e architettura script fino allo shuffle '''

''' Qualora vi siano dati categorici (parole) o in formato string occorrerebbe adattarlo con un modulo di normalizzazione sopra citati -> possibilità di integrazione? '''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

### Caricamento dataset per training e validazione tramite parametri di accuratezza
df_train_data = pd.read_csv("D:\\Ema Pontoglio\Salbertrand\DATASET TRAINING - TESTING PYTHON\Training 2.0\RE_NIR_Training2_INTEGER.txt", sep=';', error_bad_lines=False, low_memory=False)
df_train_data = df_train_data.dropna()
feature_columns = ['R_int','G_int','B_int']
features = df_train_data[feature_columns]
labels = df_train_data['Class']
#print(labels.value_counts())

### Porzionamento se si hanno labels sbilanciate - altrimenti non usarlo
portion = df_train_data.groupby('Class', sort=False).head(330000)
features = portion[feature_columns]
labels = portion['Class']

### Traduzione in matrice semplice e mescolamento (shuffling) del training dataset
x_sparse = coo_matrix(features)
features, x_sparse, labels = shuffle(features, x_sparse, labels, random_state=0)

### Addestramento modello ed accuratezza sul training dataset (capacità di generalizzazione) -> inserire CV e GridSearchCV per ottimizzazioen iperparametri RF
model = RandomForestClassifier(n_estimators=10, random_state=0)
t1 = time.time()
model.fit(feature, labels)
t2 = time.time()
print ('\n\n### %s ### \nTraining Time: %s s' % (RandomForestClassifier, round(t2-t1,5)))
y_pred_train = model.predict(feature)
accuracy = round(accuracy_score(labels, y_pred_train) * 100,2)
print ("\tAccuracy on Train: %s" % (accuracy))

### Classificazione ed esportazione 2° dataset vergine
df_testing = pd.read_csv("D:\Ema Pontoglio\Salbertrand\DATASET TRAINING - TESTING PYTHON\Terza porzione\RE_NIR_terza_porzione_INTEGER.txt", sep=';', error_bad_lines=False, low_memory=False)
df_testing_clean = df_testing.dropna().drop(columns=['FID', 'X', 'Y'])
feature_columns_test = ['R_int','G_int','B_int']
features_test = df_testing_clean[feature_columns_test]
CLASSIFIED_DATASET = model.predict(df_testing_clean)

tot_df = pd.concat([df_testing,df_testing_clean],axis=1).dropna()
my_data_1 = np.vstack((tot_df.T.drop(['R_int','G_int','B_int']),CLASSIFIED_DATASET))
my_data_1 = my_data_1.T
np.savetxt('INSERIRE NOME DESIDERATO.txt',my_data_1,fmt='%s',delimiter=';',header='PointID;X;Y;Class')
