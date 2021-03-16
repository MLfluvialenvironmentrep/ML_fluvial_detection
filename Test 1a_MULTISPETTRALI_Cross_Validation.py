import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
import time

### Caricamento dataset per training e validazione tramite parametri di accuratezza
df_train_data = pd.read_csv("D:\\Ema Pontoglio\Salbertrand\DATASET TRAINING - TESTING PYTHON\Training 2.0\RE_NIR_Training2_INTEGER.txt", sep=';', error_bad_lines=False, low_memory=False).dropna()
features = df_train_data.drop(columns=['FID', 'pointid', 'Class'], axis='columns')
labels = df_train_data['Class']

### Separazione dataset di training/testing per la CV --- normalizzazione features tramite modulo StandardScaler() --- randomizzazione training dataset (shuffle)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

features_scaler = StandardScaler()
X_train = features_scaler.fit_transform(X_train)
X_test = features_scaler.transform(X_test)

X_train, y_train = shuffle(X_train, y_train)

### Scelta algoritmo (RF), calcolo della CV del modello (cross_val_score), printing dei risultati.
t1 = time.time()
classifier = RandomForestClassifier(n_estimators=10, random_state=0)
all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=4)
print("Accuracy on training dataset with differents cv=4 folds:", all_accuracies)
print("Mean accuracy on training dataset through CV:", all_accuracies.mean())
print("Standard deviation reached in trainig dataset (if <1% is good -> very LOW variance):", all_accuracies.std())
precision = cross_val_score(classifier,X_train,y_train, cv=4, scoring='precision_macro')
recall = cross_val_score(classifier,X_train,y_train, cv=4, scoring='recall_macro')
f1 = cross_val_score(classifier,X_train,y_train, cv=4, scoring='f1_macro')
print("Precion train CV=4:", precision)
print("Recall train CV=4:", recall)
print("F1 train CV=4:", f1)
t2 = time.time()
print ('\n\n### %s ### \nCV and statistical parameters Time: %s s' % (RandomForestClassifier, round(t2-t1,5)))

### Dopo la CV, si calcolano i migliori parametri di addestramento dell'algoritmo scelto (RF): si devono scegliere i parametri da utilizzare (grid_param)
### con i valori scelti, creando le diverse conbinazioni che si vogliono testare. Successivamente, tramite il modulo GridSearchCV, si possono ottenere i
### risultati in termini di accuratezza ottenuti per ogni combinazione scelta. Infine, si printa la migliore combinazione ottenuta (.best_params_).
t3 = time.time()
grid_param = {
	'n_estimators': [10, 25, 50],
	'criterion': ['gini', 'entropy'],
	'max_features': ['auto', 'log2'],
        'min_samples_split': [5, 7, 10],
        'min_samples_leaf': [4, 6, 10],
        'random_state': [None, 0, 42]
}
                                                                 #estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                                                                                 # class_weight=None,
                                                                                                 # criterion='gini', max_depth=None,
                                                                                                 # max_features='auto',
                                                                                                 # max_leaf_nodes=None,
                                                                                                 # max_samples=None,
                                                                                                 # min_impurity_decrease=0.0,
                                                                                                 # min_impurity_split=None,
                                                                                                 # min_samples_leaf=1,
                                                                                                 # min_samples_split=2,
                                                                                                 # min_weight_fraction_leaf=0.0,
                                                                                                 # n_estimators=10, n_jobs=None,
                                                                                                 # oob_score=False, random_state=0,
                                                                                                 # verbose=0, warm_start=False),

gd_sr = GridSearchCV(estimator=classifier,
		   param_grid=grid_param,
		   scoring='accuracy',
		   cv=5,
		   n_jobs=-1)

gd_sr.fit(X_train, y_train)

best_parameters = gd_sr.best_params_
print(best_parameters)
best_result = gd_sr.best_score_
print(best_result)

t4 = time.time()
print("GridSearchCV time proceeding:", round(t3-t4,5))
