import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEMILLA = 123456789
np.random.seed(SEMILLA)

dtrain = pd.read_csv('./clasificar_train.csv')
dtest = pd.read_csv('./clasificar_test.csv')

X_train = dtrain.drop(columns=['X', 'Y', 'id', 'Cat'])
X_test = dtest.drop(columns=['X', 'Y', 'id', 'Cat'])
y_train = dtrain["Cat"]
y_test = dtest["Cat"]



#----------Decision Tree----------
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#Creamos el clasificador
classifierDT = DecisionTreeClassifier(random_state=SEMILLA)

#Entrenamos el clasificador
classifierDT.fit(X_train, y_train)

#Realizamos la prediccion
predictionsDT = classifierDT.predict(X_test)

#Mostramos el árbol de decisión
fig = plt.figure(figsize=(40, 40))
_ = tree.plot_tree(classifierDT, filled=True)

#Analizamos la eficacia del método
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
print("Decission Tree:")
print("  Accuracy: " + str(accuracy_score(y_test, predictionsDT)))
print(classification_report(y_test, predictionsDT))

fig, ax = plt.subplots(figsize=(10, 10))
disp = plot_confusion_matrix(classifierDT, X_test, y_test, ax=ax)
disp.figure_.suptitle("Matriz de confusión-Decision Tree")
plt.show()



#----------Support Vector Machine classifier----------
from sklearn.svm import SVC

#Creamos el clasificador
classifierSVM = SVC(kernel="linear", random_state=SEMILLA)

#Entrenamos el clasificador
classifierSVM.fit(X_train, y_train)

#Realizamos la prediccion
predictionsSVM = classifierSVM.predict(X_test)

#Analizamos la eficacia del método
print("SVM:")
print("  Accuracy: " + str(accuracy_score(y_test, predictionsSVM)))
print(classification_report(y_test, predictionsSVM))

fig, ax = plt.subplots(figsize=(10, 10))
disp = plot_confusion_matrix(classifierSVM, X_test, y_test, ax=ax)
disp.figure_.suptitle("Matriz de confusión-SVM")
plt.show()



#----------Random Forest Classifier----------
from sklearn.ensemble import RandomForestClassifier
#Creamos el clasificador
classifierRF = RandomForestClassifier(n_estimators=3, random_state=SEMILLA)

#Entrenamos el clasificador
classifierRF.fit(X_train, y_train)

#Realizamos la prediccion
predictionsRF = classifierRF.predict(X_test)

#Analizamos la eficacia del método
print("Random Forest:")
print("  Accuracy: " + str(accuracy_score(y_test, predictionsRF)))
print(classification_report(y_test, predictionsRF))

fig, ax = plt.subplots(figsize=(10, 10))
disp = plot_confusion_matrix(classifierRF, X_test, y_test, ax=ax)
disp.figure_.suptitle("Matriz de confusión-Random Forest")
plt.show()