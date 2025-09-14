'''INSTRUCCIONES
1. Carga y exploración de datos (1 punto)
• Carga el dataset proporcionado, que contiene información sobre temperatura media,
cambio en las precipitaciones, frecuencia de sequías y producción agrícola en
distintos países.
• Analiza la distribución de las variables y detecta posibles valores atípicos o
tendencias.'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, label_binarize, LabelBinarizer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, accuracy_score, precision_score, roc_auc_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC




df=pd.read_csv('M6/cambio_climatico_agricultura.csv')

print(f'Datos:{df.head(5)} \n')
print(f'{df.describe()}\n')
print(f'{df.info()}\n')


'''2. Preprocesamiento y escalamiento de datos (2 puntos)
• Aplica técnicas de normalización o estandarización a las variables numéricas.'''

'''• Codifica correctamente cualquier variable categórica si fuera necesario.
• Divide los datos en conjunto de entrenamiento y prueba (80%-20%).'''

X = df.drop('Producción_alimentos', axis=1)
y = df['Producción_alimentos']

categoricas=X.select_dtypes(include=['object']).columns.tolist()
print('Variables categóricas')
preprocessor= ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categoricas)
], remainder='passthrough')

X_encoded=preprocessor.fit_transform(X)

print(f'{X_encoded}')

# Revisar valores nulos
print(X.isnull().sum())
print(y.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split( X_encoded, y, test_size=0.3, random_state=42)

print('Tamaño de entrenamiento:', X_train.shape)
print('Tamaño de prueba:', X_test.shape)


scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('Media de entrenamiento (después de escalar):', X_train_scaled.mean(axis=0))
dense_array = X_test_scaled.toarray()
standard_deviation = np.std(dense_array)

print(f"Desviación estándar de entrenamiento: {standard_deviation}")


'''3. Aplicación de modelos de aprendizaje supervisado (4 puntos)
• Regresión:
o Entrena un modelo de regresión lineal para predecir la producción de
alimentos.
o Evalúa el modelo usando métricas como MAE, MSE y R2.
o Compara con otros modelos de regresión (árbol de decisión, random forest).'''

reg_model=LinearRegression()
reg_model.fit(X_train_scaled, y_train)
y_pred_reg=reg_model.predict(X_test_scaled)

mae_reg=mean_absolute_error(y_test, y_pred_reg)
mse_reg=mean_squared_error(y_test, y_pred_reg)
r2_score_reg=r2_score(y_test, y_pred_reg)


print(f'mae de reg: {mae_reg}')
print(f'mse de reg: {mse_reg}')
print(f'r2_score de reg: {r2_score_reg}')

#Comparación con otros modelos

#Árbol de decisión

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

print('Exactitud Árbol de Decisión:', accuracy_score(y_test, y_pred_dt))
print('Matriz de confusión Árbol de Decisión:')
print(confusion_matrix(y_test, y_pred_dt))

mae_reg2=mean_absolute_error(y_test, y_pred_dt)
mse_reg2=mean_squared_error(y_test, y_pred_dt)
r2_score_reg2=r2_score(y_test, y_pred_dt)


print(f'mae de reg: {mae_reg2}')
print(f'mse de reg: {mse_reg2}')
print(f'r2_score de reg: {r2_score_reg2}')


#Random Forest

modelo_arbol=RandomForestClassifier(max_depth=10, random_state=42)
modelo_arbol.fit(X_train_scaled, y_train)
y_pred2=modelo_arbol.predict(X_test_scaled)

mae_reg2=mean_absolute_error(y_test, y_pred2)
mse_reg2=mean_squared_error(y_test, y_pred2)
r2_score_reg2=r2_score(y_test, y_pred2)


print(f'mae de reg: {mae_reg2}')
print(f'mse de reg: {mse_reg2}')
print(f'r2_score de reg: {r2_score_reg2}')


'''• Clasificación:
o Crea una nueva variable categórica que clasifique los países en "Bajo",
"Medio" y "Alto" impacto climático en la producción agrícola.
o Entrena modelos de clasificación como K-Nearest Neighbors, Árbol de
Decisión y Support Vector Machine.
o Evalúa el desempeño usando matriz de confusión, precisión, sensibilidad y
curva ROC-AUC.'''



df['Categorías_vulnerabilidad'] = '' # Inicializar la columna
df.loc[df['Producción_alimentos'] <= 500000, 'Categorías_vulnerabilidad'] = 'Bajo'
df.loc[(df['Producción_alimentos'] > 500000) & (df['Producción_alimentos'] <= 790000), 'Categorías_vulnerabilidad'] = 'Medio'
df.loc[df['Producción_alimentos'] > 790000, 'Categorías_vulnerabilidad'] = 'Alto'


print(df)

#---------------------------------------------------------------------------------------------------------------------------------------------------------

# K-NN (k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print('Exactitud K-NN:', accuracy_score(y_test, y_pred_knn))
print('Matriz de confusión K-NN:')
print(confusion_matrix(y_test, y_pred_knn))

precision= precision_score(y_test, y_pred_knn, average='micro')
accuracy=accuracy_score(y_test, y_pred_knn)
print(f'Precisión: {precision}')
print(f'Exactitud: {accuracy}\n')

#Curva ROC-AUC


n_classes = len(np.unique(y_test))
fpr = dict()
tpr = dict()
thresholds = dict()

for i in range(n_classes):
    # Binarize y_test for the current class (One-vs-Rest)
    y_test_binarized = label_binarize(y_test, classes=range(n_classes))[:, i]
   
    y_score_class_i = df['Categorías_vulnerabilidad'] #knn.predict_proba(X_test_scaled)[:, 1]
    
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_test_binarized, y_score_class_i)


auc=roc_auc_score(y_test_binarized, y_score_class_i)
print(f'AUC:{auc}')

plt.figure(figsize=(7,5))
plt.plot(fpr[i], tpr[i], label='K-NN (AUC = {:.2f})'.format(np.trapezoid(tpr[i], fpr[i])))
plt.plot([0, 1], [0, 1], 'k--', label='Línea aleatoria')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend()
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------

#Árbol de decisión

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

print('Exactitud Árbol de Decisión:', accuracy_score(y_test, y_pred_dt))
print('Matriz de confusión Árbol de Decisión:')
print(confusion_matrix(y_test, y_pred_dt))
precision= precision_score(y_test, y_pred_dt, average='micro')
accuracy=accuracy_score(y_test, y_pred_dt)
print(f'Precisión: {precision}')
print(f'Exactitud: {accuracy}\n')

#Curva ROC-AUC

n_classes = len(np.unique(y_test))
fpr = dict()
tpr = dict()
thresholds = dict()

for i in range(n_classes):
    # Binarize y_test for the current class (One-vs-Rest)
    y_test_binarized = label_binarize(y_test, classes=range(n_classes))[:, i]
    y_score_class_i = dt.predict_proba(X_test_scaled)[:, 1]
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_test_binarized, y_score_class_i)

auc=roc_auc_score(y_test_binarized, y_score_class_i)
print(f'AUC:{auc}')

plt.figure(figsize=(7,5))
plt.plot(fpr[i], tpr[i], label='Árbol de Decisión (AUC = {:.2f})'.format(np.trapezoid(tpr[i], fpr[i])))
plt.plot([0, 1], [0, 1], 'k--', label='Línea aleatoria')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend()
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------

# SVM
svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=42) 
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

print('Exactitud SVM:', accuracy_score(y_test, y_pred_svm))
print('Matriz de confusión SVM:')
print(confusion_matrix(y_test, y_pred_svm))
precision= precision_score(y_test, y_pred_svm, average='micro')
accuracy=accuracy_score(y_test, y_pred_svm)
print(f'Precisión: {precision}')
print(f'Exactitud: {accuracy}\n')


#Curva ROC-AUC

n_classes = len(np.unique(y_test))
fpr = dict()
tpr = dict()
thresholds = dict()

for i in range(n_classes):
    # Binarize y_test for the current class (One-vs-Rest)
    y_test_binarized = label_binarize(y_test, classes=range(n_classes))[:, i]
   
    y_score_class_i = svm.predict_proba(X_test_scaled)[:, 1]
    
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_test_binarized, y_score_class_i)


auc=roc_auc_score(y_test_binarized, y_score_class_i)
print(f'AUC:{auc}')

plt.figure(figsize=(7,5))
plt.plot(fpr[i], tpr[i], label='K-NN (AUC = {:.2f})'.format(np.trapezoid(tpr[i], fpr[i])))
plt.plot([0, 1], [0, 1], 'k--', label='Línea aleatoria')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend()
plt.show()


'''4. Optimización de modelos (2 puntos)
• Ajusta hiperparámetros utilizando validación cruzada y búsqueda en grilla.
• Aplica técnicas de regularización y analiza su impacto en los modelos.'''



'''5. Análisis de resultados y conclusiones (1 punto)
• Compara los modelos utilizados y justifica cuál ofrece mejores resultados para la
predicción y clasificación.
• Relaciona los hallazgos con posibles implicaciones en la seguridad alimentaria
global'''