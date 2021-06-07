
# Importar paquetes utiles
import os  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.getcwd()  
os.chdir('C:\\Ber\\Universidades\\UdeSA\\Big Data\\Trabajos Practicos\\TP 1')

# Abrir base de datos
eph = pd.read_csv("indiv.csv")
print(eph)


# Eliminar observaciones con edad negativa
eph = eph[eph.CH06 >= 0]

# Eliminar observaciones con ingreso negativos
eph = eph[eph.ITF >= 0]
eph = eph[eph.IPCF >= 0]


# Grafico de barras mostrando composición por sexo
varones = sum(map(lambda x : x == 1, eph.CH04))
mujeres = sum(map(lambda x : x == 2, eph.CH04))
generos = ['Varón', 'Mujer']
cantidades = [varones, mujeres]
plt.barh(generos, cantidades, height=0.8)


# Matriz de correlación
import seaborn
corr = eph[['CH04', 'CH07', 'CH08', 'NIVEL_ED', 'ESTADO', 'CAT_INAC', 'IPCF']].corr()
ax = seaborn.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=seaborn.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# Cantidad de desocupados en la muestra
desocupados = sum(map(lambda x : x == 2, eph.ESTADO))
desocupados
        
# Cantidad de inactivos en la muestra
inactivos = sum(map(lambda x : x == 3, eph.ESTADO))
inactivos

# Media de IPCF según estado
eph[['ESTADO', 'IPCF']].groupby('ESTADO').mean()


# Agregar columna 'ad_equiv'
edades = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 30, 46, 61, 76]
ad_equiv_varones = [0.35, 0.37, 0.46, 0.51, 0.55, 0.6, 0.64,0.66,0.68,0.69,0.79,0.82,0.85,0.9,0.96,1.00,1.03,1.04,1.02,1.00,1.00,0.83,0.74]
ad_equiv_mujeres = [0.35,0.37,0.46,0.51,0.55,0.6,0.64,0.66,0.68,0.69,0.7,0.72,0.74,0.76,0.76,0.77,0.77,0.77,0.76,0.77,0.76,0.67,0.63]

n = 0
for i in edades:
    eph.loc[(eph.CH06 >= i) & (eph.CH04 == 1), 'ad_equiv'] = ad_equiv_varones[n]
    n = n+1
    
n= 0
for i in edades:
    eph.loc[(eph.CH06 >= i) & (eph.CH04 == 2), 'ad_equiv'] = ad_equiv_mujeres[n]
    n = n+1


# Agregar columna 'ad_equiv_hogar'

eph['codigohogar'] = eph['CODUSU'] + str(eph['NRO_HOGAR'])
aeh = eph[['codigohogar', 'ad_equiv']].groupby('codigohogar').sum()
aeh.reset_index(level=0, inplace=True)
aeh.rename(columns={'ad_equiv':'ad_equiv_hogar'}, inplace=True)
eph = pd.merge(eph, aeh, on='codigohogar')


# Cantidad de observaciones que no respondieron ITF
sum(map(lambda x : x == 0, eph.ITF))

# Guardar base de los que respondieron el ITF
respondieron = eph[eph.ITF > 0]

# Guardar base de los que no respondieron el ITF
norespuesta = eph[eph.ITF == 0]


# Agregar columna 'ingreso_necesario'
respondieron['ingreso_necesario'] = respondieron['ad_equiv_hogar']*8928


# Agregar columna 'pobre'
respondieron.loc[respondieron.ITF >= respondieron.ingreso_necesario, 'pobre'] = 0
respondieron.loc[respondieron.ITF < respondieron.ingreso_necesario, 'pobre'] = 1

# Cantidad de pobres en la muestra
respondieron[['pobre', 'CODUSU']].groupby('pobre').count()


# Eliminar variables de ingresos en las bases 'respondieron' y 'norespuesta'
respondieron = respondieron.drop(["P21", "DECOCUR", "IDECOCUR", "RDECOCUR", "GDECOCUR", "PDECOCUR", "ADECOCUR", "TOT_P12", "P47T", "DECINDR", "IDECINDR", "RDECINDR", "GDECINDR", "PDECINDR", "ADECINDR", 'V2_M','V3_M','V4_M','V5_M','V8_M','V9_M','V10_M','V11_M','V12_M','V18_M','V19_AM','V21_M','T_VI','ITF','DECIFR','IDECIFR','RDECIFR','GDECIFR','PDECIFR','ADECIFR','IPCF','DECCFR','IDECCFR','RDECCFR','GDECCFR','PDECCFR','ADECCFR', 'ad_equiv', 'ad_equiv_hogar', 'ingreso_necesario'], axis=1)
norespuesta = norespuesta.drop(["P21", "DECOCUR", "IDECOCUR", "RDECOCUR", "GDECOCUR", "PDECOCUR", "ADECOCUR", "TOT_P12", "P47T", "DECINDR", "IDECINDR", "RDECINDR", "GDECINDR", "PDECINDR", "ADECINDR", 'V2_M','V3_M','V4_M','V5_M','V8_M','V9_M','V10_M','V11_M','V12_M','V18_M','V19_AM','V21_M','T_VI','ITF','DECIFR','IDECIFR','RDECIFR','GDECIFR','PDECIFR','ADECIFR','IPCF','DECCFR','IDECCFR','RDECCFR','GDECCFR','PDECCFR','ADECCFR', 'ad_equiv', 'ad_equiv_hogar'], axis=1)


# Dividir base en entrenamiento y test
from sklearn.model_selection import train_test_split
train, test = train_test_split(respondieron, test_size=0.30, random_state=101)


# Separar variables dependientes e independientes y agregar intercepto
# Base de entrenamiento
y_train=train.pobre
X_train=train
X_train=X_train.drop('pobre', axis=1)
intercepto = np.ones(len(train.index))
X_train['intercepto']=intercepto

# Base de prueba
y_test=test.pobre
X_test=test
X_test=X_test.drop('pobre', axis=1)
intercepto = np.ones(len(test.index))
X_test['intercepto']=intercepto


# Eliminar variables no numéricas y problematicas en las bases de regresoras
X_train=X_train.drop(['CODUSU', 'codigohogar','MAS_500', 'ANO4','TRIMESTRE', 'REGION', 'AGLOMERADO', 'IMPUTA', 'NRO_HOGAR', 'Unnamed: 0'], axis=1)
X_test=X_test.drop(['CODUSU', 'codigohogar','MAS_500', 'ANO4','TRIMESTRE', 'REGION', 'AGLOMERADO', 'IMPUTA', 'NRO_HOGAR', 'Unnamed: 0'], axis=1)


# Detectar columnas linealmente independientes en las bases de regresoras
from numpy.linalg import matrix_rank
matrix_rank(X_train)

import sympy 
reduced_form, inds = sympy.Matrix(X_train.values).rref()
inds

# Eliminar columnas linealmente dependientes en las bases de regresoras
X_train= X_train.drop(X_train.columns[[56, 87, 94, 95, 96, 97]], axis=1)
X_test= X_test.drop(X_test.columns[[56, 87, 94, 95, 96, 97]], axis=1)

# Convertir en missing los valores no numericos
def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False

X_train=X_train[X_train.applymap(isnumber)]
X_test=X_test[X_test.applymap(isnumber)]

# Convertir en 0 los missings. Una alternativa a esto es eliminar los missings
# al estimar el logit, agregando la opcion missing = ‘drop’. Sin embargo, esto
# hace que se pierdan muchas observaciones
X_train=X_train.fillna(0)
X_test=X_test.fillna(0)

# Convertir a valores numericos la columna CH05
X_test['CH05'] = pd.to_numeric(X_test['CH05'])
X_train['CH05'] = pd.to_numeric(X_train['CH05'])

# Logit
# Estimar logit
import statsmodels.api as sm
logit_model=sm.Logit(y_train.astype(float),X_train.astype(float))
result_logit=logit_model.fit()
print(result_logit.summary2())

# Predecir probabilidades sobre la base de test con el modelo logit estimado
y_pred_logit = result_logit.predict(X_test)

# Aplicar clasificador de Bayes a las probabilidades predichas por logit
y_pred_logit=np.where(y_pred_logit>0.5, 1, y_pred_logit)
y_pred_logit=np.where(y_pred_logit<=0.5, 0, y_pred_logit)

# Matriz de confusion para predicciones de logit
from sklearn.metrics import confusion_matrix

confusion_matrix_logit = confusion_matrix(y_pred_logit, y_test) 
print(confusion_matrix_logit)

# Accuracy de logit
accuracy_logit = (confusion_matrix_logit[0][0] + confusion_matrix_logit[1][1])/confusion_matrix_logit.sum()
print(accuracy_logit)

# Curva ROC logit
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr): 
    plt.plot(fpr, tpr, color='orange', label='ROC') 
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

auc_logit = roc_auc_score(y_test, y_pred_logit)
print('AUC: %.2f' % auc_logit)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_logit) 
plot_roc_curve(fpr, tpr) 


# Análisis discriminante lineal (LDA)
# Estimar función discriminante lineal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
resultslda=clf.predict(X_test)

# Predecir a partir de la función discriminante lineal estimada
y_pred_lda=pd.Series(resultslda.tolist())

# Matriz de confusion para predicciones de LDA
confusion_matrix_lda = confusion_matrix(y_pred_lda, y_test) 
print(confusion_matrix_lda)

# Accuracy de LDA
accuracy_lda = (confusion_matrix_lda[0][0] + confusion_matrix_lda[1][1])/confusion_matrix_lda.sum()
print(accuracy_lda)

# Curva ROC LDA
auc_lda = roc_auc_score(y_test, y_pred_lda)
print('AUC: %.2f' % auc_lda)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_lda) 
plot_roc_curve(fpr, tpr) 

# Análisis discriminante cuadrático
# Estimar función discriminante cuadrática
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

cqf = QuadraticDiscriminantAnalysis()
cqf.fit(X_train, y_train)
resultsqda=cqf.predict(X_test)

# Predecir a partir de la función discriminante cuadrática estimada
y_pred_qda=pd.Series(resultsqda.tolist())

# Matriz de confusion para predicciones de QDA
confusion_matrix_qda = confusion_matrix(y_pred_qda, y_test) 
print(confusion_matrix_qda)

# Accuracy de QDA
accuracy_qda = (confusion_matrix_qda[0][0] + confusion_matrix_qda[1][1])/confusion_matrix_qda.sum()
print(accuracy_qda)

# Curva ROC QDA
auc_qda = roc_auc_score(y_test, y_pred_qda)
print('AUC: %.2f' % auc_qda)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_qda) 
plot_roc_curve(fpr, tpr) 

# K Vecinos más cercanos
# Clasificar en base a K vecinos más cercanos
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Matriz de confusion para predicciones de KNN
confusion_matrix_knn = confusion_matrix(y_pred_knn, y_test) 
print(confusion_matrix_knn)

# Accuracy de KNN
accuracy_knn = (confusion_matrix_knn[0][0] + confusion_matrix_knn[1][1])/confusion_matrix_knn.sum()
print(accuracy_knn)

# Curva ROC KNN
auc_knn = roc_auc_score(y_test, y_pred_knn)
print('AUC: %.2f' % auc_knn)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_knn) 
plot_roc_curve(fpr, tpr) 


# Agregar intercepto a la base 'norespuesta'
intercepto = np.ones(len(norespuesta.index))
norespuesta['intercepto']=intercepto

# Seleccionar en la base 'norespuesta' las mismas columnas que en X_train
norespuesta=norespuesta[X_train.columns]

# Corregir missings y variables no numéricas en 'norespuesta'
norespuesta=norespuesta[norespuesta.applymap(isnumber)]
norespuesta=norespuesta.fillna(0)
norespuesta['CH05'] = pd.to_numeric(norespuesta['CH05'])

# Predecir probabilidades sobre la base 'norespuesta' con el modelo logit estimado
y_pred_logit_norespuesta = result_logit.predict(norespuesta)

# Aplicar clasificador de Bayes a las probabilidades predichas por logit
y_pred_logit_norespuesta=np.where(y_pred_logit_norespuesta>0.5, 1, y_pred_logit_norespuesta)
y_pred_logit_norespuesta=np.where(y_pred_logit_norespuesta<=0.5, 0, y_pred_logit_norespuesta)

# Calcular proporción de pobres predichos en la base no respuesta
y_pred_logit_norespuesta = pd.DataFrame(y_pred_logit_norespuesta)
y_pred_logit_norespuesta.columns = ['pred']
contador = np.ones(len(y_pred_logit_norespuesta.index))
y_pred_logit_norespuesta['contador']=contador
contar_predic=y_pred_logit_norespuesta[['pred', 'contador']].groupby('pred').count()
proporcion_predic = contar_predic.at[1.0,'contador']/(contar_predic.at[1.0,'contador']+contar_predic.at[0.0,'contador'])
print(proporcion_predic)


# Convertir en una tabla el output del modelo logit
logit_variables=(result_logit.summary2().tables[1])

# Crear una lista con los nombres de las variables que tienen un p-valor menor
# a 0,10 en la regresión logit (variables significativas)
variables_seleccionadas = list(logit_variables[logit_variables['P>|z|']<=0.10].index)[1:]

# Seleccionar las variables significativas en las bases de train y test
X_train2=X_train[variables_seleccionadas]
X_test2=X_test[variables_seleccionadas]

# Estimar logit con variables seleccionadas
logit_model=sm.Logit(y_train.astype(float),X_train2.astype(float))
result_logit2=logit_model.fit()
print(result_logit2.summary2())

# Predecir probabilidades sobre la base de test con el modelo logit estimado (variables seleccionadas)
y_pred_logit2 = result_logit2.predict(X_test2)

# Aplicar clasificador de Bayes a las probabilidades predichas por logit (variables seleccionadas)
y_pred_logit2=np.where(y_pred_logit2>0.5, 1, y_pred_logit2)
y_pred_logit2=np.where(y_pred_logit2<=0.5, 0, y_pred_logit2)

# Matriz de confusion para predicciones de logit  (variables seleccionadas)
confusion_matrix_logit2 = confusion_matrix(y_pred_logit2, y_test) 
print(confusion_matrix_logit2)

# Accuracy de logit (variables seleccionadas)
accuracy_logit2 = (confusion_matrix_logit2[0][0] + confusion_matrix_logit2[1][1])/confusion_matrix_logit2.sum()
print(accuracy_logit2)

# Curva ROC y AUC logit (variables seleccionadas)
auc_logit2 = roc_auc_score(y_test, y_pred_logit2)
print('AUC: %.2f' % auc_logit2)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_logit2) 
plot_roc_curve(fpr, tpr) 

# Clasificar a las observaciones en la base 'norespuesta' en base al nuevo
# modelo logit

# Seleccionar en la base 'norespuesta' las mismas columnas que en X_train2
norespuesta2=norespuesta[X_train2.columns]

# Predecir probabilidades sobre la base 'norespuesta' con el modelo logit estimado
y_pred_logit_norespuesta2 = result_logit2.predict(norespuesta2)

# Aplicar clasificador de Bayes a las probabilidades predichas por logit
y_pred_logit_norespuesta2=np.where(y_pred_logit_norespuesta2>0.5, 1, y_pred_logit_norespuesta2)
y_pred_logit_norespuesta2=np.where(y_pred_logit_norespuesta2<=0.5, 0, y_pred_logit_norespuesta2)

# Calcular proporción de pobres predichos en la base no respuesta
y_pred_logit_norespuesta2 = pd.DataFrame(y_pred_logit_norespuesta2)
y_pred_logit_norespuesta2.columns = ['pred']
contador = np.ones(len(y_pred_logit_norespuesta2.index))
y_pred_logit_norespuesta2['contador']=contador
contar_predic2=y_pred_logit_norespuesta2[['pred', 'contador']].groupby('pred').count()
proporcion_predic2 = contar_predic2.at[1.0,'contador']/(contar_predic2.at[1.0,'contador']+contar_predic2.at[0.0,'contador'])
print(proporcion_predic2)

