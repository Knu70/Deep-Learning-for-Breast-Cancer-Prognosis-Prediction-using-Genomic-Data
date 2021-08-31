# Luis Díaz del Valle

# -----------------------------------------------
# IMPORTACIÓN DE LIBRERÍAS Y MÓDULOS NECESARIOS
# -----------------------------------------------

from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.under_sampling import NearMiss
from keras import callbacks
from keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.python.debug.examples.debug_mnist import tf

# -----------------------------------------------
# CREACIÓN Y EXPLORACIÓN DEL CONJUNTO DE DATOS
# -----------------------------------------------

dataset = np.loadtxt('dataset/METABRIC_gene_exp_1980.txt', delimiter=' ')
labels = np.loadtxt('dataset/METABRIC_label_5year_positive491.txt', delimiter=' ').reshape(-1, 1)

# Gráfico para visualizar la distribución de las clases
def plotClasses(title, labels):
    plt.title(title)
    sns.countplot(labels.flatten())
    plt.show()

plotClasses('Distribución de clases objetivo desbalanceadas', labels)

print('Número total de muestras:', len(dataset))  # 1980
print('Tamaño de los datos:', dataset.shape[1])  # 400
# print('Número total de etiquetas:', len(labels)) # 1980
# print('Tamaño de las etiquetas:', labels.shape[1]) # 400

# Función que cuenta el número de muestras de cada clase
def numClasses(labels):
    posit, negat = 0, 0
    for i in labels:
        if i == 0:
            negat += 1
        else:
            posit += 1
    print('\tNúmero total de muestras positivas:', posit)  # 491
    print('\tNúmero total de muestras negativas:', negat)  # 1489


print('\nNumero de clases desbalanceadas:')
numClasses(labels)


# -----------------------------------------------
# PREPROCESO DE LOS DATOS
# -----------------------------------------------

# Submuestreo de los datos para balancear las clases
us = NearMiss(version=2, n_neighbors=3)
dataset, labels = us.fit_resample(dataset, labels)

print('\nNumero de clases balanceadas:')
numClasses(labels)

plotClasses('Distribución de clases objetivo balanceadas', labels)

# Se usa un 80% de los datos para el entrenamiento y un 20% para la validación
X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

print('\nNúmero de datos de entrenamiento 80%:', len(X_train))
print('Número de datos de validación 20%:', len(X_test))

# Asignacion de variables unidimensionales para el modelo RandomForest
X_train_RF = X_train
Y_train_RF = Y_train
X_test_RF = X_test
Y_test_RF = Y_test

# Agrega número de canales a los datos
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Parsea números a floats
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')

# Convierte los vectores de las etiquetas en categóricos
Y_train = to_categorical(Y_train, num_classes=2)
Y_test = to_categorical(Y_test, num_classes=2)

num_classes = Y_test.shape[1]  # 2


# -----------------------------------------------
# FUNCIONES DE GRÁFICOS
# -----------------------------------------------

def plotAUC_ROC(Y_true, Y_pred, a, b, c, d):
    fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(d, Y_pred)
    auc_cnn = auc(fpr_cnn, tpr_cnn)

    """ Para que la trama sea más significativa se entrena otro clasificador binario con RandomForest
        y se compara con el clasificador de la red neuronal convolucional.
        Modelo con 100 arboles """

    rf = RandomForestClassifier(n_estimators=100,
                                bootstrap=True,
                                # verbose=1,
                                max_features='sqrt')
    rf.fit(a, b.ravel())

    y_pred_rf = rf.predict_proba(c)[:, 1]
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(d, y_pred_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_cnn, tpr_cnn, label='CNN (area = {:.3f})'.format(auc_cnn))
    plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('Tasa Falsos Positivos')
    plt.ylabel('Tasa Verdaderos Positivos')
    plt.title('curva ROC')
    plt.legend(loc='best')
    plt.show()


def plotLearningCurve(history):
    plt.figure(figsize=(10, 6))

    # precisión del modelo
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Datos de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Datos de Validación')
    plt.title('Precisión')
    plt.ylabel('%')
    plt.xlabel('No. épocas')
    plt.legend(['entrenamiento', 'validación'], loc='upper left')

    # pérdidia del modelo
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Datos de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Datos de Validación')
    plt.title('Pérdida')
    plt.ylabel('Valor de pérdida')
    plt.xlabel('No. épocas')
    plt.legend(['entrenamiento', 'validación'], loc='upper left')
    plt.show()


def plotConfusionMatrix(cm, normalize=False, cmap=plt.cm.Blues):
    title = 'Matriz de Confusión'
    classes = {0: 'Supervivencia > 5 años', 1: 'Supervivencia < 5 años'}
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Valor real')
    plt.xlabel('Predicción')
    plt.show()


# -----------------------------------------------
# CONSTRUCCIÓN Y ENTRENAMIENTO DE LOS MODELOS
# -----------------------------------------------

# Contrucción y entrenamiento RandomForest
def RandomForest(a, b, c, d):
    # Crear el modelo con 100 arboles
    model = RandomForestClassifier(n_estimators=100,
                                   bootstrap=True,
                                   verbose=2,
                                   max_features='sqrt')

    # a entrenar!
    model.fit(a, b.ravel())

    # Predicción sobre el conjunto de validación
    Y_pred = model.predict_proba(c)[:, 1]
    Y_predicted = model.predict(c)

    print('Precisión:', accuracy_score(d, Y_predicted))

    # Ver la matriz de confusión para los datos de validacion y los de predicción
    print('Matric de confusión:\n', confusion_matrix(d, Y_predicted))

    # Ver el repote clasificatorio
    print(classification_report(d, Y_predicted))

    confusion_mtx = confusion_matrix(d, Y_predicted)
    plotConfusionMatrix(confusion_mtx)

    return model


# Construcción Red Neuronal Convolucional
def building_CNN():
    # La forma de entrada de las muestras va a ser de 400*1 (400)
    inputShape = (400, 1)
    classes = 2
    model = Sequential()

    # Primera convolución
    model.add(Conv1D(32, 3, activation='relu', input_shape=inputShape, kernel_regularizer=l2(0.01)))

    model.add(MaxPooling1D(3))
    # model.add(Dropout(0.2))

    # Segunda convolución
    model.add(Conv1D(64, 3, activation='relu', kernel_regularizer=l2(0.01)))

    model.add(MaxPooling1D(3))
    # model.add(Dropout(0.3))

    # Aplanar resultados
    model.add(Flatten())

    # 128 neuronas en la capa oculta
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))

    model.add(Dropout(0.4))

    model.add(Dense(classes, activation='sigmoid', kernel_regularizer=l2(0.01)))

    return model


# Entrenamiento del modelo CNN
def train_CNN(model, a, b, c, d):
    batch_size = 32
    epochs = 15

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)  # estimación de momento adaptativo
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics='accuracy')

    model.summary()

    earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                            mode="min", patience=5,
                                            restore_best_weights=True)

    H = model.fit(a, b,
                  validation_data=(c, d),
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=1)
    #             ,callbacks=[earlystopping])

    model.save('CNN' + '.h5')

    scores = model.evaluate(c, d, batch_size=batch_size, verbose=0)
    print('\n precisión:', scores[1])
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    Y_pred = model.predict(c)
    map_characters = {0: 'S. > 5 años', 1: 'S. < 5 años'}

    print('\n', classification_report(np.where(d > 0)[1], np.argmax(Y_pred, axis=1),
                                      target_names=list(map_characters.values())), sep='')
    Y_predicted = np.argmax(Y_pred, axis=1)

    Y_true = np.argmax(d, axis=1)

    Y_pred_prob = model.predict(X_test)[:, 1]
    plotAUC_ROC(Y_true, Y_pred_prob, X_train_RF, Y_train_RF, X_test_RF, Y_test_RF)
    plotLearningCurve(H)
    confusion_mtx = confusion_matrix(Y_true, Y_predicted)
    plotConfusionMatrix(confusion_mtx)


# -----------------------------------------------
# EJECUCIÓN
# -----------------------------------------------

# CNN
train_CNN(building_CNN(), X_train, Y_train, X_test, Y_test)

# RandomForest
##RandomForest(X_train_RF, Y_train_RF, X_test_RF, Y_test_RF)
