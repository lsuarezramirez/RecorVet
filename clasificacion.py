# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:18:59 2023

@author: Jhon Jairo Vega Diaz

Algoritmo para realizar un proceso de clasificación pasando una varaible objetivo
y las variables independientes, Primero optimiza el mejor método de sklearn y luego
optimiza una red neuronal, para finalmente guardar el modelo con mejor AUC.

Entrada: (variable objetivo, varaibles independientes)

imprime tablas de procesos y gráficos de curvas ROC

Salida: Guarda el modelo más eficiente de la red neuronal y el modelo sklearn


"""

import pandas as pd
import numpy as np
import os 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from pycaret.classification import (setup, compare_models, create_model, pull,
                                    tune_model, plot_model, predict_model, 
                                    save_model, load_model)
from sklearn.metrics import ConfusionMatrixDisplay,roc_curve
from sklearn.preprocessing import StandardScaler
import joblib 
import tensorflow as tf
from tensorflow import keras
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score
import plotly.figure_factory as ff
#from optbinning.scorecard import plot_cap, plot_ks
import warnings
warnings.filterwarnings("ignore")
import gc
gc.collect()



def make_model(inicial, train_features, capa, output_bias=None):   
    metrics = [
          keras.metrics.TruePositives(name='tp'),
          keras.metrics.FalsePositives(name='fp'),
          keras.metrics.TrueNegatives(name='tn'),
          keras.metrics.FalseNegatives(name='fn'), 
          keras.metrics.BinaryAccuracy(name='accuracy'),
          keras.metrics.Precision(name='precision'),
          keras.metrics.Recall(name='recall'),
          keras.metrics.AUC(name='auc'),
          keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
    
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(train_features.shape[-1],)),
        keras.layers.GaussianNoise(stddev=0.25, seed=123),
        keras.layers.Dense(int(inicial*capa[0]), activation='relu'),          
        keras.layers.Dense(int(inicial*capa[1]), activation='relu'),
        keras.layers.GaussianDropout(0.5, seed=123),
        keras.layers.Dense(1, activation='sigmoid',
                           bias_initializer=output_bias),
        ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)
    return model

    
def make_ds(features, labels, BUFFER_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()
    ds = ds.shuffle(BUFFER_SIZE).repeat()
    return ds

def NN(train_df,test_df,bal,corrida, capa):
    print("configuracion :", corrida)

    X_tl, y_tl = bal.fit_resample(train_df.drop(columns="Target"), train_df["Target"])
    X_tl["Target"]=y_tl 
    train_df=X_tl.copy()
    train_df=train_df.dropna()
    print(Counter(train_df["Target"]))
    
    X_train, X_test, y_train, y_test = train_test_split(train_df.drop(columns="Target"),
                                                train_df["Target"], test_size=0.2, random_state=123)
    X_train["Target"]=y_train
    train_df=X_train.copy()
    train_df=train_df.dropna()
    X_test["Target"]=y_test
    val_df=X_test.copy()    
    
    neg, pos = np.bincount(train_df['Target'])
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))
    
    train_labels = np.array(train_df.pop('Target'))
    bool_train_labels = train_labels != 0
    val_labels = np.array(val_df.pop('Target'))
    
    train_features = np.array(train_df)
    val_features = np.array(val_df)
    
    test_labels = np.array(test_df.pop('Target'))
    test_features = np.array(test_df)
    
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)
    joblib.dump(scaler, "std_scaler_NN.bin", compress=True)
    
    EPOCHS = 100
    BATCH_SIZE = 2048
    tf.random.set_seed(123)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)    
    pos_features = train_features[bool_train_labels]
    neg_features = train_features[~bool_train_labels]
    
    pos_labels = train_labels[bool_train_labels]
    neg_labels = train_labels[~bool_train_labels]
    
    ids = np.arange(len(pos_features))
    choices = np.random.choice(ids, len(neg_features))
    
    res_pos_features = pos_features[choices]
    res_pos_labels = pos_labels[choices]
    resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
    resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)
    
    order = np.arange(len(resampled_labels))
    np.random.shuffle(order)
    resampled_features = resampled_features[order]
    resampled_labels = resampled_labels[order]
    
    BUFFER_SIZE = 100000

    
    pos_ds = make_ds(pos_features, pos_labels, BUFFER_SIZE)
    neg_ds = make_ds(neg_features, neg_labels, BUFFER_SIZE)
    resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
    resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)
    resampled_steps_per_epoch = np.ceil(2.0*neg/BATCH_SIZE)
    
    resampled_model = make_model(len(train_df.columns), train_features, capa)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(2) 
    
    # Reset the bias to zero, since this dataset is balanced.
    output_layer = resampled_model.layers[-1] 
    output_layer.bias.assign([0])
    
    resampled_history = resampled_model.fit(
    resampled_ds,
    # These are not real epochs
    steps_per_epoch=20,
    epochs=10*EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_ds),
    verbose=-1)    
    resampled_model.save("modelo_NN.h5")
    
    test_predictions_resampled = resampled_model.predict(test_features, batch_size=BATCH_SIZE)
    
    
    
    resampled_results = resampled_model.evaluate(test_features, test_labels,
                                       batch_size=BATCH_SIZE, verbose=-1)
    for name, value in zip(['loss', 'compile_metrics','tp','tn','fn','accuracy',
                            'precision','recall','auc','prc'], resampled_results):
        if name == "accuracy":
            ac=value
        if name == "precision":
            prec=value
        if name == "recall":
            re=value
        if name == "auc":
            au=value 
    print(au)
    data = {corrida: [au,ac,prec,re]}
    return (test_labels, test_predictions_resampled, data)



def optimizar(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.15,random_state =123)
    
    train=X_train.copy()
    train["Target"]=y_train
    train=train.dropna()
    ################# Modelos sklearn ####################
    
    s = setup(train,
            #ignore_features = [], # Campos que no se tendrán en cuenta
             target = 'Target' #Definimos la variable objetivo
            
             ,fold_strategy = 'kfold' #Definimos lla estrategia para validacion cruzada creando a partir del dataset de Train, varios dataset de Validation
             ,fold = 10 #Definimos el numero total de folds a realizar
             ,train_size = 0.7
            
             ,fix_imbalance= True # SMOTE por defecto
             ,fix_imbalance_method = 'SMOTE' #Balanceamos la muesra utilizando la metodología SMOTE (crear muesrtas sintéticas aleatorias en función de la distribución actual)
            
             ,remove_outliers = True
             ,outliers_method = 'iforest'
             ,outliers_threshold = 0.05
            
             ,normalize = True # normalizamos las variables metricas con z score para evitar una alta varianza entre las variables y que se nos sesguen los resultados al comportamiento de alguna en particular
             ,normalize_method = 'zscore'
            
             ,remove_multicollinearity = True
             ,multicollinearity_threshold = 0.8 # Variables con correlación superior al 90% son eliminadas, solo permanece la que tenga mayor corr con Y
            
             ,feature_selection_method = 'univariate' #Definimos el feature selecction: rankeo de las variables por medio de un test de Chi2
             ,n_features_to_select = 0.2 # Maximo nos podemos quedar con el 80% de las variables X que tenemos ordenadas según su importancia
              
             ,session_id = 123 #semilla
            
             ,n_jobs = -1
             ,use_gpu = False
             
             ,verbose=1
             
             )
    
    best = compare_models(sort = 'AUC', exclude=["ridge"])
    
    results = pull() 
    results.sort_values('AUC', ascending= False)
    print(results)
    
    model = create_model(best)
    model_name=type(model).__name__
    print("modelo seleccionado: "+ model_name)
    results = pull() 
    results.sort_values('AUC', ascending= False)
    print(results)
    
    tuned_model = tune_model(model)
    
    results = pull() 
    results.sort_values('AUC', ascending= False)
    print(results)
    
    try:
        plot_model(tuned_model, plot = 'auc') 
        plot_model(tuned_model, plot = 'feature')
    except:
        print("falló gráfico")
        pass
    
    
    resultados_test=predict_model(tuned_model, data=X_test ,raw_score=True)
    
    RocCurveDisplay.from_predictions(
        y_test,
        resultados_test["prediction_score_1"],
        name="Target",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="Nivel de azar (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("Tasa de falso Positivo")
    plt.ylabel("Tasa de verdadero Positivo")
    plt.title("Curva ROC Uno-vs-Resto :\n"+model_name+"\nBase de test")
    plt.legend()
    plt.savefig("ROC_sklearn.png")
    plt.show()
    
    cm = confusion_matrix(y_test, resultados_test["prediction_label"]) 
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
    cm_display.plot()
    plt.grid(False)
    plt.title("Matrix de confusión\n"+model_name+"\nBase de test")
    plt.savefig("confusion_sklearn.png")
    plt.show()
    print ("Confusion Matrix : \n", cm) 
    skl_auc = roc_auc_score(y_test, resultados_test["prediction_score_1"])
    print('Roc_auc_score modelo '+model_name+' :' , roc_auc_score(y_test, resultados_test["prediction_score_1"]))

    # Create the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, resultados_test["prediction_score_1"])
    print('GINI a 0.5 auc = ',2 * roc_auc_score(y_test, resultados_test["prediction_score_1"]) - 1)


    # Calculate the Youden's J statistic
    youdenJ = tpr - fpr

    # Calculate the G-mean
    gmean = np.sqrt(tpr * (1 - fpr))
    # Find the optimal threshold
    index = np.argmax(youdenJ)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    youdenJOpt = round(gmean[index], ndigits = 4)
    fprOpt = round(fpr[index], ndigits = 4)
    tprOpt = round(tpr[index], ndigits = 4)
    print('Mejor Threshold: {} con Youden J statistic: {} para el modelo sklearn'.format(thresholdOpt, youdenJOpt))
    print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

    cm = confusion_matrix(y_test, resultados_test["prediction_score_1"] > thresholdOpt) 
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
    cm_display.plot()
    plt.grid(False)
    plt.title("Matrix de confusión\n"+model_name+"\nBase de test\nPunto de corte: "+str(thresholdOpt))
    plt.savefig("confusion_sklearn_opt.png")
    plt.show()
    print ("Confusion Matrix : \n", cm)  

    # Graficar la distribución acumulada de los datos
    positive_values = resultados_test["prediction_score_1"][y_test == 0]
    negative_values = resultados_test["prediction_score_1"][y_test == 1]
    print('KS sklearn a '+ str(thresholdOpt)+':',stats.ks_2samp(positive_values, negative_values))
    """
    sns.histplot(positive_values, cumulative=True, kde=True, label="Positivo", stat="density", color="#D98F08")
    sns.histplot(negative_values, cumulative=True, kde=True, label="Negativo", stat="density", color="#0067B1")
    plt.title("Prueba de Kolmogorov-Smirnov")
    plt.legend()
    plt.xlabel("Probabilidad")
    plt.savefig("KS_sklearn_opt.png")
    plt.show()
    """
    plot_cap(y_test, resultados_test["prediction_score_1"],savefig=True, fname="Gini_sklearn_opt.png")
    plt.show()
    plot_ks(y_test, resultados_test["prediction_score_1"],savefig=True, fname="KS_sklearn_opt.png")
    plt.show()


    datos=pd.DataFrame({'y_val':y_test, 'predicciones': resultados_test["prediction_score_1"]})
    colores=["#D98F08", "#0067B1"]
    fig = ff.create_distplot([list(datos[datos['y_val']==0]['predicciones']),list(datos[datos['y_val']==1]['predicciones'])], ["Negativo","Positivo"],colors=colores,bin_size=.02, show_rug=False)
    fig.layout.update({'title': 'Target Vs probabilidad - Test'})
    fig.layout.yaxis.update({'title': 'Cantidad de registros'})
    fig.layout.xaxis.update({'title': 'Probabilidad'})
    fig.show()


    save_model(tuned_model, 'modelo_optimizado_sklearn')
    
    ################# Red Neuronal ####################
    
    metrics = pd.DataFrame(columns=["AUC_val", "accuracy_val","precision_val","recall_val"])
    (tail,head)=os.path.split(os.getcwd())
    
    capas=[[1,0.5], [2,0.333],[2,0.5],[2,0.333],[0.5,0.5],[0.5,0.333],[0.5,1]]
    
    for balanceo in range(2):
        #balanceo poblaciones
        if balanceo==0:  
            bal = TomekLinks(sampling_strategy='majority', n_jobs=-1)       
        else:
            bal = SMOTE(n_jobs=-1)
        for value,capa in enumerate(capas):
            train_df= train.copy()
            X_test["Target"]=y_test
            test_df=X_test.copy()
            print(Counter(train_df["Target"]))
            corrida=str(balanceo)+str(value)
            try:
                test_labels, test_predictions_resampled, data = NN(train_df,test_df,bal,corrida, capa)
                metrics_model = pd.DataFrame.from_dict(data, orient='index',columns=metrics.columns)
                metrics = pd.concat([metrics, metrics_model], sort=False, ignore_index=False)
            except:
                pass
    train_df= train.copy()
    X_test["Target"]=y_test
    test_df=X_test.copy()
    metrics = metrics.sort_values(by='AUC_val', ascending=False)
    print(metrics)
    metrics = metrics.reset_index()
    for balanceo in range(2):
        if balanceo==0:  
            bal = TomekLinks(sampling_strategy='majority', n_jobs=-1)       
        else:
            bal = SMOTE(n_jobs=-1)
        for value,capa in enumerate(capas):
            corrida=str(balanceo)+str(value)
            if metrics.loc[0]["index"] == corrida:
                test_labels, test_predictions_resampled, data = NN(train_df,test_df,bal,corrida, capa)     

    cm = confusion_matrix(test_labels, test_predictions_resampled > 0.5) 
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
    cm_display.plot()
    plt.grid(False)
    plt.title("Matrix de confusión\nRed nueronal\nBase de test")
    plt.savefig("confusion_nn.png")
    #plt.show()
    print ("Confusion Matrix : \n", cm)         
    
    RocCurveDisplay.from_predictions(
        test_labels,
        test_predictions_resampled,
        name="Target",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="Nivel de azar (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("Tasa de falso Positivo")
    plt.ylabel("Tasa de verdadero Positivo")
    plt.title("Curva ROC Uno-vs-Resto :\nRed nueronal\nBase de test")
    plt.legend()
    plt.savefig("ROC_nn.png")
    #plt.show()

    # Create the ROC curve
    fpr, tpr, thresholds = roc_curve(test_labels, test_predictions_resampled)
    print('GINI a 0.5 auc = ',2 * roc_auc_score(test_labels, test_predictions_resampled) - 1)
    # Calculate the Youden's J statistic
    youdenJ = tpr - fpr
    # Calculate the G-mean
    gmean = np.sqrt(tpr * (1 - fpr))
    # Find the optimal threshold
    index = np.argmax(youdenJ)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    youdenJOpt = round(gmean[index], ndigits = 4)
    fprOpt = round(fpr[index], ndigits = 4)
    tprOpt = round(tpr[index], ndigits = 4)
    
    NN_auc = roc_auc_score(test_labels, test_predictions_resampled)

    print('Mejor Threshold: {} con Youden J statistic: {} para la red Neuronal'.format(thresholdOpt, youdenJOpt))
    print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

    cm = confusion_matrix(test_labels, test_predictions_resampled > thresholdOpt) 
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
    cm_display.plot()
    plt.grid(False)
    plt.title("Matrix de confusión\nRed nueronal\nBase de test\nPunto de corte: "+str(thresholdOpt))
    plt.savefig("confusion_nn_opt.png")
    plt.show()
    print ("Confusion Matrix : \n", cm)  

    # Graficar la distribución acumulada de los datos
    positive_values = test_predictions_resampled[test_labels == 0]
    negative_values = test_predictions_resampled[test_labels == 1]
    positive_values=positive_values.T[0]
    negative_values=negative_values.T[0]
    print('KS NN a '+ str(thresholdOpt)+':',stats.ks_2samp(positive_values, negative_values))
    """
    sns.histplot(positive_values, cumulative=True,kde=True, label="Positivo", stat="density", color="#D98F08")
    sns.histplot(negative_values, cumulative=True,kde=True, label="Negativo", stat="density", color="#0067B1")
    plt.title("Prueba de Kolmogorov-Smirnov")
    plt.legend()
    plt.xlabel("Probabilidad")
    plt.savefig("KS_nn_opt.png")
    plt.show()
    """
    plot_cap(test_labels, test_predictions_resampled.T[0], savefig=True, fname="Gini_nn_opt.png" )
    plt.show()
    plot_ks(test_labels, test_predictions_resampled.T[0], savefig=True, fname="KS_nn_opt.png")
    plt.show()

    datos=pd.DataFrame({'y_val':test_labels, 'predicciones': test_predictions_resampled.T[0]})
    colores=["#D98F08", "#0067B1"]
    fig = ff.create_distplot([list(datos[datos['y_val']==0]['predicciones']),list(datos[datos['y_val']==1]['predicciones'])], ["Negativo","Positivo"],colors=colores,bin_size=.02, show_rug=False)
    fig.layout.update({'title': 'Target Vs probabilidad - Test'})
    fig.layout.yaxis.update({'title': 'Cantidad de registros'})
    fig.layout.xaxis.update({'title': 'Probabilidad'})    
    fig.show()

    print('Roc_auc_score Red neuronal : ', roc_auc_score(test_labels, test_predictions_resampled))
    print('Roc_auc_score modelo '+model_name+' :' , roc_auc_score(y_test, resultados_test["prediction_score_1"]))
    print("Escoja el modelo más eficiente")
    if skl_auc > NN_auc:
        return "skl"
    else: 
        return "NN"
    
def prediccion_modelo(X,optimo):
    if optimo=="NN":
        X=np.array(X)
        scaler=joblib.load("std_scaler_NN.bin") # abrir modelo de transformación
        model_NN = tf.keras.models.load_model("modelo_NN.h5")
        prediction_features=scaler.transform(X) # transformar datos      
        predictions = model_NN.predict(prediction_features) # realizar las predicciones 
        return predictions.T[0] 
    if optimo=="skl":
        loaded_model = load_model('modelo_optimizado_sklearn')
        print(loaded_model)
        predictions = predict_model(loaded_model, data=X, raw_score=True)
        return np.array(predictions["prediction_score_1"])
    return []

def ejemplo():
    base=pd.read_csv("../base_multi_objetivo.csv")
    y=base["BGI_0"]
    columnas=[]
    for i in range(30):
        columnas.append("var_"+str(i))
    X=base[columnas]
    res=optimizar(X, y)
    resultados=prediccion_modelo(X,res)
    print(resultados)

if __name__ == "__main__":
    ejemplo()    
