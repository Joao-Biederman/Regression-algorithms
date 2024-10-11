import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import scipy.stats as stats

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import mannwhitneyu
import time

rlm_mae = []
rlm_mse = []
rlm_rmse = []

svr_mae = []
svr_mse = []
svr_rmse = []

knr_mae = []
knr_mse = []
knr_rmse = []

mlp_mae = []
mlp_mse = []
mlp_rmse = []

rf_mae = []
rf_mse = []
rf_rmse = []

gb_mae = []
gb_mse = []
gb_rmse = []



data = pd.read_csv('Wine/winequality-red.csv')
for k in range(20):
    #=========================== DATA ============================================
    data = shuffle(data)

    x = data.iloc[:, 0:11]
    y = data.iloc[:, 11]

    value_counts = y.value_counts().sort_index()
    print("Contagem de valores únicos na última coluna (qualidade):")
    print(value_counts)

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(x)
    x = X_normalized

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.5, stratify=y)
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp)

    accuracy = []

    corr_matrix = data.corr()
    print(corr_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title('Heatmap de Correlação')
    #plt.show()


    #=========================== DATA ============================================

    #=========================== REGRESSÃO LINEAR MULTIPLA ============================================
    print("==========RLM=====================================\n")

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)

    y_pred = np.round(y_pred).astype(int)
    y_pred = np.clip(y_pred, y.min(), y.max())

    mae_test = mean_absolute_error(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred, squared=True)
    rmse_test = mean_squared_error(y_test, y_pred, squared=False)

    print(f"Execução {k}: MAE para Regressão Linear Múltipla: {mae_test:.2f}")
    print(f"Execução {k}: MSE para Regressao Linear Múltipla: {mse_test:.2f}")
    print(f"Execução {k}: RMSE para Regressão Linear Múltipla: {rmse_test:.2f}")

    
    rlm_mae.append(mae_test)
    rlm_mse.append(mse_test)
    rlm_rmse.append(rmse_test)
    
    # Abrindo arquivo para escrita (adicionando ao final do arquivo)
    with open('resultados_rlm.txt', 'a') as file:
        file.write(f"Execução {k}:\n")
        file.write(f"MAE para Regressão Linear Múltipla: {mae_test:.2f}\n")
        file.write(f"MSE para Regressão Linear Múltipla: {mse_test:.2f}\n")
        file.write(f"RMSE para Regressão Linear Múltipla: {rmse_test:.2f}\n")
        file.write("\n")

    # Calculando as médias após todas as execuções
    mae_media_rlm = sum(rlm_mae) / len(rlm_mae)
    mse_media_rlm = sum(rlm_mse) / len(rlm_mse)
    rmse_media_rlm = sum(rlm_rmse) / len(rlm_rmse)



    #=========================== RLM - MATRIX/STATS ============================================
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=value_counts.index, yticklabels=value_counts.index)
    plt.title('Matriz de Confusão')
    plt.xlabel('Classes Previstas')
    plt.ylabel('Classes Reais')
    #plt.show()

    sensibilidades_rlm = []
    especifidades_rlm = []
    
    for i in range(len(conf_matrix)):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)
        
        sensibilidade = TP / (TP + FN) 
        especifidade = TN / (TN + FP)
        
        sensibilidades_rlm.append(sensibilidade)
        especifidades_rlm.append(especifidade)

    print("Sensibilidades:", sensibilidades_rlm)
    print("Especificidades:", especifidades_rlm)

    acuracia = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acuracia:.2f}")
    acuracia_rlm = []
    acuracia_rlm.append(acuracia)

    #=========================== RLM - MATRIX/STATS ============================================

    #=========================== REGRESSÃO LINEAR MULTIPLA ============================================

    #=========================== KNR ===========================================================
    print("==========KNR=====================================\n")

    #========================== KNR - GRID SEARCH ==============================================
    param_grid_knr = {
        'n_neighbors': list(range(1, 50)),
        'metric': ['euclidean', 'manhattan']  
    }

    best_rmse = float('inf')
    best_params = {}

    start_time = time.time()
    for n_neighbors in param_grid_knr['n_neighbors']:
        for metric in param_grid_knr['metric']:  
            knr = KNeighborsRegressor(n_neighbors=n_neighbors, metric=metric)
            knr.fit(x_train, y_train)
            
            opiniao = knr.predict(x_val)
            rmse_val = mean_squared_error(y_val, opiniao, squared=False) 

            
            if rmse_val < best_rmse:  
                best_rmse = rmse_val
                best_params = {'n_neighbors': n_neighbors, 'metric': metric}  

    end_time = time.time()

    execution_time = end_time - start_time
    print("Melhores Parâmetros: ", best_params)
    print("Melhor RMSE no conjunto de validação: ", best_rmse)
    print(f"Execution time: {execution_time:.6f} seconds")

    #========================== KNR - GRID SEARCH ==============================================

    #=========================== KNR - TREINAMENTO ==============================================

    KNR_best = KNeighborsRegressor(n_neighbors=best_params['n_neighbors'], metric=best_params['metric'])
    KNR_best.fit(x_train, y_train)

    y_pred = KNR_best.predict(x_test)
    mae_test = mean_absolute_error(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred, squared=True)
    rmse_test = mean_squared_error(y_test, y_pred, squared=False)

    print(f"MAE no conjunto de teste: {mae_test:.2f}")
    print(f"MSE no conjunto de teste: {mse_test:.2f}")
    print(f"RMSE no conjunto de teste: {rmse_test:.2f}")

    knr_mae.append(mae_test)
    knr_mse.append(mse_test)
    knr_rmse.append(rmse_test)

    # Abrindo arquivo para escrita (adicionando ao final do arquivo)
    with open('resultados_knr.txt', 'a') as file:
        file.write(f"Execução {k}:\n")
        file.write(f"MAE para KNR: {mae_test:.2f}\n")
        file.write(f"MSE para KNR: {mse_test:.2f}\n")
        file.write(f"RMSE para KNR: {rmse_test:.2f}\n")
        file.write(f"Execution time: {execution_time:.6f} seconds")
        file.write("\n")

    # Calculando as médias após todas as execuções
    mae_media_knr = sum(knr_mae) / len(knr_mae)
    mse_media_knr = sum(knr_mse) / len(knr_mse)
    rmse_media_knr = sum(knr_rmse) / len(knr_rmse)

    y_pred = np.round(y_pred).astype(int)
    y_pred = np.clip(y_pred, y.min(), y.max())

     # ========================= KNR - TREINAMENTO ==============================================

    #=========================== KNR - MATRIX/STATS ============================================

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(conf_matrix)

    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=value_counts.index, yticklabels=value_counts.index)
    plt.title('Matriz de Confusão')
    plt.xlabel('Classes Previstas')
    plt.ylabel('Classes Reais')
    #plt.show()

    sensibilidades_knr = []
    especifidades_knr = []
    
    
    for i in range(len(conf_matrix)):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)
        
        sensibilidade = TP / (TP + FN) 
        especifidade = TN / (TN + FP)
        
        sensibilidades_knr.append(sensibilidade)
        especifidades_knr.append(especifidade)


    print("Sensibilidades:", sensibilidades_knr)
    print("Especificidades:", especifidades_knr)

    acuracia = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acuracia:.2f}")
    acuracia_knr = []
    acuracia_knr.append(acuracia)

    #=========================== KNR - MATRIX/STATS ============================================

    #=========================== KNR ===========================================================

    #=========================== SVR ===========================================================

    print("==========SVR=====================================\n")

    #========================== SVR - GRID SEARCH ==============================================
    param_grid_svr = {
        'C': list(range(1, 2)),
        'kernel': ['linear', 'rbf', 'poly'],
    }

    best_rmse = float('inf')
    best_params = {}

    start_time = time.time()
    for C in param_grid_svr['C']:
        for kernel in param_grid_svr['kernel']:
            svr = SVR(C=C, kernel=kernel)
            svr.fit(x_train, y_train)
            
            opiniao = svr.predict(x_val)
            rmse_val = mean_squared_error(y_val, opiniao, squared=False)  
            
            if rmse_val < best_rmse:  
                best_rmse = rmse_val
                best_params = {'C': C, 'kernel': kernel}
    end_time = time.time()
    execution_time = end_time - start_time

    print("Melhores Parâmetros: ", best_params)
    print("Melhor RMSE no conjunto de validação: ", best_rmse)
    print(f"Execution time: {execution_time:.6f} seconds")

    #========================== SVR - GRID SEARCH ==============================================

    #========================= SVR - TREINAMENTO ==============================================

    SVR_best = SVR(C=best_params['C'], kernel=best_params['kernel'])
    SVR_best.fit(x_train, y_train)

    y_pred = SVR_best.predict(x_test)
    mae_test = mean_absolute_error(y_test, y_pred) 
    mse_test = mean_squared_error(y_test, y_pred, squared=True)
    rmse_test = mean_squared_error(y_test, y_pred, squared=False)

    print(f"MAE no conjunto de teste: {mae_test:.2f}")
    print(f"MSE no conjunto de teste: {mse_test:.2f}")
    print(f"RMSE no conjunto de teste: {rmse_test:.2f}")

    svr_mae.append(mae_test)
    svr_mse.append(mse_test)
    svr_rmse.append(rmse_test)

    # Abrindo arquivo para escrita (adicionando ao final do arquivo)
    with open('resultados_svr.txt', 'a') as file:
        file.write(f"Execução {k}:\n")
        file.write(f"MAE para SVR: {mae_test:.2f}\n")
        file.write(f"MSE para SVR: {mse_test:.2f}\n")
        file.write(f"RMSE para SVR: {rmse_test:.2f}\n")
        file.write(f"Execution time: {execution_time:.6f} seconds")
        file.write("\n")

    # Calculando as médias após todas as execuções
    mae_media_svr = sum(svr_mae) / len(svr_mae)
    mse_media_svr = sum(svr_mse) / len(svr_mse)
    rmse_media_svr = sum(svr_rmse) / len(svr_rmse)

    y_pred = np.round(y_pred).astype(int)
    y_pred = np.clip(y_pred, y.min(), y.max())  

    #========================= SVR - TREINAMENTO ==============================================

    #=========================== SVR - MATRIX/STATS ============================================

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=value_counts.index, yticklabels=value_counts.index)
    plt.title('Matriz de Confusão')
    plt.xlabel('Classes Previstas')
    plt.ylabel('Classes Reais')
    #plt.show()

    sensibilidades_svr = []
    especifidades_svr = []
    
    for i in range(len(conf_matrix)):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)
        
        sensibilidade = TP / (TP + FN) 
        especifidade = TN / (TN + FP)
        
        sensibilidades_svr.append(sensibilidade)
        especifidades_svr.append(especifidade)

   
    print("Sensibilidades:", sensibilidades_svr)
    print("Especificidades:", especifidades_svr)

    acuracia = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acuracia:.2f}")
    acuracia_svr = []
    acuracia_svr.append(acuracia)

    #=========================== SVR - MATRIX/STATS ============================================
    
    #=========================== SVR ==========================================================

    #========================== MLP ===========================================================
    print("==========MLP=====================================\n")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    #=========================== MLP - GRID SEARCH ===========================================================

    param_grid_mlp = {
        'hidden_layer_sizes': [[6, 6], [6, 11], [11, 6], [11, 11], [6, 6, 6], [6, 6, 11], [6, 11, 6], [6, 11, 11], [11, 6, 6], [11, 6, 11], [11, 11, 6], [11, 11, 11]],
        'activation': ['relu', 'tanh', 'logistic'],
        'max_iter': [200, 300],
        'learning_rate_init': [0.001, 0.01]
    }

    best_rmse = float('inf')
    best_params = {}

    start_time = time.time()
    for hidden_layer_sizes in param_grid_mlp['hidden_layer_sizes']:
        for activation in param_grid_mlp['activation']:
            for max_iter in param_grid_mlp['max_iter']:
                for learning_rate in param_grid_mlp['learning_rate_init']:
                    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                                    activation=activation, 
                                    max_iter=max_iter, 
                                    learning_rate_init=learning_rate)
                    
                    mlp.fit(x_train, y_train)

                    opiniao_val = mlp.predict(x_val)
                    rmse_val = mean_squared_error(y_val, opiniao_val, squared=False)

                    if rmse_val < best_rmse:
                        best_rmse = rmse_val
                        best_params = {'hidden_layer_sizes': hidden_layer_sizes, 
                                    'activation': activation, 
                                    'max_iter': max_iter, 
                                    'learning_rate_init': learning_rate}

    end_time = time.time()
    execution_time = end_time - start_time
    print("Melhores Parâmetros: ", best_params)
    print("Melhor RMSE no conjunto de validação: ", best_rmse)
    print(f"Execution time: {execution_time:.6f} seconds")

    MLP_best = MLPRegressor(hidden_layer_sizes=best_params['hidden_layer_sizes'],
                            activation=best_params['activation'], 
                            max_iter=best_params['max_iter'], 
                            learning_rate_init=best_params['learning_rate_init'])
    
    #=========================== MLP - GRID SEARCH ===========================================================

    #=========================== MLP - TREINAMENTO ===========================================================

    MLP_best.fit(x_train, y_train)

    y_pred = MLP_best.predict(x_test)
    mae_test = mean_absolute_error(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred, squared=True)
    rmse_test = mean_squared_error(y_test, y_pred, squared=False)

    print(f"MAE no conjunto de teste: {mae_test:.2f}")
    print(f"MSE no conjunto de teste: {mse_test:.2f}")
    print(f"RMSE no conjunto de teste: {rmse_test:.2f}")

    mlp_mae.append(mae_test)
    mlp_mse.append(mse_test)
    mlp_rmse.append(rmse_test)

    # Abrindo arquivo para escrita (adicionando ao final do arquivo)
    with open('resultados_mlp.txt', 'a') as file:
        file.write(f"Execução {k}:\n")
        file.write(f"MAE para MLP: {mae_test:.2f}\n")
        file.write(f"MSE para MLP: {mse_test:.2f}\n")
        file.write(f"RMSE para MLP: {rmse_test:.2f}\n")
        file.write(f"Execution time: {execution_time:.6f} seconds")
        file.write("\n")

    # Calculando as médias após todas as execuções
    mae_media_mlp = sum(mlp_mae) / len(mlp_mae)
    mse_media_mlp = sum(mlp_mse) / len(mlp_mse)
    rmse_media_mlp = sum(mlp_rmse) / len(mlp_rmse)

    y_pred = np.round(y_pred).astype(int)
    y_pred = np.clip(y_pred, y.min(), y.max()) 

    #=========================== MLP - TREINAMENTO ===========================================================

    # ========================== MLP - MATRIX/STATS ===========================================================

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=value_counts.index, yticklabels=value_counts.index)
    plt.title('Matriz de Confusão')
    plt.xlabel('Classes Previstas')
    plt.ylabel('Classes Reais')
    #plt.show()

    sensibilidades_mlp = []
    especifidades_mlp = []
    
    for i in range(len(conf_matrix)):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)
        
        sensibilidade = TP / (TP + FN) 
        especifidade = TN / (TN + FP)
        
        sensibilidades_mlp.append(sensibilidade)
        especifidades_mlp.append(especifidade)

   
    print("Sensibilidades:", sensibilidades_mlp)
    print("Especificidades:", especifidades_mlp)

    acuracia = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acuracia:.2f}")
    acuracia_mlp = []
    acuracia_mlp.append(acuracia)

    # ========================== MLP - MATRIX/STATS ===========================================================

    # ========================== MLP  ===========================================================

    #================================== RF ===========================================================
    print("==========RF=====================================\n")
    
    #=========================== RF - GRID SEARCH ===========================================================

    param_grind_rf = {
        'n_estimators': [100, 200, 300],
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    best_rmse = float('inf')
    best_params = {}

    start_time = time.time()
    for n_estimators in param_grind_rf['n_estimators']:
        for criterion in param_grind_rf['criterion']:
            for max_depth in param_grind_rf['max_depth']:
                for min_samples_split in param_grind_rf['min_samples_split']:
                    for min_samples_leaf in param_grind_rf['min_samples_leaf']:
                        rf = RandomForestRegressor(n_estimators=n_estimators, 
                                                max_depth=max_depth, 
                                                min_samples_split=min_samples_split, 
                                                min_samples_leaf=min_samples_leaf)
                        
                        rf.fit(x_train, y_train)

                        opiniao_val = rf.predict(x_val)
                        rsme_val = mean_squared_error(y_val, opiniao_val, squared=False)

                        if rsme_val < best_rmse:
                            best_rmse = rsme_val
                            best_params = {'n_estimators': n_estimators, 
                                        'criterion': criterion, 
                                        'max_depth': max_depth, 
                                        'min_samples_split': min_samples_split, 
                                        'min_samples_leaf': min_samples_leaf}

    end_time = time.time()
    execution_time = end_time - start_time
    print("Melhores Parâmetros: ", best_params)
    print("Melhor RMSE no conjunto de validação: ", best_rmse)
    print(f"Execution time: {execution_time:.6f} seconds")

    RF_BEST = RandomForestRegressor(n_estimators=best_params['n_estimators'], criterion = best_params['criterion'], max_depth = best_params['max_depth'], min_samples_split = best_params['min_samples_split'], min_samples_leaf = best_params['min_samples_leaf'])                       
        
    #=========================== RF - GRID SEARCH ===========================================================

    #=========================== RF - TREINAMENTO ===========================================================

    RF_BEST.fit(x_train, y_train)

    y_pred = RF_BEST.predict(x_test)
    mae_test = mean_absolute_error(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred, squared=True)
    rmse_test = mean_squared_error(y_test, y_pred, squared=False)

    print(f"MAE no conjunto de teste: {mae_test:.2f}")
    print(f"MSE no conjunto de teste: {mse_test:.2f}")
    print(f"RMSE no conjunto de teste: {rmse_test:.2f}")
    
    rf_mae.append(mae_test)
    rf_mse.append(mse_test)
    rf_rmse.append(rmse_test)

    # Abrindo arquivo para escrita (adicionando ao final do arquivo)
    with open('resultados_rf.txt', 'a') as file:
        file.write(f"Execução {k}:\n")
        file.write(f"MAE para RF: {mae_test:.2f}\n")
        file.write(f"MSE para RF: {mse_test:.2f}\n")
        file.write(f"RMSE para RF: {rmse_test:.2f}\n")
        file.write(f"Execution time: {execution_time:.6f} seconds")
        file.write("\n")

    # Calculando as médias após todas as execuções
    mae_media_rf = sum(rf_mae) / len(rf_mae)
    mse_media_rf = sum(rf_mse) / len(rf_mse)
    rmse_media_rf = sum(rf_rmse) / len(rf_rmse)

    y_pred = np.round(y_pred).astype(int)
    y_pred = np.clip(y_pred, y.min(), y.max()) 

    #=========================== RF - TREINAMENTO ===========================================================

    # ========================== RF - MATRIX/STATS ===========================================================

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=value_counts.index, yticklabels=value_counts.index)
    plt.title('Matriz de Confusão')
    plt.xlabel('Classes Previstas')
    plt.ylabel('Classes Reais')
    #plt.show()

    sensibilidades_rf = []
    especifidades_rf = []
    
    for i in range(len(conf_matrix)):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)
        
        sensibilidade = TP / (TP + FN) 
        especifidade = TN / (TN + FP)
        
        sensibilidades_rf.append(sensibilidade)
        especifidades_rf.append(especifidade)

    print("Sensibilidades:", sensibilidades_rf)
    print("Especificidades:", especifidades_rf)

    acuracia = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acuracia:.2f}")
    acuracia_rf = []
    acuracia_rf.append(acuracia)

    # ========================== RF - MATRIX/STATS ===========================================================

    # ========================== RF ===========================================================

    # ========================= GB ===========================================================

    print("==========GB=====================================\n")

    # ========================= GB - GRID SEARCH ===========================================================

    param_grind_gb = {
        'n_estimators': [100, 200, 300],
        'loss': ['absolute_error', 'huber', 'quantile'],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    best_rmse = float('inf')
    best_params = {}

    start_time = time.time()
    for n_estimators in param_grind_gb['n_estimators']:
        for loss in param_grind_gb['loss']:
            for learning_rate in param_grind_gb['learning_rate']:
                for max_depth in param_grind_gb['max_depth']:
                    for min_samples_split in param_grind_gb['min_samples_split']:
                        for min_samples_leaf in param_grind_gb['min_samples_leaf']:
                            gb = GradientBoostingRegressor(n_estimators=n_estimators, 
                                                    learning_rate=learning_rate, 
                                                    max_depth=max_depth, 
                                                    min_samples_split=min_samples_split, 
                                                    min_samples_leaf=min_samples_leaf)
                            
                            gb.fit(x_train, y_train)

                            opiniao_val = gb.predict(x_val)
                            rmse_val = mean_squared_error(y_val, opiniao_val, squared=False)

                            if rmse_val < best_rmse:
                                best_rmse = rmse_val
                                best_params = {'n_estimators': n_estimators, 
                                            'loss': loss, 
                                            'learning_rate': learning_rate, 
                                            'max_depth': max_depth, 
                                            'min_samples_split': min_samples_split, 
                                            'min_samples_leaf': min_samples_leaf}

    end_time = time.time()
    execution_time = end_time - start_time
    print("Melhores Parâmetros: ", best_params)
    print("Melhor RMSE no conjunto de validação: ", best_rmse)
    print(f"Execution time: {execution_time:.6f} seconds")

    GB_BEST = GradientBoostingRegressor(n_estimators= best_params['n_estimators'],
                                        loss = best_params['loss'],
                                        learning_rate = best_params['learning_rate'],
                                        max_depth = best_params['max_depth'],
                                        min_samples_split = best_params['min_samples_split'],
                                        min_samples_leaf = best_params['min_samples_leaf'])

    # ========================= GB - GRID SEARCH ===========================================================

    # ========================= GB - TREINAMENTO ===========================================================

    GB_BEST.fit(x_train, y_train)

    y_pred = GB_BEST.predict(x_test)
    mae_test = mean_absolute_error(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred, squared=True)
    rmse_test = mean_squared_error(y_test, y_pred, squared=False)

    print(f"MAE no conjunto de teste: {mae_test:.2f}")
    print(f"MSE no conjunto de teste: {mse_test:.2f}")
    print(f"RMSE no conjunto de teste: {rmse_test:.2f}")

    gb_mae.append(mae_test)
    gb_mse.append(mse_test)
    gb_rmse.append(rmse_test)

    # Abrindo arquivo para escrita (adicionando ao final do arquivo)
    with open('resultados_gb.txt', 'a') as file:
        file.write(f"Execução {k}:\n")
        file.write(f"MAE para GB: {mae_test:.2f}\n")
        file.write(f"MSE para GB: {mse_test:.2f}\n")
        file.write(f"RMSE para GB: {rmse_test:.2f}\n")
        file.write(f"Execution time: {execution_time:.6f} seconds")
        file.write("\n")

    # Calculando as médias após todas as execuções
    mae_media_gb = sum(gb_mae) / len(gb_mae)
    mse_media_gb = sum(gb_mse) / len(gb_mse)
    rmse_media_gb = sum(gb_rmse) / len(gb_rmse)

    y_pred = np.round(y_pred).astype(int)
    y_pred = np.clip(y_pred, y.min(), y.max()) 

    # ========================= GB - TREINAMENTO ===========================================================

    # ========================== GB - MATRIX/STATS ===========================================================

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=value_counts.index, yticklabels=value_counts.index)
    plt.title('Matriz de Confusão')
    plt.xlabel('Classes Previstas')
    plt.ylabel('Classes Reais')
    #plt.show()

    sensibilidades_gb = []
    especifidades_gb = []

    for i in range(len(conf_matrix)):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)
        
        sensibilidade = TP / (TP + FN) 
        especifidade = TN / (TN + FP)
        
        sensibilidades_gb.append(sensibilidade)
        especifidades_gb.append(especifidade)

    print("Sensibilidades:", sensibilidades_gb)
    print("Especificidades:", especifidades_gb)

    acuracia = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acuracia:.2f}")
    acuracia_gb = []
    acuracia_gb.append(acuracia)

    # ========================== GB - MATRIX/STATS ===========================================================

    # ========================== GB ===========================================================

    # ========================== PRINTANDO AS MEDIAS =====================================================

# Escrevendo as médias no arquivo
with open('resultados_rlm.txt', 'a') as file:
    file.write(f"Média MAE: {mae_media_rlm:.2f}\n")
    file.write(f"Média MSE: {mse_media_rlm:.2f}\n")
    file.write(f"Média RMSE: {rmse_media_rlm:.2f}\n")

# Escrevendo as médias no arquivo
with open('resultados_knr.txt', 'a') as file:
    file.write(f"Média MAE: {mae_media_knr:.2f}\n")
    file.write(f"Média MSE: {mse_media_knr:.2f}\n")
    file.write(f"Média RMSE: {rmse_media_knr:.2f}\n")

# Escrevendo as médias no arquivo
with open('resultados_svr.txt', 'a') as file:
    file.write(f"Média MAE: {mae_media_svr:.2f}\n")
    file.write(f"Média MSE: {mse_media_svr:.2f}\n")
    file.write(f"Média RMSE: {rmse_media_svr:.2f}\n")

# Escrevendo as médias no arquivo
with open('resultados_mlp.txt', 'a') as file:
    file.write(f"Média MAE: {mae_media_mlp:.2f}\n")
    file.write(f"Média MSE: {mse_media_mlp:.2f}\n")
    file.write(f"Média RMSE: {rmse_media_mlp:.2f}\n")

# Escrevendo as médias no arquivo
with open('resultados_rf.txt', 'a') as file:
    file.write(f"Média MAE: {mae_media_rf:.2f}\n")
    file.write(f"Média MSE: {mse_media_rf:.2f}\n")
    file.write(f"Média RMSE: {rmse_media_rf:.2f}\n")

# Escrevendo as médias no arquivo
with open('resultados_gb.txt', 'a') as file:
    file.write(f"Média MAE: {mae_media_gb:.2f}\n")
    file.write(f"Média MSE: {mse_media_gb:.2f}\n")
    file.write(f"Média RMSE: {rmse_media_gb:.2f}\n")

# Abrindo o arquivo para gravação
with open('stats.txt', 'w') as f:

    # ========================== KRUSKAL-WALLIS ====================================

    # Realizando o teste Kruskal-Wallis
    rmse = [rlm_rmse, knr_rmse, svr_rmse, mlp_rmse, rf_rmse, gb_rmse]
    stat, p_value = stats.kruskal(rmse)

    # Gravando os resultados no arquivo
    f.write(f'Estatística de Kruskal-Wallis: {stat}\n')
    f.write(f'Valor-p: {p_value}\n')

    alpha = 0.05
    if p_value < alpha:
        f.write("Há diferença estatisticamente significativa entre os classificadores.\n")
    else:
        f.write("Não há diferença estatisticamente significativa entre os classificadores.\n")

    f.write('\n# ========================== KRUSKAL-WALLIS ====================================\n\n')

    # ========================== MANN-WHITNEY =====================================================

    # Definindo os valores RMSE de cada modelo
    rmse_values = {
        'RLM': rlm_rmse,
        'KNR': knr_rmse,
        'SVR': svr_rmse,
        'MLP': mlp_rmse,
        'RF': rf_rmse,
        'GB': gb_rmse
    }

    modelos = list(rmse_values.keys())

    # Realizando o teste de Mann-Whitney para cada par de classificadores
    for i in range(len(modelos)):
        for j in range(i + 1, len(modelos)):
            model1 = modelos[i]
            model2 = modelos[j]
            stat, p_value = mannwhitneyu(rmse_values[model1], rmse_values[model2])

            # Gravando os resultados no arquivo
            f.write(f'Teste de Mann-Whitney: {model1} vs {model2}\n')
            f.write(f'Estatística: {stat}, Valor-p: {p_value}\n')

            if p_value < alpha:
                f.write(f'Diferença estatisticamente significativa entre {model1} e {model2}.\n')
            else:
                f.write(f'Não há diferença estatisticamente significativa entre {model1} e {model2}.\n')

            f.write('---------------------------------------------------\n')
    
#========================== AVALIAÇÃO DOS MODELOS ====================================