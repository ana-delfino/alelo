from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import optuna 
import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,  LabelEncoder
from typing import Optional, List

def apply_scaler(df:pd.DataFrame, numeric_features:List[str]) -> pd.DataFrame:
    scaler = MinMaxScaler()

    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    return df

def apply_label_encoder(df:pd.DataFrame) -> pd.DataFrame:
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'bool']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df     

def format_int_columns_into_categorical(df:pd.DataFrame):
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = df[col].astype('category')  
        

def prediction(model, x_test, y_test, encoder):
    
    df = pd.DataFrame(encoder.inverse_transform(y_test))
    pred_prob = pd.DataFrame(model.predict_proba(x_test))
    pred = pd.DataFrame(encoder.inverse_transform(model.predict(x_test)))
    pred = pd.concat([df,pred_prob,pred],axis=1, ignore_index=True)
    pred.columns = ['Actual','0', '1','Y_pred']
    return(pred) 

def evaluate_in_test_binary(model, x_test, y_test, encoder):
    predictions = prediction(model, x_test, y_test, encoder) 
    accuracy = accuracy_score(predictions['Y_pred'],predictions['Actual']) 
    f1_score_weighted = f1_score(predictions['Actual'],predictions['Y_pred'],average='weighted')
    text_accuracy = "Acurácia: %.2f%%" % (accuracy * 100.0)
    text_f1_score_weighted = "f1 score weighted: %.2f%%" % (f1_score_weighted * 100.0)
    cl_report= classification_report(predictions['Actual'], predictions['Y_pred'],digits=4)
    print('Model Performance in test')
    print(cl_report)
    print(text_accuracy)
    print(text_f1_score_weighted)
    return 

def evaluate_in_test(model, x_test, y_test):
    predictions = prediction(model, x_test, y_test) 
    accuracy = accuracy_score(predictions['Y_pred'],predictions['Actual']) 
    f1_score_weighted = f1_score(predictions['Actual'],predictions['Y_pred'],average='weighted')
    text_accuracy = "Acurácia: %.2f%%" % (accuracy * 100.0)
    text_f1_score_weighted = "f1 score weighted: %.2f%%" % (f1_score_weighted * 100.0)
    cl_report= classification_report(predictions['Actual'], predictions['Y_pred'],digits=4)
    print('Model Performance in test')
    print(cl_report)
    print(text_accuracy)
    print(text_f1_score_weighted)
    return 

def print_best_model_in_train(best_model, score):
    print('Best model in train: \n {} \n f1 score weighted: {:%}'.format(best_model, score))
    return

def plot_feature_importance(model, num_features,feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    ind=[]
    num_features = num_features
    for i in indices:
        ind.append(feature_names[i]) 
    plt.figure(figsize=(20,30))
    plt.title('Feature Importances')

    # only plot the customized number of features
    plt.barh(range(num_features), importances[indices[-num_features:]], color='darkorange', align='center')
    plt.yticks(range(num_features), [feature_names[i] for i in indices[-num_features:]])
    plt.xlabel('Relative Importance')
    return plt.show()


def objective_rfc(trial, x_treino, y_treino):
    """return the f1-score"""

    dict_search_space = {
        'n_estimators': trial.suggest_int('n_estimators', low=500, high=2000, step=10),
        'min_samples_split': trial.suggest_int('min_samples_split', low=2, high=4, step=1),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', low=2, high=4, step=1),
        'max_depth': trial.suggest_int('max_depth', low=10, high=200, step=10),
        'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt']),
        'score_metric':'f1_weighted'
    }

    # random forest classifier object
    rfc = RandomForestClassifier(
          n_estimators = dict_search_space['n_estimators'], 
          min_samples_split = dict_search_space['min_samples_split'],
          min_samples_leaf = dict_search_space['min_samples_leaf'],
          max_depth = dict_search_space['max_depth'],
          max_features = dict_search_space['max_features'],
          random_state = 42)
    
    score =  cross_val_score(estimator = rfc, 
                             X = x_treino, 
                             y = y_treino, 
                             scoring = dict_search_space['score_metric'], #'accuracy',
                             cv=3,
                             n_jobs=-1).mean()
    
    return score

def fit_best_rf_model(best_trial_params,x_treino,y_treino):
    b_op_best_params = best_trial_params 
    b_op_model = RandomForestClassifier(
        n_estimators = b_op_best_params['n_estimators'],
        min_samples_split = b_op_best_params['min_samples_split'],
        min_samples_leaf=b_op_best_params['min_samples_leaf'],
        max_depth=b_op_best_params['max_depth'],
        max_features=b_op_best_params['max_features'],random_state=42).fit(x_treino,y_treino)
    return b_op_model


def objective_etc(trial, x_treino, y_treino):
    """return the f1-score"""

    dict_search_space = {
        'n_estimators': trial.suggest_int('n_estimators', low=100, high=1000, step=10),
        'min_samples_split': trial.suggest_int('min_samples_split', low=2, high=10, step=1),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', low=2, high=4, step=1),
        'max_depth': trial.suggest_int('max_depth', low=10, high=200, step=10),
        'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'score_metric':'f1_weighted'
    }

    # random forest classifier object
    etc = ExtraTreesClassifier(
        n_estimators = dict_search_space['n_estimators'], 
        min_samples_split = dict_search_space['min_samples_split'],
        min_samples_leaf = dict_search_space['min_samples_leaf'],
        max_depth = dict_search_space['max_depth'],
        max_features = dict_search_space['max_features'],
        bootstrap = dict_search_space['bootstrap'],
        random_state = 42)

    score =  cross_val_score(estimator = etc,
                             X = x_treino,
                             y = y_treino,
                             scoring = dict_search_space['score_metric'], #'accuracy',
                             cv=3,
                             n_jobs=-1).mean()

    return score

def fit_best_etc_model(best_trial_params, x_treino, y_treino):
    b_op_best_params = best_trial_params 
    b_op_model = ExtraTreesClassifier(
          n_estimators = b_op_best_params['n_estimators'], 
          min_samples_split = b_op_best_params['min_samples_split'],
          min_samples_leaf = b_op_best_params['min_samples_leaf'],
          max_depth = b_op_best_params['max_depth'],
          max_features = b_op_best_params['max_features'],
          bootstrap = b_op_best_params['bootstrap'],
          random_state = 42).fit(x_treino,y_treino)
    return b_op_model


def objective_gb(trial, x_treino, y_treino):
    #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
    #https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_integration.py
    #https://github.com/dataman-git/codes_for_articles/blob/master/A%20wide%20variety%20of%20models%20for%20multi-class%20classification.ipynb
    #https://medium.com/all-things-ai/in-depth-parameter-tuning-for-gradient-boosting-3363992e9bae
    #https://github.com/benai9916/Handle-imbalanced-data/blob/master/handle-imbalance-data.ipynb
    #https://www.datacareer.de/blog/parameter-tuning-in-gradient-boosting-gbm/
    """return the f1-score"""
    
    dict_search_space = {
        'n_estimators': trial.suggest_int('n_estimators', low=100, high=2000, step=20),
        'learning_rate': trial.suggest_float('learning_rate',0.05,0.5),
        'max_depth': trial.suggest_int('max_depth', low=1, high=8, step=1),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', low=2, high=10, step=1),
        'min_samples_split': trial.suggest_int('min_samples_split', low=2, high=10, step=1),
        'score_metric':'f1_weighted'
    }
   

    
    # random forest classifier object
    gb = GradientBoostingClassifier(n_estimators = dict_search_space['n_estimators'],
                                    learning_rate = dict_search_space['learning_rate'],
                                    max_depth = dict_search_space['max_depth'],
                                    min_samples_leaf = dict_search_space['min_samples_leaf'],
                                    min_samples_split = dict_search_space['min_samples_split'],
                                    random_state=0)
    
    score =  cross_val_score(estimator = gb, 
                             X = x_treino, 
                             y = y_treino, 
                             scoring = dict_search_space['score_metric'], #'accuracy',
                             cv = 3,
                             n_jobs = -1).mean()
    
    return score

def fit_best_GB_model(best_trial_params,x_treino,y_treino):
    b_op_best_params = best_trial_params 
    b_op_model = GradientBoostingClassifier(
        n_estimators = b_op_best_params['n_estimators'],
        learning_rate = b_op_best_params['learning_rate'],
        max_depth = b_op_best_params['max_depth'],
        min_samples_leaf = b_op_best_params['min_samples_leaf'],
        min_samples_split = b_op_best_params['min_samples_split'],
        random_state=0).fit(x_treino,y_treino)
    return b_op_model



