import numpy as np
import pandas as pd

import os

import seaborn as sns
import pylab as plt
import plots

from sklearn import preprocessing, metrics,model_selection,linear_model,svm,ensemble
from sklearn.pipeline import Pipeline
from tsfresh import select_features

def run_classification(data,features,covs,target='pd',external_data=[],save='/scratch/c.c21013066/data/ppmi/analyses/classifyPDHC'):
    if len(features)==0:
        save = f'{save}/baseline'
    # Define a pipeline to search for the best classifier regularization.
    # Define a Standard Scaler to normalize inputs
    cvouter = model_selection.StratifiedKFold(n_splits=5,random_state=12,shuffle=True)
    preds = np.hstack([features,covs])
    coefs_df = pd.DataFrame(columns=['coef'],index=pd.MultiIndex.from_product([np.arange(5),preds],names=['cv','predictor']))
    test_scores_df = pd.DataFrame(index=np.arange(5),columns=['AUPRC'])
    fitted_params_df = pd.DataFrame(index=np.arange(5),columns=['C','L1_ratio'])
    
    for cvo,(train_index, test_index) in enumerate(cvouter.split(data[preds], data[target])):
        X_train, X_test = data.loc[train_index,preds], data.loc[test_index,preds]
        y_train, y_test = data.loc[train_index,target], data.loc[test_index,target]
        predictors = select_features(X_train[features],y_train)
        predictors = predictors.columns
        scaler = preprocessing.StandardScaler()
        cv = model_selection.StratifiedKFold(n_splits=5,random_state=123,shuffle=True)
        # set the tolerance to a large value to make the example faster
        logistic = linear_model.LogisticRegression(max_iter=10000, tol=0.1,random_state=4,penalty='elasticnet',solver='saga')
        pipe = Pipeline(steps=[("scaler", scaler), ("logistic", logistic)])

        param_grid = {
            "logistic__C": np.logspace(1, 4, 5),
            "logistic__l1_ratio": np.linspace(0, 1, 5),
        }
        search = model_selection.GridSearchCV(pipe, param_grid, n_jobs=-1,cv=cv,scoring='average_precision')
        search.fit(X_train[np.hstack([predictors,covs])],y_train)

        # For each number of components, find the best classifier results
        coefs_df.loc[(cvo,np.hstack([predictors,covs])),:] = search.best_estimator_['logistic'].coef_.T
        test_scores_df.loc[cvo,'AUPRC'] = search.best_score_
        fitted_params_df.loc[cvo,['C']] = search.best_params_['logistic__C']
        fitted_params_df.loc[cvo,['L1_ratio']] = search.best_params_['logistic__l1_ratio']
        #if cvo==0:
        #    plots.plot_performance(search.cv_results_,save=[])
        if len(external_data)>0:
            external_test = test_external(search,external_data,np.hstack([predictors,covs]),cvo=cvo,save=save)
 
    if save:
        if not os.path.exists(save):
            os.makedirs(save)
        # save the dataframes to the directory as CSV files
        coefs_df.to_csv(os.path.join(save, 'coefs.csv'))
        test_scores_df.to_csv(os.path.join(save, 'test_scores.csv'))
        fitted_params_df.to_csv(os.path.join(save, 'fitted_params.csv'))
        try:
            plots.plot_coefs(coefs_df,save=save)
        except:
            print('plotting failed')
    
    return coefs_df, search, external_test


def run_classification_models(data,features,covs,target='pd',external_data=[],saveing='/scratch/c.c21013066/data/ppmi/analyses/classifyPDHC'):
            
    logistic = linear_model.LogisticRegression(max_iter=10000, tol=0.001,random_state=4,penalty='elasticnet',solver='saga')
    logistic_param_grid = {
            "logreg__C": np.logspace(1, 4, 5),
            "logreg__l1_ratio": np.linspace(0, 1, 5),
        }
    poly_svm = svm.SVC(kernel='poly',gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=4)
    poly_svm_param_grid = {
            "poly_svm__C": np.logspace(1, 4, 5),
            "poly_svm__degree": np.linspace(3, 5, 3).astype(int),
        }
    rbf_svm = svm.SVC(kernel='rbf',gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=4)
    rbf_svm_param_grid = {
            "rbf_svm__C": np.logspace(1, 4, 5),
        }
    rf = ensemble.RandomForestClassifier(criterion='gini', min_samples_split=2, min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=4, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
    rf_param_grid = {
            "rf__n_estimators": np.linspace(50, 200, 3).astype(int),
            "rf__max_depth": np.linspace(15,100,3).astype(int),
        }
    xgb = ensemble.GradientBoostingClassifier(loss='log_loss', subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0, init=None, random_state=4, max_features=None, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.001)
    xgb_param_grid = {
            "xgb__n_estimators": np.linspace(50, 200, 3),
            "xgb__learning_rate": np.logspace(-3, -1, 3),
            "xgb__max_depth": np.linspace(15,100,3).astype(int),
        }
    names = ['logreg','poly_svm','rbf_svm','rf','xgboost']
    models = [logistic,poly_svm,rbf_svm,rf,xgb]
    param_grids = [logistic_param_grid,poly_svm_param_grid,rbf_svm_param_grid,rf_param_grid,xgb_param_grid]
    for name,model,param_grid in zip(names,models,param_grids):
        save_ = f'{saveing}/{name}/'
        if not os.path.exists(save_):
            os.makedirs(save_)
        # Define a pipeline to search for the best classifier regularization.
        # Define a Standard Scaler to normalize inputs
        cvouter = model_selection.StratifiedKFold(n_splits=5,random_state=12,shuffle=True)
        preds = np.hstack([features,covs])
        #coefs_df = pd.DataFrame(columns=['coef'],index=pd.MultiIndex.from_product([np.arange(5),preds],names=['cv','predictor']))
        test_scores_df = pd.DataFrame(index=np.arange(5),columns=['AUPRC'])
        #fitted_params_df = pd.DataFrame(index=np.arange(5),columns=['C','L1_ratio'])

        for cvo,(train_index, test_index) in enumerate(cvouter.split(data[preds], data[target])):
            X_train, X_test = data.loc[train_index,preds], data.loc[test_index,preds]
            y_train, y_test = data.loc[train_index,target], data.loc[test_index,target]
            predictors = select_features(X_train[features],y_train)
            predictors = predictors.columns
            scaler = preprocessing.StandardScaler()
            cv = model_selection.StratifiedKFold(n_splits=5,random_state=123,shuffle=True)
            pipe = Pipeline(steps=[("scaler", scaler), (name, model)])

            search = model_selection.GridSearchCV(pipe, param_grid, n_jobs=-1,cv=cv,scoring='average_precision')
            search.fit(X_train[np.hstack([predictors,covs])],y_train)

            # For each number of components, find the best classifier results
            #try:
            #    coefs_df.loc[(cvo,np.hstack([predictors,covs])),:] = search.best_estimator_[name].coef_.T
            #except:
            #    pass
            test_scores_df.loc[cvo,'AUPRC'] = search.best_score_
            #fitted_params_df.loc[cvo,['C']] = search.best_params_['logistic__C']
            #fitted_params_df.loc[cvo,['L1_ratio']] = search.best_params_['logistic__l1_ratio']
            #if cvo==0:
            #    plots.plot_performance(search.cv_results_,save=[])
            if len(external_data)>0:
                external_test = test_external(search,external_data,np.hstack([predictors,covs]),cvo=cvo,savepath=save_)

        if save_:
            if not os.path.exists(save_):
                os.makedirs(save_)
            # save the dataframes to the directory as CSV files
            #coefs_df.to_csv(os.path.join(save_, 'coefs.csv'))
            test_scores_df.to_csv(os.path.join(save_, 'test_scores.csv'))
            #fitted_params_df.to_csv(os.path.join(save_, 'fitted_params.csv'))
            #try:
            #    plots.plot_coefs(coefs_df,save=save_)
            #except:
            #    print('plotting failed')

def test_external(search,data,preds,cvo='',savepath=[]):
    # check external test to which class
    data['pred'] = search.predict(data[preds])
    data['pred_proba'] = search.predict_proba(data[preds])[:,1]
    try:
        plots.plot_predproba_diag(data,save=savepath)
    except:
        print('plotting failed')
    if savepath:
        data.to_csv(f'{savepath}predictions{cvo}.csv')
    return data