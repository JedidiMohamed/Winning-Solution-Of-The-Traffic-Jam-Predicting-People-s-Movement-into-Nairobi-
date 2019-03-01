import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle
import os
from sklearn.model_selection import KFold,RepeatedKFold
import gc
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as  plt
import random
import pandas as pd
import itertools


class Xgboost_model(object):
    def __init__(self,
                 Train_df,
                 Test_df,
                 Target_name,
                 feature_names,
                 feval_metrics=None,
                 Val_df=None,

                 params={},
                 keys=[],
                 col_to_remove=[],
                 nbr_fold=10,
                 nbr_run=1,
                 predict_train=True,
                 verbose_eval=10,
                 early_stopping_rounds=10,
                 maximize=True,
                 num_boost_round=10000,
                 test_size=0.1,
                 path="Model/Xgboost/",
                 model_name="model1.p",
                 load_model=False,
                 nbr_nthread_data=6,
                 do_eval=False,
                 random_state=1994,
                 seed=1994,
                 nbr_RepeatedKFold=5
                 ):

        self.Train_df = Train_df
        self.Test_df = Test_df
        self.Val_df = Val_df
        self.feval_metrics = feval_metrics
        self.Target_name = Target_name
        self.nbr_fold = nbr_fold
        self.nbr_run = nbr_run
        self.keys = keys
        self.params = params
        self.logs = []
        self.col_to_remove = col_to_remove
        self.verbose_eval = verbose_eval
        self.maximize = maximize
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.test_size = test_size
        self.path = path
        self.load_model = load_model
        self.model_name = model_name
        self.predict_train = predict_train
        self.feature_names = feature_names
        self.nbr_nthread_data=nbr_nthread_data
        self.eta = [0.05, 0.1, 0.15]
        self.min_child_weight = [5]
        self.max_depth = [10]
        self.subsample = [0.8]
        self.colsample_bytree = [0.85]
        self.gamma = [0]
        self.scale_pos_weight = [1]
        self.LAMBDA = [1]
        self.alpha = [0]
        self.seed = seed
        self.params["seed"] = random_state
        self.random_state = random_state
        self.best_params_df = pd.DataFrame()
        self.do_eval=do_eval
        self.nbr_RepeatedKFold=nbr_RepeatedKFold
        # self.params["reg_alpha"]=self.seed
        # self.params["reg_lambda"]=self.seed

        if self.load_model:
            self.Load_model(self.path, self.model_name)

        print("Shape of Train", self.Train_df.shape)
        print("Shape of Test", self.Test_df.shape)

    def feval_xgb(self, predection, data_set):
        if self.feval_metrics:
            labels = data_set.get_label()
            metric = self.feval_metrics(labels, predection)
            return [('metric', metric)]
        else:
            return None

    def CreateDMatrix(self, DF, is_test=False):
        return xgb.DMatrix(DF[self.feature_names], label=DF[self.Target_name].values,
                           feature_names=self.feature_names, nthread=self.nbr_nthread_data) if not is_test else xgb.DMatrix(
            DF[self.feature_names],
            feature_names=self.feature_names, nthread=5)

    def fit(self, train=None, val=None):
        print("Create DMatrix ")
        self.D_Train = self.CreateDMatrix(DF=train)
        self.D_Val = self.CreateDMatrix(DF=val)

        del train, val
        gc.collect()
        print("train")
        self.watchlist = [(self.D_Train, 'train'), (self.D_Val, 'valid')]
        if ( self.feval_metrics is not None) & (self.do_eval) :
            self.xgboost = xgb.train(self.params, self.D_Train, self.num_boost_round,
                                     evals=self.watchlist, early_stopping_rounds=self.early_stopping_rounds,
                                     feval=self.feval_xgb, maximize=self.maximize,
                                     verbose_eval=self.verbose_eval)
        else :

            self.xgboost = xgb.train(self.params, self.D_Train, self.num_boost_round,
                                     evals=self.watchlist, early_stopping_rounds=self.early_stopping_rounds,
                                      maximize=self.maximize,
                                     verbose_eval=self.verbose_eval)
        self.Save_model(self.path, self.model_name)
        return self.xgboost

    def Xgboost_fit(self):

        if self.Val_df is not None:
            print("Create DMatrix ")
            self.d_train = self.CreateDMatrix(DF=self.Train_df)
            self.d_valid = self.CreateDMatrix(DF=self.Val_df)
            self.d_test = self.CreateDMatrix(DF=self.Test_df, is_test=True)

        else:
            train, val = train_test_split(self.Train_df, test_size=self.test_size, random_state=self.random_state)
            print("Create DMatrix ")
            self.d_train = self.CreateDMatrix(DF=train)
            self.d_valid = self.CreateDMatrix(DF=val)
            self.d_test = self.CreateDMatrix(DF=self.Test_df, is_test=True)
            del train, val
            gc.collect()

        self.watchlist = [(self.d_train, 'train'), (self.d_valid, 'valid')]
        if( self.feval_metrics is not None) & (self.do_eval) :
            self.xgboost = xgb.train(self.params, self.d_train, self.num_boost_round,
                                 evals=self.watchlist, early_stopping_rounds=self.early_stopping_rounds,
                                 feval=self.feval_xgb, maximize=self.maximize,
                                 verbose_eval=self.verbose_eval)
        else :
            self.xgboost = xgb.train(self.params, self.d_train, self.num_boost_round,
                                     evals=self.watchlist, early_stopping_rounds=self.early_stopping_rounds,
                                     maximize=self.maximize,
                                     verbose_eval=self.verbose_eval)
        self.Save_model(self.path, self.model_name)
        return self.xgboost

    def print_log(self, logs):
        for log in self.logs:
            print(log)

    def update_params(self):
        for par in self.best_params.keys():
            setattr(self, par, [self.best_params[par]])

    def Save_model(self, path, name="model1"):
        if not os.path.isdir(path):
            print('creating checkpoint directory {}'.format(path))
            os.makedirs(path)
        pickle.dump(self.xgboost, open(os.path.join(path, name), "wb"))

    def Load_model(self, name="model1.p"):
        self.xgboost = pickle.load(open(os.path.join(self.path, name), "rb"))
        print("Model loaded")

    def plot_importance(self,figsize=(10,10)):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        xgb.plot_importance(self.xgboost, ax=ax)
        plt.show()

    def update_best_params_DF(self, parms, score):
        seris = pd.Series(parms)
        seris["score"] = score
        self.best_params_df = self.best_params_df.append(seris, ignore_index=True)

    def get_nbr_of_all_iteration(self, Values):
        prod = 1
        for val in Values:
            prod *= len(val)
        return prod

    def predict_df(self, data_set="test"):

        if data_set == "test":
            self.Test_df[self.Target_name] = self.xgboost.predict(self.d_test,
                                                                  ntree_limit=self.xgboost.best_ntree_limit)
            return self.Test_df[[self.Target_name] + self.keys]
        if data_set == "train":
            self.Train_df[self.Target_name + str("pred")] = self.xgboost.predict(self.d_train,
                                                                                 ntree_limit=self.xgboost.best_ntree_limit)
            return self.Train_df[[self.Target_name, self.Target_name + str("pred")] + self.keys]
        if data_set == "val":
            self.Val_df[self.Target_name + str("pred")] = self.xgboost.predict(self.d_valid,
                                                                               ntree_limit=self.xgboost.best_ntree_limit)
            return self.Val_df[[self.Target_name, self.Target_name + str("pred")] + self.keys]

    def Xgboost_Kfold(self):
        i = 0
        self.d_test = self.CreateDMatrix(DF=self.Test_df, is_test=True)
        self.Train_df.reset_index(inplace=True, drop=True)
        kf = KFold(n_splits=self.nbr_fold, shuffle=True,random_state=self.random_state)
        kf.get_n_splits(self.Train_df)
        self.Pred_train = np.zeros((len(self.Train_df)))
        self.Pred_test = np.zeros((len(self.Test_df)))
        List_validation_fold = []
        List_Train_fold = []
        self.logs = []
        for (train_index, val_index) in (kf.split(self.Train_df)):
            i += 1
            self.logs.append("#" * 50 + "fold:" + str(i) + "#" * 50)
            List_train_run = []
            List_validation_run = []
            Train_fold, Val_flod = self.Train_df.loc[train_index, :], self.Train_df.loc[val_index, :]

            for run in range(self.nbr_run):
                clear_output()
                if self.nbr_run >0 :
                    self.params["seed"] = random.randint(1, 10000)
                self.print_log(self.logs)

                self.xgboost = self.fit(Train_fold, Val_flod)

                train_metrics, val_metrics, Val_pred, Test_pred = self.eval_xgboost()
                List_train_run.append(train_metrics)
                List_validation_run.append(val_metrics)

                self.logs.append(
                    "run " + str(run) + " train metrics :" + str(train_metrics) + " val metrics : " + str(val_metrics))
                self.Pred_train[val_index] += Val_pred
                self.Pred_test += Test_pred

                clear_output()

            List_validation_fold.append(np.mean(List_validation_run))
            List_Train_fold.append(np.mean(List_train_run))
            self.logs.append(
                "\n" + "fold-" + str(i) + " train metrics :" + str(np.mean(List_train_run)) + " val metrics : " + str(
                    np.mean(List_validation_run)))

            clear_output()
            self.print_log(self.logs)

        Val_mtercis = np.mean(List_validation_fold)
        Train_mtercis = np.mean(List_Train_fold)
        self.Pred_test /= (self.nbr_fold * self.nbr_run)
        self.Pred_train /=  self.nbr_run

        print("End Training with  train metrics :" + str(Train_mtercis) + " val metrics : " + str(Val_mtercis))

        return self.get_output(self.Pred_test, self.Pred_train)

    def get_output(self, Pred_test, Pred_train):
        self.Train_df[self.Target_name + "pred"] = self.Pred_train
        self.Test_df[self.Target_name] = self.Pred_test
        return self.Train_df[[self.Target_name,self.Target_name + "pred" ]+ self.keys ], self.Test_df[self.keys+[ self.Target_name]]

    def eval_xgboost(self):
        pred = self.xgboost.predict(self.D_Train, ntree_limit=self.xgboost.best_ntree_limit)
        label = self.D_Train.get_label()
        train_metrics = self.feval_metrics(label, pred)

        Val_pred = self.xgboost.predict(self.D_Val, ntree_limit=self.xgboost.best_ntree_limit)
        label = self.D_Val.get_label()
        val_metrics = self.feval_metrics(label, Val_pred)
        del pred, label
        gc.collect()

        Test_pred = self.xgboost.predict(self.d_test, ntree_limit=self.xgboost.best_ntree_limit)

        return train_metrics, val_metrics, Val_pred, Test_pred
    
    def Xgboost_RepeatedKFold(self):
        i = 0
        self.d_test = self.CreateDMatrix(DF=self.Test_df, is_test=True)
        self.Train_df.reset_index(inplace=True, drop=True)
        kf = RepeatedKFold(n_splits=self.nbr_fold,random_state=self.random_state,n_repeats=self.nbr_RepeatedKFold)
        kf.get_n_splits(self.Train_df)
        self.Pred_train = np.zeros((len(self.Train_df)))
        self.Pred_test = np.zeros((len(self.Test_df)))
        List_validation_fold = []
        List_Train_fold = []
        self.logs = []
        for (train_index, val_index) in (kf.split(self.Train_df)):
            i += 1
            self.logs.append("#" * 50 + "fold:" + str(i) + "#" * 50)
            List_train_run = []
            List_validation_run = []
            Train_fold, Val_flod = self.Train_df.loc[train_index, :], self.Train_df.loc[val_index, :]

            for run in range(self.nbr_run):
                clear_output()
                if self.nbr_run >0 :
                    self.params["seed"] = random.randint(1, 10000)
                self.print_log(self.logs)

                self.xgboost = self.fit(Train_fold, Val_flod)

                train_metrics, val_metrics, Val_pred, Test_pred = self.eval_xgboost()
                List_train_run.append(train_metrics)
                List_validation_run.append(val_metrics)

                self.logs.append(
                    "run " + str(run) + " train metrics :" + str(train_metrics) + " val metrics : " + str(val_metrics))
                self.Pred_train[val_index] += Val_pred
                self.Pred_test += Test_pred

                clear_output()

            List_validation_fold.append(np.mean(List_validation_run))
            List_Train_fold.append(np.mean(List_train_run))
            self.logs.append(
                "\n" + "fold-" + str(i) + " train metrics :" + str(np.mean(List_train_run)) + " val metrics : " + str(
                    np.mean(List_validation_run)))

            clear_output()
            self.print_log(self.logs)

        Val_mtercis = np.mean(List_validation_fold)
        Train_mtercis = np.mean(List_Train_fold)
        self.Pred_test /= (self.nbr_fold * self.nbr_run*self.nbr_RepeatedKFold)
        self.Pred_train /=  (self.nbr_run*self.nbr_RepeatedKFold)

        print("End Training with  train metrics :" + str(Train_mtercis) + " val metrics : " + str(Val_mtercis))

        return self.get_output(self.Pred_test, self.Pred_train)
