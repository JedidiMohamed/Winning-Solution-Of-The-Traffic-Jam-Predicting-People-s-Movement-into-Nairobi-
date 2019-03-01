import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle
import os
from sklearn.model_selection import KFold , StratifiedKFold
import gc
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as  plt
import random
import pandas as pd
import itertools


class lightgbm_model(object):
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
                 path="Model/lightgbm/",
                 model_name="model1.p",
                 load_model=False,
                 nbr_nthread_data=6,
                 eval_kfold=False,
                 random_state=1994,
                 seed=1994
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
        self.eval_kfold=eval_kfold
        self.seed = seed
        self.params["seed"] = self.seed
        self.random_state = random_state



        if self.load_model:
            self.Load_model(self.path, self.model_name)

        print("Shape of Train", self.Train_df.shape)
        print("Shape of Test", self.Test_df.shape)

    def feval_lgb(self, predection, data_set):
        if self.feval_metrics:
            labels = data_set.get_label()
            metric = self.feval_metrics(labels, predection)
            return [('metric', metric,self.maximize)]
        else:
            return None

    def CreateDMatrix(self, DF, is_test=False):
        return lgb.Dataset(DF[self.feature_names], label=DF[self.Target_name].values,
                           feature_name =self.feature_names)

    def fit(self, train=None, val=None):
        print("Create DMatrix ")
        self.D_Train = self.CreateDMatrix(DF=train)
        self.D_Val = self.CreateDMatrix(DF=val)

        del train, val
        gc.collect()
        print("train")
        self.valid_sets = [self.D_Train, self.D_Val]
        self.valid_names = ['train', 'valid']
        if (self.feval_metrics is not None) & self.eval_kfold :
            self.lightgbm = lgb.train(self.params,
                                      self.D_Train,
                                      self.num_boost_round,
                                      valid_sets=self.valid_sets,
                                      valid_names=self.valid_names,
                                
                                      early_stopping_rounds=self.early_stopping_rounds,
                                      feval=self.feval_lgb,
                                      verbose_eval=self.verbose_eval)
        else :

            self.lightgbm = lgb.train(self.params, self.D_Train, self.num_boost_round,
                                      valid_sets=self.valid_sets,
                                      valid_names=self.valid_names,
                                   
                                      early_stopping_rounds=self.early_stopping_rounds,
                                      verbose_eval=self.verbose_eval)
        self.Save_model(self.path, self.model_name)
        return self.lightgbm

    def lightgbm_fit(self):

        if self.Val_df is not None:
            print("Create DMatrix ")
            self.d_train = self.CreateDMatrix(DF=self.Train_df)
            self.d_valid = self.CreateDMatrix(DF=self.Val_df)

        else:
            train, val = train_test_split(self.Train_df, test_size=self.test_size, random_state=self.random_state)
            print("Create DMatrix ")
            self.d_train = self.CreateDMatrix(DF=train)
            self.d_valid = self.CreateDMatrix(DF=val)
            del train, val
            gc.collect()

        self.valid_sets = [self.d_train, self.d_valid]
        self.valid_names = ['train', 'valid']
        if (self.feval_metrics is not None) & self.eval_kfold :
            self.lightgbm = lgb.train(self.params, self.d_train, self.num_boost_round,
                                      valid_sets=self.valid_sets,
                                      valid_names=self.valid_names,
                                      early_stopping_rounds=self.early_stopping_rounds,
                                     feval=self.feval_lgb,
                                     verbose_eval=self.verbose_eval)
        else :
            self.lightgbm = lgb.train(self.params, self.d_train, self.num_boost_round,
                                      valid_sets=self.valid_sets,
                                      valid_names=self.valid_names,
                                      early_stopping_rounds=self.early_stopping_rounds,
                                      verbose_eval=self.verbose_eval

                                      )
        self.Save_model(self.path, self.model_name)
        return self.lightgbm

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
        pickle.dump(self.lightgbm, open(os.path.join(path, name), "wb"))

    def Load_model(self, name="model1.p"):
        self.lightgbm = pickle.load(open(os.path.join(self.path, name), "rb"))
        print("Model loaded")

    def plot_importance(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        lgb.plot_importance(self.lightgbm, ax=ax)
        plt.show()
    def plot_tress(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        lgb.plot_tree(self.lgbm, ax=ax)
        plt.show()
    def plot_metric(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        lgb.plot_metric(self.lgbm, metric="metric",ax=ax)
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

    def predict_df(self, data_set="test",ntree_limit=None):
        if ntree_limit is None :
            ntree_limit=self.lightgbm.best_iteration
        print(ntree_limit)
        if data_set == "test":
            self.Test_df[self.Target_name] = self.lightgbm.predict(self.Test_df[self.feature_names],
                                                                  ntree_limit=ntree_limit)
            return self.Test_df[  self.keys+[self.Target_name]]
        if data_set == "train":
            self.Train_df[self.Target_name + str("pred")] = self.lightgbm.predict(self.Train_df[self.feature_names],
                                                                                 ntree_limit=ntree_limit)
            return self.Train_df[self.keys+[self.Target_name, self.Target_name + str("pred")]]
        if data_set == "val":
            self.Val_df[self.Target_name + str("pred")] = self.lightgbm.predict(self.Val_df[self.feature_names],
                                                                               ntree_limit=ntree_limit)
            return self.Val_df[self.keys+[self.Target_name, self.Target_name + str("pred")] ]

    def lightgbm_Kfold(self):
        i = 0

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
            self.Train_fold, self.Val_flod = self.Train_df.loc[train_index, :], self.Train_df.loc[val_index, :]

            for run in range(self.nbr_run):
                clear_output()
                if self.nbr_run >0 :
                    self.params["seed"] = random.randint(1, 10000)
              
                self.print_log(self.logs)

                self.lightgbm = self.fit(self.Train_fold, self.Val_flod)

                train_metrics, val_metrics, Val_pred, self.Test_pred = self.eval_lightgbm()
                List_train_run.append(train_metrics)
                List_validation_run.append(val_metrics)

                self.logs.append(
                    "run " + str(run) + " train metrics :" + str(train_metrics) + " val metrics : " + str(val_metrics))
                self.Pred_train[val_index] += Val_pred
                self.Pred_test += self.Test_pred

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
        self.Pred_train /= self.nbr_run

        print("End Training with  train metrics :" + str(Train_mtercis) + " val metrics : " + str(Val_mtercis))

        return self.get_outpust(self.Pred_test, self.Pred_train)
    
    def lightgbm_StratifiedKFold(self):
        i = 0

        self.Train_df.reset_index(inplace=True, drop=True)
        kf = StratifiedKFold(n_splits=self.nbr_fold, shuffle=True,random_state=self.random_state)
        kf.get_n_splits(self.Train_df)
        self.Pred_train = np.zeros((len(self.Train_df)))
        self.Pred_test = np.zeros((len(self.Test_df)))
        List_validation_fold = []
        List_Train_fold = []
        self.logs = []
        for (train_index, val_index) in (kf.split(self.Train_df,self.Train_df[self.Target_name])):
            i += 1
            self.logs.append("#" * 50 + "fold:" + str(i) + "#" * 50)
            List_train_run = []
            List_validation_run = []
            self.Train_fold, self.Val_flod = self.Train_df.loc[train_index, :], self.Train_df.loc[val_index, :]

            for run in range(self.nbr_run):
                clear_output()
                self.params["seed"] = random.randint(1, 10000)
                self.print_log(self.logs)

                self.lightgbm = self.fit(self.Train_fold, self.Val_flod)

                train_metrics, val_metrics, Val_pred, self.Test_pred = self.eval_lightgbm()
                List_train_run.append(train_metrics)
                List_validation_run.append(val_metrics)

                self.logs.append(
                    "run " + str(run) + " train metrics :" + str(train_metrics) + " val metrics : " + str(val_metrics))
                self.Pred_train[val_index] += Val_pred
                self.Pred_test += self.Test_pred

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
        self.Pred_train /= self.nbr_run

        print("End Training with  train metrics :" + str(Train_mtercis) + " val metrics : " + str(Val_mtercis))

        return self.get_outpust(self.Pred_test, self.Pred_train)

    def get_outpust(self, Pred_test, Pred_train):
        self.Train_df[self.Target_name + "pred"] = Pred_train
        self.Test_df[self.Target_name] = Pred_test
        return self.Train_df[[self.Target_name,self.Target_name + "pred" ]+ self.keys ], self.Test_df[self.keys+[ self.Target_name]]

    def eval_lightgbm(self):
        pred = self.lightgbm.predict(self.Train_fold[self.feature_names], ntree_limit=self.lightgbm.best_iteration)
        label = self.D_Train.get_label()
        train_metrics = self.feval_metrics(label, pred)

        Val_pred = self.lightgbm.predict(self.Val_flod[self.feature_names], ntree_limit=self.lightgbm.best_iteration)
        label = self.D_Val.get_label()
        val_metrics = self.feval_metrics(label, Val_pred)
        del pred, label
        gc.collect()

        Test_pred = self.lightgbm.predict(self.Test_df[self.feature_names], ntree_limit=self.lightgbm.best_iteration)

        return train_metrics, val_metrics, Val_pred, Test_pred
    
    def lightgbm_Kfold_agg(self,var_to_agg,vars_be_agg,func,fillnan=True):
        i = 0

        self.Train_df.reset_index(inplace=True, drop=True)
        kf = KFold(n_splits=self.nbr_fold, shuffle=True,random_state =self.random_state)
        kf.get_n_splits(self.Train_df)
        self.Pred_train = np.zeros((len(self.Train_df)))
        self.Pred_test = np.zeros((len(self.Test_df)))
        List_validation_fold = []
        List_Train_fold = []
        self.logs = []
        self.original_features=self.feature_names.copy()
        for (train_index, val_index) in (kf.split(self.Train_df)):
            i += 1
            self.logs.append("#" * 50 + "fold:" + str(i) + "#" * 50)
            List_train_run = []
            List_validation_run = []
            self.Test_fold=self.Test_df.copy()
            print("Split train ")
            self.Train_fold, self.Val_flod = self.Train_df.loc[train_index, :], self.Train_df.loc[val_index, :]
            print("create agg  features ")
            self.feature_names=self.original_features.copy()
            self.Train_fold, self.Val_flod,self.Test_fold=self.agg_funcation(self.Train_fold,self.Val_flod,
                                                                             self.Test_fold,var_to_agg,
                                                                             vars_be_agg,func,fillnan
                                                                            )
            for run in range(self.nbr_run):
                clear_output()
                if self.nbr_run >0 :
                    self.params["seed"] = random.randint(1, 10000)
              
                self.print_log(self.logs)

                self.lightgbm = self.fit(self.Train_fold, self.Val_flod)

                train_metrics, val_metrics, Val_pred, self.Test_pred = self.eval_lightgbm_bagging(self.Test_fold)
                List_train_run.append(train_metrics)
                List_validation_run.append(val_metrics)

                self.logs.append(
                    "run " + str(run) + " train metrics :" + str(train_metrics) + " val metrics : " + str(val_metrics))
                self.Pred_train[val_index] += Val_pred
                self.Pred_test += self.Test_pred

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
        self.Pred_train /= self.nbr_run

        print("End Training with  train metrics :" + str(Train_mtercis) + " val metrics : " + str(Val_mtercis))

        return self.get_outpust(self.Pred_test, self.Pred_train)
    
    def eval_lightgbm_bagging(self,test):
        pred = self.lightgbm.predict(self.Train_fold[self.feature_names], ntree_limit=self.lightgbm.best_iteration)
        label = self.D_Train.get_label()
        train_metrics = self.feval_metrics(label, pred)

        Val_pred = self.lightgbm.predict(self.Val_flod[self.feature_names], ntree_limit=self.lightgbm.best_iteration)
        label = self.D_Val.get_label()
        val_metrics = self.feval_metrics(label, Val_pred)
        del pred, label
        gc.collect()

        Test_pred = self.lightgbm.predict(test[self.feature_names], ntree_limit=self.lightgbm.best_iteration)

        return train_metrics, val_metrics, Val_pred, Test_pred
    def agg_funcation(self,Train_fold,Val_flod,Test_fold,var_to_agg,
                      vars_be_agg,func,fillnan):
        for var in var_to_agg : 
            print(var)
            agg=Train_fold.groupby(var)[vars_be_agg].agg(func)
            if isinstance(var, list):
                agg.columns = pd.Index([vars_be_agg+"_by_" + "_".join(var) + "_" + str(e) for e in agg.columns.tolist()])
            else:
                agg.columns = pd.Index([vars_be_agg+"_by_" + var + "_" + str(e) for e in agg.columns.tolist()]) 

            Train_fold=Train_fold.merge(agg,on=var,how="left")
            Val_flod=Val_flod.merge(agg,on=var,how="left")
            self.feature_names.extend(agg.columns.tolist())
            print( self.feature_names)
            if fillnan : 
                for col in agg.columns  :  
                    Val_flod[col].fillna(agg[col].mean(),inplace=True)
            del agg 
            gc.collect()
            agg=pd.concat([Train_fold,Val_flod]).groupby(var)[vars_be_agg].agg(func)
            if isinstance(var, list):
                agg.columns = pd.Index([vars_be_agg+"_by_" + "_".join(var) + "_" + str(e) for e in agg.columns.tolist()])
            else:
                agg.columns = pd.Index([vars_be_agg+"_by_" + var + "_" + str(e) for e in agg.columns.tolist()]) 

            Test_fold=Test_fold.merge(agg,on=var,how="left")
            if fillnan : 
                for col in agg.columns  :  
                    Test_fold[col].fillna(agg[col].mean(),inplace=True)
            del agg 
            gc.collect()
        return Train_fold ,Val_flod ,Test_fold
            


            