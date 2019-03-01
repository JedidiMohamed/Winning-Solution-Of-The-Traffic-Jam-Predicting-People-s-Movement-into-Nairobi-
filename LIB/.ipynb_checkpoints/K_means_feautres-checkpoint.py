import pandas as pd 
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import gc 
class K_means_feautres(object):
    def __init__(self,
               n_clusters=10,
               batch_size=2048,
               Train_df=None,
               Test_df=None,
               Val_df=None,
              Class_Prediction=False,
              features_names=[],
               ids=[],
               range_of_clustres=[],
                nor=False):
        self.n_clusters=n_clusters
        self.batch_size=batch_size
        self.Class_Prediction=Class_Prediction
        self.features_names=features_names
        self.range_of_clustres=range_of_clustres
        self.ids=ids
        self.Data=Train_df.copy()
        self.nor=nor
        if Val_df is not None :
            self.Data=pd.concat([self.Data,Val_df])
        if Test_df is not None :
            self.Data=pd.concat([self.Data,Test_df])
        del Train_df ,Test_df,Val_df
        if self.nor==True :
            self.data=normalize(self.Data[self.features_names].values)
        else :
            self.data=self.Data[self.features_names].values
        gc.collect()
    def one_kmeans(self):
        self.kmeans=MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size,verbose =True).fit(self.data)
    def transform(self):
        output=self.Data[self.ids]
        if  self.Class_Prediction :  
            
            output["kmeans_cluster_"+str(self.n_clusters)]=self.kmeans.predict(self.data)
            return output
        else :  
            columns=[ "Kmeans_features_"+str(i) for i in range(self.n_clusters)]
            data=self.kmeans.transform(self.data)
#             return pd.concat([pd.DataFrame(data=data,columns=columns),self.Data[self.ids]],axis=1).reset_index()
            output[columns]=pd.DataFrame(data=data,columns=columns,index=output.index)
            return output
    def multi_Kmeans(self, range_of_clustres):
        self.Class_Prediction=True
        data=[]
        output=self.Data[self.ids]
        for i in range_of_clustres :
            print(i)
            self.n_clusters=i
            self.one_kmeans()
            output["kmeans_cluster_"+str(self.n_clusters)]=self.kmeans.predict(self.data)
         
        return output
        
