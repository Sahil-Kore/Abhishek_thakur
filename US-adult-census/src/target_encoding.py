import copy 
from sklearn import model_selection
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd 

def mean_target_encoding(data):
    df = copy.deepcopy(data)
    
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
        ]
            
    target_mapping = {
        ' <=50K':0,
        ' >50K':1
    }
    
    df[" <=50K"] = df[" <=50K"] .map(target_mapping)
    features = [f for f in df.columns if f not in ('kfold' , ' <=50K') and f not in num_cols]

    for col in features :
        if col not in num_cols:
            df[col] =df[col].astype(str).fillna("NONE")    
    
    for col in features :
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df[col] = lbl.transform(df[col])
    
    encoded_dfs = []
    
    for fold in range (5):
        df_train  = df[df['kfold']!=fold].reset_index(drop = True)
        df_val = df[df["kfold"]== fold ].reset_index(drop = True)
        
        for column in features :
            mapping_dict = dict (
                df_train.groupby(column)[" <=50K"].mean()
                                 ) 
            df_val[column + "_enc"] = df_val[column].map(mapping_dict)
        
        encoded_dfs.append(df_val)
    
    encoded_dfs = pd.concat(encoded_dfs , axis =0)
    return encoded_dfs

def run(df , fold):
    df_train = df[df["kfold"] != fold].reset_index(drop =True)
    df_val = df[df["kfold"] == fold].reset_index(drop =True)

    features = [f for f in df.columns if f not in ( "kfold" , " <=50K")]
    X_train = df_train[features].values
    
    X_val = df_val[features].values
    
    model = xgb.XGBClassifier(n_jobs =-1 ,max_depth=7)
    
    model.fit(X_train , df_train[" <=50K"])    
    val_preds = model.predict_proba(X_val)[:,1]

    auc = metrics.roc_auc_score(df_val[" <=50K"].values , val_preds)
    
    print(f"Fold {fold}  ,auc {auc}")


if __name__=="__main__":
    df = pd.read_csv("../inputs/adult.data_kfold.csv")
    df = mean_target_encoding(df)
    for fold in range(5):
        run(df , fold)