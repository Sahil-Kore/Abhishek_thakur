import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run (fold ):
    df= pd.read_csv("../inputs/train_inputs_kfold.csv")
    features = [f for f in df.columns if f not in ["id","target","kfold"]]

    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")
    
    df_train = df[df["kfold"]!= fold].reset_index(drop =True)
    df_val = df[df["kfold"] == fold].reset_index(drop = True)
    
    ohe = preprocessing.OneHotEncoder()
    
    full_data = pd.concat([df_train[features] , df_val[features]]
                        ,axis=0)
    ohe.fit(full_data[features])
    
    X_train = ohe.transform(df_train[features])
    X_val = ohe.transform(df_val[features])
    
    model = linear_model.LogisticRegression()
    
    model.fit(X_train , df_train["target"])

    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    val_preds = model.predict_proba(X_val)[:,1]
    
    auc = metrics.roc_auc_score(df_val["target"].values , val_preds)
    print(f"Fold = {fold}, AUC = {auc}")
    
if __name__ == "__main__":
    for fold_ in range(0,5):
        run(fold=fold_)