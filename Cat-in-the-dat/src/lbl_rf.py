from cuml import ensemble
from cuml import metrics
from cuml import preprocessing

import cudf
 

def run(fold):
    df = cudf.read_csv("../inputs/train_inputs_kfold.csv")
    
    features = [f for f in df.columns if f not in ["id" , "target" ,"kfold"]]
    
    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")
    
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df[col] = lbl.transform(df[col])
    
    df_train = df[df["kfold"]!= fold].reset_index(drop =True)
    df_val = df[df["kfold"]== fold].reset_index(drop =True)

    X_train = df_train[features]
    X_val = df_val [features]
    
    model = ensemble.RandomForestClassifier()
    model.fit(X_train ,df_train["target"].values)
    
    raw_preds = model.predict_proba(X_val)
    val_preds = cudf.Series(raw_preds.loc[:, 1], index=X_val.index)
    
    auc = metrics.roc_auc_score(df_val["target"].values ,val_preds)
    print(f"Fold {fold} , AUC ={auc}")

if __name__ == "__main__":
    for fold_ in range(0,5):
        run(fold=fold_)