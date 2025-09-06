from cuml import linear_model
import cudf
from cuml import metrics
from cuml import preprocessing

def run (fold ):
    df = cudf.read_csv("../inputs/adult.data_kfold.csv")
    num_cols = [col for col in df.select_dtypes(include ="number").columns.tolist() if col !="kfold" ]
    df = df.drop(num_cols , axis=1)
    target_mapping = {
        " <=50K":0,
        " >50K":1
    }
    
    df[" <=50K"] = df[" <=50K"].map(target_mapping)
    
    features = [f for f in df.columns if f not in ["kfold"," <=50K"]]
    
    for col in features :
        df[col] = df[col].astype(str).fillna("NONE")
    
    df_train = df[df["kfold"] != fold].reset_index(drop =True)
    df_val = df[df["kfold"] == fold].reset_index(drop = True)
    ohe  = preprocessing.OneHotEncoder()
    
    full_data = cudf.concat(
        [df_train[features] , df_val[features]],
        axis=0
    )
    
    ohe.fit(full_data[features])
    x_train = ohe.transform(df_train[features])
    x_val = ohe.transform (df_val[features])
    
    model = linear_model.LogisticRegression()
    model.fit(x_train , df_train[" <=50K"].values)
    
    raw_preds = model.predict_proba(x_val)
    val_preds = cudf.Series(raw_preds[:, 1], index=df_val.index)
    auc = metrics.roc_auc_score(df_val [" <=50K"] , val_preds)
    print(f"Fold {fold}, AUC={auc}")

if __name__ =="__main__":
    for fold in range (5):
        run(fold)
    
    
    