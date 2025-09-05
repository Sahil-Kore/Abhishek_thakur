# from zipfile import ZipFile

# with ZipFile("../inputs/cat-in-the-dat-ii.zip") as zip_ref:
#     zip_ref.extractall("../inputs")
    
# import os 
# os.remove("../inputs/cat-in-the-dat-ii.zip")

import pandas as pd
df = pd.read_csv("../inputs/train.csv")
df

#creating folds

from sklearn.model_selection import StratifiedKFold

if __name__ =="__main__":
    df= pd.read_csv("../inputs/train.csv")
    df["kfold"]=-1
    
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["target"].values
    kf = StratifiedKFold(n_splits=5)
    
    for fold,(train_idx,val_idx) in enumerate(kf.split(X = df, y=y)):
        df.loc[val_idx,"kfold"] =fold
        
    df.to_csv("../inputs/train_inputs_kfold.csv",index =False)
    
    df = pd.read_csv("../inputs/train_inputs_kfold.csv")
    df["kfold"].value_counts()
    
    for fold in df["kfold"].unique().tolist():
        sub_df= df[df["kfold"]==fold]
        print(f"The value couunt for fold {fold} is ")
        print(sub_df["target"].value_counts())
