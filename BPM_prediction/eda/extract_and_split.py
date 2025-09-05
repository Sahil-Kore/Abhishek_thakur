from zipfile import ZipFile
with ZipFile("../Inputs/playground-series-s5e9.zip") as zip_ref:
    zip_ref.extractall("../Inputs")
    
import os 
os.remove("../Inputs/playground-series-s5e9.zip")

import pandas as pd 
df=pd.read_csv("../Inputs/train.csv")
df.corr()["BeatsPerMinute"]

df=df.drop(["id"],axis=1)

df["fold"]=-1

df= df.sample(frac=1).reset_index(drop=True)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)


for fold,(train_idx,val_idx) in enumerate(kf.split(df)):
    print(fold)
    df.loc[val_idx,"fold"]= fold
    
df

df.to_csv("../Inputs/kfold_train.csv",index=False)