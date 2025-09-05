# from zipfile import ZipFile

# with ZipFile("../inputs/archive (4).zip") as zip_ref:
#     zip_ref.extractall("../inputs")
# import os 
# os.remove("../inputs/archive (4).zip")

from sklearn import model_selection
import pandas as pd 

df = pd.read_csv("../inputs/adult.data")
df

df["kfold"] =-1
df = df.sample(frac=1).reset_index(drop= True)
kf = model_selection.StratifiedKFold()
targets = df[" <=50K"]


for fold ,(train_idx , val_idx )in enumerate(kf.split(df ,targets)):
    df.loc[val_idx, "kfold"] = fold 
    
df.to_csv("../inputs/adult.data_kfold.csv",index= False)