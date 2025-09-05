import joblib 
import pandas as pd 
import train_config
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
import os 
import argparse
import model_dispatcher

def run(fold, model):
    df = pd.read_csv(train_config.INPUT_PATH)  

    df_train  = df[df["fold"]!=fold].reset_index(drop = True)
    df_val =df[df["fold"]==fold].reset_index(drop = True)
    
    X_train = df_train.drop(["BeatsPerMinute","fold"] ,axis=1)
    y_train = df_train["BeatsPerMinute"]
    X_val = df_val.drop(["BeatsPerMinute","fold"] ,axis=1)
    y_val = df_val["BeatsPerMinute"]
    model = model_dispatcher.models[model]
    model.fit(X_train , y_train)
    preds = model.predict(X_val)
    
    rmse = root_mean_squared_error(y_val , preds)
    print(f"Fold {fold } , rmse={rmse}")
    joblib.dump(
        model,
        os.path.join(train_config.MODEL_OUTPUT_DIR, f"dt_{fold}.bin")
    )


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type = int,
    )
    parser.add_argument(
        "--model",
        type = str,
    )
    args = parser.parse_args()
    run(fold=args.fold, model = args.model)