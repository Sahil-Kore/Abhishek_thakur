from sklearn import tree 
from sklearn import ensemble
models ={
    "decision_tree_squared_error":tree.DecisionTreeRegressor(
        criterion="squared_error"
    ),
    "decision_tree_poisson":tree.DecisionTreeRegressor(
        criterion="poisson"
    ),
    "rf":ensemble.RandomForestRegressor(),
    "ef": ensemble.ExtraTreesRegressor()
}
