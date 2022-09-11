import mlflow
import pandas as pd
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error

path_utils = os.path.join(os.getcwd(), "utilities")
if path_utils not in sys.path:
    sys.path.append(path_utils)
    
from utils import Regroup, CreateCatsFilled, FrequencyEncoder, MeanEncoder, DropGarbage, CustomOneHot

# ("mean_encode_cats", MeanEncoder(fields=["gender", 
#                                          "marital_status", 
#                                          "city_category", 
#                                          "product_category_1"])),


#%%
n_folds = 3
seed = 420


#%%
#fold spec for cv
folder = KFold(n_splits=n_folds,
               shuffle=True,
               random_state=seed)


#%%
train = pd.read_csv("./data/train.csv")
train.columns = [c.lower() for c in train.columns]

#consider moving these steps into the pipeline too...
#from EDA... possible values of product_category 1 to recast as "low"... and the remainder as "high"
low_group = [19, 20, 13, 12, 4, 18, 11, 5, 8]
pcat1_groups = {"A":low_group, 
                "B":[c for c in train["product_category_1"].unique() if c not in low_group]}

X = train[["gender", 
           "marital_status",
           "city_category",
           "product_category_1",
           "product_category_2",
           "product_category_3",
           "purchase"]].reset_index(drop=True)

y = train[["purchase"]].reset_index(drop=True)


#%%
#steps to transform data ready for 
pipe = Pipeline([("cats_filled", CreateCatsFilled())
                 ,("group_categories", Regroup(field="product_category_1", 
                                              groups=pcat1_groups))
                 ,("custom_one_hot", CustomOneHot(fields=["gender",
                                                         "marital_status",
                                                         "city_category",
                                                         "product_category_1"]))
                 ,("drop_garbage", DropGarbage(fields=["product_category_2", 
                                                      "product_category_3",
                                                      "gender",
                                                      "marital_status",
                                                      "city_category",
                                                      "product_category_1"]))
                 ,("regressor", RandomForestRegressor(n_estimators=50,
                                                      random_state=seed, 
                                                      n_jobs=-1))])
#op = pipe.fit_transform(X)


#%% 
#search space for cross validation
parameters = {
              "regressor__criterion": ["squared_error", "poisson"]
              ,"regressor__max_depth":[None, 10, 50, 100]
              #,"regressor__min_samples_split":[2, 0.2, 0.4, 0.6, 0.8]
              #,"regressor__min_samples_leaf":[1, 2, 4, 8, 16]
              #,"regressor__max_features":["sqrt", "log2", None]
              }


#%%
#grid search combined with cross validation procedure
gs = GridSearchCV(pipe, 
                  param_grid=parameters, 
                  cv=folder,
                  verbose=4,
                  scoring="neg_root_mean_squared_error")

gs.fit(X=X, y=y.values.ravel())


#%%
#rmse evaluation
rmse = mean_squared_error(y_true=y.values.ravel(), y_pred=gs.predict(X))**0.5
