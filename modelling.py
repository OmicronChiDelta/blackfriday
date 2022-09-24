import mlflow
import pandas as pd
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

path_utils = os.path.join(os.getcwd(), "utilities")
if path_utils not in sys.path:
    sys.path.append(path_utils)
    
from utils import Regroup, CreateCatsFilled, FrequencyEncoder, MeanEncoder, DropGarbage, CustomOneHot, median_ordered_boxplots, engineer_frame

# ("mean_encode_cats", MeanEncoder(fields=["gender", 
#                                          "marital_status", 
#                                          "city_category", 
#                                          "product_category_1"])),


#%%
n_folds = 3
seed = 420
test_frac = 0.25


#%%
#fold spec for cv
folder = KFold(n_splits=n_folds,
               shuffle=True,
               random_state=seed)


#%%
data = pd.read_csv("./data/train.csv")
data = engineer_frame(data)
raw_fields = data.columns.to_list()

#siphon off a completely independent test set
test = data.sample(frac=test_frac, replace=False, random_state=seed)
train = data.loc[~data["row_idx"].isin(test["row_idx"].values)].reset_index(drop=True)


#%%
#verify that the eda we did is not working just because of test leakage
plt.close("all")
#_, fig, ax = median_ordered_boxplots(train, field_group="product_category_1", do_sort=True)
_, fig, ax = median_ordered_boxplots(train, field_group="product_id", do_sort=True, minimal_vis=True)



#%%
#consider moving these steps into the pipeline too...
#from EDA... possible values of product_category 1 to recast as "low"... and the remainder as "high"
low_group = [19, 20, 13, 12, 4, 18, 11, 5, 8]
pcat1_groups = {"A":low_group, 
                "B":[c for c in train["product_category_1"].unique() if c not in low_group]}

#features
y_train = train[["purchase"]].reset_index(drop=True)
y_test = test[["purchase"]].reset_index(drop=True)


#%%
#steps to transform data ready for 
pipe = Pipeline([("cats_filled", CreateCatsFilled())
                ,("group_categories", Regroup(field="product_category_1", 
                                              groups=pcat1_groups))
                ,("custom_one_hot", CustomOneHot(fields=["gender",
                                                         "marital_status",
                                                         "city_category",
                                                         "product_category_1"]))
                ,("drop_garbage", DropGarbage(fields=[c.lower() for c in raw_fields]))
                ,("regressor", RandomForestRegressor(n_estimators=50,
                                                      random_state=seed, 
                                                      n_jobs=-1))
                ])

#verify steps work
# op = pipe.fit_transform(train)


#%% 
#search space for cross validation
parameters = {
              #"regressor__criterion": ["squared_error", "poisson"]
              "regressor__max_depth":[None, 10, 50, 100]
              ,"regressor__min_samples_split":[2, 0.2, 0.4, 0.8]
              #,"regressor__min_samples_leaf":[1, 2, 4, 8, 16]
              ,"regressor__max_features":["sqrt", "log2", None]
              }


#%%
#grid search combined with cross validation procedure
gs = GridSearchCV(pipe, 
                  param_grid=parameters, 
                  cv=folder,
                  verbose=4,
                  scoring="neg_root_mean_squared_error")

gs.fit(X=train, y=y_train.values.ravel())


#%%
#report best model
print("\nbest parameters found:")
for ii in gs.best_params_.items():
    print(f"{ii[0]}: {ii[1]}")
print()

#rmse stability?
rmse_train = mean_squared_error(y_true=y_train.values.ravel(), y_pred=gs.predict(train))**0.5
rmse_test = mean_squared_error(y_true=y_test.values.ravel(), y_pred=gs.predict(test))**0.5
print(f"\ntraining rmse: {rmse_train}")
print(f"testing rmse: {rmse_test}")
print(f"test/train performance ratio: {rmse_test/rmse_train}")


#%%
#prediction on challenge test set
heldout = pd.read_csv("./data/test.csv")
heldout = engineer_frame(heldout)

#included as dummy for compatibility
heldout["purchase"] = None

heldout["Purchase"] = gs.predict(heldout)
heldout = heldout[["Purchase", "user_id", "product_id"]].reset_index(drop=True)
heldout.rename({"user_id":"User_ID", "product_id":"Product_ID"}, axis=1, inplace=True)