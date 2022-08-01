import mlflow
import pandas as pd
import sys
import os
from sklearn.pipeline import Pipeline

path_utils = os.path.join(os.getcwd(), "utilities")
if path_utils not in sys.path:
    sys.path.append(path_utils)
    
from utils import Regroup, CreateCatsFilled, FrequencyEncoder, MeanEncoder, DropGarbage


#%%
raw = pd.read_csv("./data/train.csv")
raw.columns = [c.lower() for c in raw.columns]

low_group = [19, 20, 13, 12, 4, 18, 11, 5, 8]
pcat1_groups = {"A":low_group, 
                "B":[c for c in raw["product_category_1"].unique() if c not in low_group]}

X = raw[["gender", 
         "marital_status",
         "city_category",
         "product_category_1",
         "product_category_2",
         "product_category_3",
         "purchase"]].reset_index(drop=True)

y = raw[["purchase"]].reset_index(drop=True)


#%%
pipe = Pipeline([("cats_filled", CreateCatsFilled()),
                 ("group_categories", Regroup(field="product_category_1", 
                                              groups=pcat1_groups)),
                 ("mean_encode_cats", MeanEncoder(fields=["gender", "marital_status", "city_category", "product_category_1"])),
                 ("drop_garbage", DropGarbage(fields=["product_category_2", "product_category_3"]))])

op = pipe.fit_transform(X)
