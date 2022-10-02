import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans

path_utils = os.path.join(os.getcwd(), "utilities")
if path_utils not in sys.path:
    sys.path.append(path_utils)
    
from utils import get_envelope, get_q1, get_q3, median_ordered_boxplots


#%%
"""
predict "purchase" (quantity purchased) given categorical information about the customer, and the product they might wish to purchase

as a hunch, gender, age, occupation are probably strongly coupled with purchase
categroy precision reflects how "unique" the product is - products spanning 3 categories are more broad than one which can only fit into a single
category

columns:
gender              - cat
city_category       - cat
marital_status      - cat
occupation          - cat

age                 - ordinal
stay_years          - ordinal

product_category_{} - cat
cats_filled         - ordinal

marital_status helps partition purchase at higher levels of purchase, not so much at lower

among the ordinal features, levels don't seem to split purchase cleanly. looking like we're going to need to lean on most available features

** generally, the more categories are filled in, the higher the typical purchase amount

** product_category_1 is very discriminative - certain categories typically accumulate very low/high purchase amounts
not a lot of point in including category 2 and 3, don't have nearly as much variance across the labels compared with 1
consider placing 19, 20, 13, 12, 4, 18, 11, 5, 8 with a macro label (A) and the remeinder with (B) - would still capture a lot of the variance without having 20 or so levels, wouldn't be as much one-hot sparsity
target encoding? leave-one-out target encoding? frequency encoding?

assuming in general that the test dataset will feature the same number/span of categorical labels as training...

stay_years and age have barely any capacity to split up purchase
"""


#%%
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")


#%%
#could we actually make things ever more accurate by using the User ID? Are training users preset in test?
#...yes. In other words, we're looking at a fixed pool of customers, each buying different products in different quantities
# 
personal_features = ["User_ID", "Gender", "Age", "Occupation", "Marital_Status", "City_Category", "Stay_In_Current_City_Years"]
overlap = train[personal_features].drop_duplicates().merge(test[personal_features].drop_duplicates(), 
                                                           how="inner", 
                                                           on=personal_features)
print(f"unique training users: {len(train['User_ID'].unique())}")
print(f"unique testing users: {len(test['User_ID'].unique())}") 
print(f"users present in both train/test datasets: {len(overlap)}")  


#%% feature exploration/engineering
clean = train.copy()
clean.columns = [c.lower() for c in clean.columns]
clean.rename({"stay_in_current_city_years":"stay_years"}, axis=1, inplace=True)
clean["cats_filled"] = 3 - clean[['product_category_1',
                                  'product_category_2', 
                                  'product_category_3']].isna().sum(axis=1)

#note: cats_filled is already ordinal
ordinals = {"stay_years":["0", "1", "2", "3", "4+"],
            "age":["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]}

enc = OrdinalEncoder()
for o in ordinals.keys():
    clean[f"ord_{o}"] = enc.fit_transform(clean[o].values.reshape(-1, 1))


#%%
plt.close("all")


#%%
for t in ["gender", "marital_status", "city_category"]:
    fig, ax = plt.subplots()
    groups = clean.groupby(t)
    for g in groups:
        x, dens = get_envelope(g[1])
        ax.plot(x, dens(x), label=g[0])
    ax.legend()
    ax.set_title(t)


#%%
fig, ax = plt.subplots()
groups = clean.groupby(["gender", "marital_status"])
for g in groups:
    x, dens = get_envelope(g[1])
    ax.plot(x, dens(x), label=g[0])
ax.legend()


#%% boxplots for medium number of categories
_, fig, ax = median_ordered_boxplots(clean, field_group="product_id", do_sort=True)
_, fig, ax = median_ordered_boxplots(clean, field_group="product_category_1", do_sort=True)
_, fig, ax = median_ordered_boxplots(clean, field_group="occupation", do_sort=True)


#%% ordinal level comparisons
_, fig, ax = median_ordered_boxplots(clean, field_group="cats_filled")
_, fig, ax = median_ordered_boxplots(clean, field_group="ord_stay_years")
_, fig, ax = median_ordered_boxplots(clean, field_group="ord_age")


#%%
#product id clustering
for n in range(9, 10):
    
    #cluster_data = clean["purchase"]
    cluster_data = clean.groupby("product_id").agg({"purchase":"median"}).reset_index()["purchase"]
    
    kmeans = KMeans(n_clusters=n, random_state=420).fit(cluster_data.values.reshape(-1, 1))
    
    #product IDs, ordered by median purchase
    _, fig, ax = median_ordered_boxplots(clean, field_group="product_id", do_sort=True, minimal_vis=False)
    
    #overlay cluster boundaries
    for k in kmeans.cluster_centers_:
        ax.axhline(k[0], ls="--", c="r")
        
    #1 dimensional clustering, so establish cluster boundaries with min/max
    #clean["product_group"] = kmeans.labels_
    #group_bounds = clean.groupby("product_group").agg({"purchase":["min", "max"]}).reset_index()
    #group_bounds.columns = ["_".join(c) for c in group_bounds.columns]
    #group_bounds.sort_values(by="purchase_min", ascending=True, inplace=True)
   
    
