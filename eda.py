import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

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
raw = pd.read_csv("./data/train.csv")


#%% feature exploration/engineering
clean = raw.copy()
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