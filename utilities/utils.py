# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:52:45 2022

@author: Alex White
"""
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#%%
def get_q1(x):
    """
    25%ile for pandas groupby
    """
    return np.quantile(x, 0.25)


def get_q3(x):
    """
    75%ile for pandas groupby
    """
    return np.quantile(x, 0.75)


def get_envelope(df, field_vis="purchase", precision=100):
    """
    kde estimate at a given level of precision 
    """
    x_min = df[field_vis].min()
    x_max = df[field_vis].max()
    dens = gaussian_kde(df[field_vis])
    x_vals = np.arange(x_min, x_max, (x_max - x_min)/precision).tolist()
    return x_vals, dens


def median_ordered_boxplots(df, field_group, field_vis="purchase", do_sort=False, minimal_vis=False):
    """
    boxplots of group data, optionally ordered ascending by group median
    """        
    box_data = {c[0]:c[1][field_vis].dropna().values.tolist() for c in df.groupby(field_group)}
    if do_sort:
        box_data = {key: value for key, value in sorted(box_data.items(), key=lambda item: np.median(item[1]))}
        
    fig, ax = plt.subplots()
    if minimal_vis:
        ax.plot([np.median(i) for i in box_data.values()])
    else:
        ax.boxplot(box_data.values(), showfliers=False)
        ax.set_xticklabels(box_data.keys(), rotation=90)
    ax.set_xlabel(field_group)
    ax.set_ylabel(field_vis)
    return box_data, fig, ax


def engineer_frame(df):
    df.columns = [c.lower() for c in df.columns]
    df.reset_index(inplace=True)
    df.rename({"index":"row_idx"}, axis=1, inplace=True)
    return df


class Regroup():
    """
    within a field of interest, bag up members into coarser groups
    """
    def __init__(self, field, groups):
        self.field = field
        self.groups = groups

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for g in self.groups.keys():
            X[self.field] = X[self.field].replace(self.groups[g], g)
        return X
  
    
class CreateCatsFilled():
    """
    create an ordinal feature indicating how many product category fields have been populated
    """
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["cats_filled"] = 3 - X[["product_category_1",
                                  "product_category_2", 
                                  "product_category_3"]].isna().sum(axis=1)
        return X
   
    
class FrequencyEncoder():
    """
    replace categorical fields of interest with label proportions
    """
    def __init__(self, fields):
        self.fields = fields

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for f in self.fields:
            X[f] = X[f].fillna("unknown")
            freq_table = X[f].value_counts(normalize=True).to_dict()
            X[f] = X[f].replace(freq_table)
        return X
    
    
class MeanEncoder():
    """
    replace categorical fields of interest with mean purchase price within each label
    """
    def __init__(self, fields):
        self.fields = fields

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for f in self.fields:
            X[f] = X[f].fillna("unknown")
            mean_table = X.groupby(f).agg({"purchase":"mean"})["purchase"].to_dict()
            X[f] = X[f].replace(mean_table)
        return X


class CustomOneHot():
    """
    Use dataframe-friendly way of generating one-hot-encoded categorical fields
    """
    def __init__(self, fields):
        self.fields = fields
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for f in self.fields:
            one_hot = pd.get_dummies(X[f], prefix=f+"_")
            X = pd.concat([X, one_hot], axis=1)
        return X
    
    
class DropGarbage():
    """
    remove features no longer required at the end of the processing chain
    """
    def __init__(self, fields):
        self.fields = fields

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.fields, axis=1)
    
    
class ProductCluster():
    """
    remove features no longer required at the end of the processing chain
    """
    def __init__(self, bounds):
        self.bounds = bounds

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.merge(self.bounds, how="inner", on="product_id")

    
def cluster_products(df, k, random_state):
    #obtain a representative purchase per id
    cluster_bounds = df.groupby("product_id").agg({"purchase":"median"}).reset_index()
    
    kmeans = KMeans(n_clusters=k, random_state=random_state).fit(cluster_bounds["purchase"].values.reshape(-1, 1))
           
    #1 dimensional clustering, so establish cluster boundaries with min/max
    cluster_bounds["product_group"] = kmeans.labels_
    cluster_bounds = cluster_bounds[["product_id", "product_group"]].drop_duplicates()
    return cluster_bounds