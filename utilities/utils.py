# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:52:45 2022

@author: Alex White
"""
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


#%%
def get_q1(x):
    return np.quantile(x, 0.25)

def get_q3(x):
    return np.quantile(x, 0.75)

def get_envelope(df, field_vis="purchase", precision=100):
    x_min = df[field_vis].min()
    x_max = df[field_vis].max()
    dens = gaussian_kde(df[field_vis])
    x_vals = np.arange(x_min, x_max, (x_max - x_min)/precision).tolist()
    return x_vals, dens

def median_ordered_boxplots(df, field_group, field_vis="purchase", do_sort=False):
    box_data = {c[0]:c[1][field_vis].dropna().values.tolist() for c in df.groupby(field_group)}
    if do_sort:
        box_data = {key: value for key, value in sorted(box_data.items(), key=lambda item: np.median(item[1]))}
    fig, ax = plt.subplots()
    ax.boxplot(box_data.values())
    ax.set_xticklabels(box_data.keys(), rotation=90)
    ax.set_xlabel(field_group)
    ax.set_ylabel(field_vis)
    return fig, ax