#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import pandas as pd

import galsim #install with conda install -c conda_forge galsim

import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.cm as cm
import matplotlib.colors as norm
from matplotlib.gridspec import SubplotSpec
import seaborn as sns

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline #This allows one to build different steps together
from sklearn.preprocessing import StandardScaler, RobustScaler

from tqdm import tqdm 

from pysr import PySRRegressor

from scipy.optimize import curve_fit


# In[2]:


def spin_vdisk2_func(spin_vdisk, a, b, spin_min):
    # spin_vdisk is a 2,M array that contains HalopropSpin and GalpropNormVdisk
    # spin is spin_vdisk[0]
    # vdisk is spin_vdisk[1]
    
    spin_use = np.copy(spin_vdisk[0])
    vdisk = np.copy(spin_vdisk[1])
    spin_use[spin_use < spin_min] = spin_min
    
    size = a + b*spin_use/vdisk**2

    return size


# In[54]:


def disks_physical_formula_func (df, group):
    
    """ This function takes a df as an input, then: a) extracts the size, spin and vdisk columns from the df, 
    b) forms a 2,M array with the spin & vdisk columns (called spin_vdisk)
    c) uses curve_fit to find the optimal parameters of spin_vdisk2_func (whose input is spin_vdisk and true size of the galaxies from the df)
    d) calculates the R2 score of true size vs predicted size by the spin_vdisk2_func on the complete dataset, df
    e) calculates the R2 score of true size vs predicted size by the spin_vdisk2_func on on the galaxies with low spin (spin<0.021)
    f) Plots the figure of the true size vs predicted size for each group of galaxy morphology (noted in the title)
    
    It returns: the spin_vdisk 2,M array, the optimal parameters and pcov from curve_fit, the size array obtained
    by applying spin_vdisk2_func on spin_vdisk with parameters obtained from curve_fit, and the
    figure comparing true size vs predicted size
    
    Note: the group input has to be in a string format"""
    
#   a)
    Size_true = np.array(df.loc[:, 'GalpropNormHalfRadius'])
    HalopropSpin = np.array(df.loc[:, 'HalopropSpin'])
#     HalopropSpin = np.array(HalopropSpin)
    GalpropNormVdisk = np.array(df.loc[:, 'GalpropNormVdisk'])
#     GalpropNormVdisk = np.array(GalpropNormVdisk)
#   b) 
    spin_vdisk = [HalopropSpin, GalpropNormVdisk]
    spin_vdisk = np.array (spin_vdisk) # convert list into array
#   c) 
    popt, pcov = curve_fit(spin_vdisk2_func, spin_vdisk, Size_true, p0=[0.5, 1000, 0.002])

    size_func = spin_vdisk2_func(spin_vdisk, *popt)
#   d) 
    r2_score_df = r2_score(Size_true, size_func)
    r2_score_df
#   e) 
    df_spin_size = df.loc[:,['GalpropNormHalfRadius', 'HalopropSpin']]
    df_spin_size.loc[:, "Predicted"] = size_func
    zz = df_spin_size[df_spin_size.loc[:,'HalopropSpin']<=0.021]
    actual_size= zz['GalpropNormHalfRadius']
    predicted_size = zz['Predicted']
    r2_score_lowspin = r2_score(actual_size, predicted_size)

#   f) Plot the figure
    fig_prediction, ax = plt.subplots(figsize=(7, 5))
    
    ax = plt.subplot()
    im = ax.scatter(Size_true, size_func, marker='.', s=10, alpha=0.7, 
                c  = df.loc[:,'BulgeMstar_ratio'],  
                cmap='Spectral',
                label=' colorbar: Mbulge/Mstar ratio \n fit: a=%5.3f, b=%5.3f, \n spin_min=%5.3f' % tuple(popt))
    ax.plot([0.0, 150], [0.0, 150], color = 'black', linewidth = 2)
    ax.set_xlim([0.0,150])
    ax.set_ylim([0.0,150])
    ax.text(10, 130, 'R2 score=' + '{:.2f}'.format(r2_score_df), size=12)
    ax.text(10, 120, 'R2_lowspin_gals=' + '{:.2f}'.format(r2_score(actual_size, predicted_size)), size=12)
    ax.set_title('Eqn size= a+b*spin/vdisk^2 vs True Size \n {} '.format(group))
    ax.set_ylabel('Size as predicted by spin/vdisk^2 function')
    ax.set_xlabel('True Size')
    ax.legend(loc = 'lower right', shadow=True)
    fig_prediction.colorbar(im, ax=ax)
    
    fig_prediction.tight_layout()
    # plt.savefig('BulgierDisks_TrueSize_vs_FunctionSize_15_wsmallgals.jpeg', dpi=500)
    plt.show()
    
    return spin_vdisk, popt, pcov, size_func, fig_prediction, r2_score_df, r2_score_lowspin


# In[50]:


# note to myself: I can make the small spin galaxies' R2 score a function that calls spin_min, 
# if needed, in the future

