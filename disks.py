#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


df_disks = pd.read_csv('Disks Dataset_as defined in notebook v13')


# In[ ]:


X_disks = df_disks.drop(columns=['GalpropNormHalfRadius', 'BulgeMstar_ratio'])


y_disks = df_disks.loc[:,'GalpropNormHalfRadius']


# In[ ]:


df_disks_sample = df_disks.sample(n = 5000, random_state = 2022) #choose a subset of randomly sampled data


# In[ ]:


# choose the 7 most important features from analysis above to be used for SR modelling

X_disks_imp = df_disks_sample.loc[:, ['HalopropSpin', 'GalpropNormVdisk',
                              'GalpropNormMHI', 'HalopropC_nfw'
                              ]]


y_disks_imp = df_disks_sample.loc[:, 'GalpropNormHalfRadius']


# In[ ]:


# choose the Symbolic Regression model; choose the mathematical operations allowed
model_disks_imp = PySRRegressor(

    niterations=100,
    binary_operators=["+", "*", "pow", "/"],
    unary_operators=["exp", "square", "cube"
#         "inv(x) = 1/x",  # Custom operator (julia syntax)
    ],
    constraints={
        "pow": (-1, 1),
        "/": (-1, 4),
#         "sqrt": 5,
    },
    nested_constraints={
        "pow": {"pow": 1, "exp": 0},
    },
    maxsize=30,
    multithreading=False,
    model_selection="best", # Result is mix of simplicity+accuracy
    loss="loss(x, y) = (x - y)^2"  # Custom loss function (julia syntax)
)


# In[ ]:


start_time = time.time()

model_disks_imp.fit(X_disks_imp, np.array(y_disks_imp))

elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the Disks SymbolicRegression fitting: {elapsed_time:.3f} seconds")


# In[ ]:


print(model_disks_imp)


# In[ ]:


model_disks_imp.equations_
disks_eqns = model_disks_imp.equations_
# disks_eqns.to_csv('Disk_equations_n_iter_10000_4feat')

# In[ ]:


disks_pred = model_disks_imp.predict(X_disks_imp)
disks_pred = pd.DataFrame(disks_pred)
# disks_pred.to_csv('Predicted_disk_sizes_SR_n_iter_10000_4feat')


# In[ ]:

print(model_disks_imp.sympy())

plt.scatter(y_disks_imp, model_disks_imp.predict(X_disks_imp),
            c = df_disks_sample['BulgeMstar_ratio'],  cmap='Spectral',
            s=10, marker='.', alpha=0.7, label = 'Bulge/Mstar')
plt.plot([0.0, 150], [0.0, 150], color = 'black', linewidth = 2)
plt.axis([0.0,50, 0.0,50])
plt.title('Predicted vs True Galaxy Size with SR \n Disks only')
plt.xlabel('True Galaxy Size [kpc]')
plt.ylabel('Predicted Galaxy Size [kpc] ')
plt.legend(loc='lower right' , shadow=True)
plt.colorbar()
# plt.savefig('SR_disks_predicted vs true gal size.jpeg', dpi=500)
plt.show()

