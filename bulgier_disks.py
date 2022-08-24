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


df_bulgier_disks = pd.read_csv('Bulgier Disks Dataset_as defined in notebook v13')


# In[ ]:


X_bulgier_disks = df_bulgier_disks.drop(columns=['GalpropNormHalfRadius', 'BulgeMstar_ratio'])


y_bulgier_disks = df_bulgier_disks.loc[:,'GalpropNormHalfRadius']


# In[ ]:


# choose the 7 most important features from analysis above to be used for SR modelling

X_bulgier_disks_imp = df_bulgier_disks.loc[:, ['HalopropSpin', 'GalpropNormSigmaBulge', 
                                                'GalpropNormMHI', 'GalpropNormMbulge',
                                                'GalpropNormMstar', 'GalpropNormVdisk',
                                                'GalpropZcold'
                              ]]

y_bulgier_disks_imp = df_bulgier_disks.loc[:, 'GalpropNormHalfRadius']


# In[ ]:


# choose the Symbolic Regression model; choose the mathematical operations allowed
model_bulgier_disks_imp = PySRRegressor(

    niterations=2000,
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

model_bulgier_disks_imp.fit(X_bulgier_disks_imp, np.array(y_bulgier_disks_imp))

elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the Bulgier Disks SymbolicRegression fitting: {elapsed_time:.3f} seconds")


# In[ ]:


print(model_bulgier_disks_imp)


# In[ ]:


model_bulgier_disks_imp.equations_
bulgier_disks_eqns = model_bulgier_disks_imp.equations_
bulgier_disks_eqns.to_csv('Bulgier Disk_equations_n_iter_2000')

# In[ ]:


bulgier_disks_pred = model_bulgier_disks_imp.predict(X_bulgier_disks_imp)
bulgier_disks_pred = pd.DataFrame(bulgier_disks_pred)
bulgier_disks_pred.to_csv('Predicted_bulgier_disk_sizes_SR_n_iter_2000')


# In[ ]:

print(model_bulgier_disks_imp.sympy())

r2_score_bulgier_disks=r2_score(y_bulgier_disks_imp, model_bulgier_disks_imp.predict(X_bulgier_disks_imp))

plt.scatter(y_bulgier_disks_imp, model_bulgier_disks_imp.predict(X_bulgier_disks_imp),
            c = df_bulgier_disks['BulgeMstar_ratio'],  cmap='Spectral',
            s=10, marker='.', alpha=0.7, label = 'Bulge/Mstar')
plt.plot([0.0, 150], [0.0, 150], color = 'black', linewidth = 2)
plt.axis([0.0,100, 0.0,100])
plt.text(10, 90, 'R2 score=' + '{:.2f}'.format(r2_score_bulgier_disks), size=12)
plt.title('Predicted vs True Galaxy Size with SR niter=2000 \n Bulgier Disks (0.3<B/Mstar<=0.5)')
plt.xlabel('True NormGalaxy Size')
plt.ylabel('Predicted NormGalaxy Size ')
plt.legend(loc='lower right' , shadow=True)
plt.colorbar()
plt.savefig('SR_bulgier_disks_predicted vs true gal size_niter2000.jpeg', dpi=500)
plt.show()

# The best equation (n_iter=100, time took 236sec) for Bulgier Disks from SR: 270.25516*HalopropSpin + 11.172674

# The best equation (n_iter=700, time took 954sec) for Bulgier Disks from SR: 638.72766*Abs(HalopropSpin)**GalpropNormVdisk