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


df_disks = pd.read_csv('Disks Dataset_as defined in notebook v13')

X_disks = df_disks.drop(columns=['GalpropNormHalfRadius'])


y_disks = df_disks.loc[:,'GalpropNormHalfRadius']

df_disks_sample = df_disks.sample(n = 7000, random_state = 2022) #choose a subset of randomly sampled data

# choose only the 7 most important features from feature ranking in notebook v13,
# in order to reduce the time to run SR modelling

X_disks_imp = df_disks_sample.loc[:, ['HalopropSpin', 'GalpropNormVdisk',
                              'GalpropNormMHI', 'HalopropC_nfw', 'GalpropNormMbulge',
                              'GalpropNormMH2', 'GalpropNormMstar', 'BulgeMstar_ratio']]


y_disks_imp = df_disks_sample.loc[:, 'GalpropNormHalfRadius']

# choose the Symbolic Regression model; choose the mathematical operations allowed
model_disks_imp = PySRRegressor(
    
    niterations=20000,
    
    unary_operators=["exp", "square", "cube", "log10_abs", "log1p"] ,#   "inv(x) = 1/x"
#         "inv(x) = 1/x",  # Custom operator (julia syntax)
         
    
    binary_operators=["+", "*", "pow", "/"], #"mylogfunc(x)=log(1-(x))" ],


        constraints={
        "pow": (4, 1),
        "/": (-1, 4),
        "log1p": 4,
    },
    
    # extra_sympy_mappings={'mylogfunc': lambda x: log1p(x)},
    
    nested_constraints={
        "pow": {"pow": 0, "exp": 0},
        "square": {"square": 0, "cube": 0, "exp": 0},
        "cube": {"square": 0, "cube": 0, "exp": 0},
        "exp": {"square": 0, "cube": 0, "exp": 0},
        "log1p": {"pow": 0, "exp": 0},
    },
    
    maxsize=30,
    multithreading=False,
    model_selection="best", # Result is mix of simplicity+accuracys
    loss="loss(x, y) = (x - y)^2"  # Custom loss function (julia syntax)

)

start_time = time.time()

model_disks_imp.fit(X_disks_imp, np.array(y_disks_imp))

elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the Disks SymbolicRegression fitting: {elapsed_time:.3f} seconds")

model_disks_imp.equations_
disks_eqns = model_disks_imp.equations_
disks_eqns.to_csv('Disks_equations_n_iter_20000')

disks_pred = model_disks_imp.predict(X_disks_imp)
disks_pred = pd.DataFrame(disks_pred)
disks_pred.to_csv('Disks_Predicted_sizes_SR_n_iter_20000')


print(model_disks_imp.sympy())


r2_score_disks=r2_score(y_disks_imp, model_disks_imp.predict(X_disks_imp))


with open('Disks_bestequation_n_iter_20000.txt', 'w', encoding='utf-8') as txt_save:
    txt_save.write(str(X_disks_imp.columns.to_list()))
    txt_save.write('\n \n')
    txt_save.write(str(model_disks_imp.sympy()))
    txt_save.write('\n \n')
    txt_save.write('R2 score=' + '{:.2f}'.format(r2_score_disks))
    txt_save.write('\n \n')
    txt_save.write('Elapsed time to compute the Disks SR =' + '{:.3f}'.format(elapsed_time) + 'seconds')




plt.scatter(y_disks_imp, model_disks_imp.predict(X_disks_imp),
            c = df_disks_sample['BulgeMstar_ratio'],  cmap='Spectral',
            s=10, marker='.', alpha=0.7, label = 'Bulge/Mstar') #,label= label, vmin=-2, vmax=1.0)
plt.plot([0.0, 150], [0.0, 150], color = 'black', linewidth = 2)
plt.axis([0.0,100, 0.0,100])
plt.text(5, 80, 'R2 score=' + '{:.2f}'.format(r2_score_disks), size=12)
plt.text(5, 90, 'eqn=' + '{}'.format(model_disks_imp.sympy()), size=10)
plt.title('Predicted vs True Galaxy Size with SR \n Disks (B/M <=0.15)')
plt.xlabel('True Galaxy Size')
plt.ylabel('Predicted Galaxy Size ')
plt.legend(loc='lower right' , shadow=True)
plt.colorbar()
plt.savefig('Disks_SR_predicted vs true gal size_n_iter_20000.jpeg', dpi=500)
plt.show()


