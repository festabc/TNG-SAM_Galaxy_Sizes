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


df_ellipticals = pd.read_csv('Ellipticals Dataset_as defined in notebook v13')

X_ellipticals = df_ellipticals.drop(columns=['GalpropNormHalfRadius'])


y_ellipticals = df_ellipticals.loc[:,'GalpropNormHalfRadius']

# choose only the 4 most important features from feature ranking in notebook v13,
# in order to reduce the time to run SR modelling

X_ellipticals_imp = X_ellipticals.loc[:, ['GalpropNormSigmaBulge',
                                                   'GalpropNormMstar_merge', 'HalopropSpin',
                                                   'HalopropC_nfw',
                                                   'HalopropMetal_ejected', 'BulgeMstar_ratio']]


y_ellipticals_imp = y_ellipticals

# choose the Symbolic Regression model; choose the mathematical operations allowed
model_ellipticals_imp = PySRRegressor(
    
    niterations=7000,
    
    unary_operators=["exp", "square", "cube", "log10_abs", "log1p"] ,#   "inv(x) = 1/x"
#         "inv(x) = 1/x",  # Custom operator (julia syntax)
         
    
    binary_operators=["+", "*", "pow", "/"], #"mylogfunc(x)=log(1-(x))" ],


        constraints={
        "pow": (4, 1),
        "/": (-1, 4),
#         "sqrt": 5,
    },
    
    # extra_sympy_mappings={'mylogfunc': lambda x: log1p(x)},
    
    nested_constraints={
        "pow": {"pow": 0, "exp": 0},
        "square": {"square": 0, "cube": 0, "exp": 0},
        "cube": {"square": 0, "cube": 0, "exp": 0},
        "exp": {"square": 0, "cube": 0, "exp": 0},
    },
    
    maxsize=30,
    multithreading=False,
    model_selection="best", # Result is mix of simplicity+accuracys
    loss="loss(x, y) = (x - y)^2"  # Custom loss function (julia syntax)

)

start_time = time.time()

model_ellipticals_imp.fit(X_ellipticals_imp, np.array(y_ellipticals_imp))

elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the Ellipticals SymbolicRegression fitting: {elapsed_time:.3f} seconds")

model_ellipticals_imp.equations_
ellipticals_eqns = model_ellipticals_imp.equations_
ellipticals_eqns.to_csv('Elliptical_equations_n_iter_7000')

ellipticals_pred = model_ellipticals_imp.predict(X_ellipticals_imp)
ellipticals_pred = pd.DataFrame(ellipticals_pred)
ellipticals_pred.to_csv('Predicted_elliptical_sizes_SR_n_iter_7000')


print(model_ellipticals_imp.sympy())


r2_score_ellipticals=r2_score(y_ellipticals_imp, model_ellipticals_imp.predict(X_ellipticals_imp))


with open('Ellipticals_bestequation_n_iter_7000.txt', 'w', encoding='utf-8') as txt_save:
    txt_save.write( str(X_ellipticals_imp.columns.to_list()))
    txt_save.write('\n \n')
    txt_save.write(str(model_ellipticals_imp.sympy()))
    txt_save.write('\n \n')
    txt_save.write('R2 score=' + '{:.2f}'.format(r2_score_ellipticals))
    txt_save.write('\n \n')
    txt_save.write('Elapsed time to compute the Ellipticals SR =' + '{:.3f}'.format(elapsed_time) + 'seconds')




plt.scatter(y_ellipticals_imp, model_ellipticals_imp.predict(X_ellipticals_imp),
            c = df_ellipticals['BulgeMstar_ratio'],  cmap='Spectral',
            s=10, marker='.', alpha=0.7, label = 'Bulge/Mstar') #,label= label, vmin=-2, vmax=1.0)
plt.plot([0.0, 150], [0.0, 150], color = 'black', linewidth = 2)
plt.axis([0.0,80, 0.0,80])
plt.text(2, 55, 'R2 score=' + '{:.2f}'.format(r2_score_ellipticals), size=12)
plt.text(2, 75, 'eqn=' + '{}'.format(model_ellipticals_imp.sympy()), size=10)
plt.title('Predicted vs True Galaxy Size with SR \n Ellipticals only')
plt.xlabel('True Galaxy Size')
plt.ylabel('Predicted Galaxy Size ')
plt.legend(loc='lower right' , shadow=True)
plt.colorbar()
plt.savefig('SR_ellipticals_predicted vs true gal size_n_iter_7000.jpeg', dpi=500)
plt.show()

#.        Using ['GalpropNormSigmaBulge', 'GalpropNormMstar','GalpropNormMstar_merge', 'HalopropSpin',
#        'HalopropC_nfw', 'GalpropNormMbulge', 'HalopropMetal_ejected'] as important features:

# Best SR eqn for Ellipticals (with niter=1000, time=1257 sec) is: (308.606066516334*GalpropNormMstar + GalpropNormMstar**3/GalpropNormMbulge**3)/GalpropNormSigmaBulge**3
# The R2=0.71 for this eqn

# Best SR eqn for Ellipticals (with niter=15000, time=14441.4 sec) is: 1111.70006510116*Abs(GalpropNormMbulge - 1.0620177*GalpropNormMstar)**GalpropNormSigmaBulge
# The R2=0.77 for this eqn

# Best SR eqn for Ellipticals w Bulge/Mstar feature added(with niter=20, time=179.4 sec) is:
#(109.702541889589*Abs(GalpropNormMstar)**BulgeMstar_ratio + 1.97220178845976)/GalpropNormSigmaBulge**3
# The R2=0.76 for this eqn

