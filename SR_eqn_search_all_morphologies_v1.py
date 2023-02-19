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

#### Physical Model Equation search in all galaxies (that is, disks + ellipticals at once)

# Normalized dataset: all masses divided by halo mass (Mvir)
df_normalized_31 = pd.read_csv('Halo Mass Normalized Dataset w 31 features.csv')

# Remove non-physical galaxies whose Mstar/Mvir (GalpropNormMstar) > 0.2
df_all = df_normalized_31[df_normalized_31.GalpropNormMstar < 0.2]

X_all = df_all.drop(columns=['GalpropNormHalfRadius'])


y_all = df_all.loc[:,'GalpropNormHalfRadius']

df_all_sample = df_all.sample(n = 10000, random_state = 2023) #choose a subset of randomly sampled data, 
# fix random seed here because we don't want to have the random seed effect when we repeat the eqn search with more iterations
# choose a subset of 10,000 randomly sampled galaxies because we expect including 30% of the dataset to be a good representation
# of the dataset (we have ~35,000 galaxies in TNG-SAM (Santa Cruz SAM)


# choose only the 7 most important features from feature ranking in notebook v13,
# in order to reduce the time to run SR modelling

X_all_imp = df_all_sample.loc[:, ['HalopropSpin', 'GalpropNormMstar_merge', 'GalpropNormMstar',
                                  'GalpropNormVdisk','GalpropNormMbulge', 'GalpropNormSigmaBulge',
                                  'GalpropOutflowRate_Mass']]

# Most important features for All Morphologies and their corresponding R2 scores
#  1 HalopropSpin 0.38981842549718637
#  2 GalpropNormMstar_merge 0.5687884206620755
#  3 GalpropNormMstar 0.7074391163134375
#  4 GalpropNormVdisk 0.8172425437842289
#  5 GalpropNormMbulge 0.9139774683331688
#  6 GalpropNormSigmaBulge 0.9268403936483627
#  7 GalpropOutflowRate_Mass 0.9321074877316505


y_all_imp = df_all_sample.loc[:, 'GalpropNormHalfRadius']

# choose the Symbolic Regression model; choose the mathematical operations allowed
model_all_imp = PySRRegressor(
    
    niterations=5000,
    
    unary_operators=[ "square"], #, "exp", "cube", "log10_abs", "log1p"] ,#   "inv(x) = 1/x"
#         "inv(x) = 1/x",  # Custom operator (julia syntax)
         
    
    binary_operators=["+", "-", "*", "pow", "/"], #"mylogfunc(x)=log(1-(x))" ],


    constraints={
        "pow": (4, 1),
        "/": (-1, 4),
#         "log1p": 4,
    },
    
    # extra_sympy_mappings={'mylogfunc': lambda x: log1p(x)},
    
    nested_constraints={
        "pow": {"pow": 0}, #, "exp": 0},
        "square": {"square": 0} #, "cube": 0, "exp": 0},
#         "cube": {"square": 0, "cube": 0, "exp": 0},
#         "exp": {"square": 0, "cube": 0, "exp": 0},
#         "log1p": {"pow": 0, "exp": 0},
    },
    
    maxsize=30,
    multithreading=False,
    model_selection="best", # Result is mix of simplicity+accuracys
    loss="loss(x, y) = (x - y)^2"  # Custom loss function (julia syntax)

)

start_time = time.time()

model_all_imp.fit(X_all_imp, np.array(y_all_imp))

elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the SymbolicRegression fitting for all morphologies: {elapsed_time:.3f} seconds")

# Run1 with n_iter=1,000, n_galaxies = 7,000, random_state not fixed, unary and binary operators included 
# (unary_operators=["exp", "square", "cube", "log10_abs", "log1p"],  binary_operators=["+", "-", "*", "pow", "/"], constraints={"pow": (4, 1), "/":(-1, 4), "log1p": 4,}, 
#  nested_constraints={ "pow": {"pow": 0, "exp": 0}, "square": {"square": 0, "cube": 0, "exp": 0},      
#   "cube": {"square": 0, "cube": 0, "exp": 0}, "exp": {"square": 0, "cube": 0, "exp": 0}, "log1p": {"pow": 0, "exp": 0},  })    
# Run2 with n_iter=5,000, n_galaxies = 7,000, random_state not fixed, unary and binary operators same as in Run1 included
# Run3 with n_iter=15,000, n_galaxies = 9,000, random_state not fixed, unary and binary operators same as in Run1 included 
# Run4 with n_iter=1,000, n_galaxies = 10,000, random_state fixed to 2023, binary operators only, "-" added 
# Run5 with n_iter=5,000, everything else same as in Run4
# Run6 with n_iter=10,000, everything else same as in Run4
# Run7 with n_iter=1,000, everything else same as in Run4 except for including "square" among unary operators
# Run8 with n_iter=5,000, everything same as in Run7
# Run9 with n_iter=15,000, everything same as in Run7

model_all_imp.equations_
all_eqns = model_all_imp.equations_
all_eqns.to_csv('SR_run8_All_morphologies_equations_n_iter_5000')

all_pred = model_all_imp.predict(X_all_imp)
all_pred = pd.DataFrame(all_pred)
all_pred.to_csv('SR_run8_All_morphologies_Predicted_sizes_n_iter_5000')


print(model_all_imp.sympy())


r2_score_all=r2_score(y_all_imp, model_all_imp.predict(X_all_imp))


with open('SR_run8_All_morphologies_bestequation_n_iter_5000.txt', 'w', encoding='utf-8') as txt_save:
    txt_save.write('The most important features used by SR to find the best equation are:')
    txt_save.write(str(X_all_imp.columns.to_list()))
    txt_save.write('\n \n')
    txt_save.write('Run9: The best equation with n_iter 15,000 is:')
    txt_save.write(str(model_all_imp.sympy()))
    txt_save.write('\n \n')
    txt_save.write('R2 score=' + '{:.2f}'.format(r2_score_all))
    txt_save.write('\n \n')
    txt_save.write('Elapsed time to compute the All Morphologies SR =' + '{:.3f}'.format(elapsed_time) + 'seconds')




plt.scatter(y_all_imp, model_all_imp.predict(X_all_imp),
            c = df_all_sample['GalpropNormMbulge']/df_all_sample['GalpropNormMstar'],  cmap='Spectral',
            s=10, marker='.', alpha=0.7, label = 'Bulge/Mstar', vmin=0.0, vmax=0.5) #,label= label,
plt.plot([0.0, 150], [0.0, 150], color = 'black', linewidth = 2)
plt.axis([0.0,100, 0.0,100])
plt.text(5, 80, 'R2 score=' + '{:.2f}'.format(r2_score_all), size=12)
plt.text(5, 90, 'eqn=' + '{}'.format(model_all_imp.sympy()), size=10)
plt.title('Predicted vs True Galaxy Size with SR \n All Morphologies')
plt.xlabel('True Galaxy Size')
plt.ylabel('Predicted Galaxy Size ')
plt.legend(loc='lower right' , shadow=True)
plt.colorbar()
plt.savefig('SR_run8_All_morphologies_predicted_vs_true_size_n_iter_5000.jpeg', dpi=500)
plt.show()


