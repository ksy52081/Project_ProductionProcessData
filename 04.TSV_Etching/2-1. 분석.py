########  정리된 X25 다시 불러오기
import pandas as pd
#Data= pd.read_csv('F:/JupyterLab/AdvancedIntelligence19/AI_Homework02/TSV ETCH/data25.csv')
Data= pd.read_csv('F:/JupyterLab/AdvancedIntelligence19/AI_Homework02/TSV ETCH/data80.csv')
####인덱스 부분 제거
Data= Data.iloc[:,1:]
### mean centering -> 차원축소 -> 학습과 테스트 나누기 -> 학습하기 -> 평가하기 ####
print(Data.shape)
Describing_Data = Data.describe().transpose()
''' Data Mean Centering + scaling = Nomalizing '''
from sklearn.preprocessing import scale
import numpy as np
X = np.array(scale(Data.iloc[:,:-1])) ## 
y = np.array(Data.iloc[:,-1])
#Describing_DataX_Scaled = scale(Data.iloc[:,:-1]).describe().transpose()


'''
(1) 3차원 데이터셋 (샘플수x변수x가공시간)을 2차원으로 unfolding한 후 
(kernel) PCA를 수행한 후 PCR과 Gaussian process regression의 성능을 
leave-one-out cross validation을 통해 측정하시오(R2, MAE, MAPE).

(2) Unfolding된 데이터에 대해서 
PLS의 성능을 leave-one-out cross validation을 통해측정하시오.

(3) 여러분이 알고 있는 feature extraction 기법을 통해
 feature를 추출하여 X에 대해 Y를 예측하는 회귀 모델을 제시하고,
 모델의 성능을 leave-one-out cross validation을 통해 측정하시오.
'''



'''
PCA 혹은 KernelPCA 수행하기
'''
from sklearn.decomposition import PCA, KernelPCA 
pca_rbf = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_pca_rbf = pca_rbf.fit_transform(X)
X_pca_rbf_back = pca_rbf.inverse_transform(X_pca_rbf)
pca = PCA()
X_pca = pca.fit_transform(X)


import numpy as np


from sklearn.utils.validation import (check_array, check_consistent_length)
import sklearn

def _check_reg_targets(y_true, y_pred, multioutput):
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("y_true and y_pred have different number of output "
                         "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ('raw_values', 'uniform_average',
                               'variance_weighted')
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. "
                             "You provided multioutput={!r}".format(
                                 allowed_multioutput_str,
                                 multioutput))
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in "
                             "multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(("There must be equally many custom weights "
                              "(%d) as outputs (%d).") %
                             (len(multioutput), n_outputs))
    y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'

    return y_type, y_true, y_pred, multioutput


from sklearn.utils import check_array
def mean_absolute_percentage_error(y_test, y_pred, sample_weight=None, multioutput='uniform_average'):
    y_type, y_test, y_pred, multioutput = _check_reg_targets(y_test, y_pred, multioutput)
    y_test, y_pred = check_array(y_test, y_pred)
    check_consistent_length(y_test, y_pred, sample_weight)
    output_errors = np.average((np.abs((y_test, y_pred)/y_test))*100, weights=sample_weight, axis=0)
    
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None
    return np.average(output_errors, weights=multioutput)



'''Q1-1-1 PCA+PCR'''
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut  
from sklearn import metrics

loo = LeaveOneOut()
ytests = []
ypreds = []
for train_idx, test_idx in loo.split(X_pca):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx] #requires arrays
    y_train, y_test = y[train_idx], y[test_idx]
    model = LinearRegression()
    
    model.fit(X = X_train, y = y_train) 
    y_pred = model.predict(X_test)
        
    # there is only one y-test and y-pred per iteration over the loo.split, 
    # so to get a proper graph, we append them to respective lists.
    ytests += list(y_test)
    ypreds += list(y_pred)
        
Score_R2 = metrics.r2_score(ytests, ypreds)
Score_MAE = metrics.mean_absolute_error(ytests, ypreds)
Score_MSE = metrics.mean_squared_error(ytests, ypreds)
Score_MAPE = mean_absolute_percentage_error(ytests, y_pred)
        
print("Leave One Out Cross Validation")
print("R^2: {:.5f}%, MSE: {:.5f}".format(Score_R2*100, Score_MSE))


'''Q1-1-2 KernelPCA+PCR'''
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut  
from sklearn import metrics

loo = LeaveOneOut()
ytests = []
ypreds = []
for train_idx, test_idx in loo.split(X_pca_rbf):
    X_train, X_test = X_pca_rbf[train_idx], X_pca_rbf[test_idx] #requires arrays
    y_train, y_test = y[train_idx], y[test_idx]
    model = LinearRegression()
    
    model.fit(X = X_train, y = y_train) 
    y_pred = model.predict(X_test)
        
    # there is only one y-test and y-pred per iteration over the loo.split, 
    # so to get a proper graph, we append them to respective lists.
    ytests += list(y_test)
    ypreds += list(y_pred)
        
Score_R2 = metrics.r2_score(ytests, ypreds)
Score_MAE = metrics.mean_absolute_error(ytests, ypreds)
Score_MSE = metrics.mean_squared_error(ytests, ypreds)
#Score_MAPE = mean_absolute_percentage_error(ytests, y_pred)
        
print("Leave One Out Cross Validation")
print("R^2: {:.5f}%, MSE: {:.5f}".format(Score_R2*100, Score_MSE))



'''Q1-2 Kernel + GPR'''
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
kernel = DotProduct() + WhiteKernel()
loo = LeaveOneOut()
ytests = []
ypreds = []
for train_idx, test_idx in loo.split(X_pca):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx] #requires arrays
    y_train, y_test = y[train_idx], y[test_idx]
    model = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)
    model.fit(X = X_train, y = y_train) 
    y_pred = model.predict(X_test)
        
    # there is only one y-test and y-pred per iteration over the loo.split, 
    # so to get a proper graph, we append them to respective lists.
    ytests += list(y_test)
    ypreds += list(y_pred)
        
Score_R2 = metrics.r2_score(ytests, ypreds)
Score_MAE = metrics.mean_absolute_error(ytests, ypreds)
Score_MSE = metrics.mean_squared_error(ytests, ypreds)


'''Q2 PLS '''
numofPC = 17
from sklearn.cross_decomposition import PLSRegression
loo= LeaveOneOut()
ytests=[]
ypreds=[]
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx] #requires arrays
    y_train, y_test = y[train_idx], y[test_idx]
    model = PLSRegression(n_components= numofPC)
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    # there is only one y-test and y-pred per iteration over the loo.split, 
    # so to get a proper graph, we append them to respective lists.
    ytests += list(y_test)
    ypreds += list(y_pred)
        
Score_R2 = metrics.r2_score(ytests, ypreds)
Score_MAE = metrics.mean_absolute_error(ytests, ypreds)
Score_MSE = metrics.mean_squared_error(ytests, ypreds)


'''Q3-1 Decision Tree Regressor'''
from sklearn.tree import DecisionTreeRegressor
loo= LeaveOneOut()
ytests=[]
ypreds=[]
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx] #requires arrays
    y_train, y_test = y[train_idx], y[test_idx]
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    # there is only one y-test and y-pred per iteration over the loo.split, 
    # so to get a proper graph, we append them to respective lists.
    ytests += list(y_test)
    ypreds += list(y_pred)
        
Score_R2 = metrics.r2_score(ytests, ypreds)
Score_MAE = metrics.mean_absolute_error(ytests, ypreds)
Score_MSE = metrics.mean_squared_error(ytests, ypreds)

'''Q3-2 SVR'''
from sklearn.svm import SVR
loo= LeaveOneOut()
ytests=[]
ypreds=[]
for train_idx, test_idx in loo.split(X_pca):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx] #requires arrays
    y_train, y_test = y[train_idx], y[test_idx]
    model =  SVR(gamma='scale', C=1.0, epsilon=0.2)
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    # there is only one y-test and y-pred per iteration over the loo.split, 
    # so to get a proper graph, we append them to respective lists.
    ytests += list(y_test)
    ypreds += list(y_pred)
        
Score_R2 = metrics.r2_score(ytests, ypreds)
Score_MAE = metrics.mean_absolute_error(ytests, ypreds)
Score_MSE = metrics.mean_squared_error(ytests, ypreds)


'''Q3-3 Random Forest Regressor'''
from sklearn.ensemble import RandomForestRegressor
loo= LeaveOneOut()
ytests=[]
ypreds=[]
for train_idx, test_idx in loo.split(X_pca):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx] #requires arrays
    y_train, y_test = y[train_idx], y[test_idx]
    model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    # there is only one y-test and y-pred per iteration over the loo.split, 
    # so to get a proper graph, we append them to respective lists.
    ytests += list(y_test)
    ypreds += list(y_pred)
        
Score_R2 = metrics.r2_score(ytests, ypreds)
Score_MAE = metrics.mean_absolute_error(ytests, ypreds)
Score_MSE = metrics.mean_squared_error(ytests, ypreds)


'''Q3-4 Gradient Boost Regressor'''
from sklearn import ensemble
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
loo= LeaveOneOut()
ytests=[]
ypreds=[]
for train_idx, test_idx in loo.split(X_pca):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx] #requires arrays
    y_train, y_test = y[train_idx], y[test_idx]
    model = ensemble.GradientBoostingRegressor(**params)
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    # there is only one y-test and y-pred per iteration over the loo.split, 
    # so to get a proper graph, we append them to respective lists.
    ytests += list(y_test)
    ypreds += list(y_pred)
        
Score_R2 = metrics.r2_score(ytests, ypreds)
Score_MAE = metrics.mean_absolute_error(ytests, ypreds)
Score_MSE = metrics.mean_squared_error(ytests, ypreds)






import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut

loocv = model_selection.LeaveOneOut()
model = LinearRegression()
results_loocv = model_selection.cross_val_score(model, X_pca, y, cv=loocv )
print("Accuracy: %.2f%%" % (results_loocv.mean()*100.0))


model_selection.cross_val_score()


'''
변수들 끼리의 correlation 보기
PCA 직접 수행하기
skip 합니다.
'''


###PLS Regression ### 





'''


from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

loo = LeaveOneOut()
loo.get_n_splits(X_kpca)
for train_index, test_index in loo.split(X_kpca):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_kpca[train_index], X_kpca[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train, X_test, y_train, y_test)
    model = LinearRegression()
    scores_MAE = cross_val_score(model, X_train, y_train, scoring="neg_mean_absolute_error", cv=loo)
    scores_R2 = sklearn.model_selection.cross_val_score(model, X, y, scoring = 'r2')
    
'''





'''
#### SVR RFE #####
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.feature_selection import RFE
>>> from sklearn.svm import SVR
>>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
>>> estimator = SVR(kernel="linear")
>>> selector = RFE(estimator, 5, step=1)
>>> selector = selector.fit(X, y)
>>> selector.support_ 
array([ True,  True,  True,  True,  True, False, False, False, False,
       False])
>>> selector.ranking_
array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])
'''