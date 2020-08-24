import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import GridSearchCV

df=pd.read_csv('C:/Users/Nadia/Documents/ds_salary_pr/Data_exp.csv')

df.columns

df.select_dtypes(include=[np.number]).dtypes

df_models=df[['avg_salary','Rating','Size','Type_of_ownership','Industry','Sector','Revenue','State',
              'HeadQuarter_in_jobState','JobDescrip_len','python','spark','aws','sql','oracle','JobSimp','seniority','age']]

'''df_models=df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector',
             'Revenue','num_comp','hourly','employer_provided','job_state', 'same_state', 'age', 'python_yn',
             'spark', 'aws', 'excel', 'job_simp', 'seniority', 'desc_len']]'''

#df_models.hist(figsize=(30,20))

#df_models.boxplot()

#df_models=df_models.query("Size != '-1' ")  # df_models=df_models[df_models['Size'] != '-1']

'''for col in df_models.columns:
    print('-' * 40 + col +'-' * 40 ,end='-')
    display(df_models[col].value_counts())'''
#df_models.Size.value_counts()   
#mm=df_models.dscribe()
'''
# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Fill missing values with 0
df.LotFrontage = df.LotFrontage.fillna(value=0)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == object)

# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(df[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()
# Apply LabelEncoder to categorical columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())

# Import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(categorical_features=categorical_mask,sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = ohe.fit_transform(df)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(df_encoded[:5, :])

# Print the shape of the original DataFrame
print(df.shape)

# Print the shape of the transformed array
print(df_encoded.shape)

#Encoding categorical columns III: DictVectorizer two step process  - LabelEncoder followed by OneHotEncoder - can be simplified by using a DictVectorizer
# Import DictVectorizer
from sklearn.feature_extraction import DictVectorizer

# Convert df into a dictionary: df_dict
df_dict = df.to_dict('records')

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df: df_encoded
df_encoded = dv.fit_transform(df_dict)

# Print the resulting first five rows
print(df_encoded[:5,:])

# Print the vocabulary
print(dv.vocabulary_)
'''

dummy_var=pd.get_dummies(df_models)

X=dummy_var.drop('avg_salary',axis=1)
y=dummy_var.avg_salary.values

dummy_var.to_csv('DumyV.csv')


yy=y.reshape(len(y),1)

x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

# data normalization with sklearn
'''norm = MinMaxScaler().fit(x_train)
x_train_norm= norm.transform(x_train)
x_test_norm=norm.transform(x_test)'''

#Standardization using sklearn
x_train_stand = x_train.copy()
x_test_stand = x_test.copy()

stanSc = StandardScaler().fit(x_train_stand)
x_trainS= stanSc.transform(x_train_stand)

x_testS=stanSc.transform(x_test_stand)
#***********************************
'''X_sm=X=sm.add_constant(X)
model=sm.OLS(y,X_sm)
model.fit().summary()'''
#********************************
lr=LinearRegression()
lr.fit(x_trainS,y_train)
cross_val_score(lr,x_trainS,y_train, scoring='neg_mean_absolute_error',cv=3)
lr.score(x_testS,y_test)
np.sqrt(mean_squared_error(y_test, lr.predict(x_testS)))
lr_predic=lr.predict(x_testS)
#****************************************

lasso=Lasso(alpha=0.55)
lasso.fit(x_trainS,y_train)
np.mean(cross_val_score(lasso,x_trainS,y_train, scoring='neg_mean_absolute_error',cv=3))
lasso_predic=lasso.predict(x_testS)

alpha=[]
error=[]
for i in range(1,100):
    alpha.append(i/100)
    lasso_1=Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lasso_1,x_trainS,y_train,scoring='neg_mean_absolute_error',cv=3)))
plt.plot(alpha,error)

err=tuple(zip(alpha, error))
df_err=pd.DataFrame(err,columns=['Aplha','Error'])
df_err[df_err['Error']==max(df_err.Error)]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Fitting RandomForest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=90,criterion='mse',max_features='auto',bootstrap=True)
rf.fit(x_train,y_train)
rf.score(x_test,y_test)

importance_feat=pd.Series(rf.feature_importances_ ,index=X.columns)
Sorted_import_feat=importance_feat.sort_values()
Sorted_import_feat[-30:-1].plot(kind='barh',color='lightgreen');plt.show()

scor=cross_val_score(rf,x_train,y_train,scoring='neg_mean_absolute_error',cv=3)
np.mean(scor)


parameters = {'n_estimators':range(10,100,10), 'criterion':('mse', 'mae'),'max_features':('auto', 'sqrt', 'log2')}
gs=GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(x_trainS,y_train)
gs.best_score_

rf_predic=gs.best_estimator_.predict(x_testS)

print(mean_absolute_error(y_test,lr_predic))
print(mean_absolute_error(y_test,lasso_predic))
print(mean_absolute_error(y_test,rf_predic))

print(mean_squared_error(y_test,lr_predic))
print(mean_squared_error(y_test,lasso_predic))
print(mean_squared_error(y_test,rf_predic))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from sklearn.svm import SVR
#svr=SVR(kernel = 'rbf')
#svr.fit(x_trainS,y_train)
svr=SVR(kernel='poly',C=10,gamma=0.01)

# Instantiate the GridSearchCV object and run the search
parameters = {'kernel':['rbf','linear','poly'],'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svr, parameters)
searcher.fit(x_trainS,y_train)
# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(x_testS, y_test))
print("Best CV params", searcher.best_estimator_)

print("Best CV accuracy", searcher.best_score_)# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
np.mean(cross_val_score(svr,x_trainS,y_train, scoring='neg_mean_absolute_error',cv=3))
svr_predic=svr.predict(x_testS)

svr.score(x_testS,y_test)
print(mean_absolute_error(y_train,svr.predict(x_trainS)))
print(mean_absolute_error(y_test,svr_predic)) # looks like "HIGH VARIANCE"

#Bias:errortermthattellsyou,onaverage,howmuch≠f.f^
#Variance: tells you how much  is inconsistent over different training sets.f^

''' VARIANCE PROBLEM --- If F^  suffers from high variance:CV error of  F^ > training set error of F^.
F^ is said to overt the training set. To remedy overtting:
    decrease model complexity,
    for ex: decrease max depth, 
    increase min samples per leaf, ...gather more data'''

''' BIAS PROBLEM ---if F^ suffers from high bias:CV error of F^≈ training set error of F^>> desired error. 
F^ is said to undert the training set. To remedy undertting:
    increase model complexity
    for ex: increase max depth, decrease min samples per leaf, ...gather more relevant features'''

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#decision Tree 


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()

#dtr.fit(x_trainS,y_train)

np.mean(cross_val_score(dtr,x_train,y_train, scoring='neg_mean_absolute_error',cv=3))
dtr_predic=dtr.predict(x_test)
dtr.score(x_testS,y_test)
print(mean_absolute_error(y_test,dtr_predic))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from sklearn.ensemble import VotingRegressor

regressor=[('lasso',lasso),('random forest',rf),('super vector',svr),('decision tree',dtr)]
Vr=VotingRegressor(estimators=regressor)
Vr.fit(x_trainS,y_train)
y_predic=Vr.predict(x_testS)
print(mean_squared_error(y_test,y_predic))
print(mean_absolute_error(y_test,y_predic))

for Vr_name,Vr in regressor:
    Vr.fit(x_trainS,y_train)
    yy_pred=Vr.predict(x_testS)
    
    print('{:s}: {:.3f}'.format(Vr_name,mean_absolute_error(y_test,yy_pred)))
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#XGBoost

import xgboost as xgb

'''parameters_for_testing = {
   'colsample_bytree':[0.4,0.6,0.8],
   'gamma':[0,0.03,0.1,0.3],
   'min_child_weight':[1.5,6,10],
   'learning_rate':[0.1,0.07],
   'max_depth':[3,5],
   'n_estimators':[10000],
   'reg_alpha':[1e-5, 1e-2,  0.75],
   'reg_lambda':[1e-5, 1e-2, 0.45],
   'subsample':[0.6,0.95]}'''
    
xgb_model = xgb.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)

gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,iid=False, verbose=10,scoring='neg_mean_squared_error')

    
xgb1 = xgb.XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)


xgb_grid.fit(x_train,y_train)
print('best params')
print (xgb_grid.best_params_)
print('best score')
print (xgb_grid.best_score_)
   
    
