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
rf=RandomForestRegressor()

scor=cross_val_score(rf,x_trainS,y_train,scoring='neg_mean_absolute_error',cv=3)
np.mean(scor)

from sklearn.model_selection import GridSearchCV
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


