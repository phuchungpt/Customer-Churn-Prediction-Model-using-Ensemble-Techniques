from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score,classification_report


#function to create a model using SMOTE
def prepare_model_smote(df,class_col,cols_to_exclude): 
#Synthetic Minority Oversampling Technique. Generates new instances from existing minority cases that you supply as input. 
  cols=df.select_dtypes(include=np.number).columns.tolist() 
  X=df[cols]
  X = X[X.columns.difference([class_col])]
  X = X[X.columns.difference(cols_to_exclude)]
  y=df[class_col] ##Selecting y as a column
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
  sm = SMOTE(random_state=0, sampling_strategy=1.0)
  X_train, y_train = sm.fit_resample(X_train, y_train) 
  return(X_train, X_test, y_train, y_test)


# define the models
def run_model(model,X_train,X_test,y_train,y_test):
  if model=='random': # random forest model
    ##Fitting the random forest
    randomforest = RandomForestClassifier(max_depth=5)
    randomforest.fit(X_train, y_train)
    ##Predicting y valuesn
    y_pred = randomforest.predict(X_test)
    randomforest_roc_auc = roc_auc_score(y_test, randomforest.predict(X_test))
    print(classification_report(y_test, y_pred))
    print("The area under the curve is: %0.2f"%randomforest_roc_auc)
    return (randomforest,y_pred)

  elif model=='adaboost': # adabosst model
    ##Fitting the adaboost
    adaboost = AdaBoostClassifier(n_estimators = 100)
    adaboost.fit(X_train, y_train)
    ##Predicting y values
    y_pred = adaboost.predict(X_test)
    adaboost_roc_auc = roc_auc_score(y_test, adaboost.predict(X_test))
    print(classification_report(y_test, y_pred))
    print("The area under the curve is: %0.2f"%adaboost_roc_auc)
    return (adaboost,y_pred)

  elif model=='gradient': # gradient boosting model
    ##Fitting the logistic regression
    gradientboost = GradientBoostingClassifier()
    gradientboost.fit(X_train, y_train)
    ##Predicting y values
    y_pred = gradientboost.predict(X_test)
    gradientboost_roc_auc = roc_auc_score(y_test, gradientboost.predict(X_test))
    print(classification_report(y_test, y_pred))
    print("The area under the curve is: %0.2f"%gradientboost_roc_auc)
    return (gradientboost,y_pred)
  
  else:
    print("none")


















