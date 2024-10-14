# import the required libraries
import pickle
from ML_Pipeline.utils import read_data,inspection,null_values
from ML_Pipeline.ml_model import prepare_model_smote,run_model
from ML_Pipeline.evaluate_metrics import confusion_matrix,roc_curve
from ML_Pipeline.lime import lime_explanation
import matplotlib.pyplot as plt

# Read the initial datasets
df = read_data("../input/data_regression.csv")

# Inspection and cleaning the data
x = inspection(df)

# Drop the null values
df = null_values(df)

### Run the decision tree model with sklearn ###

##Selecting only the numerical columns and excluding the columns we specified in the function
X_train, X_test, y_train, y_test = prepare_model_smote(df,class_col='churn',
                                                 cols_to_exclude=['customer_id','phone_no', 'year']) 

# run the model

model_rf,y_pred = run_model('random',X_train,X_test,y_train,y_test) # change model accordingly

## performance metric ##
conf_matrix = confusion_matrix(y_test,y_pred) # generate confusion matrix
#print(conf_matrix)

roc_val = roc_curve(model_rf,X_test,y_test) # plot the roc curve
plt.savefig("../output/ROC_curves/ROC_Curve_rf.png") # plot the feature importance graph

## Save the model ##f
pickle.dump(model_rf, open('../output/models/model_rf.pkl', 'wb'))

# generating a lime report for all the models
lime_exp = lime_explanation(model_rf,X_train,X_test,['Not Churn','Churn'],1)
lime_exp.savefig('../output/LIME_reports/lime_report_rf.jpg')

             


