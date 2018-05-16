# -*- coding: utf-8 -*-
"""
Created on Fri May 11 23:12:08 2018

@author: Rahul
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import scipy.stats as stats
from imblearn.over_sampling import SMOTE 
os.chdir("C:/Users/584845/Desktop/edwisor")
data=pd.read_csv("caravan-insurance-challenge.csv")
#data_train=data.loc[data["ORIGIN"]=="train"]
#data_test=data.loc[data["ORIGIN"]=="test"]
data=data.drop(['ORIGIN'],axis=1)
sns.set(style="darkgrid")
ax = sns.countplot(x="CARAVAN", data=data)
#Target data is highly imbalanced so we need to opt oversampling technique
null_columns=data.isnull().any()
#It does not contains any null value so we are not implementing any imputing algorithm
testColumns = data.columns.tolist()
testColumns.remove('CARAVAN')
class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        k=0
        if self.p>alpha:
            k=1
        return k
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        return self._print_chisquare_result(colX,alpha)
        
cT = ChiSquare(data)
unimportantColumns=[]
for var in testColumns:
    if (cT.TestIndependence(colX=var,colY="CARAVAN" )==1): 
        unimportantColumns.append(var)
def removeColumns(listofcolumns,dataframe):
    for i in listofcolumns:
        dataframe=dataframe.drop([i],axis=1) 
    return dataframe

def false_nagative_rate(y_actual, y_hat):
    TP = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    FNR=(FN/(FN+TP))*100

    return(FNR)

transformedDF=removeColumns(unimportantColumns,data)
X=transformedDF.drop(['CARAVAN'],axis=1)
Y=transformedDF['CARAVAN']
training_features, test_features,training_target, test_target, = train_test_split(X,Y,test_size = .1,random_state=12)
model = RandomForestClassifier()
model.fit(training_features,training_target);
score_bf_removal=model.score(test_features, test_target)
## display the relative importance of each attribute
feature_importances = pd.DataFrame(model.feature_importances_,index = training_features.columns,columns=['importance']).sort_values('importance',ascending=False)
feature_importances=feature_importances.reset_index()     
feature_importances = feature_importances.rename(columns={'index': 'Column_name'})
important_feature_df=feature_importances[(feature_importances.importance < 0.02)]
removal_columns=important_feature_df["Column_name"].tolist()
retransformedDF=removeColumns(removal_columns,transformedDF)
X1=retransformedDF.drop(['CARAVAN'],axis=1)
Y1=retransformedDF['CARAVAN']
training_features1, test_features1,training_target1, test_target1, = train_test_split(X1,Y1,test_size = .1,random_state=12)
model1 = RandomForestClassifier()
model1.fit(training_features1,training_target1)
score_af_removal=model1.score(test_features1, test_target1)
#as before removal and after removal accuracy is closest we choose threshold as 0.02
X_original=retransformedDF.drop(['CARAVAN'],axis=1)
Y_original=retransformedDF['CARAVAN']
train_features, testing_features,train_target, testing_target, = train_test_split(X_original,Y_original,test_size = .1,random_state=12)
sm = SMOTE(kind='regular')
X_oversampled, y_oversampled= sm.fit_sample(train_features, train_target)
#######################################Testing for different model #########################################
#########################################random_forest###################################################
actual_model=RandomForestClassifier()
actual_model.fit(X_oversampled,y_oversampled)
score_rf=actual_model.score(testing_features, testing_target)
testing_predicted=actual_model.predict(testing_features).tolist()
testing_actual=testing_target.tolist()
False_negative_rate_rf=false_nagative_rate(testing_actual,testing_predicted)
###################################decision tree classifier################################
clf_DT = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=10, 
                                min_samples_split=2, min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, max_features=None, 
                                max_leaf_nodes=None, min_impurity_split=1e-07)
clf_DT.fit(X_oversampled, y_oversampled)
y_pred_DT = clf_DT.predict(testing_features).tolist()
y_actual= testing_target.tolist()
fals_ng_dt=false_nagative_rate(y_actual,y_pred_DT)
score_dt=clf_DT.score(testing_features,testing_target)

###############################Naive Bayes Classifier######################################

clf_NB = BernoulliNB()
clf_NB.fit(X_oversampled, y_oversampled)
y_pred_NB = clf_NB.predict(testing_features).tolist()
y_actual_nb = testing_target.tolist()
fals_ng_nb=false_nagative_rate(y_actual_nb,y_pred_NB)
score_nb=clf_NB.score(testing_features,testing_target)

############################Logistic Regression Classifier################################

clf_Log = LogisticRegression(solver='liblinear', max_iter=1000, 
                             random_state=42,verbose=2,class_weight='balanced')

clf_Log.fit(X_oversampled, y_oversampled)
y_pred_Log = clf_Log.predict(testing_features).tolist()
y_actual_log=testing_target.tolist()
fals_ng_log=false_nagative_rate(y_actual_log,y_pred_Log)
score_log=clf_Log.score(testing_features,testing_target)

#######################################SVM########################################

clf_SVM = SVC(C=10, class_weight='balanced', gamma='auto', kernel='rbf',
              max_iter=-1, probability=True, random_state=42, verbose=True)
clf_SVM.fit(X_oversampled, y_oversampled)
y_pred_SVM = clf_SVM.predict(testing_features).tolist()
y_actual_svm = testing_target.tolist()
fals_ng_svm=false_nagative_rate(y_actual_svm,y_pred_SVM)
score_svm=clf_SVM.score(testing_features,testing_target)

#####################################Neural network #####################################

clf_MLP = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(64))
clf_MLP.fit(X_oversampled, y_oversampled)
y_pred_MLP = clf_MLP.predict(testing_features).tolist()
y_actual_MLP=testing_target.tolist()
fals_ng_MLP=false_nagative_rate(y_actual_MLP,y_pred_MLP)
score_MLP=clf_MLP.score(testing_features,testing_target)

summary= pd.DataFrame({'Model_name': ["RF","NN","SVM","LR","DT","NB"], 'Accuracy': [score_rf,score_MLP,score_svm,score_log,score_dt,score_nb], 'False_negative_rate': [False_negative_rate_rf,fals_ng_MLP,fals_ng_svm,fals_ng_log,fals_ng_dt,fals_ng_nb]})

fig = plt.figure() # Create matplotlib figure


ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.4

summary.Accuracy.plot(kind='bar', color='red', ax=ax, width=width, position=1)
summary.False_negative_rate.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)
#x_labels=summary["Model_name"]
ax.set_ylabel('Accuracy')
ax.set_xticklabels(summary.Model_name)
ax2.set_ylabel('False_negative_rate')



plt.show()



#As Logistic regression giving us most optimized result so we are choosing Logistic regression as our final model

