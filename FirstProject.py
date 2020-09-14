#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.stats import ttest_ind
from scipy import stats
pd.set_option("display.max_rows", None, "display.max_columns", None)

#Importing Data
data = pd.read_csv('C:/Users/Derek/Documents/Data/datasets_727551_1263738_heart_failure_clinical_records_dataset.csv')

#Looking at the Data

print(data.head())
print(data.keys())
print(data.info())
print(data.describe())
print(data.isna().any())    #No NA's

data2= pd.pivot_table(data, values = ['age', 'anaemia', 'diabetes', 'creatinine_phosphokinase','ejection_fraction',
                                      'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
                                      'smoking'], index = 'DEATH_EVENT', aggfunc= 'mean')
print(data2)
print('\n')
#Making some visualizations

sns.set(style = 'darkgrid')
for column in data2:
    sns.catplot(x='DEATH_EVENT', y=column, data=data, kind='bar', ci=99)

plt.plot()
plt.show()


#Hypothesis Testing
alive = data[data['DEATH_EVENT'] == 0]
dead = data[data['DEATH_EVENT'] == 1]

cp_dead = dead['creatinine_phosphokinase']
cp_alive = alive[ 'creatinine_phosphokinase']

ef_dead = dead['ejection_fraction']
ef_alive = alive['ejection_fraction']

platlets_dead = dead['platelets']
platlets_alive = alive['platelets']

serum_creatinine_dead = dead['serum_creatinine']
serum_creatinine_alive = alive['serum_creatinine']

serum_sodium_dead = dead['serum_sodium']
serum_sodium_alive = alive['serum_sodium']

ttest_cp, pval_cp = ttest_ind(cp_alive, cp_dead)
ttest_ef, pval_ef = ttest_ind(ef_alive, ef_dead)
ttest_p, pval_p = ttest_ind(platlets_alive, platlets_dead)
ttest_sc, pval_sc = ttest_ind(serum_creatinine_alive, serum_creatinine_dead)
ttest_ss, pval_ss = ttest_ind(serum_sodium_alive, serum_sodium_dead)

hypo_results = {'P-Value for Creatinine Phosphokinase: ' : pval_cp , 'P-Value for Ejection Fraction: ' : pval_ef,
                'P-Value for Platelets: ' : pval_p, 'P-value for serum_creatinine: ':pval_sc, 'P-value for serum sodium: ': pval_ss}
for k, v in hypo_results.items():
    print(k + ' ' + str(v))
    print('\n')

#print(pval_p, pval_cp, pval_ef, pval_sc, pval_ss)
#It seems that platlet count, creatinine-phosphokinase, ejection_fraction are not significantly different between the dead and alive

#Chi-square testing for the catagorical variablers
contingency_table_smoking = pd.crosstab(data['DEATH_EVENT'], data['smoking'])
contingency_table_age = pd.crosstab(data['DEATH_EVENT'], data['age'])
contingency_table_anaemia = pd.crosstab(data['DEATH_EVENT'], data['anaemia'])
contingency_table_hbp = pd.crosstab(data['DEATH_EVENT'], data['high_blood_pressure'])
contingency_table_sex = pd.crosstab(data['DEATH_EVENT'], data['sex'])

Observed_values = 0
Expected_Values = 0
ddof = 0
alpha  = 0
from scipy.stats import chi2
def chi_square_test(crosstable):
    Observed_values = crosstable.values
    b = stats.chi2_contingency(crosstable)
    Expected_Values = b[3]
    no_of_rows = len(crosstable.iloc[0:2, 0])
    no_of_columns = len(crosstable.iloc[0,0:2])
    ddof = (no_of_rows-1)*(no_of_columns-1)
    alpha = 0.05
    chi_square = sum([(o-e)**2./e for o, e in
                 zip(Observed_values,Expected_Values)])
    chi_square_statistic = chi_square[0] + chi_square[1]
    critical_value = chi2.ppf(q = 1-alpha, df=ddof)
    p_value = 1-chi2.cdf(x=chi_square_statistic, df=ddof)
    print(p_value, chi_square_statistic, critical_value)
    print('\n')


print('This is the p-value, chi-square statistic and critical value for smoking')
chi_square_test(contingency_table_smoking)

print('This is the p-value, chi-square statistic and critical value for age')
chi_square_test(contingency_table_age)

print('This is the p-value, chi-square statistic and critical value for anaemia')
chi_square_test(contingency_table_anaemia)

print('This is the p-value, chi-square statistic and critical value for high-blood pressure')
chi_square_test(contingency_table_hbp)

print('This is the p-value, chi-square statistic and critical value for sex')
chi_square_test(contingency_table_sex)


#Creating a model
X = data.drop('DEATH_EVENT', axis = 1)
y = data.DEATH_EVENT
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2)

param_grid = [
    {'n_estimators': [10,20,30], 'max_features': [4,8,12]},
    {'bootstrap' : [False], 'n_estimators' : [10,20,30,40,100], 'max_features': [4,8,12]}
]
data_clf = RandomForestClassifier()
grid_search = GridSearchCV(data_clf, param_grid, cv = 10, scoring = 'f1')
grid_search.fit(X_train,y_train)

#Displaying importance scores next to their corresponding attribute names
feature_importances = grid_search.best_estimator_.feature_importances_
print('Below are important features')
attributes = list(data)
print(sorted(zip(feature_importances, attributes), reverse = True))
print('\n')

#Predicting target and scoring model
y_train_pred = cross_val_predict(grid_search, X_train, y_train, cv=10)
y_pred = grid_search.predict(X_test)
print('Score:' + ' ' + str(grid_search.score(X_test, y_test)) + '\n')
print('Below are best parameters and estimators' + '\n')
print('Best parameters' + ' ' + str(grid_search.best_params_) + '\n')
print('Best estimator' + ' ' + ' ' + str(grid_search.best_estimator_) + '\n')
print(grid_search.predict_proba(X_test))


# Confusion Matrix
from sklearn.metrics import confusion_matrix
cf_mat = confusion_matrix(y_train, y_train_pred)
print(cf_mat)
print('\n')
plt.matshow(cf_mat, cmap = plt.cm.gray)         #Plotting the confusion matrix
plt.show()

# Plotting the errors
row_sums = cf_mat.sum(axis = 1, keepdims = True)
norm_cf_mx = cf_mat/row_sums
np.fill_diagonal(norm_cf_mx, 0)
plt.matshow(norm_cf_mx, cmap = plt.cm.gray)
plt.show()


# Accuracy
print('Accuracy score')
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
print('\n')

# Recall
y_train_1 = (y_train == 1)
y_test_1 = (y_test == 1)
from sklearn.metrics import recall_score
print('Recall Score')
print(recall_score(y_train_1, y_train_pred, average=None))
print('\n')

# Precision
from sklearn.metrics import precision_score
print('Precision Score')
print(precision_score(y_train_1, y_train_pred, average=None))
print('\n')
