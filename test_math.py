## Basics
import numpy as np
import pandas as pd

## Visualization
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns

## ML
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, plot_roc_curve

## Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

data_raw = pd.read_csv('https://raw.githubusercontent.com/lucamarcelo/Churn-Modeling/main/Customer%20Churn%20Data.csv', set_index='CustomerID')

## Split the data to create a train and test set
train, test = train_test_split(data_raw, test_size=0.25)

data_raw.dtypes

## We have one variable that's wrongly encoded as string
data_raw['TotalCharges'] =pd.to_numeric(data_raw['TotalCharges'], errors='coerce')

def missing_values(data):
  df = pd.DataFrame()
  for col in list(data):
    unique_values = data[col].unique()
    try:
      unique_values = np.sort(unique_values)
    except:
      pass
    nans = round(pd.isna(data[col]).sum()/data.shape[0]*100, 1)
    zeros = round((data[col] == 0).sum()/data.shape[0]*100, 1)
    #empty = round((data[data[col]] '').sum()/data.shape[0]*100,1)
    df = df.append(pd.DataFrame([col, len(unique_values), nans,  zeros]).T, ignore_index = True)
  return df.rename(columns = {0: 'variable',
				1: 'Unique values',
				2: 'Nan %',
				3: 'zeros %',
				4: 'empty'}).sort_values('Nan %', ascending=False)

missing_values(data_raw)

fig, ax = plt.subplots(4, 5, figsize=(15, 12))

plt.subplots_adjust(left=None, bottom=None, right=None, top=1, wspace=0.3, hspace=0.4)

for variable, subplot in zip(data_raw.columns, ax.flatten()):
  sns.histplot(data_raw[variable], ax=subplot)
  
# Numerical-numerical variables

sns.pairplot(data = data_raw, hue='Churn')
plt.show()

# Categorical-numerical variables
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

## Are churned customers likely to get charged more?
plt.subplot(1,3,1)
sns.boxplot(data_raw['Churn'], data_raw['MonthlyCharges'])
plt.title('MonthlyCharges vs Churn')

## When do customers churn?
plt.subplot(1,3,2)
sns.boxplot(data_raw['Churn'], data_raw['tenure'])
plt.title('Tenure vs Churn')

## Are senior citizen more likely to churn?
plt.subplot(1,3,3)
counts = (data_raw.groupby(['Churn'])['SeniorCitizen']
  .value_counts(normalize=True)
  .rename('percentage')
  .mul(100)
  .reset_index())

plot = sns.barplot(x="SeniorCitizen", y="percentage", hue="Churn", data=counts).set_title('SeniorCitizen vs Churn')

for col in data_raw.select_dtypes(exclude=np.number):
	sns.catplot(x=col, kind='count', hue='Churn',data=data_raw.select_dtypes(exclude=np.number))

data_raw['Churn'] = data_raw['Churn'].map( {'No': 0, 'Yes': 1} ).astype(int)

# How does PaymentMethod and Contract type affect churn?

## Create pivot table
result = pd.pivot_table(data=data_raw, index='PaymentMethod', columns='Contract',values='Churn')

## create heat map of education vs marital vs response_rate
sns.heatmap(result, annot=True, cmap = 'RdYlGn_r').set_title('How does PaymentMethod and Contract type affect churn?')

plt.show()

train, test = train_test_split(data_raw, test_size=0.25, random_state=123)

X = train.drop(columns='Churn', axis=1)
y = train['Churn']

## Selecting categorical and numeric features
numerical_ix = X.select_dtypes(include=np.number).columns
categorical_ix = X.select_dtypes(exclude=np.number).columns

## Create preprocessing pipelines for each datatype
numerical_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='median')),
('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
('encoder', OrdinalEncoder()),
('scaler', StandardScaler())])

## Putting the preprocessing steps together
preprocessor = ColumnTransformer([
('numerical', numerical_transformer, numerical_ix),
('categorical', categorical_transformer, categorical_ix)],
remainder='passthrough')

## Creat list of classifiers we're going to try out
classifiers = [
KNeighborsClassifier(),
SVC(random_state=123),
DecisionTreeClassifier(random_state=123),
RandomForestClassifier(random_state=123),
AdaBoostClassifier(random_state=123),
GradientBoostingClassifier(random_state=123)
]

classifier_names = [
'KNeighborsClassifier()',
'SVC()',
'DecisionTreeClassifier()',
'RandomForestClassifier()',
'AdaBoostClassifier()',
'GradientBoostingClassifier()'
]

model_scores = []

## Looping through the classifiers
for classifier, name in zip(classifiers, classifier_names):
  pipe = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('selector', SelectKBest(k=len(X.columns))),
  ('classifier', classifier)])
  score = cross_val_score(pipe, X, y, cv=10, scoring='roc_auc').mean()
  model_scores.append(score)
  
model_performance = pd.DataFrame({
  'Classifier':
    classifier_names,
  'Cross-validated AUC':
    model_scores
}).sort_values('Cross-validated AUC', ascending = False, ignore_index=True)

display(model_performance)

pipe = Pipeline(steps=[
('preprocessor', preprocessor),
('selector', SelectKBest(k=len(X.columns))),
('classifier', GradientBoostingClassifier(random_state=123))
])

grid = {
  "selector__k": k_range,
  "classifier__max_depth":[1,3,5],
  "classifier__learning_rate":[0.01,0.1,1],
  "classifier__n_estimators":[100,200,300,400]
}

gridsearch = GridSearchCV(estimator=pipe, param_grid=grid, n_jobs= 1, scoring='roc_auc')

gridsearch.fit(X, y)

print(gridsearch.best_params_)
print(gridsearch.best_score_)

{'classifier__learning_rate': 0.1, 'classifier__max_depth': 1, 'classifier__n_estimators': 400, 'selector__k': 16}

0.8501488378467128

## Separate features and target for the test data
X_test = test.drop(columns='Churn', axis=1)
y_test = test['Churn']

## Refitting the training data with the best parameters
gridsearch.refit

## Creating the predictions
y_pred = gridsearch.predict(X_test)
y_score = gridsearch.predict_proba(X_test)[:, 1]

## Looking at the performance
print('AUCROC:', roc_auc_score(y_test, y_score), '\nAccuracy:', accuracy_score(y_test, y_pred))

# Plotting the ROC curve
plot_roc_curve(gridsearch, X_test, y_test)
plt.show()

AUCROC: 0.8441470897420832
Accuracy: 0.7927314026121521


