# DataScience_TermProject(2022-1)
<h3>❤Heart disease prediction🩺</h3>
<h4>Summarized following the 'End to End process' and more details can be found in 'TP_Report.docx.'</h4>

1️⃣**Project Objective**

>Healthcare research using big data has been continuously conducted, and it has been applied to actual daily life and is used by many people. By analyzing and linking health data such as clinical, genetic, and daily life of numerous people, it is possible to provide the most suitable solution for an individual's future health. Predicting the probability of heart disease due to the usual health condition, age, and the same characteristics of heart disease patients.

**In this project, we will identify health factors that affect heart disease development.**
***
2️⃣**Data Curation**
Source : Personal Key Indicators of Heart Disease | Kaggle
<https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease>

***
3️⃣**Data Inspection**
>This dataset has 18 attributes(5 Numerical, 13 Categorical data) and 319795 rows.

*'HeartDisease'* is target data which is the goal for learning model training

>Data visualization code is also in this process.
***

4️⃣**Data Preprocessing**

1. There's no dirty data(outlier, misssing data, wrong data), but for education purpose we considered some categorical data to NaN values.
```python
df['Diabetic'] = df['Diabetic'].replace('No, borderline diabetes', np.NaN)
df['Diabetic'] = df['Diabetic'].replace('Yes (during pregnancy)', np.NaN)

# check number of NaN values created
print("===Check isnull()===")
print(df.isnull().sum().sort_values(ascending=False))
```
And it is checked that data has 9340 null values in column *'Diabetic'*

2. Cleaning NaN values
```python
# Conservative judgment is made on the criteria for health with respect to Diabetic
df['Diabetic'] = df['Diabetic'].fillna(value='Yes')
```

3. Encoding Categorical data to numerical data (which has 2 values)
```
# Smoking, AlcoholDrinking, Stroke, DiffWalking, Sex have 2 values, so use replace function to (1, 0)
df =  df[df.columns].replace({'Yes':1, 'No':0, 'Male':1,'Female':0})
# Use the median of category range to use the age category information, which is categorical data, as numerically
df['AgeCategory'] = df['AgeCategory'].replace('18-24', 21)
df['AgeCategory'] = df['AgeCategory'].replace('25-29', 27)
df['AgeCategory'] = df['AgeCategory'].replace('30-34', 32)
df['AgeCategory'] = df['AgeCategory'].replace('35-39', 37)
df['AgeCategory'] = df['AgeCategory'].replace('40-44', 42)
df['AgeCategory'] = df['AgeCategory'].replace('45-49', 47)
df['AgeCategory'] = df['AgeCategory'].replace('50-54', 52)
df['AgeCategory'] = df['AgeCategory'].replace('55-59', 57)
df['AgeCategory'] = df['AgeCategory'].replace('60-64', 62)
df['AgeCategory'] = df['AgeCategory'].replace('65-69', 67)
df['AgeCategory'] = df['AgeCategory'].replace('70-74', 72)
df['AgeCategory'] = df['AgeCategory'].replace('75-79', 77)
df['AgeCategory'] = df['AgeCategory'].replace('80 or older', 80)
```

4. Encoding Categorical data using LabelEncoder
```python
enc = LabelEncoder() 

encoded_categ = df[['Race']]
encoded_categ = pd.DataFrame(enc.fit_transform(encoded_categ))
# Linking the encoed_cateh with the df
encoded_categ = pd.concat([df, encoded_categ], axis = 1)

encoded_categ2 = df[['GenHealth']]
encoded_categ2 = pd.DataFrame(enc.fit_transform(encoded_categ2))
encoded_categ = pd.concat([encoded_categ, encoded_categ2], axis = 1)
# Dropping the categorical features after encoding
encoded_categ = encoded_categ.drop(columns = ['Race', 'GenHealth'], axis = 1)

```
5. Data Normalization
>Using 3 scaling (standard scaling, min-max scaling, robust scaling)

these scaling results had approximately same results.

6. Find corrleation between *'HeartDisease'* feature and others.
>Using heatmap and bar graph, find features which have highest corrleation with heart disease presence.
```python
correlation = df.corr().round(2)
plt.figure(figsize = (14,7))
sns.heatmap(correlation, annot = True, cmap = 'RdPu')

sns.set_style('white')
sns.set_palette('RdPu')
plt.figure(figsize = (13,6))
plt.title('Heatmap correlation of features')
abs(correlation['HeartDisease']).sort_values()[:-1].plot.barh()
plt.show()
```
For more accurate analysis, we dropped the feature which has lowest correlation.
***

5️⃣**Data Analysis**
>We repeated prediction 3 times using different data(3 scaled data).
Model : ***RandomForest Regressor, Decision Tree Classifier, KNeighbors Classifier***
**First, split data to train dataset and test dataset.**
```python
 train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=size, random_state=34, stratify = target, shuffle=True)

```
**Second, train a model using trained data**
1. RandomForest Regressor
```python
	f_param = {'n_estimators':(100, 200), 'max_depth':(5, 8), 'min_samples_leaf':(8, 18), 'min_samples_split':(5, 10)}
  rf = RandomForestRegressor(random_state = 0, n_jobs=-1)
  grid_rf = GridSearchCV(rf, param_grid=f_param, cv=KFold(5), n_jobs=-1)
  grid_rf.fit(train_x, train_y)
```

2. Decision Tree Classifier
```python
  kfold = KFold(5, shuffle=True)
  dec = tree.DecisionTreeClassifier()
  score = cross_val_score(dec, X=train_x, y=train_y, cv=kfold) #train and predict
```

3. KNeighbors Classifier
```python
  knn = KNeighborsClassifier()
  h_param = {'n_neighbors':np.arange(1, 10, 2)} #list [1, 3, 5, 7, 9]
    
  grid_knn = GridSearchCV(knn, param_grid=h_param, cv=KFold(5), n_jobs=-1)
  grid_knn.fit(train_x, train_y)
```

6️⃣**Data Evaluation**
>For evaluation, we used ***gridSearchCV and KFold model***.

1. KFold
```python
 kfold = KFold(5, shuffle=True)
 score = cross_val_score(dec, X=train_x, y=train_y, cv=kfold) #train and predict
 print(score)
```
2. gridSearchCV
```python
grid_rf = GridSearchCV(rf, param_grid=f_param, cv=KFold(5), n_jobs=-1)
print('best params: ', grid_rf.best_params_)
print('best score: ', grid_rf.best_score_)
print('best estimator: ', grid_rf.best_estimator_)
```

>Make confusion matrix
```python
#make function
def build_cm(test, predict):
    cm = confusion_matrix(test, predict)
    matrix = pd.DataFrame(data = cm, columns = ['Predicted:0', 'Predicted:1'], index = ['Actual:0', 'Actual:1'])
    plt.figure(figsize = (8, 5))
    sns.heatmap(matrix, annot=True, fmt='d')
    plt.show()

#use
build_cm(test_y, pred_y)
```

7️⃣**Result**
>Prediction accuracy

1) Random Forest Regressor model & GridSearchCV validation
-> this model has the lowest accuracy(14%), so not suitable for prediction
2) Decision Tree Classification model & KFold validation
-> 90% accuracy in most cases
3) K-Nearest Neighbors Classification model & GridSearchCV validation
-> 90% accuracy in best score when n_neighbor is 9

>Failure analysis

Although our model has been almost 90% accurate, *it cannot be concluded that this model accurately predicts heart disease.* 
Because the feature in the data set used cannot be considered to be the cause of all heart disease.
There are several researchers that suggest that various environmental and genetic factors may cause heart disease, 
and *these factors are not reflected in the dataset.*

**For realistic analysis, a process of finding and studying papers or other analyses to refer to is necessary.**

8️⃣**Reference**
1. <https://www.kaggle.com/code/ahmadsoliman94/heart-disease-prediction-rf-93>
2. <https://www.kaggle.com/code/andls555/heart-disease-prediction/notebook>
