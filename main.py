# 1. importing library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor

import warnings 
warnings.filterwarnings("ignore")

# 2. Reading the dataset (heart_2020_cleaned.csv) 
df = pd.read_csv("heart_2020_cleaned.csv")

"""
Data exploration(For dataset description)
"""
# 3. Data cleaning - set NaN values and check with isnull()
df['Diabetic'] = df['Diabetic'].replace('No, borderline diabetes', np.NaN)
df['Diabetic'] = df['Diabetic'].replace('Yes (during pregnancy)', np.NaN)

# check number of NaN values created
print("===Check isnull()===")
print(df.isnull().sum().sort_values(ascending=False))

# 4. Data inspection
print("===describe===")
print(df.describe())
print("===info===")
print(df.info())
print("===nunique===")
print(df.nunique())

# 5. Cleaning dirty data (Our dirty data is in column 'Diabetic')
# Conservative judgment is made on the criteria for health with respect to Diabetic
df['Diabetic'] = df['Diabetic'].fillna(value='Yes')
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

# check number of NaN values created
print("===Check isnull()===")
print(df.isnull().sum().sort_values(ascending=False))


# 6. Data Visualization
# 1) Visualization of Categorical features
# Sex
fig, ax = plt.subplots(figsize = (13, 6))
ax.hist(df[df["HeartDisease"]==1]["Sex"], bins=2, alpha=0.5, color="#86E57F", label="HeartDisease")
ax.hist(df[df["HeartDisease"]==0]["Sex"], bins=2, alpha=0.5, color="#FFB2D9", label="Normal")
ax.set_xlabel("Sex")
ax.set_ylabel("Frequency")
fig.suptitle("Distribution of heart disease by Sex")
ax.legend();

# Smoking
fig, ax = plt.subplots(figsize = (13,6))
ax.hist(df[df["HeartDisease"]==1]["Smoking"], bins=10, alpha=0.5, color="#86E57F", label="HeartDisease")
ax.hist(df[df["HeartDisease"]==0]["Smoking"], bins=10, alpha=0.5, color="#FFB2D9", label="Normal")
ax.set_xlabel("Smoking")
ax.set_ylabel("Frequency")
fig.suptitle("Distribution of heart disease by Smoking")
ax.legend();

# Race
plt.figure(figsize = (13,6))
sns.countplot( x= df['Race'], hue = 'HeartDisease', data = df, palette = 'pastel')
plt.xlabel('Race')
plt.ylabel('Frequency')
fig.suptitle("Distribution of heart disease by Race")

# Age Category
plt.figure(figsize = (13,6))
sns.countplot(x = df['AgeCategory'], hue = 'HeartDisease', data = df, palette = 'pastel')
fig.suptitle("Distribution of heart disease by AgeCategory")
plt.xlabel('AgeCategory')
plt.ylabel('Frequency')

# Gen Health
gen =  df['GenHealth'].unique()
gen_cnt = df['GenHealth'].value_counts()
sns.barplot(x=gen_cnt,y=gen,data=df, palette = 'pastel')
fig.suptitle("Distribution of heart disease by AgeCategory")

# KidneyDisease
fig, ax = plt.subplots(figsize = (13,6))

ax.hist(df[df["HeartDisease"]==1]["KidneyDisease"], bins=15, alpha=0.5, color="#86E57F", label="HeartDisease")
ax.hist(df[df["HeartDisease"]==0]["KidneyDisease"], bins=15, alpha=0.5, color="#FFB2D9", label="Normal")
ax.set_xlabel("KidneyDisease")
ax.set_ylabel("Frequency")
fig.suptitle("Distribution of heart disease by kidneydisease")
ax.legend();

# SkinCancer
fig, ax = plt.subplots(figsize = (13,6))

ax.hist(df[df["HeartDisease"]==1]["SkinCancer"], bins=15, alpha=0.5, color="#86E57F", label="HeartDisease")
ax.hist(df[df["HeartDisease"]==0]["SkinCancer"], bins=15, alpha=0.5, color="#FFB2D9", label="Normal")
ax.set_xlabel("SkinCancer")
ax.set_ylabel("Frequency")
fig.suptitle("Distribution of heart disease based on previous exposure to skin cancer")
ax.legend();

# Stroke
fig, ax = plt.subplots(figsize = (13,6))

ax.hist(df[df["HeartDisease"]==1]["Stroke"], bins=15, alpha=0.5, color="#86E57F", label="HeartDisease")
ax.hist(df[df["HeartDisease"]==0]["Stroke"], bins=15, alpha=0.5, color="#FFB2D9", label="Normal")
ax.set_xlabel("Stroke")
ax.set_ylabel("Frequency")
fig.suptitle("Distribution of heart disease based on previous exposure to Stroke")
ax.legend();

# Diabetic
fig, ax = plt.subplots(figsize = (13,6))

ax.hist(df[df["HeartDisease"]==1]["Diabetic"], bins=15, alpha=0.5, color="#86E57F", label="HeartDisease")
ax.hist(df[df["HeartDisease"]==0]["Diabetic"], bins=15, alpha=0.5, color="#FFB2D9", label="Normal")
ax.set_xlabel("Diabetic")
ax.set_ylabel("Frequency")
fig.suptitle("Distribution of heart disease based on previous exposure to Diabetic")
ax.legend();

plt.show()

# 2) Visualization of Numerical Features
# BMI
fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(df[df["HeartDisease"]==1]["BMI"], alpha=0.5,shade = True, color="#86E57F", label="HeartDisease", ax = ax)
sns.kdeplot(df[df["HeartDisease"]==0]["BMI"], alpha=0.5,shade = True, color="#FFB2D9", label="Normal", ax = ax)
plt.title('Distribution of BMI', fontsize = 16)
ax.set_xlabel("BodyMass")
ax.set_ylabel("Frequency")
ax.legend()

# Sleep time
fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(df[df["HeartDisease"]==1]["SleepTime"], alpha=0.5,shade = True, color="#86E57F", label="HeartDisease", ax = ax)
sns.kdeplot(df[df["HeartDisease"]==0]["SleepTime"], alpha=0.5,shade = True, color="#FFB2D9", label="Normal", ax = ax)
plt.title('Distribution of SleepTime values', fontsize = 16)
ax.set_xlabel("SleepTime")
ax.set_ylabel("Frequency")
ax.legend()

# Physical Health
fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(df[df["HeartDisease"]==1]["PhysicalHealth"], alpha=0.5,shade = True, color="#86E57F", label="HeartDisease", ax = ax)
sns.kdeplot(df[df["HeartDisease"]==0]["PhysicalHealth"], alpha=0.5,shade = True, color="#FFB2D9", label="Normal", ax = ax)
plt.title('Distribution of PhysicalHealth state for the last 30 days', fontsize = 16) 
ax.set_xlabel("PhysicalHealth")
ax.set_ylabel("Frequency")
ax.legend()

# Mental Health
fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(df[df["HeartDisease"]==1]["MentalHealth"], alpha=0.5,shade = True, color="#86E57F", label="HeartDisease", ax = ax)
sns.kdeplot(df[df["HeartDisease"]==0]["MentalHealth"], alpha=0.5,shade = True, color="#FFB2D9", label="Normal", ax = ax)
plt.title('Distribution of MenalHealth state for the last 30 days', fontsize = 16)
ax.set_xlabel("MentalHealth")
ax.set_ylabel("Frequency")
ax.legend()
plt.show()

# 3) Check correlation with heatmap
correlation = df.corr().round(2)
plt.figure(figsize = (14,7))
sns.heatmap(correlation, annot = True, cmap = 'RdPu')

sns.set_style('white')
sns.set_palette('RdPu')
plt.figure(figsize = (13,6))
plt.title('Heatmap correlation of features')
abs(correlation['HeartDisease']).sort_values()[:-1].plot.barh()
plt.show()

"""
Data preprocessing
"""

# 7. Encoding for data(for categorical data) 
enc = LabelEncoder() 

# Encoding categorical features - dataframe name : categ
encoded_categ = df[['Race']]
encoded_categ = pd.DataFrame(enc.fit_transform(encoded_categ))
# Linking the encoed_cateh with the df
encoded_categ = pd.concat([df, encoded_categ], axis = 1)

encoded_categ2 = df[['GenHealth']]
encoded_categ2 = pd.DataFrame(enc.fit_transform(encoded_categ2))
encoded_categ = pd.concat([encoded_categ, encoded_categ2], axis = 1)
# Dropping the categorical features after encoding
encoded_categ = encoded_categ.drop(columns = ['Race', 'GenHealth'], axis = 1)

# 8. Data Normalization
'''
StandardScaling change dataset by StandardScaler
'''
#input : columnName(numerical columns name), df (data frame)
#output : df(After scaled dataset)
def StandardScaling(columnName,df):
    scaler=StandardScaler()
    scaler.fit(df[columnName])
    df[columnName]=scaler.transform(df[columnName])
    return df
'''
MinMaxScaling change dataset by MinMaxScaler
'''
#input : columnName(numerical columns name), df (data frame)
#output : df(After scaled dataset)
def MinMaxScaling(columnName,df):
    scaler=MinMaxScaler()
    scaler.fit(df[columnName])
    df[columnName]=scaler.transform(df[columnName])
    return df

'''
RobustScaling change dataset by RobustScaler
'''
#input : columnName(numerical columns name), df (data frame)
#output : df(After scaled dataset)
def RobustScaling(columnName,df):
    scaler=RobustScaler()
    scaler.fit(df[columnName])
    df[columnName]=scaler.transform(df[columnName])
    return df

# Make scaling dataset (for numerical data)
stTrainData=StandardScaling(['BMI','PhysicalHealth','MentalHealth','AgeCategory','SleepTime'],encoded_categ)
mmTrainData=MinMaxScaling(['BMI','PhysicalHealth','MentalHealth','AgeCategory','SleepTime'],encoded_categ)
rbTrainData=RobustScaling(['BMI','PhysicalHealth','MentalHealth','AgeCategory','SleepTime'],encoded_categ)

data_list = [stTrainData, mmTrainData, rbTrainData]
# low_col : under correlation 1.5
low_col = ['SleepTime', 'AlcoholDrinking', 'BMI', 'Sex', 'SkinCancer', 'PhysicalActivity', 'Smoking']
size = 0.2

from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

"""
Show confusion matrix figure
input : test data, predict data
output : none
"""
def build_cm(test, predict):
    cm = confusion_matrix(test, predict)
    matrix = pd.DataFrame(data = cm, columns = ['Predicted:0', 'Predicted:1'], index = ['Actual:0', 'Actual:1'])
    plt.figure(figsize = (8, 5))
    sns.heatmap(matrix, annot=True, fmt='d')
    plt.show()

# Use 3 normalized data
for i in range(3):
    # 9. Prepare for split train dataset and test dataset
    data = data_list[i].drop(['HeartDisease'], axis=1) # drop target data
    data = data.drop(low_col, axis = 1) # drop data which has low correlation
    target = data_list[i]['HeartDisease']

    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=size, random_state=34, stratify = target, shuffle=True)

    #######################################################
    # 10. Build a model
    # 1) Random Forest Regressor model & GridSearchCV
    f_param = {'n_estimators':(100, 200), 'max_depth':(5, 8), 'min_samples_leaf':(8, 18), 'min_samples_split':(5, 10)}
    rf = RandomForestRegressor(random_state = 0, n_jobs=-1)
    grid_rf = GridSearchCV(rf, param_grid=f_param, cv=KFold(5), n_jobs=-1)
    grid_rf.fit(train_x, train_y)
    # 11. Model evalution
    # 1) Find best score 
    print("Model 1. RandomForest Regression")
    print('best params: ', grid_rf.best_params_)
    print('best score: ', grid_rf.best_score_)
    print('best estimator: ', grid_rf.best_estimator_)

    #######################################################
    # 10. Build a model
    # 2) Decision Tree model & KFold validation
    kfold = KFold(5, shuffle=True)
    dec = tree.DecisionTreeClassifier()
    score = cross_val_score(dec, X=train_x, y=train_y, cv=kfold) #train and predict

    # 11. Model evalution
    # 1) Find validation score
    print("Model 2. Decision Tree")
    print("cross validation scroe: ", score)
    print("Mean score: ", np.mean(score))

    # 2) Confusion matrix
    dec.fit(train_x, train_y)
    pred_y = dec.predict(test_x)
    build_cm(test_y, pred_y)

    #######################################################
    # 10. Build a model
    # 3) KNN classifier model & GridSearchCV
    knn = KNeighborsClassifier()
    h_param = {'n_neighbors':np.arange(1, 10, 2)} #list [1, 3, 5, 7, 9]
    
    grid_knn = GridSearchCV(knn, param_grid=h_param, cv=KFold(5), n_jobs=-1)
    grid_knn.fit(train_x, train_y)
    # 11. Model evalution
    # 1) Find best score
    print("Model 3. KNN classifier")
    print('best params: ', grid_knn.best_params_)
    print('best score: ', grid_knn.best_score_)
    print('best estimator: ', grid_knn.best_estimator_)
    # 2) Confusion matrix
    build_cm(test_y, grid_knn.predict(test_x))
