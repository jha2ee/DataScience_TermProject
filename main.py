# 1. importing library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import OneHotEncoder 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# 2. Reading the dataset (heart_2020_cleaned.csv) 
df = pd.read_csv("heart_2020_cleaned.csv")

"""
Data exploration(For dataset description)
"""
# 3. Data cleaning - set NaN values and check with isnull()
df['Diabetic'] = df['Diabetic'].replace('No, borderline diabetes', np.NaN)
df['Diabetic'] = df['Diabetic'].replace('Yes (during pregnancy)', np.NaN)
print(df.isnull().sum().sort_values(ascending=False))

# 4. Data inspection
print(df.head())
print("describe")
print(df.describe())
print("info")
print(df.info())
print("nunique")
print(df.nunique())

# 5. Cleaning dirty data (Our dirty data is in column 'Diabetic')
# Conservative judgment is made on the criteria for health with respect to Diabetic
df['Diabetic'] = df['Diabetic'].fillna(value='Yes')
# Smoking, AlcoholDrinking, Stroke, DiffWalking, Sex have 2 values, so use replace function to (1, 0)
df =  df[df.columns].replace({'Yes':1, 'No':0, 'Male':1,'Female':0,'No, borderline diabetes':'0','Yes (during pregnancy)':'1' })
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

# 6. Data Visualization
# 1) Visualization of Categorical features
# Sex
fig, ax = plt.subplots(figsize = (13, 6))
ax.hist(df[df["HeartDisease"]==1]["Sex"], bins=10, alpha=0.5, color="#86E57F", label="HeartDisease")
ax.hist(df[df["HeartDisease"]==0]["Sex"], bins=10, alpha=0.5, color="#FFB2D9", label="Normal")

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
plt.show()
# Age Category
plt.figure(figsize = (13,6))
sns.countplot(x = df['AgeCategory'], hue = 'HeartDisease', data = df, palette = 'pastel')
fig.suptitle("Distribution of heart disease by AgeCategory")
plt.xlabel('AgeCategory')
plt.ylabel('Frequency')
plt.show()
# Gen Health
gen =  df['GenHealth'].unique()
gen_cnt = df['GenHealth'].value_counts()
sns.barplot(x=gen_cnt,y=gen,data=df, palette = 'pastel')
fig.suptitle("Distribution of heart disease by AgeCategory")
plt.show()

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

# 2) Visualization of Numerical Features
# BMI
fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(df[df["HeartDisease"]==1]["BMI"], alpha=0.5,shade = True, color="#86E57F", label="HeartDisease", ax = ax)
sns.kdeplot(df[df["HeartDisease"]==0]["BMI"], alpha=0.5,shade = True, color="#FFB2D9", label="Normal", ax = ax)
plt.title('Distribution of BMI', fontsize = 16)
ax.set_xlabel("BodyMass")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()
# Sleep time
fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(df[df["HeartDisease"]==1]["SleepTime"], alpha=0.5,shade = True, color="#86E57F", label="HeartDisease", ax = ax)
sns.kdeplot(df[df["HeartDisease"]==0]["SleepTime"], alpha=0.5,shade = True, color="#FFB2D9", label="Normal", ax = ax)
plt.title('Distribution of SleepTime values', fontsize = 16)
ax.set_xlabel("SleepTime")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()
# Physical Health
fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(df[df["HeartDisease"]==1]["PhysicalHealth"], alpha=0.5,shade = True, color="#86E57F", label="HeartDisease", ax = ax)
sns.kdeplot(df[df["HeartDisease"]==0]["PhysicalHealth"], alpha=0.5,shade = True, color="#FFB2D9", label="Normal", ax = ax)
plt.title('Distribution of PhysicalHealth state for the last 30 days', fontsize = 16) 
ax.set_xlabel("PhysicalHealth")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()
# Mental Health
fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(df[df["HeartDisease"]==1]["MentalHealth"], alpha=0.5,shade = True, color="#86E57F", label="HeartDisease", ax = ax)
sns.kdeplot(df[df["HeartDisease"]==0]["MentalHealth"], alpha=0.5,shade = True, color="#FFB2D9", label="Normal", ax = ax)
plt.title('Distribution of MenalHealth state for the last 30 days', fontsize = 16)
ax.set_xlabel("MentalHealth")
ax.set_ylabel("Frequency")
ax.legend();
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
# 6. Data Normalization
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
stTrainData=StandardScaling(['BMI','PhysicalHealth','MentalHealth','AgeCategory','SleepTime'],df)
mmTrainData=MinMaxScaling(['BMI','PhysicalHealth','MentalHealth','AgeCategory','SleepTime'],df)
rbTrainData=RobustScaling(['BMI','PhysicalHealth','MentalHealth','AgeCategory','SleepTime'],df)

# 7. Encoding for data(for categorical data)
enc = OneHotEncoder() 

# Encoding categorical features - dataframe name : categ
encoded_categ = mmTrainData[['Race', 'GenHealth']]
encoded_categ = pd.DataFrame(enc.fit_transform(encoded_categ).toarray())
# Linking the encoed_cateh with the df
encoded_categ = pd.concat([mmTrainData, encoded_categ], axis = 1)
# Dropping the categorical features after encoding
encoded_categ = encoded_categ.drop(columns = ['Race', 'GenHealth'], axis = 1)



# 8. Split train dataset and test dataset
size = 0.2
data = encoded_categ.drop(['HeartDisease'], axis=1) # drop target data
target = encoded_categ['HeartDisease']
x_train, x_valid, y_train, y_valid = train_test_split(data, target, test_size=size, random_state=34, stratify = target, shuffle=True)
# for checking
print('Shape of training feature:', x_train.shape)
print('Shape of testing feature:', x_valid.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_valid.shape)


# 9. Build a model
"""
name : evaluate_model
input : model, x_test, y_test
result : accuracy information with model
"""
def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}


# 1) KNN classifier model
knn = KNeighborsClassifier(n_neighbors = 5)
# kfold = KFold(5, shuffle=True)
knn.fit(x_train, y_train)
knn_eval = evaluate_model(knn, x_valid, y_valid)

print('Accuracy:', knn_eval['acc'])
print('Precision:', knn_eval['prec']) # 분류기가 참으로 분류한 결과 중에서 실제 참의 비율
print('Recall:', knn_eval['rec']) # 실제 참 중에서 분류기가 참으로 분류한 비율
print('F1 Score:', knn_eval['f1']) # Precision과 Recall의 조화평균으로 주로 분류 클래스 간 데이터가 심각한 불균형을 이루는 경우에 사용
print('Area Under Curve:', knn_eval['auc'])
print('Confusion Matrix:\n', knn_eval['cm'])

from sklearn import tree

# 2) Decision Tree model 
dc = tree.DecisionTreeClassifier(random_state=0)
dc.fit(x_train, y_train)
dc_eval = evaluate_model(dc, x_valid, y_valid)

# Print result
print('Accuracy:', dc_eval['acc'])
print('Precision:', dc_eval['prec'])
print('Recall:', dc_eval['rec'])
print('F1 Score:', dc_eval['f1'])
print('Area Under Curve:', dc_eval['auc'])
print('Confusion Matrix:\n', dc_eval['cm'])

# 3) Linear Regression
# reg = LinearRegression()
# kfold = KFold(5, shuffle=True)



# 10. Model evalution
# 1) Confusion matrix
data = {'y_Predicted': knn.predict(x_valid), 'y_Actual': y_valid}
df = pd.DataFrame(data,
columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'],
df['y_Predicted'], rownames=['Actual'],
colnames=['Predicted'], margins = True)
sns.heatmap(confusion_matrix, annot=True)

# 2) ROC curve
fig, ax = plt.subplots(figsize = (13,5))
ax.plot(dc_eval['fpr'], dc_eval['tpr'], label='Decision Tree, auc = {:0.5f}'.format(dc_eval['auc']))
ax.plot(knn_eval['fpr'], knn_eval['tpr'], label='K-Nearest Nieghbor, auc = {:0.5f}'.format(knn_eval['auc']))
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax.legend(loc=4)

plt.show()
