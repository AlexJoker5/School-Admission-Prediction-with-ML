import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sns
import matplotlib.pyplot as plt


df =pd.read_csv('Admission_Predict.csv')

df.head()
df.tail()
df.info()
df.isnull().sum()
df.nunique()

sns.countplot(x='Research',data=df)
admitted = df['Research'].value_counts()[1]
notAdmitted = df['Research'].value_counts()[0]
print("Admitted ",admitted)
print("Not admitted",notAdmitted)

plt.show()

df['University Rating'].unique()
df.value_counts('University Rating')
print("Group by : ",df.groupby(['Research', 'University Rating'])['University Rating'].count())


sns.countplot(x="University Rating", hue="Research", palette="Set3", data=df)
plt.show()

#GRE_Score by ranking
median_GRE_Score_by_ranking = df.groupby('University Rating')['GRE Score'].median()

print("Median : ", median_GRE_Score_by_ranking)

#df['GRE Score'] = df['GRE Score'].fillna(df.groupby('University Rating')['GRE Score'].transform('median'))

print("Max : ",df['GRE Score'].max())

print("Min : ",df['GRE Score'].min())


print(median_GRE_Score_by_ranking)

GRE_Score_ranges = [290, 300, 310, 320, 330, 340]

df['GRE ScoreRange'] = pd.cut(df['GRE Score'], bins=GRE_Score_ranges)

GRE_Score_range_counts = df['GRE ScoreRange'].value_counts().sort_index()

admitted_counts = df[df['Research'] == 1]['GRE ScoreRange'].value_counts().sort_index()

print('GRE Score Range Counts:')
print(GRE_Score_range_counts)
print('\nAdmitted Counts:')
print(admitted_counts)


plt.bar(GRE_Score_range_counts.index.astype(str), GRE_Score_range_counts.values, label='Total Applicants')

plt.bar(admitted_counts.index.astype(str), admitted_counts.values, label='Admitted Students')

plt.xlabel('GRE Score Range')
plt.ylabel('Count')
plt.title('Applicant Count vs. Admitted Count in Different GRE Score Ranges')

plt.legend()

plt.show()



#TOEFL_Score by ranking
median_TOEFL_Score_by_ranking = df.groupby('University Rating')['TOEFL Score'].median()

print("Median : ", median_TOEFL_Score_by_ranking)

df['TOEFL Score'] = df['TOEFL Score'].fillna(df.groupby('University Rating')['TOEFL Score'].transform('median'))


df['TOEFL Score'].max()
df['TOEFL Score'].min()
print(median_TOEFL_Score_by_ranking)

TOEFL_Score_ranges = [90, 95, 100, 105, 110, 115, 120]

df['TOEFL ScoreRange'] = pd.cut(df['TOEFL Score'], bins=TOEFL_Score_ranges)

TOEFL_Score_range_counts = df['TOEFL ScoreRange'].value_counts().sort_index()

admitted_counts = df[df['Research'] == 1]['TOEFL ScoreRange'].value_counts().sort_index()

print('TOEFL Score Range Counts:')
print(TOEFL_Score_range_counts)
print('\nAdmitted Counts:')
print(admitted_counts)

plt.bar(TOEFL_Score_range_counts.index.astype(str), TOEFL_Score_range_counts.values, label='Total Applicants')

plt.bar(admitted_counts.index.astype(str), admitted_counts.values, label='Admitted Students')

plt.xlabel('TOEFL Score Range')
plt.ylabel('Count')
plt.title('Applicant Count vs. Admitted Count in Different TOEFL Score Ranges')

plt.legend()

plt.show()



#SOP by ranking 
median_CGPA_by_ranking = df.groupby('University Rating')['CGPA'].median()

print("Median : ", median_CGPA_by_ranking)

df['CGPA'] = df['CGPA'].fillna(df.groupby('University Rating')['CGPA'].transform('median'))


df['CGPA'].max()
df['CGPA'].min()
print(median_CGPA_by_ranking)

CGPA_ranges = [6, 7, 8, 9, 10]

df['CGPARange'] = pd.cut(df['CGPA'], bins=CGPA_ranges)

CGPA_range_counts = df['CGPARange'].value_counts().sort_index()

admitted_counts = df[df['Research'] == 1]['CGPARange'].value_counts().sort_index()

print('CGPA Range Counts:')
print(CGPA_range_counts)
print('\nAdmitted Counts:')
print(admitted_counts)

plt.bar(CGPA_range_counts.index.astype(str), CGPA_range_counts.values, label='Total Applicants')

plt.bar(admitted_counts.index.astype(str), admitted_counts.values, label='Admitted Students')

plt.xlabel('CGPA Range')
plt.ylabel('Count')
plt.title('Applicant Count vs. Admitted Count in Different CGPA Ranges')

plt.legend()

plt.show()


X = df[['University Rating','GRE Score','TOEFL Score','CGPA']]
y = df['Research']

X.head()
y.head()
X.info()
y.info()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8 , random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#Perception
ppn = Perceptron(max_iter=200,eta0=0.1, random_state=1)
#default max_iter is 100, now we try with 200, the result doesn't imporve
ppn.fit(X_train, y_train)

y_pred_ppn = ppn.predict(X_test)
print('Misclassified examples: %d' % (y_test != y_pred_ppn).sum())

cm = confusion_matrix(y_test, y_pred_ppn)

# Create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

# Set labels, title, and ticks
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])
plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])

# Show the plot
plt.show()

accuracy_ppn = accuracy_score(y_test, y_pred_ppn)
print(accuracy_ppn)

report = classification_report(y_test, y_pred_ppn)
print(report)

#Logistic Regression
LR_model = LogisticRegression(max_iter=200, random_state=1, solver='liblinear')
LR_model.fit(X_train,y_train)
y_pred_lr=LR_model.predict(X_test)
print('Misclassified examples: %d' % (y_test != y_pred_lr).sum())

cm = confusion_matrix(y_test, y_pred_lr)

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])
plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])

plt.show()

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(accuracy_lr)

report = classification_report(y_test, y_pred_lr)
print(report)


#Support Vector Machine
svm = SVC(kernel='linear', C=1.0, random_state=1) 
svm.fit(X_train, y_train)

y_pred_svm=svm.predict(X_test)
print('Misclassified examples: %d' % (y_test != y_pred_svm).sum())

cm = confusion_matrix(y_test, y_pred_svm)

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])
plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])

plt.show()

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(accuracy_svm)

report = classification_report(y_test, y_pred_svm)
print(report)

#Decision Tree
tree_model = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)

tree_model.fit(X_train, y_train)

y_pred_dt=tree_model.predict(X_test)
print('Misclassified examples: %d' % (y_test != y_pred_dt).sum())

cm = confusion_matrix(y_test, y_pred_dt)

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])
plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])

plt.show()

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(accuracy_dt)

report = classification_report(y_test, y_pred_dt)
print(report)


#Random Forest
forest = RandomForestClassifier(criterion='gini', n_estimators=300,random_state=1,n_jobs=2)#42

forest.fit(X_train, y_train)

y_pred_rf=forest.predict(X_test)
print('Misclassified examples: %d' % (y_test != y_pred_rf).sum())

cm = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])
plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])

plt.show()

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(accuracy_rf)

report = classification_report(y_test, y_pred_rf)
print(report)

from sklearn.metrics import mean_squared_error

rmse_ppn = np.sqrt(mean_squared_error(y_test, y_pred_ppn))
print("RMSE of Perceptron",rmse_ppn)
      
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print("RMSE of Logistic Regression",rmse_lr)
      
rmse_svm = np.sqrt(mean_squared_error(y_test, y_pred_svm))
print("RMSE of SVM",rmse_svm)
      
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
print("RMSE of Decision Tree",rmse_dt)
      
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("RMSE of Ramdom forest",rmse_rf)

accuracy = {"Perceptron":accuracy_ppn,"Logistic":accuracy_lr,
            "SVM":accuracy_svm,"Decision Tree":accuracy_dt,"Random Forest":accuracy_rf}
rmse={"Perceptron":rmse_ppn,"Logistic":rmse_lr,
            "SVM":rmse_svm,"Decision Tree":rmse_dt,"Random Forest":rmse_rf}

labels = list(accuracy.keys())
values = list(accuracy.values())

plt.bar(labels, values)

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Bar Plot of Accuracy')

plt.show()


labels = list(rmse.keys())
values = list(rmse.values())

plt.bar(labels, values)

plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Bar Plot of Error')

plt.show()


import joblib

joblib_file = "AdmissionDataset_Model.pkl"  
joblib.dump(LR_model, joblib_file)

AdmissionDataset_Model = joblib.load("AdmissionDataset_Model.pkl")

rating=(int)(input("What is your university ranking(1~5) : "))
GRE_Score=(int)(input("What is your GRE Score? : "))
TOEFL_Score=(int)(input("What is your TOEFL Score? : "))
CGPA=(float)(input("What is your high school or college GPA? : "))

data=[rating,GRE_Score,TOEFL_Score,CGPA]

result=AdmissionDataset_Model.predict([data])

print("------------------------")
if result==0:
    print("Not admitted")
else:
    print("Admitted")

print("------------------------")
