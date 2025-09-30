# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings("ignore")

# %%
df = pd.read_csv("heart.csv")
df

# %%
df.info()

# %%
df.columns

# %%
df.describe()

# %%
plt.boxplot(df["RestingBP"])

# %%
plt.boxplot(df["Cholesterol"])

# %%
plt.boxplot(df["MaxHR"])

# %%
plt.boxplot(df["Oldpeak"])

# %%
df["Cholesterol"]

# %%
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound,df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound,df[column])


# %%
remove_outliers(df,["Cholesterol"])

# %%
plt.boxplot(df["Cholesterol"])

# %%
remove_outliers(df,["RestingBP"])
remove_outliers(df,["Oldpeak"])
remove_outliers(df,["MaxHR"])

# %%
plt.boxplot(df["RestingBP"])

# %%
plt.hist(df.Age)

# %%
sns.set_style('whitegrid')
sns.histplot(df['ChestPainType'],color ='red', bins = 10)

# %%
plt.hist(df.RestingECG)

# %%
fig = px.pie(df, names='Sex')
fig.show()

# %%
df["ChestPainType"].value_counts()

# %%
df["RestingECG"].unique()

# %%
df

# %%
df["ChestPainType"] = df["ChestPainType"].map({'ASY': 0 ,'NAP': 1, 'ATA':2,'TA':3})

# %%
df["Sex"] = df["Sex"].map({'M': 1 ,'F': 0})

# %%
df["RestingECG"] = df["RestingECG"].map({'Normal': 0 ,'ST': 1, 'LVH':2})

# %%
df["ExerciseAngina"].value_counts()

# %%
df["ExerciseAngina"] = df["ExerciseAngina"].map({'N': 0 ,'Y': 1})

# %%
df["ST_Slope"].value_counts()

# %%
df["ST_Slope"] = df["ST_Slope"].map({'Flat': 0 ,'Up': 1,'Down':2})

# %%
df

# %%


# %%
corr = df.corr()
plt.figure(figsize=(24, 20))
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.title("Correlation matrix ")
plt.show()

# %%
threshold = 0.01
high_corr_features = corr.index[abs(corr["HeartDisease"]) > threshold].tolist()
high_corr_features.remove("HeartDisease")
print("Selected features based on correlation with target:")
print(high_corr_features)

X_selected = df[high_corr_features]
y = df["HeartDisease"]

# %%


# %%
X_selected

# %%
X_selected_train,X_selected_test,y_train,y_test = train_test_split(X_selected,y, test_size = 0.3 , random_state = 42)

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_selected_train)
X_test_scaled = scaler.transform(X_selected_test)

# %%
model_L = LogisticRegression()
model_L.fit(X_train_scaled , y_train)

# %%
model_L.score(X_train_scaled,y_train)

# %%
y_pred= model_L.predict(X_test_scaled)

# %%
accuracy_score(y_test , y_pred)

# %%
print(classification_report (y_test , y_pred))

# %%
# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# Create a GridSearchCV object
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)

# Fit the model
grid.fit(X_train_scaled, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", grid.best_params_)
print("Best cross-validation score: ", grid.best_score_)

# Evaluate the model on the test set
print("Test set score: ", grid.score(X_test_scaled, y_test))

# %%
print("Test set score: ", grid.score(X_test_scaled, y_test))

# %%
y_pred2 = grid.predict(X_test_scaled)

# %%
accuracy_score(y_test , y_pred2)

# %%
accuracy=accuracy_score(y_pred, y_pred)
conf_matrix=confusion_matrix(y_test, y_pred)
class_report=classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# %%
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.show()

# %%
from sklearn.neighbors import KNeighborsClassifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# %%
knn.fit(X_train_scaled, y_train)

# %%
y_pred_knn = knn.predict(X_test_scaled)

# %%
accuracy = accuracy_score(y_test, y_pred_knn)
print(f'Accuracy: {accuracy * 100:.2f}%')

# %%
classification_report(y_test, y_pred_knn)

# %%
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# %%
rf_classifier.fit(X_train_scaled, y_train)

# %%
y_pred4 = rf_classifier.predict(X_test_scaled)

# %%
accuracy = accuracy_score(y_test, y_pred4)
print(f'Accuracy: {accuracy * 100:.2f}%')

# %%
# Confusion Matrix
cm1 = confusion_matrix(y_test, y_pred4)

# Visualizing the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.show()

# %%
# Data baru sebagai dictionary, pastikan kolomnya sesuai dengan df.columns
data_baru = {
    'Age': [52],
    'Sex': [1],
    'ChestPainType': [0],
    'RestingBP': [125],
    'Cholesterol': [212],
    'FastingBS': [0],
    'RestingECG': [1],
    'MaxHR': [168],
    'ExerciseAngina': [0],
    'Oldpeak': [1.0],
    'ST_Slope': [2]
}

# Buat dataframe dari data baru
df_baru = pd.DataFrame(data_baru)

# Standarisasi atau normalisasi jika model Anda memerlukannya
df_baru_scaled = scaler.transform(df_baru)  # asumsikan variabel scaler sudah ada

# Prediksi
prediksi = rf_classifier.predict(df_baru_scaled)
print("Hasil prediksi:", prediksi[0])

# %%
# Prediksi probabilitas
proba = rf_classifier.predict_proba(df_baru_scaled)

# Tampilkan prediksi kelas & probabilitasnya
prediksi = rf_classifier.predict(df_baru_scaled)
confidence = proba[0][prediksi[0]] * 100

print(f"Prediksi: {prediksi[0]} (Confidence: {confidence:.2f}%)")


import joblib
joblib.dump(rf_classifier, 'model.pkl')
# %%
