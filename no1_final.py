"""
Step0:
# 到虛擬環境下執行以下程式
$cd hw2_env\Scripts

# 啟動venv
$.\Activate.ps1

"""

import numpy as np
import pandas as pd
import re
import io
import requests
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.impute import SimpleImputer  # 用於填補 NaN 值
import optuna

# 1. Business Understanding
# Goal: Predict survival on the Titanic dataset using feature selection and hyperparameter optimization.

# 2. Data Understanding & Loading
url = "https://github.com/thisisjasonjafari/my-datascientise-handcode/raw/master/005-datavisualization/titanic.csv"
s = requests.get(url).content
data = pd.read_csv(io.StringIO(s.decode('utf-8')))

# 3. Data Cleaning and Encoding
data['name'] = data['name'].apply(lambda x: re.sub('"', '', x))
data['sex'] = data['sex'].map({'male': 0, 'female': 1})

# 定義特徵和目標變量
X = data[['pclass', 'sex', 'age', 'sibsp', 'fare']]
y = data['survived']

# 使用 SimpleImputer 填補 NaN 值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 4. Data Preparation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature Selection with RFE
model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=3)  # 定義 RFE 模型
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)  # 應用 RFE 選擇的特徵

# 5. Modeling & Hyperparameter Optimization
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 1, 32)
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train_rfe, y_train)  # 使用 RFE 選擇的特徵
    preds = model.predict(X_test_rfe)
    
    accuracy = accuracy_score(y_test, preds)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
print(f"Best params: {best_params}")

# Train final model with best parameters
final_model = RandomForestClassifier(**best_params)
final_model.fit(X_train_rfe, y_train)
predictions = final_model.predict(X_test_rfe)

# 6. Evaluation
conf_matrix = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

plt.imshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0,1], ["0", "1"])
plt.yticks([0,1], ["0", "1"])
plt.savefig('C:\\中興大學\\物聯網應用與資料分析\\aiot_hw2\\conf_matrix.png')  # 儲存混淆矩陣圖片

print(f"Accuracy: {accuracy:.2f}")

# 7. Deployment
# 讀取測試資料
test_data = pd.read_csv('C:\\中興大學\\物聯網應用與資料分析\\aiot_hw2\\test.csv')
warnings.filterwarnings('ignore')

# 預處理測試資料
test_data['name'] = test_data['Name'].apply(lambda x: re.sub('"', '', x))
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})  # 將 'Sex' 欄位轉換為數值
test_data = test_data[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Fare']].fillna(0)  # 選取需要的欄位並填補缺失值

# Scale test data (僅標準化數值欄位)
test_data_scaled = scaler.transform(test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']])

# 對測試資料進行相同的特徵選擇（RFE）
test_data_rfe = rfe.transform(test_data_scaled)

# Predict survival
submission = final_model.predict(test_data_rfe)

# 建立包含 PassengerId 和 Survived 的 DataFrame
submission_df = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': submission})

# 輸出到 CSV
submission_df.to_csv('C:\\中興大學\\物聯網應用與資料分析\\aiot_hw2\\submission_final.csv', index=False)
print("Submission saved to CSV.")

