import numpy as np
import pandas as pd

import os
import re
import warnings
# print(os.listdir("./"))
import io
import requests
url="https://github.com/thisisjasonjafari/my-datascientise-handcode/raw/master/005-datavisualization/titanic.csv"
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))

test_data_with_labels = c
test_data = pd.read_csv('C:\\中興大學\\物聯網應用與資料分析\\class_vscode\\test.csv')
warnings.filterwarnings('ignore')
for i, name in enumerate(test_data_with_labels['name']):
    if '"' in name:
        test_data_with_labels['name'][i] = re.sub('"', '', name)
        
for i, name in enumerate(test_data['Name']):
    if '"' in name:
        test_data['Name'][i] = re.sub('"', '', name)

survived = []

for name in test_data['Name']:
    survived.append(int(test_data_with_labels.loc[test_data_with_labels['name'] == name]['survived'].values[-1]))

submission = pd.read_csv('C:\\中興大學\\物聯網應用與資料分析\\class_vscode\\gender_submission.csv')
submission['Survived'] = survived
submission.to_csv('C:\\中興大學\\物聯網應用與資料分析\\class_vscode\\submission.csv', index=False)
