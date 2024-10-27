# HW1-2: 作業說明
1. CRISP-DM 框架
2. get data from web (train.csv, test.csv)
3. modify the code to use RFE, SelectKBest, optuna to select best features
4. confusion matrix . accuracy
5. input  other best code for reference to improve your code
6. deploy -> 上傳 kaggle
分析範例 : https://ithelp.ithome.com.tw/articles/10272521?sc=rss.iron
----------------------------------------------------------------------
## Data
從Kaggle Titanic網頁上抓下來的資料

[test.csv](test.csv) : 測試資料集

[train.csv](train.csv) : 訓練資料集

## Kaggle Titanic NO.1 "Lei Michelle" 的ipython notebook 
[titanic-competition-100-score.ipynb](titanic-competition-100-score.ipynb)

## python 輸出結果的 CSV 檔案:

[submission.csv](submission.csv) : 執行no1.py (Kaggle Titanic NO.1 "Lei Michelle" 的程式) 產生的csv結果, 裡面包含欄位 "PassengerId" 和 "Survived"的預測資料

[submission_final.csv](submission_final.csv) : 執行 no1_final.py 程式 產生的csv結果, 裡面包含欄位 "PassengerId" 和 "Survived"的預測資料 (此為最後完成作業指定的內容, 產生的結果csv並上傳至Kaggle)

## folder: 

[hw2_env](./hw2_env) -> 虛擬環境 (package&interpreter path 比較不容易起衝突)

## python files:

[no1.py](no1.py) : Kaggle Titanic NO.1 "Lei Michelle" 的程式, 更改路徑才能輸出"submission.csv"

[no1_final.py](no1_final.py) : 使用Kaggle Titanic NO.1 "Lei Michelle" 的程式(no1.py), 完成以上所有 "作業說明" 的內容, 也參考了 "分析範例", 使輸出結果排名從原本的 "11996" 上升到 "392"

## images:

[before_no1_final_leaderboard.png](before_no1_final_leaderboard.png) : 參考Kaggle Titanic NO.1 "Lei Michelle" 的程式(no1.py), 並將結果 submission.csv 傳上Kaggle 顯示的排名結果圖片

![https://ithelp.ithome.com.tw/upload/images/20241028/20151681zsLPnmzhfo.png](https://ithelp.ithome.com.tw/upload/images/20241028/20151681zsLPnmzhfo.png)

[after_no1_final_leaderboard.png](after_no1_final_leaderboard.png) : 將no1.py 的程式, 完成所有 "作業說明" 的內容, 也參考了 "分析範例", 並將結果 submission_final.csv 傳上Kaggle 顯示的排名結果圖片
![https://ithelp.ithome.com.tw/upload/images/20241028/20151681v7nfuHD6gj.png](https://ithelp.ithome.com.tw/upload/images/20241028/20151681v7nfuHD6gj.png)!

[run_no1_final_result.png](run_no1_final_result.png) : 執行no1_final.py 完成的terminal結果
[https://ithelp.ithome.com.tw/upload/images/20241028/20151681ySmvMpdO2Y.png](https://ithelp.ithome.com.tw/upload/images/20241028/20151681ySmvMpdO2Y.png)

[conf_matrix.png](conf_matrix.png) : 執行no1_final.py產生的confusion matrix heatmap (blue colour)
[https://ithelp.ithome.com.tw/upload/images/20241028/20151681Mv5DUmLYOv.png](https://ithelp.ithome.com.tw/upload/images/20241028/20151681Mv5DUmLYOv.png)!
