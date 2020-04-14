###### Date: 2020.04.07 

# 前情提要

## 環境設定
- 安裝軟體
  - python 3.7
  - pycharm 2018.03
  - [Graphviz - Graph Visualization Software](https://www.graphviz.org/download/)
  - python packages:
    - jupyter
    - scikit-learn
    - pandas
    - matplotlib
    - seaborn
    - graphviz ─ 決策樹
    - [`opencc-python-reimplemented`](https://pypi.org/project/opencc-python-reimplemented/) ─ 簡繁轉換 
    - [`jieba`](https://github.com/fxsjy/jieba) ─ 中文分詞 

+ 設定環境變數（Windows）
  - python.exe  -> C:\Users\user\AppData\Local\Programs\Python\Python37
  - jupyter.exe -> C:\Users\user\AppData\Local\Programs\Python\Python37\Scripts
  - graphviz　  -> C:\Program Files (x86)\Graphviz2.38\bin


### Jupyter
- 使用 Jupyter 學習 Machine Learning，較方便觀察每一步的結果
- 在 Terminal 輸入「jupyter notebook」，就可以使用網頁開啟 Jupyter

### Markdown
- 語法說明文件：https://markdown.tw

### LaTeX Mathematics
- 數學排版語法：https://en.wikibooks.org/wiki/LaTeX/Mathematics

</br></br>

# AI 基礎概念
### 簡介
+ 認知 = 真理（真實不變的公式）+ 經驗（實踐得來的知識或技能）
+ 輸入 -> 公式 -> 輸出  
  運用輸入（input）和輸出（output），透過演算法，做出可以擬合輸入和輸出的「公式」 

- **專家系統**（Expert system）
- **機器學習**（Machine Learning）
  - 傳統統計機率 
  - 擅長處理表格（結構化資料）  
  - 資料量需要「千」筆以上，也考慮問題難度，問題若簡單則資料量可不用太多 
- **深度學習**（Deep Learning）
  - 傳統統計無法做到「組合」
  - 神經網路/模仿人類大腦
  - 擅長處理圖片、文字、語言等抽象型資料（非結構化資料）
  - 資料量需要「萬」筆以上，也考慮問題難度，問題困越難資料量需要越多

### 範例
+ 圖片模型範例：
  - [style2paints](https://github.com/lllyasviel/style2paints)
  - [stargan](https://github.com/yunjey/stargan)
  - [edge cat](https://affinelayer.com/pixsrv/)  
+ 語言模型範例：
  - [Talk to Transformer](https://talktotransformer.com/)   
  - BERT (Google)
  - GPT-2 (OpenAI)

</br></br>

# ML model 建立步驟

### Step1: 準備訓練資料 
- 資料類型 data：
  1. **監　督式學習**（Supervised Learning）：  
     資料有正確答案，常用演算法為分類、回歸。

  2. **非監督式學習**（Unsupervised Learning ）：  
     資料沒正確答案（因資料太多，或不知道該如何填入），常用演算法為分群。

  3. **強化學習/半監督式**（Reinforcement Learning）：  
     正確答案是由環境產生/反饋，如自駕車、機器人、遊戲。

### Step2: 建立訓練模型 
- 篩選特徵 feature？  
  - 大多數的演算法會自動決定**特徵重要性**、自動選擇合適的特徵來建立模型；  
    因此，**並不需要**特別去篩選特徵欄位，或刪除可能沒用的特徵欄位。

  - 決策樹會自己決定特徵的重要性，而單純貝氏會透過**機率**決定特徵的重要性。

### Step3: 利用模型預測  
- 預測類型 predict：
  1. **分類（classification）**：  
     - 有附答案的選擇題（無大小關係），如明天會不會下雨
     - [`sklearn.tree.DecisionTreeClassifier`](https://github.com/yalonw/Machine_Learning/blob/master/classification.ipynb)
     - [`sklearn.naive_bayes.MultinomialNB`](https://github.com/yalonw/Machine_Learning/blob/master/poem_naive_bayes.ipynb)（文字型資料）

  2. **分群（cluster）**：  
     - 沒附答案的選擇題（無大小關係）
     - [`sklearn.cluster.KMeans`](https://github.com/yalonw/Machine_Learning/blob/master/cluster.ipynb)

  3. **回歸（regression）**：  
     - 必有答案的計算題（有大小關係），如明天降雨機率多少
     - [`sklearn.tree.DecisionTreeRegressor`](https://github.com/yalonw/Machine_Learning/blob/master/regression.ipynb)

