# 玉山人工智慧公開挑戰賽2021冬季賽 - 信用卡消費類別推薦 - 第一名方法
[比賽連結](https://tbrain.trendmicro.com.tw/Competitions/Details/18)
我在該比賽中單人參賽獲得 1st / 859, 在 Public LB與 Private LB 皆為第一，且提交次數遠低於其他前幾名的參賽者。
## 怎麼 Train
1. 寫好 `src/config.json`
2. 改好 `src/train_5fold.sh` 的路徑
3.  ```
    cd src
    . ./train_5fold.sh
    ```

## 實驗紀錄
請看: [experiments_record.md](https://github.com/AxotZero/esun_consumption_prediction_1st_place_solution/blob/main/experiments_record.md)


## Data 介紹
- 各個 column 的報告: [下載連結](https://raw.githubusercontent.com/AxotZero/esun_consumption_prediction_1st_place_solution/main/reports/minimal_report.html)，請載下來後以瀏覽器開啟
- Data Shape: (32975653, 53)
- 以多個 numerical 與 categorical features 組成。
- example of data
![](https://i.imgur.com/w5A5nY3.png)
- 最重要的 4 個 features:
    | feature_name   | explanation | 
    | -------- | -------- |
    | dt     | 月份，為 1~24 的數字，比賽要預測的是 dt=25 時的txn_amt 的 NDCG@3     |
    | chid     | 消費者id，共有 500K 個消費者，同時也是我們拿來訓練及要預測的所有消費者     |
    | shop_tag     | 消費類別，共有49種，但比賽只要預測其中16種     |
    | txn_amt     | 消費金額，透過`txn_amt`與`shop_tag`即可算出我們的 NDCG@3     |
- 一個 row 代表: 消費者(chid)在該月份(dt)的該消費類別(shop_tag) 的消費資訊及消費者基本訊息
- data 的問題
    - 一個`chid`會有多種`dt`，最多24種，數量不一，原因為有可能消費者在某些月份沒有消費紀錄
    - 一個`(chid, dt)`會有多個 row
        - 一個chid會在一個月消費多種類別，且數量不一
        - 同時代表不只每個 chid 的 seq_len 不一樣，連同 (chid, dt) 的 seq_len 也不一樣。

## Idea
- 其實可以把它想像成多分類任務，目標便是預測每個月消費的比例
    - 就算預測錯了，若是預測的類別的`txn_amt`與我們 target 的`txn_amt`差距不大，NDCG依然會趨近於1
    - 使用 Soft Cross Entropy Loss
- 想辦法把 data 都塞進模型就好了
## 硬體資源
> 這裡就不把實驗室資源講出來了，不過我個人所用的資源如下
- GeForce GTX 1080 Ti * 1
- 16G Ram

## Preprocess
1. 新增一個feature: “txn_amt_pct”:
    - 代表該 chid 的該 shop_tag 在該 dt 的 txn_amt 的比例
    - 第 N 個月的 `txn_amt_pct` 即為 1~(N-1) 月的 data 的 target
2. 對於 Numerical features (excludes dt and pct columns): 
    1. 用 Quantile Transform 將他們 Mapping 到 Normal Distribution
    2. 再來對於 Missing Value 填 0
3. 對於 Categorical features:
    - 對於 missing value 填一個 special index
    - 把所有 index mapping 到 0 ~ num_classes-1

## Training Strategy
1. loss function: **Soft Cross Entropy Loss**
2. optimizer: **ranger**
3. early_stopping: 
    - monitor: val_loss
    - step: 4
4. 給 1 到 n-1 月的資料去預測第 n 個月的 `txn_amt_pct`
5. end to end (直接預測24個月)
6. 先 train 在 49 類上，再 fine-tuned 在 16 類上 (沒有freeze)
7. 5 fold cross validation
    - 每個 fold 拿 40 萬個消費者當 Train，10 萬人當 Validation 
    - average the 5 fold test output.

## Multi Index Model (簡稱mm)
![](https://i.imgur.com/Q4OJQY3.png)
其中主要有四個 Component
### 1. Row Encoder
![](https://i.imgur.com/j7X2qHD.png)
Row Encoder 會將一個 Row 的data 轉成一個embedding，主要有三個步驟:
1. 經過一個 Fixed Embedder 將所有的 feature 轉成 32 維，得到一個52*32=1664維的輸出:
    - Fixed Embedder 會把每個 feature 轉成 32 維，對於categorical feature 使用 embedding layer (num_classes, 32), 對於 numerical 我們使用 linear layer (1, 32), 每個 feature 使用的 layer 是獨立的。
2. 過一個 MLP 或是 CNN 得到一個 emb_dim 的 embedding
3. 前面得到的 Embedding 與 txn_amt_pct 相乘
    - 不用讓模型一定要從前兩個 component 來去知道這個 embedding 的重要性

### 2. IndexMapper
![](https://i.imgur.com/jfBXlf4.png)
index mapper 創建一個 shape **(24, 49, emb_dim) 的全為 0 的 tensor**, 並將 row_emb 丟到他對應的 `dt` 和 `shop_tag` 的位置上。

### 3. Month Aggregator
![](https://i.imgur.com/4D424M0.png)
month_aggregator 將同 dt 的 49 類的 embeddings  aggregate, 得到  (24, emb_dim) 的 tensor, 這 24 個 embeddings 分別代表某顧客在這24個月份的消費資訊。

### 4. GRU+MLP
![](https://i.imgur.com/jImEKoN.png)
最後再把24個月的 embedding，丟到 GRU 和 MLP, 來去預測每個下個月的消費比例

## Result + Ensemble
以下為各個結果 model_name 的 h 後面接的數字代表前面模型所說的 emb_dim
Training Record 點進去會有 `base` 和 `fine` 資料夾
- `base`: train 49 類
- `fine`: 拿 base 當 pretrained 再 fine-tuned 在 16 類上
他們都會有 5 個 fold，其中包含 config 和 log。

| Model_Name | 5 fold valid mean | Public LB | Private LB| Note|
| -------- | -------- | -------- | -------- | -------- | 
| mm_cnn_h256<br>[Training Record](https://github.com/AxotZero/esun_consumption_prediction_1st_place_solution/tree/main/save_dir/mm_cnn_hidden256_5fold)  | 0.72656     | 0.72782 | 0.7268 | **單模型最強的。Public, Private皆為最好** |
| mm_nnbn_h192<br>[Training Record](https://github.com/AxotZero/esun_consumption_prediction_1st_place_solution/tree/main/save_dir/mm_nnbn_h192)      | 0.72632     |  0.72724 | 0.7260      |  nn 版本忘了加 batch_norm，另外太過於overfitting，因此降低 hidden_dim，**該模型 Public 能成為第一，但Private會輸**    |
| nn3_attn_h128<br>[Training Record](https://github.com/AxotZero/esun_consumption_prediction_1st_place_solution/tree/main/save_dir/nn3_attn_5fold)      | 0.72604     | 0.72652 | 0.7263     | 另一種模型架構，沒有Index Map，Month Aggregator 的部分是使用 Transformer Encoder， 且 Row Encoder 很小。  **該模型 Private 其實能以些微差距成為第一，且沒那麼 overfitting，不過架構又更複雜所以就不介紹了** |
| mm_cnn \* 0.5 + <br> mm_nnbn \* 0.35 + <br> nn3_attn \* 0.15     | -     | 0.728289     | 0.727069 | 人工試了20 幾種 weights 裡面 Public LB 最好的。**大約有四成的提交次數花在這上面** |

## 試了沒效的東西
> 以下都是只 train 一個 fold 得出來的感覺沒用或變爛的東東(說不定 5fold會變好)

| 方法 |  備註 |
| -------- |  -------- |
| Transformer RowEncoder | [SAINT](https://arxiv.org/abs/2106.01342)這篇論文的方法, FixedEmbedder 的想法也是取自於這，或許因為我沒法像他那樣建那麼大的模型所以比較爛     |
| Train with one target     | Train 的時候只拿第 24 個月當 target, Test 時再預測第 25 個月     | 
| 直接train在16類上     | 甚至比 train 49類還爛     |
| Top3 Cross Entropy loss | 算 loss 時只取 top3 的比例, 其他設 0     |
| Label Smoothing     | 希望別太 focus 在特定消費類別上     |
| Add dt%12 to input      | 想看有沒有週期性，不過加到 row_encoder 和 gru 都變差     |
| mask features and rows     | 就是 mask，想說增加資料多樣性，不過變爛  |



## Reference
- 程式碼是從 [Pytorch Template](https://github.com/victoresque/pytorch-template) 一路改成這樣的。
- 其他有被我引用到的程式碼的來源:
    * [Kaggle-MoA-2nd-Place-Solution](https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution)
    * [Soft Cross Entropy](https://blog.csdn.net/Hungryof/article/details/93738717)
    * [Tabnet](https://github.com/dreamquark-ai/tabnet)
    * [Ranger Optimizer](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)