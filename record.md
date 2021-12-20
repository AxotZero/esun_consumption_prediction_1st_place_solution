### train 23 predict 24
1. gru_baseline, 0.732, 0.710
2. self_attn_baseline, 0.720, 0.700
3. nn_attn_gru_cheater: 0.775, 0.700
4. nn_attn_gru_generate_mask: 0.733, 0.711
#### validation 和 LB 差太多了, 作弊的傢伙也不夠強
- 最後一個月的分佈或許和 test 差太多了 -> seq2seq
- row_encoder + rows_aggregator 太爛了 -> txn_amt+shop_tag的資訊不夠明確的傳給後面，embedder對cat feats太偏心了
    -> 用 saint 的 fixed-dim embedder 的方式 再把 nn 加深

### seq2seq, train first 23, predict last 23
1. fixednn_attn_gru_wrong_target: 0.735, 0.7217
2. fixednn3_attn_gru: 0.7364, 0.7235
3. add_dt: 0.7361, 0.7232
4. resume_nn3_ce16: 0.7376, 0.725   
5. nn3_ce16: 0.7378, 0.7216         
6. resume16_gru_add_dt: 0.7376, 0.7247 
7. 1dcnn_attn_gru: 0.7377, 0.7233
8. resume_1dcnn_higher_dropout: 0.7370, 0.7225
9. resume_1dcnn_higher_dropout_ce16: 0.7382, 0.7238
10. 1dcnn_label_smooth: 0.7375, 0.7234
11. nn_label_smooth: 0.7365, 0.7226
12. 1dcnn_mask_feats_rows: 0.7314, 0.716   # 18 就被我斷了 # 我有寫錯
13. 1dcnn_no_bn: 0.7372, 0.720 #s 越來越爛了ㄟ= =
14. 1dcnn_preserve_mask: 0.7371, 0.719
15. 1dcnn_smaller_hidden: 0.7371, 0.72234

#### 
- 1,2: seq2seq 就是比較好
- 4,5: train 49 再 fine-tuned 16類 會變好，之後都靠這招，直接 train 16 反而 test 蠻爛的
- 3,6: 加入 embedding(dt mod 12) 沒什麼用，反而變差, 3是加在row, 6是加在 gru 的 input embedding 
- 7,8,9: 套1D CNN，收斂較快, val_ndcg 提升, 但 test 反而變差
- 10,11: 套label_smooth, val其實差不多, 但test變差, 在這裡其不只是為了映證label_smooth好不好，1dcnn為了加速及省空間，我把被mask的東西不丟到row_encoder裡面, 我也將nn改成這種形式，結果得到的結果就是兩者雖然val差不多，但test變差，看來被mask的東西對模型泛化能力也有些微幫助，讓row_encoder學到說哪些東西就是會被mask。
- 12: mask_feats_rows 沒啥用



### right ndcg
1. nn3_larger_hidden: 0.7254, 0.7202
2. resume_nn3_larger_hidden_ce16: 0.7271,0.7228 
3. multi_index_model_larger_hidden: 0.7255, 0.7213
4. mm_hidden128_deeper_leakyrelu: 0.7257, 0.7213
5. mm_hidden256_deeper_leakyrelu: 0.7263, 0.7241
6. mm_hidden256_deeper_leakyrelu_resume16: 0.7273, 0.7247