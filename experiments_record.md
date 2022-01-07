# 實驗記錄
## Train with one fold
### train 23 predict 24
| model_name | valid | public lb |
| -------- | -------- | -------- |
| gru_baseline     | 0.732     | 0.710     |
| self_attn_baseline     | 0.72     | 0.700     |
| nn_attn_gru_cheater     | 0.775     | 0.700     |
| nn_attn_gru_generate_mask     | 0.733     | 0.711     |

### seq2seq, train first 23, predict last 23
| model_name | valid | public lb |
| -------- | -------- | -------- |
| fixednn_attn_gru_wrong_target | 0.735 | 0.7217     |
| fixednn3_attn_gru | 0.7364 | 0.7235 |
| add_dt | 0.7361 | 0.7232|
| resume_nn3_ce16 | 0.7376 | 0.725 |
| nn3_ce16 | 0.7378 | 0.7216 |
| resume16_gru_add_dt | 0.7376 | 0.7247 | 
| 1dcnn_attn_gru | 0.7377 | 0.7233 |
| resume_1dcnn_higher_dropout | 0.7370 | 0.7225 |
| resume_1dcnn_higher_dropout_ce16 | 0.7382 | 0.7238 |
| 1dcnn_label_smooth | 0.7375 | 0.7234|
| nn_label_smooth| 0.7365| 0.7226|
| 1dcnn_mask_feats_rows | 0.7314 | 0.716 |
| 1dcnn_no_bn| 0.7372 | 0.720 |
| 1dcnn_preserve_mask | 0.7371 | 0.719 |
| 1dcnn_smaller_hidden | 0.7371| 0.72234 |

## 上面的 valid 都是錯的，我的 NDCG 寫錯了

### right ndcg
| model_name | valid | public lb |
| -------- | -------- | -------- |
| nn3_larger_hidden | 0.7254 | 0.7202 |
| resume_nn3_larger_hidden_ce16| 0.7271 |0.7228| 
| multi_index_model_larger_hidden | 0.7255 | 0.7213 |
| mm_hidden128_deeper_leakyrelu | 0.7257 | 0.7213|
| mm_hidden256_deeper_leakyrelu | 0.7263 | 0.7241 |
| mm_hidden256_deeper_leakyrelu_resume16 | 0.7273 | 0.7249 |

### 5 fold + ensemble
| model_name | valid | public lb |
| -------- | -------- | -------- |
| mm_nn_h256|         [0.7273, 0.7262, 0.7270, 0.7267, 0.7255]| 0.72697|
| mm_cnn_h256|        [0.7275, 0.7263, 0.7271, 0.7265, 0.7254]| 0.72782|
| nn3_attn_h128|          [0.7267, 0.7258, 0.7267, 0.7261, 0.7249]| 0.72652|
| mm_CnnAggBn_h256|   [0.7273, 0.7262, 0.7267, 0.7264, 0.7254]| 0.72669|
| mm_nnbn_h192|            [0.7270, 0.7260, 0.7267, 0.7265, 0.7254]| 0.72724|

### ensemble
我忘記存所有的 ensemble 權重，總共試了20幾組，不過最好的參數如下: 
- mm_cnn_h256 * 0.5 + mm_nnbn_h192 * 0.35 + nn3_attn_h128 * 0.15
