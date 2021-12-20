device=1
base_config=config.json
save_dir='../save_dir/mm_cnn_hidden256_5fold'

# train 5fold
for fold_idx in {1..2}
do  
    echo === fold $fold_idx ===
    base_output_dir=$save_dir/base/fold$fold_idx

    # train 49
    python3 train.py \
        -c=config.json \
        -d=$device \
        --fold_idx=$fold_idx \
        --save_dir=$base_output_dir
    
    python3 test.py \
        --config=$base_output_dir/config.json \
        --device=$device \
        --resume=$base_output_dir/model_best.pth \
        --output_dir=$base_output_dir 

    # fine tuned 16
    fine_output_dir=$save_dir/fine/fold$fold_idx
    python3 train.py \
        -c=config.json \
        -d=$device \
        --fold_idx=$fold_idx \
        --resume=$base_output_dir/model_best.pth \
        --save_dir=$fine_output_dir \
        --loss=seq2seq_soft_ce16 \
        --lr=0.00005

    python3 test.py \
        --config=$fine_output_dir/config.json \
        --device=$device \
        --resume=$fine_output_dir/model_best.pth \
        --output_dir=$fine_output_dir 
done

. ./merge_logits.py
