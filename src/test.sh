output_dir='../save_dir/EmbNN_Attn_Gru'
python3 test.py \
    --config=$output_dir/config.json \
    --resume=$output_dir/model_best.pth \
    --output_dir=$output_dir \
    --device=5