output_dir='../save_dir/nn3_larger_hidden_ce16'
python3 test.py \
    --config=$output_dir/config.json \
    --resume=$output_dir/model_best.pth \
    --output_dir=$output_dir \
    --device=2
