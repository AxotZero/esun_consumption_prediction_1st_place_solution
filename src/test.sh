output_dir='../save_dir/mm_hidden256_deeper_leakyrelu_resume16'
python3 test.py \
    --config=$output_dir/config.json \
    --resume=$output_dir/model_best.pth \
    --output_dir=$output_dir \
    --device=3
