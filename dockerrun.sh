docker run \
        --volume=/media/hd03/axot_data/cc:/workspace/cc \
        --volume=/media/md01/home/axot/cc:/workspace/cc/md01_data \
        --shm-size=10g \
        --gpus=all \
        --name=ccv1 \
        -it cc/v1 \
        bash
