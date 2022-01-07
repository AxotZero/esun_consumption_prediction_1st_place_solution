FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
RUN pip3 install pandas numpy scikit-learn tensorboard>=1.14 tqdm easydict jupyterlab