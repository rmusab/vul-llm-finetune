#FROM python:3.9-slim-buster
FROM nvidia/cuda:10.2-runtime-ubuntu18.04


RUN apt-get update && apt-get install -y --fix-missing\
     sudo \
     python3-pip \
     python3.8 \
     python3.8-dev

RUN update-ca-certificates
# Set Python 3.8 as the default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
COPY requirements.txt ./
# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Create a user
RUN useradd -m -d /home/ma-user -s /bin/bash -g 100 -u 1000 ma-user
USER ma-user
WORKDIR /home/ma-user

RUN mkdir -p /home/ma-user/saved_models/base_model && chown -R ma-user:users /home/ma-user/
# Copy your application files to the container
COPY --chown=ma-user:users defect_detection /home/ma-user/defect_detection
COPY --chown=ma-user:users multiclass /home/ma-user/multiclass

# Add the bash script for running the model
COPY --chown=ma-user:users run_model.sh ./

COPY saved_models/pretrain_models/ContraBERT_C /home/ma-user/saved_models/pretrain_models/ContraBERT_C
# https://drive.google.com/drive/u/0/folders/1t8VX6aYchpJolbH4mkhK3IQGzyHrDD3C
COPY saved_models/pretrain_models/ContraBERT_G /home/ma-user/saved_models/pretrain_models/ContraBERT_G
# Make the script executable
RUN chmod +x run_model.sh
RUN chmod +x /home/ma-user/multiclass/run.sh

COPY --chown=ma-user:users load_huggingface_models.py ./

RUN python3 load_huggingface_models.py

ENV DATA_PATH /home/ma-user/data
ENV OUTPUT_PATH /home/ma-user/output

# Set the environment variables
ENV MODEL_NAME ContraBERT_G
ENV PRETRAIN_DIR /home/ma-user/saved_models/pretrain_models
ENV EPOCH 20
ENV BLOCK_SIZE 400
ENV TRAIN_BATCH_SIZE 32
ENV EVAL_BATCH_SIZE 64
ENV LEARNING_RATE 2e-5
ENV MAX_GRAD_NORM 1.0
ENV SEED 123456

ENTRYPOINT ["/bin/bash"]
CMD ["run_model.sh"]