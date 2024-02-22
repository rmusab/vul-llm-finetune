FROM huggingface/transformers-pytorch-gpu:latest

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    sudo \
    python3-pip && pip3 install --upgrade pip

RUN useradd -m -d /home/ma-user -s /bin/bash -g 100 -u 1000 ma-user
USER ma-user
WORKDIR /home/ma-user


# Copy your application files to the container
COPY --chown=ma-user:users requirements.txt /home/ma-user/
RUN pip3 install -r requirements.txt
COPY --chown=ma-user:users starcoder /home/ma-user/modelarts/inputs/code_1/

ENTRYPOINT ["/bin/bash"]