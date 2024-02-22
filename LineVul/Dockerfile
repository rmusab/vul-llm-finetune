FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel


ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

USER root
RUN useradd -m -d /home/ma-user -s /bin/bash -g 100 -u 1000 ma-user
RUN chown -R 1000:100 /home/ma-user && chmod 777 -R /home/ma-user

WORKDIR /home/ma-user

RUN cat /etc/os-release

RUN pwd \
 && apt-get clean \
 && apt-get update -y --fix-missing \
 && apt-get install -y locales \
 && apt-get install -y ca-certificates \
 && locale-gen en_US.UTF-8 \
 && dpkg-reconfigure locales \
 && echo "LC_ALL=en_US.UTF-8\nLANG=en_US.UTF-8\nLANGUAGE=en_US.UTF-8" > /etc/default/locale \
 && apt-get install -y software-properties-common build-essential \
 && update-ca-certificates
	
RUN apt-get install -y iproute2 libpcre3 libpcre3-dev -y
RUN apt-get install gcc libpq-dev -y
RUN apt-get install python3-wheel -y

ENV XDG_CACHE_HOME /tmp/.cache
RUN mkdir /tmp/.cache && chmod 777 /tmp/.cache
ENV PYTHONUSERBASE=/tmp/.local
RUN mkdir /tmp/.local && chmod 777 /tmp/.local

USER ma-user

RUN python3 -m pip install --upgrade pip
RUN pip3 install wheel && pip3 install torch
RUN pip3 install pip install gdown
RUN pip3 install transformers
RUN pip3 install captum
RUN pip3 install numpy
RUN pip3 install tqdm
RUN pip3 install scikit-learn
RUN pip3 install pandas
RUN pip3 install tokenizers

ENV PATH "$PATH:/tmp/.local/bin"


RUN python3 -m requests.certs
ENV REQUESTS_CA_BUNDLE /etc/ssl/certs

COPY check.py /home/ma-user/check.py
RUN python3 /home/ma-user/check.py

COPY ContraBERT_G/pytorch_model.bin /home/ma-user/linevul/contrabert/pytorch_model.bin
COPY ContraBERT_G/config.json /home/ma-user/linevul/contrabert/config.json
COPY ContraBERT_G/training_args.bin /home/ma-user/linevul/contrabert/training_args.bin

COPY linevul_experiments /home/ma-user

USER root
RUN chown -R 1000:100 /home/ma-user/linevul && chmod -R 777 /home/ma-user/linevul/
#RUN DEBIAN_FRONTEND=noninteractive apt-get -y install nvidia-driver-440
USER ma-user


WORKDIR /home/ma-user/linevul

ENTRYPOINT ["/bin/bash"]
CMD ["/home/ma-user/linevul/run_train.sh"]
