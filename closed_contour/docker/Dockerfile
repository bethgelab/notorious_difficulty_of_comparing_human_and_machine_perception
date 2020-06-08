FROM bethgelab/deeplearning:cuda9.0-cudnn7

RUN pip3 --no-cache-dir install \
    tensorboardX \
    http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl \
    torchvision==0.2.1 \
    -U protobuf \
    scikit-image \
    foolbox==1.8.0 \
    seaborn==0.9.0   
RUN sudo add-apt-repository -r ppa:jonathonf/python-3.6
RUN apt-get update -qq \
&& DEBIAN_FRONTEND=noninteractive apt-get install -yq -qq --no-install-recommends \
    python3-tk \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

