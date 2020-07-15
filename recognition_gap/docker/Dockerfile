FROM bethgelab/deeplearning:cuda9.0-cudnn7

RUN pip3 --no-cache-dir install \
    tensorboardX \
    http://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-linux_x86_64.whl \
    torchvision==0.3.0\
    -U protobuf \
    scikit-image # online datageneration   
RUN apt-get update -qq \
&& DEBIAN_FRONTEND=noninteractive apt-get install -yq -qq --no-install-recommends \
    python3-tk \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*