# Use CUDA 10 because that's only supported by the RTX2070
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# For git-lfs
RUN apt-get update
RUN apt-get install -y \
    sudo \
    wget \
    curl \
    git \
    python3-pip \
    cmake \
    build-essential \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libboost-thread-dev \
    libboost-python-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgflags-dev \
    libhdf5-dev \
    libgoogle-glog-dev \
    libgtk2.0-dev \
    libcanberra-gtk-module \
    liblmdb-dev \
    libopenblas-dev \
    libatlas-base-dev \
    libleveldb-dev \
    libsnappy-dev \
    x11-apps \
    pkg-config

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

RUN ln -s  /usr/lib/x86_64-linux-gnu/libboost_python-py35.so /usr/lib/x86_64-linux-gnu/libboost_python3.so
ENV BOOST_LIBRARYDIR /usr/lib/x86_64-linux-gnu/
ENV BOOST_ROOT /usr/lib/x86_64-linux-gnu/

RUN pip3 install --upgrade pip
# Make pip > pip3 needed for the file: install_cmake.sh
RUN echo 'alias pip="pip3"' >> ~/.bashrc

# CHECKOUT THIS REPO (Inception, I know)
WORKDIR /opt/anchormen
RUN git clone https://gitlab.anchormen.nl/MarketingDS/models/pose-slap.git
WORKDIR pose-slap
RUN pip install -r requirements.txt

########### OPENCV ###########
WORKDIR /opt
RUN git clone https://github.com/opencv/opencv.git
RUN git clone https://github.com/opencv/opencv_contrib.git && cd opencv_contrib && git checkout 3.4 && cd ..
RUN cd opencv && git checkout 3.4 && mkdir build && cd build && \
      cmake -DCMAKE_BUILD_TYPE=RELEASE \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      .. && make -j`nproc` && sudo make install && sudo sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf' && sudo ldconfig

RUN ln -s /opencv/build/lib/python3/cv2.cpython-35m-x86_64-linux-gnu.so /usr/local/lib/python3.5/dist-packages/cv2.so

########### OPENPOSE & CAFFE ###########

RUN rm /etc/apt/sources.list.d/nvidia-ml.list && rm /etc/apt/sources.list.d/cuda.list
ENV OPENPOSE_ROOT=/opt/openpose
RUN git clone -b feature/anchormen-hacks --single-branch https://gitlab.anchormen.nl/MarketingDS/openpose.git $OPENPOSE_ROOT


# build included Caffe
RUN echo "hoi"
WORKDIR $OPENPOSE_ROOT/3rdparty/caffe
RUN git clone -b feature/work-for-openpose --single-branch https://gitlab.anchormen.nl/MarketingDS/caffe-openpose.git .

# download models
WORKDIR $OPENPOSE_ROOT/models
RUN bash ./getModels.sh

# build OpenPose
WORKDIR $OPENPOSE_ROOT
RUN mkdir build
WORKDIR build
RUN cmake .. -DWITH_PYTHON3=True
# not using all available threads due to OOM error: https://github.com/yjxiong/temporal-segment-networks/issues/173
RUN make -j 16
RUN make install

########### PYOPENPOSE ###########
ENV PYOPENPOSE_ROOT=/opt/PyOpenPose
WORKDIR /opt
RUN git clone -b feature/anchormen-hacks --single-branch  https://gitlab.anchormen.nl/MarketingDS/PyOpenPose.git
WORKDIR $PYOPENPOSE_ROOT

RUN mkdir build
WORKDIR build
RUN cmake .. -DWITH_PYTHON3=True
RUN make -j 16
ENV PYTHONPATH /opt/anchormen/pose-slap
ENV PYTHONPATH $PYOPENPOSE_ROOT/build/PyOpenPoseLib/:$PYTHONPATH

########### LET HER RIP ############
RUN apt-get install git-lfs
WORKDIR /opt/anchormen/face-interpretation-with-pose

RUN git lfs pull

ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/

# This is important because the demo is Qt4 based and otherwise you'll get a weird X server error
# https://stackoverflow.com/a/35040140/543720
# https://stackoverflow.com/a/55989420/543720
ENV QT_GRAPHICSSYSTEM "native"
ENV QT_X11_NO_MITSHM 1

#ENTRYPOINT ["python3", "demo.py"]
