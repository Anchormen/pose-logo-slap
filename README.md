# Introduction
This project is for Marketing and Personal Development purposes. It is **not** production-grade code, if you use this for 
anything important, you are insane.

A game for slapping a logo through the screen

# How to build

The following command will build an image with the tag poselogoslap:latest. There are 2 flavors, CPU or GPU (Cuda10):

`docker build -t poselogoslap:latest https://github.com/Anchormen/pose-logo-slap.git/raw/master/Dockerfile.cpu`

or for GPU:

`docker build -t poselogoslap:latest https://github.com/Anchormen/pose-logo-slap.git/raw/master/Dockerfile.gpu`

# How to run

From your host system run the following to add all users to X:

`xhost local:` or `xhost +`

Run the following command for starting the image build under the poselogoslap:latest, the CPU version requires the repository to be mounted as a volume under /opt/anchormen

`docker run -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/snd:/dev/snd -v ${REPO_PATH}/pose-logo-slap:/opt/anchormen -e DISPLAY=$DISPLAY --device=/dev/video0:/dev/video0 -it pose-logo-slap:latest`

or for GPU:

`docker run --runtime=nvidia --ipc=host -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/snd:/dev/snd  -e DISPLAY=$DISPLAY --device=/dev/video0:/dev/video0 -it pose-logo-slap:latest`

The following part: `/tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY`, is for mapping the display from docker to the host screen.

Now you should be in the container's shell.

Start the game:

`python3 game.py`

