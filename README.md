# Introduction
This project is for Marketing and Personal Development purposes. 

A game for slapping a logo through the screen

# How to build

The following command will build an image with the tag poselogoslap:latest. There are 2 flavors, CPU or GPU (Cuda10):

`docker build -t poselogoslap:latest https://gitlab.anchormen.nl/MarketingDS/models/pose-logo-slap/raw/master/Dockerfile.cpu`

or for GPU:

`docker build -t poselogoslap:latest https://gitlab.anchormen.nl/MarketingDS/models/pose-logo-slap/raw/master/Dockerfile.gpu`

# How to run

From your host system run the following to add all users to X:

`xhost local:` or `xhost +`

Run the following command for starting the image build under the poselogoslap:latest, the CPU version requires the repository to be mounted as a volume under /opt/anchormen

`docker run -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/snd:/dev/snd -v ${REPO_PATH}/pose-logo-slap:/opt/anchormen -e DISPLAY=$DISPLAY --device=/dev/video0:/dev/video0 -it pose-logo-slap:latest bash`

or for GPU:

`docker run --runtime=nvidia -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/snd:/dev/snd  -e DISPLAY=$DISPLAY --device=/dev/video0:/dev/video0 -it pose-logo-slap:latest bash`

The following part: `/tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY`, is for mapping the display from docker to the host screen.

Now you should be in the container's shell.

See if the graphics system works by typing

`xeyes`

Now you should see 2 eyes following your mouse pointer. Kill it and start the game:

`python3 game.py`

TODO JV: It's unclear if xeyes is necessary anymore, or at all. I did specify two extra environment variables that seemed to fix it.


