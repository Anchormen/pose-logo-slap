"""

@author: Jeroen Vlek <j.vlek@anchormen.nl>
"""
import argparse

"""This example spawns (bouncing) balls randomly on a L-shape constructed of 
two segment shapes. Not interactive.
"""

__version__ = "$Id:$"
__docformat__ = "reStructuredText"

# Python imports
import random

# Library imports
import pygame
from pygame.key import *
from pygame.locals import *
from pygame.color import *

# pymunk imports
import pymunk
import pymunk.pygame_util

import numpy as np
from openpose import pyopenpose
import camera


class PoseLogoSlap(object):

    def __init__(self, screen_dims, model_path, cam):
        # Space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -900.0)

        # Physics
        # Time step
        self.dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self.physics_steps_per_frame = 1

        # Setup OpenPose
        params = dict()
        params["model_folder"] = model_path
        op = pyopenpose.WrapperPython()
        op.configure(params)
        op.start()
        self.op = op

        # pygame
        pygame.init()
        self.screen_dims = screen_dims
        self.screen = pygame.display.set_mode(self.screen_dims)
        self.clock = pygame.time.Clock()

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.camera = cam
        self.update_background()

        self.running = True

    def run(self):
        """
        The main loop of the game.
        :return: None
        """
        # Main loop
        while self.running:
            # Progress time forward
            for x in range(self.physics_steps_per_frame):
                self.space.step(self.dt)

            self.process_events()
            self.clear_screen()
            self.update_background()
            # TODO update pose
            # TODO update logo
            self.draw_objects()

            pygame.display.flip()
            # Delay fixed time between frames
            self.clock.tick(50)
            pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

    def process_events(self):
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self.running = False

    def clear_screen(self):
        """
        Clears the screen.
        :return: None
        """
        self.screen.fill(THECOLORS["white"])
        self.update_background()

    def draw_objects(self):
        """
        Draw the objects.
        :return: None
        """
        # self.space.debug_draw(self.draw_options)

    def update_background(self):
        _, frame = self.camera.read()
        datum = pyopenpose.Datum()
        datum.cvInputData = frame
        self.op.emplaceAndPop([datum])

        out = datum.cvOutputData
        out = np.swapaxes(out, 0, 1).astype(np.uint8)
        out = np.flip(out, axis=2)
        pygame.surfarray.blit_array(self.screen, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A live emotion recognition from webcam')
    parser.add_argument('--cam_id', default=0, type=int, choices=[0, 1, 2],
                        help='Camera ID, 0 = built-in, 1 = external')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second')
    parser.add_argument('--width', type=int, default=640, help='Capture and display width')
    parser.add_argument('--height', type=int, default=480, help='Capture and display height')
    parser.add_argument('--fullscreen', action='store_true', dest='fullscreen',
                        help='If provided, displays in fullscreen')
    parser.add_argument("--model_path", default="/opt/openpose/models/", help="Path to the model directory")
    args = parser.parse_args()

    cam = camera.get_camera_streaming(args.width, args.height)
    game = PoseLogoSlap((args.width, args.height), args.model_path, cam)
    game.run()
