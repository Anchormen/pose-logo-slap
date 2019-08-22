"""
A game to slap a logo in the other player's goal using your arms/hands as detected by OpenPose

@author: Jeroen Vlek <j.vlek@anchormen.nl>
"""

import argparse
import random
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
import numpy as np
from openpose import pyopenpose
import camera

LOGO_MASS = 10
LOGO_FRICTION = 0.9
LOGO_ELASTICITY = 0.95


class Logo(pygame.sprite.Sprite):
    """
    The logo or "ball" with which to be scored
    """
    def __init__(self, spawn_point, image_path):
        _box = self.create_logo_box()
        self.image = pygame.image.load(image_path)
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = spawn_point
        self.box = Logo.create_logo_box(self.rect)

    @staticmethod
    def create_logo_box(rect):
        """
        Create the properies for the PyMunk physics engine to use this sprite.
        :return:
        """
        inertia = pymunk.moment_for_box(LOGO_MASS, rect.size)
        body = pymunk.Body(LOGO_MASS, inertia)
        body.position = rect.center

        radius = max(rect.width, rect.height)
        box = pymunk.Poly(body, rect.size, radius)
        box.elasticity = LOGO_ELASTICITY
        box.friction = LOGO_FRICTION

        return box


class PoseLogoSlapGame(object):
    """
    Main game class
    """

    def __init__(self, screen_dims, model_path, cam, image_path):
        # Space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -900.0)

        # Physics
        # Time step
        self.dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self.physics_steps_per_frame = 1

        # pygame
        pygame.init()
        self.screen_dims = screen_dims
        self.screen = pygame.display.set_mode(self.screen_dims)
        self.clock = pygame.time.Clock()

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        # Setup OpenPose
        # params = dict()
        # params["model_folder"] = model_path
        # op = pyopenpose.WrapperPython()
        # op.configure(params)
        # op.start()
        # self.op = op
        # self.camera = cam
        # self.background = None
        # self.update_pose()

        # Setup logo
        mid_point = (self.screen_dims[0] / 2, self.screen_dims[1] / 2)
        quarter_screen_dims = (mid_point[0] / 2, mid_point[1] / 2)
        x = random.randint(mid_point[0] - quarter_screen_dims, mid_point[0] + quarter_screen_dims)
        y = random.randint(mid_point[1] - quarter_screen_dims, mid_point[1] + quarter_screen_dims)
        self.logo = Logo((x, y), image_path)
        self._space.add(self.logo.box.body, self.logo.box)

        self.running = True

    def run(self):
        """
        The main loop of the game.
        :return: None
        """

        print("Starting game loop")
        while self.running:
            # Progress time forward
            for x in range(self.physics_steps_per_frame):
                self.space.step(self.dt)

            self.process_events()
            self.clear_screen()
            # self.update_pose()
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
            # elif event.type == KEYDOWN and event.key == K_SPACE:
            #     print("Updating pose")
            #     self.update_pose()

    def clear_screen(self):
        """
        Clears the screen.
        :return: None
        """
        # pygame.surfarray.blit_array(self.screen, self.background)

    def draw_objects(self):
        """
        Draw the objects.
        :return: None
        """
        # self.space.debug_draw(self.draw_options)

    def update_pose(self):
        _, frame = self.camera.read()
        datum = pyopenpose.Datum()
        datum.cvInputData = frame
        self.op.emplaceAndPop([datum])

        out = datum.cvOutputData
        out = np.swapaxes(out, 0, 1).astype(np.uint8)
        out = np.flip(out, axis=2)
        self.background = out


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
    parser.add_argument("--image_path", default="/opt/anchormen/logo.png", help="Path to the logo")
    args = parser.parse_args()

    cam = camera.get_camera_streaming(args.width, args.height)
    game = PoseLogoSlapGame((args.width, args.height), args.model_path, cam, args.image_path)
    game.run()
