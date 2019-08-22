"""
A game to slap a logo in the other player's goal using your arms/hands as detected by OpenPose

@author: Jeroen Vlek <j.vlek@anchormen.nl>
"""

import argparse
import random
import math
import pygame
from pygame.locals import *
from pygame.color import *
import pymunk
import pymunk.pygame_util
import numpy as np
import camera
from pose_estimator import PoseEstimator

LOGO_RADIUS = 2
LOGO_MASS = 5
LOGO_FRICTION = 0.95
LOGO_ELASTICITY = 1.0
LOGO_SIZE = (60, 60)

pymunk.pygame_util.positive_y_is_up = False


class Logo(pygame.sprite.Sprite):
    """
    The logo or "ball" with which to be scored
    """

    def __init__(self, spawn_point, image_path, logo_size=LOGO_SIZE):
        raw_image = pygame.image.load(image_path)
        self.image = self.original_image = pygame.transform.scale(raw_image, logo_size)
        self.rect = self.image.get_rect(center=spawn_point)
        self.box = Logo.create_logo_box(self.rect)

    def update(self):
        self.image = pygame.transform.rotate(self.original_image, math.degrees(-self.box.body.angle))
        self.rect = self.image.get_rect(center=self.box.body.position)

    @staticmethod
    def create_logo_box(rect):
        """
        Create the properies for the PyMunk physics engine to use this sprite.
        :return:
        """
        inertia = pymunk.moment_for_box(LOGO_MASS, rect.size)
        body = pymunk.Body(LOGO_MASS, inertia)
        body.position = rect.center

        box = pymunk.Poly.create_box(body, rect.size, LOGO_RADIUS)
        box.elasticity = LOGO_ELASTICITY
        box.friction = LOGO_FRICTION

        return box


class PoseLogoSlapGame(object):
    """
    Main game class
    """

    def __init__(self, screen_dims, image_path, pose_estimator):
        # Physics
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 900.0)
        self.dt = 1.0 / 60.0
        self.physics_steps_per_frame = 1

        # PyGame
        pygame.init()
        self.screen_dims = screen_dims
        self.screen = pygame.display.set_mode(self.screen_dims)
        self.clock = pygame.time.Clock()

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        # Setup logo
        mid_point = (self.screen_dims[0] / 2, self.screen_dims[1] / 2)
        quarter_screen_dims = (mid_point[0] / 2, mid_point[1] / 2)
        x = random.randint(mid_point[0] - quarter_screen_dims[0], mid_point[0] + quarter_screen_dims[0])
        y = random.randint(mid_point[1] - quarter_screen_dims[1], mid_point[1] + quarter_screen_dims[1])
        self.logo = Logo((x, y), image_path)
        self.space.add(self.logo.box.body, self.logo.box)

        # Setup bounding box around the screen, the lines start in the top left and go clockwise
        static_body = self.space.static_body
        static_lines = [pymunk.Segment(static_body, (0, 0), (screen_dims[0], 0), 0.0),
                        pymunk.Segment(static_body, (screen_dims[0], 0), (screen_dims[0], screen_dims[1]), 0.0),
                        pymunk.Segment(static_body, (screen_dims[0], screen_dims[1]), (0, screen_dims[1]), 0.0),
                        pymunk.Segment(static_body, (0, screen_dims[1]), (0, 0), 0.0)]
        for line in static_lines:
            line.elasticity = 0.95
            line.friction = 0.9
        self.space.add(static_lines)

        self.test_ball = None

        # Setup pose estimator
        self.pose_estimator = pose_estimator
        self.background = None

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
            self.logo.update()
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
            elif event.type == MOUSEBUTTONDOWN:
                if not self.test_ball:
                    self.create_ball()
            elif event.type == MOUSEMOTION:
                if self.test_ball:
                    self.test_ball.body.position = pygame.mouse.get_pos()
            elif event.type == MOUSEBUTTONUP:
                if self.test_ball:
                    self.space.remove(self.test_ball, self.test_ball.body)
                self.test_ball = None
            elif event.type == KEYDOWN and event.key == K_SPACE:
                self.update_pose()

    def create_ball(self):
        """
        Create a ball.
        :return:
        """
        mass = 100
        radius = 25
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia, pymunk.Body.KINEMATIC)
        body.position = pygame.mouse.get_pos()
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0.95
        shape.friction = 0.9

        self.space.add(body, shape)
        self.test_ball = shape

    def clear_screen(self):
        """
        Clears the screen.
        :return: None
        """

        if self.background is not None:
            pygame.surfarray.blit_array(self.screen, self.background)
        else:
            self.screen.fill(THECOLORS["white"])

    def draw_objects(self):
        """
        Draw the objects.
        :return: None
        """
        self.space.debug_draw(self.draw_options)
        self.screen.blit(self.logo.image, self.logo.rect.topleft)

    def update_pose(self):
        datum = self.pose_estimator.grab_pose()
        out = datum.cvOutputData
        out = np.swapaxes(out, 0, 1).astype(np.uint8)
        out = np.flip(out, axis=2)
        self.background = out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A live emotion recognition from webcam')
    parser.add_argument('--cam_id', default=0, type=int, choices=[0, 1, 2],
                        help='Camera ID, 0 = built-in, 1 = external')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second')
    parser.add_argument('--width', type=int, default=1280, help='Capture and display width')
    parser.add_argument('--height', type=int, default=720, help='Capture and display height')
    parser.add_argument('--fullscreen', action='store_true', dest='fullscreen',
                        help='If provided, displays in fullscreen')
    parser.add_argument("--model_path", default="/opt/openpose/models/", help="Path to the model directory")
    parser.add_argument("--image_path", default="/opt/anchormen/logo.png", help="Path to the logo")
    args = parser.parse_args()

    camera = camera.get_camera_streaming(args.width, args.height)
    pose_estimator = PoseEstimator(args.model_path, camera)
    game = PoseLogoSlapGame((args.width, args.height), args.image_path, pose_estimator)

    game.run()
