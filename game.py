"""
A game to slap a logo in the other player's goal using your arms/hands as detected by OpenPose

@author: Jeroen Vlek <j.vlek@anchormen.nl>
"""

import argparse
import logging
import logging.config
import random

import numpy as np
import pygame
import pygame.camera
import pymunk
import pymunk.pygame_util
from pygame.color import *
from pygame.locals import *

import camera
from constants import *
from entities import ScoreCounter, GoalPost, PushBody, Player, Logo
from pose_estimator import PoseEstimator

logging.config.fileConfig('logging.conf')

pymunk.pygame_util.positive_y_is_up = False


def convert_array_to_pygame_layout(img_array):
    img_array = np.swapaxes(img_array, 0, 1).astype(np.uint8)
    img_array = np.flip(img_array, axis=2)
    return img_array


class PoseLogoSlapGame(object):
    """
    Main game class
    """

    def __init__(self, screen_dims, image_path, pose_estimator, frame_grabber, gpu_mode, debug_mode):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Physics
        self.space = pymunk.Space()
        # self.space.gravity = (0.0, 600.0)
        self.space.damping = DAMPING
        self.space.add_collision_handler(COLLTYPE_LOGO, COLLTYPE_GOAL).separate = GoalPost.goal_scored_handler

        # PyGame
        pygame.init()
        self.screen_dims = screen_dims
        self.screen = pygame.display.set_mode(self.screen_dims)
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.image_path = image_path
        self.logo = None

        self.test_push_body = None
        self.players = set()

        # Setup pose estimator
        self.pose_estimator = pose_estimator
        self.frame_grabber = frame_grabber
        self.output_frame = None
        self.pose_input_frame = None

        self.gpu_mode = gpu_mode
        self.debug_mode = debug_mode
        self.running = True
        self.fullscreen = False

    def init_game(self):
        self.setup_screen_bounds(screen_dims)

        # the right counter, is updated when the left goal gets a goal and vice versa
        right_counter = ScoreCounter((screen_dims[0] - 2 * COUNTER_MARGIN, COUNTER_MARGIN))
        left_counter = ScoreCounter((COUNTER_MARGIN, COUNTER_MARGIN))

        static_body = self.space.static_body
        goal_length = screen_dims[1] * RELATIVE_GOAL_SIZE
        self.left_goal = GoalPost(static_body, (GOAL_MARGIN, (screen_dims[1] / 2 - goal_length / 2)),
                                  (GOAL_MARGIN, (screen_dims[1] / 2 + goal_length / 2)), GOAL_MARGIN, right_counter)
        self.right_goal = GoalPost(static_body, (screen_dims[0] - GOAL_MARGIN, (screen_dims[1] / 2 - goal_length / 2)),
                                   (screen_dims[0] - GOAL_MARGIN, (screen_dims[1] / 2 + goal_length / 2)),
                                   GOAL_MARGIN, left_counter)
        self.space.add([self.left_goal, self.right_goal])

        self.init_logo()
        pygame.display.set_icon(self.logo.image)

    def init_logo(self):
        mid_point = (self.screen_dims[0] / 2, self.screen_dims[1] / 2)
        quarter_screen_dims = (mid_point[0] / 2, mid_point[1] / 2)
        x = random.randint(mid_point[0] - quarter_screen_dims[0], mid_point[0] + quarter_screen_dims[0])
        y = random.randint(mid_point[1] - quarter_screen_dims[1], mid_point[1] + quarter_screen_dims[1])
        self.logo = Logo(pymunk.Vec2d(x, y), self.image_path)
        self.space.add(self.logo.box.body, self.logo.box)

    def setup_screen_bounds(self, screen_dims):
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

    def run(self):
        """
        The main loop of the game.
        :return: None
        """

        self.logger.info("Starting game loop")
        while self.running:
            # Progress time forward
            for _ in range(PHYSICS_STEPS_PER_FRAME):
                self.space.step(DT)

            self.process_events()
            self.load_new_frame()
            if self.gpu_mode:
                self.update_poses()
            self.logo.update()

            self.clear_screen()
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
            elif event.type == KEYDOWN and event.key == K_r:
                self.reset_game()
            elif event.type == KEYDOWN and event.key == K_d:
                self.debug_mode = not self.debug_mode
            elif event.type == KEYDOWN and event.key == K_f:
                if self.fullscreen:
                    pygame.display.set_mode(self.screen_dims)
                    self.fullscreen = False
                else:
                    pygame.display.set_mode(self.screen_dims, flags=FULLSCREEN)
                    self.fullscreen = True
            elif event.type == MOUSEBUTTONDOWN:
                if not self.test_push_body:
                    pos = pygame.mouse.get_pos()
                    self.test_push_body = PushBody(pymunk.Vec2d(pos[0], pos[1]))
                    self.space.add(self.test_push_body.body, self.test_push_body.shape)
            elif event.type == MOUSEMOTION:
                if self.test_push_body:
                    pos = pygame.mouse.get_pos()
                    new_pos = pymunk.Vec2d(pos[0], pos[1])
                    self.test_push_body.move(new_pos, self.dt)
            elif event.type == MOUSEBUTTONUP:
                if self.test_push_body:
                    self.space.remove(self.test_push_body.shape, self.test_push_body.body)
                self.test_push_body = None
            elif event.type == KEYDOWN and event.key == K_SPACE:
                self.update_poses()

    def clear_screen(self):
        """
        Clears the screen.
        :return: None
        """

        if self.pose_input_frame is not None:
            # Input frame has not been processed yet
            self.output_frame = convert_array_to_pygame_layout(self.pose_input_frame)

        if self.output_frame is not None:
            pygame.surfarray.blit_array(self.screen, self.output_frame)
        else:
            self.screen.fill(THECOLORS["white"])

    def draw_objects(self):
        """
        Draw the objects.
        :return: None
        """

        if self.debug_mode:
            self.space.debug_draw(self.draw_options)

        self.screen.blit(self.logo.image, self.logo.rect.topleft)
        self.screen.blit(self.left_goal.counter.text, self.left_goal.counter.pos)
        self.screen.blit(self.right_goal.counter.text, self.right_goal.counter.pos)

        pygame.draw.line(self.screen, OBJECT_COLOR, self.right_goal.a, self.right_goal.b, GOAL_MARGIN)
        pygame.draw.line(self.screen, OBJECT_COLOR, self.left_goal.a, self.left_goal.b, GOAL_MARGIN)

    def update_poses(self):
        if self.pose_input_frame is None:
            return

        datum = self.pose_estimator.grab_pose(self.pose_input_frame)
        self.pose_input_frame = None

        num_poses = len(datum.poseKeypoints) if datum.poseKeypoints.ndim > 0 else 0
        self.logger.debug("Number of poses detected: %d", num_poses)
        if num_poses == 0:
            if len(self.players) > 0:
                self.reset_game()
            return

        new_players = set()
        for pose in datum.poseKeypoints:
            player = self.find_nearest_player(pose)
            if not player:
                player = Player(self.space)

            player.update_pose(pose, self.dt)
            new_players.add(player)

        old_players = self.players - new_players
        self.logger.debug("Removing " + str(len(old_players)) + " players")
        for old_player in old_players:
            old_player.destroy()

        self.logger.debug("Keeping/adding " + str(len(new_players)))
        self.players = new_players

        self.output_frame = convert_array_to_pygame_layout(datum.cvOutputData)

    def find_nearest_player(self, pose):
        nearest_player = None
        closest_distance = MAX_DISTANCE_THRESHOLD
        for player in self.players:
            distance = player.distance(pose)
            if distance < closest_distance:
                nearest_player = player
                closest_distance = distance

        return nearest_player

    def reset_game(self):
        self.logger.debug("Resetting game, previous scores:")
        self.logger.debug("Left team scored " + str(self.right_goal.counter.score))
        self.logger.debug("Right team scored " + str(self.left_goal.counter.score))

        self.right_goal.reset()
        self.left_goal.reset()

        for player in self.players:
            player.destroy()

        self.players = set()
        self.space.remove(self.logo.box, self.logo.box.body)
        self.init_logo()

    def load_new_frame(self):
        frame = self.frame_grabber.pop_frame()
        if frame is None:
            return

        self.pose_input_frame = frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A silly game based on OpenPose')
    parser.add_argument('--cam_id', default="0", type=int, help='Camera id (0 = built-in, 1 = external)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--width', type=int, default=1280, help='Capture and display width')
    parser.add_argument('--height', type=int, default=720, help='Capture and display height')
    parser.add_argument('--fullscreen', action='store_true', dest='fullscreen',
                        help='If provided, displays in fullscreen')
    parser.add_argument("--model_path", default="/opt/openpose/models/", help="Path to the model directory")
    parser.add_argument("--image_path", default="/opt/anchormen/logo.png", help="Path to the logo")
    parser.add_argument("--net_resolution", default="-1x368", help="Net resolution, see openpose -> flags.hpp")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info(args)

    screen_dims = (args.width, args.height)
    pose_estimator = PoseEstimator(args.model_path, args.net_resolution)
    grabber = camera.FrameGrabber(screen_dims[0], screen_dims[1], args.cam_id, args.fps)
    grabber.start()

    game = PoseLogoSlapGame(screen_dims, args.image_path, pose_estimator, grabber, args.gpu, args.debug)
    game.init_game()
    game.run()

    grabber.stop()
