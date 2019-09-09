"""
A game to slap a logo in the other player's goal using your arms/hands as detected by OpenPose

@author: Jeroen Vlek <j.vlek@anchormen.nl>
"""

import argparse
import random
import pygame
from pygame.locals import *
from pygame.color import *
import pymunk
import pymunk.pygame_util
import numpy as np
import camera
import cv2

from constants import *
from game_entities import ScoreCounter, GoalPost, PushBody, Player, Logo
from pose_estimator import PoseEstimator

pymunk.pygame_util.positive_y_is_up = False

class PoseLogoSlapGame(object):
    """
    Main game class
    """

    def __init__(self, screen_dims, image_path, pose_estimator, camera, gpu_mode, debug_mode):
        # Physics
        self.space = pymunk.Space()
        # self.space.gravity = (0.0, 600.0)
        self.space.damping = DAMPING
        self.dt = DT
        self.physics_steps_per_frame = 1
        self.space.add_collision_handler(COLLTYPE_LOGO, COLLTYPE_GOAL).separate = GoalPost.goal_scored_handler

        # PyGame
        pygame.init()
        self.screen_dims = screen_dims
        self.screen = pygame.display.set_mode(self.screen_dims)
        self.clock = pygame.time.Clock()

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        # Setup logo
        self.image_path = image_path
        self.logo = None
        self.init_logo()

        pygame.display.set_icon(self.logo.image)

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

        # dynamic objects
        self.test_push_body = None
        self.players = set()

        # Setup pose estimator
        self.pose_estimator = pose_estimator
        self.camera = camera
        self.background = None
        self.original_frame = None

        self.gpu_mode = gpu_mode
        self.debug_mode = debug_mode
        self.running = True

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

        print("Starting game loop")
        while self.running:
            # Progress time forward
            for x in range(self.physics_steps_per_frame):
                self.space.step(self.dt)

            self.process_events()
            self.get_new_frame()
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

        if self.background is not None:
            pygame.surfarray.blit_array(self.screen, self.background)
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
        datum = self.pose_estimator.grab_pose(self.original_frame)
        num_poses = len(datum.poseKeypoints) if datum.poseKeypoints.ndim > 0 else 0
        print("Number of poses detected: " + str(num_poses))
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
        print("Removing " + str(len(old_players)) + " players")
        for old_player in old_players:
            old_player.destroy()

        print("Keeping/adding " + str(len(new_players)))
        self.players = new_players

        self.background = PoseLogoSlapGame.convert_array_to_pygame_layout(datum.cvOutputData)

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
        print("Resetting game, previous scores:")
        print("Left team scored " + str(self.right_goal.counter.score))
        print("Right team scored " + str(self.left_goal.counter.score))

        self.right_goal.reset()
        self.left_goal.reset()

        for player in self.players:
            player.destroy()

        self.players = set()
        self.space.remove(self.logo.box, self.logo.box.body)
        self.init_logo()

    def get_new_frame(self):
        _, frame = self.camera.read()
        flipped = cv2.flip(frame, 1)
        self.original_frame = flipped
        self.background = PoseLogoSlapGame.convert_array_to_pygame_layout(flipped)

    @staticmethod
    def convert_array_to_pygame_layout(img_array):
        img_array = np.swapaxes(img_array, 0, 1).astype(np.uint8)
        img_array = np.flip(img_array, axis=2)
        return img_array


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
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    camera = camera.get_camera_streaming(args.width, args.height)
    pose_estimator = PoseEstimator(args.model_path)
    game = PoseLogoSlapGame((args.width, args.height), args.image_path, pose_estimator, camera, args.gpu, args.debug)

    game.run()
