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

PUSH_BODY_FRICTION = 0.9
PUSH_BODY_ELASTICITY = 1.0
PUSH_BODY_RADIUS = 50
PUSH_BODY_MASS = 100

COUNTER_MARGIN = 20

GOAL_MARGIN = 5
RELATIVE_GOAL_SIZE = 0.2

LOGO_RADIUS = 2
LOGO_MASS = 5
LOGO_FRICTION = 0.95
LOGO_ELASTICITY = 1.0
LOGO_SIZE = (60, 60)

COLLTYPE_MOUSE = 1
COLLTYPE_LOGO = 2
COLLTYPE_GOAL = 3

# Taken from: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
LEFT_WRIST_IDX = 7
RIGHT_WRIST_IDX = 4

FONT_NAME = 'Comic Sans MS'
FONT_SIZE = 60
FONT_COLOR = (229, 11, 20)

pymunk.pygame_util.positive_y_is_up = False
pygame.font.init()


class ScoreCounter(object):
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

    def __init__(self, pos):
        self.pos = pos
        self.score = 0
        self.text = ScoreCounter.font.render(str(self.score), False, FONT_COLOR)

    def reset(self):
        self.set_score(0)

    def set_score(self, score):
        self.score = score
        self.text = ScoreCounter.font.render(str(self.score), False, FONT_COLOR)

    def add_goal(self):
        self.set_score(self.score + 1)


class GoalPost(pymunk.Segment):

    def __init__(self, body, first_pos, second_pos, radius, counter):
        super().__init__(body, first_pos, second_pos, radius)
        self.elasticity = 1.0
        self.friction = 0.9
        self.collision_type = COLLTYPE_GOAL

        self.counter = counter

    def reset(self):
        self.counter.reset()

    @staticmethod
    def goal_scored_handler(arbiter, space, data):
        _, goal = arbiter.shapes
        goal.counter.add_goal()
        print("Goal scored")


class PushBody(object):
    """
    Used to push the logo, either via mouse or through pose skeletons
    """

    def __init__(self, pos):
        inertia = pymunk.moment_for_circle(PUSH_BODY_MASS, 0, PUSH_BODY_RADIUS, (0, 0))

        self.body = pymunk.Body(PUSH_BODY_MASS, inertia, pymunk.Body.KINEMATIC)
        self.body.position = pos

        self.shape = pymunk.Circle(self.body, PUSH_BODY_RADIUS, (0, 0))
        self.shape.collision_type = COLLTYPE_MOUSE
        self.shape.elasticity = PUSH_BODY_ELASTICITY
        self.shape.friction = PUSH_BODY_FRICTION

    def move(self, new_pos, dt):
        """
        Moves PushBody to new position and calculates new velocity
        """
        old_pos = self.body.position
        new_v = (new_pos - old_pos) / dt
        self.body.position = new_pos
        self.body.velocity = new_v


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
        box.collision_type = COLLTYPE_LOGO

        return box


class PoseLogoSlapGame(object):
    """
    Main game class
    """

    def __init__(self, screen_dims, image_path, pose_estimator, gpu_mode):
        # Physics
        self.space = pymunk.Space()
        # self.space.gravity = (0.0, 600.0)
        self.space.damping = 0.6
        self.dt = 1.0 / 60.0
        self.physics_steps_per_frame = 1
        self.space.add_collision_handler(COLLTYPE_LOGO, COLLTYPE_GOAL).separate = GoalPost.goal_scored_handler

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
        self.logo = Logo(pymunk.Vec2d(x, y), image_path)
        self.space.add(self.logo.box.body, self.logo.box)
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

        self.test_push_body = None

        # Setup pose estimator
        self.pose_estimator = pose_estimator
        self.background = None

        self.gpu_mode = gpu_mode
        self.running = True

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
            self.clear_screen()
            if self.gpu_mode:
                self.update_pose()
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
        self.space.debug_draw(self.draw_options)
        self.screen.blit(self.logo.image, self.logo.rect.topleft)
        self.screen.blit(self.left_goal.counter.text, self.left_goal.counter.pos)
        self.screen.blit(self.right_goal.counter.text, self.right_goal.counter.pos)

    def update_poses(self):
        datum = self.pose_estimator.grab_pose()

        print("Number of poses detected: " + str(len(datum.poseKeypoints)))

        for pose in datum.poseKeypoints:

            right_wrist_pos = None
            if pose[RIGHT_WRIST_IDX][2] > 0:
                right_wrist_pos = (int(pose[RIGHT_WRIST_IDX][0]), int(pose[RIGHT_WRIST_IDX][1]))
                right_ball = self.create_box()
                right_ball.body.position = right_wrist_pos

            left_wrist_pos = None
            if pose[LEFT_WRIST_IDX][2] > 0:
                left_wrist_pos = (int(pose[LEFT_WRIST_IDX][0]), int(pose[LEFT_WRIST_IDX][1]))
                left_ball = self.create_box()
                left_ball.body.position = left_wrist_pos

            print("Right wrist: " + str(right_wrist_pos))
            print("Left wrist: " + str(left_wrist_pos))

        out = datum.cvOutputData
        out = np.swapaxes(out, 0, 1).astype(np.uint8)
        out = np.flip(out, axis=2)
        self.background = out

    def reset_game(self):
        print("Resetting game, previous scores:")
        print("Left player scored " + str(self.right_goal.counter.score))
        print("Right player scored " + str(self.left_goal.counter.score))

        self.right_goal.reset()
        self.left_goal.reset()


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
    parser.add_argument("--gpu", action="store_true", type=bool)
    args = parser.parse_args()

    camera = camera.get_camera_streaming(args.width, args.height)
    pose_estimator = PoseEstimator(args.model_path, camera)
    game = PoseLogoSlapGame((args.width, args.height), args.image_path, pose_estimator, args.gpu)

    game.run()
