"""
All objects present in the game, including the players.

@author: Jeroen Vlek <j.vlek@anchormen.nl>
"""
import math

import numpy as np
import pygame
import pymunk

from constants import *

pygame.font.init()


class ScoreCounter(object):
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

    def __init__(self, pos):
        self.pos = pos
        self.score = 0
        self.text = ScoreCounter.font.render(str(self.score), False, OBJECT_COLOR)

    def reset(self):
        self.set_score(0)

    def set_score(self, score):
        self.score = score
        self.text = ScoreCounter.font.render(str(self.score), False, OBJECT_COLOR)

    def add_goal(self):
        self.set_score(self.score + 1)


class GoalPost(pymunk.Segment):

    def __init__(self, body, first_pos, second_pos, radius, counter):
        super().__init__(body, first_pos, second_pos, radius)
        self.elasticity = GOAL_ELASTICITY
        self.friction = GOAL_FRICTION
        self.collision_type = COLLTYPE_GOAL

        self.counter = counter

    def reset(self):
        self.counter.reset()

    @staticmethod
    def goal_scored_handler(arbiter, space, data):
        _, goal = arbiter.shapes
        goal.counter.add_goal()


class PushBody(object):
    """
    Used to push the logo, either via mouse or through pose skeletons
    """

    def __init__(self, pos):
        inertia = pymunk.moment_for_circle(PUSH_BODY_MASS, 0, PUSH_BODY_RADIUS, (0, 0))

        self.body = pymunk.Body(PUSH_BODY_MASS, inertia, pymunk.Body.KINEMATIC)
        self.body.position = pos
        self.body.velocity_func = PushBody.limit_velocity

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
        interpolated_v = (new_v + self.body.velocity) / 2
        # self.body.position = new_pos
        self.body.velocity = interpolated_v

    @staticmethod
    def limit_velocity(body, gravity, damping, dt):
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        length = body.velocity.length
        if length > PUSH_BODY_MAX_V:
            scale = PUSH_BODY_MAX_V / length
            body.velocity = body.velocity * scale


class Player(object):

    def __init__(self, space):
        self.space = space
        self.right_hand = None
        self.left_hand = None
        self.key_points = None

    def distance(self, other_key_points):
        """
        Compute summed distance to other key points
        """
        distance = math.inf
        if other_key_points[NECK_IDX][2] > 0:
            neck_pos = np.array(self.key_points[NECK_IDX][0:1])
            other_neck_pos = np.array(other_key_points[NECK_IDX][0:1])
            distance = np.linalg.norm(neck_pos - other_neck_pos)

        return distance

    def update_pose(self, new_key_points, dt):

        right_hand_pos = Player.extrapolate_hand_position(new_key_points, RIGHT_WRIST_IDX, RIGHT_ELBOW_IDX)
        if right_hand_pos is not None:

            if self.right_hand:
                self.right_hand.move(right_hand_pos, dt)
            else:
                self.right_hand = PushBody(right_hand_pos)
                self.space.add(self.right_hand.body, self.right_hand.shape)
        else:
            # No right hand found. Removing PushObject if it exists
            self.remove_right_hand()

        left_hand_pos = Player.extrapolate_hand_position(new_key_points, LEFT_WRIST_IDX, LEFT_ELBOW_IDX)
        if left_hand_pos is not None:
            if self.left_hand:
                self.left_hand.move(left_hand_pos, dt)
            else:
                self.left_hand = PushBody(left_hand_pos)
                self.space.add(self.left_hand.body, self.left_hand.shape)
        else:
            # No left hand found. Removing PushObject if it exists
            self.remove_left_hand()

        self.key_points = new_key_points

    @staticmethod
    def extrapolate_hand_position(key_points, wrist_id, elbow_id):
        if key_points[wrist_id][2] <= 0 or key_points[elbow_id][2] <= 0:
            return None

        wrist_pos = pymunk.Vec2d(int(key_points[wrist_id][0]), int(key_points[wrist_id][1]))
        elbow_pos = pymunk.Vec2d(int(key_points[elbow_id][0]), int(key_points[elbow_id][1]))

        forearm_vec = wrist_pos - elbow_pos  # went with vec instead of pos, because it's a direction

        # the 0.5 is to make sure it's the middle of the hand
        hand_vec = forearm_vec + 0.25 * HAND_FOREARM_RATIO * forearm_vec
        hand_pos = elbow_pos + hand_vec

        return hand_pos

    def remove_left_hand(self):
        if self.left_hand:
            self.space.remove(self.left_hand.shape, self.left_hand.body)
            self.left_hand = None

    def remove_right_hand(self):
        if self.right_hand:
            self.space.remove(self.right_hand.shape, self.right_hand.body)
            self.right_hand = None

    def destroy(self):
        self.remove_left_hand()
        self.remove_right_hand()


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
        """
        inertia = pymunk.moment_for_box(LOGO_MASS, rect.size)
        body = pymunk.Body(LOGO_MASS, inertia)
        body.position = rect.center

        box = pymunk.Poly.create_box(body, rect.size, LOGO_RADIUS)
        box.elasticity = LOGO_ELASTICITY
        box.friction = LOGO_FRICTION
        box.collision_type = COLLTYPE_LOGO

        return box
