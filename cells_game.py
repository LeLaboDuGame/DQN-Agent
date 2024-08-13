import random as rnd
import time

import numpy as np
import pygame
import tensorflow as tf
from pygame.math import Vector2

"""
3 butes:
-nutritions
-reproductions
-explorations/interactions

zone de 100*100
"""


class EnumObjectType:
    cell = 1
    food = 2


class Object:
    def __init__(self, obj_type, color, pos=Vector2(0, 0)):
        self.type = obj_type
        self.pos = pos
        self.color = color


class Engine:
    def __init__(self, grid_size=(200, 200), screen_factor=5):
        pygame.init()
        self.grid_size = grid_size
        self.screen_factor = screen_factor
        self.screen = pygame.display.set_mode((grid_size[0] * screen_factor, grid_size[1] * screen_factor))
        self.grid = np.zeros(grid_size)
        self.objects = []

    def add_object(self, object):
        self.objects.append(object)

    def update_grid(self, update_screen=True):
        self.grid = np.zeros(self.grid_size)
        for obj in self.objects:
            self.grid[int(obj.pos.y)][int(obj.pos.x)] = obj.obj_type
            if update_screen:
                self.update_object(obj)

        if update_screen:
            pygame.display.flip()

    def update_object(self, obj):
        pygame.draw.rect(self.screen, obj.color,
                         pygame.Rect(obj.pos.x * self.screen_factor, obj.pos.y * self.screen_factor, self.screen_factor,
                                     self.screen_factor))

    @staticmethod
    def stop_engine():
        pygame.quit()
        print("Stopping engine !")


class Cell(Object):
    def __init__(self, obj_type, color):
        super().__init__(obj_type, color)
        self.agent = self.generate_model()
        self.food = 10

    def generate_agent(self):
        def dense_layer(num_units):
            return tf.keras.layers.Dense(
                num_units,
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2.0, mode='fan_in', distribution='truncated_normal'))

        dense_layers = [dense_layer(num_units) for num_units in (128, 64)]



        return agent

    def play(self):
        pass


class Game:
    def __init__(self, engine, food_factor=5):
        self.food_factor = food_factor
        self.engine = engine
        self.cell = None
        self.reset()

    """
    Actions:
    0-> up
    1->right
    2->down
    3->left
    
    """

    def play(self):
        pass

    def reset(self):
        self.cell = Object(EnumObjectType.cell, "red", pos=Vector2(50, 50))
        self.engine.add_object(self.cell)
        self.generate_world(self.food_factor)

    def generate_world(self, food_factor):
        for i in range(int(self.engine.grid_size[0] * self.engine.grid_size[1] / 100 * 0.1)):
            self.engine.add_object(
                Object(EnumObjectType.food, "yellow", Vector2(rnd.randint(0, self.engine.grid_size[0] - 1),
                                                              rnd.randint(0, self.engine.grid_size[1] - 1))))


engine = Engine()
engine.update_grid()

game = Game(engine)

engine.update_grid()

time.sleep(5)
engine.stop_engine()
