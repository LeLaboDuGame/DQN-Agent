import random as rnd

import gym
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from copy import deepcopy

import dqn_agent

env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=42)

maps = [[1, 1, 1, 1],
        [-10, -10, -10, 5],
        [10, 1, 1, 1]]


class Game:
    def __init__(self, maps):
        self.maps = maps
        self.actual_maps = None
        self.x = 0
        self.y = 0

    def get_state(self):
        return [self.x, self.y] + list(np.array(self.actual_maps).flatten())

    def reset(self):
        self.actual_maps = deepcopy(self.maps.copy())
        self.x, self.y = rnd.randint(0, 3), rnd.randint(0, 2)
        return self.get_state()

    def play(self, action):
        actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        self.x += actions[action][0]
        self.y += actions[action][1]

        if self.x < 0 or self.y < 0 or self.x > 3 or self.y > 2:
            return self.get_state(), -10, True

        done = False
        if self.actual_maps[self.y][self.x] == -10 or self.actual_maps[self.y][self.x] == 10:
            done = True

        reward = self.actual_maps[self.y][self.x]

        self.actual_maps[self.y][self.x] = 0
        #print("MAP: ", self.actual_maps, " | ", self.maps)
        return self.get_state(), reward, done


game = Game(maps)

layers = [Dense(14, activation="relu"), Dense(64, activation="relu"), Dense(128, activation="relu"),
          Dense(4, activation="relu")]
agent = dqn_agent.DQNAgent(layers, action_size=4, learning_rate=0.01, loss="MSE", optimizer=Adam(learning_rate=0.001),
                           weights=None,
                           batch_size=100, max_memory_length=1000, gamma=0.95, epsylone=1, epsylone_decrease=0.997,
                           epsylone_min=0.1)

agent.train_agent(game.play, game.reset, n_iter=500, n_step_before_training=10)

agent.main_model.save("./Models/model.keras")

print("-------------------\n\n")
game.reset()
game.x = 0
game.y = 0
done = False
nmaps = game.maps.copy()
nmaps[game.y][game.x] = "X"
print(f"step 0:\n"
      f"{nmaps}")
i = 1
while not done:
    state = game.get_state()
    new_state, reward, done = game.play(np.argmax(agent.main_model.predict(np.array([state]))))

    nmaps = game.maps.copy()
    nmaps[game.y][game.x] = "X"
    print(f"step {i}:\n"
          f"{nmaps}")
    i += 1

print("Party finish !")
