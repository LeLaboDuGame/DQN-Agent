from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random as rnd
import dqn_agent
import numpy as np




class Game:
    def __init__(self, ):
        self.maps = [0, -1, 0, -1, 0, 0, -1, 0, 0, -1, 0, 2]  # -1 = spike | 1 = player | 2 = finished

    def reset(self):
        self.pos = np.where(self.maps)[0][rnd.randint(0, len(np.where(self.maps)[0]) - 1)]
        state = self.maps.copy()
        state[self.pos] = 1
        self.it = 0
        return state

    def play(self, action):
        if action == 0:  # Walk Left
            self.pos += 1
        if action == 1:  # Jump Left
            self.pos += 2
        if action == 2:  # Walk Right
            self.pos -= 1
        if action == 3:  # Jump Right
            self.pos -= 2

        state = self.maps.copy()


        done = False
        reward = 0
        if self.pos < 0 or self.pos > len(self.maps) - 1 or self.maps[self.pos] == -1:
            reward = -1
            done = True

        elif self.maps[self.pos] == 2:
            done = True
            reward = 2

        elif self.it == 10:
            done = True
            reward = -2

        self.pos = max(0, self.pos)
        self.pos = min(len(self.maps) - 1, self.pos)
        state[self.pos] = 1
        self.it += 1

        return state, reward, done


n_state = 12
n_action = 4
layers = [Dense(n_state, activation="relu"), Dense(64, activation="relu"), Dense(128, activation="relu"), Dense(128, activation="relu"),
          Dense(n_action, activation="linear")]
agent = dqn_agent.DQNAgent(layers, action_size=n_action, learning_rate=0.01, loss="MSE",
                           optimizer=Adam(learning_rate=0.001),
                           weights=None,
                           batch_size=400, max_memory_length=1000, gamma=0.95, epsylone=1, epsylone_decrease=0.9,
                           epsylone_min=0.1)
game = Game()

agent.train_agent(game.play, game.reset, n_iter=1000, n_step_before_training=25)


print("Testing !")
for i in range(5):
    print(f"IT: {i}")
    state = game.reset()
    done = False
    while not done:
        action = np.argmax(agent.main_model.predict(np.array([state]))[0])
        state, reward, done = game.play(action)
        print(f"Reward: {reward} | action: {action} | state: {list(state)}")
    print("End !")