import random as rnd
from collections import deque

import numpy as np
from keras import Sequential
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    def __init__(self, layers, action_size, learning_rate=0.01, loss="MSE", optimizer=Adam(learning_rate=0.001),
                 weights=None,
                 batch_size=100, max_memory_length=1000, gamma=0.95, epsylone=1, epsylone_decrease=0.001,
                 epsylone_min=0.1):
        self.epsylone_min = epsylone_min
        self.epsylone_decrease = epsylone_decrease
        self.epsylone = epsylone
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.action_size = action_size

        # Main Model
        self.main_model = self.create_model(layers, loss, optimizer)
        if weights is not None:
            print("Loading weights ...")
            self.main_model.set_weights(weights)
            print("Weight load !")

        # Batch
        self.memory = deque(maxlen=max_memory_length)

    def create_model(self, layers, loss, optimizer):
        model = Sequential(layers)
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def save_in_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        return self.memory

    def train_model(self):
        mini_batch = np.array(rnd.sample(self.memory, self.batch_size), dtype="object")

        new_q = self.main_model.predict(np.array(list(mini_batch[:, 0]), dtype="float"), verbose=0)

        q_t_plus_one_values = self.main_model.predict(np.array(list(mini_batch[:, 3]), dtype="float"), verbose=0)

        i = 0
        for state, action, reward, next_state, done in mini_batch:
            new_q_value = reward
            if not done:
                # new_q_value = Rt + Y*max(Q(t+1))
                new_q_value = reward + self.gamma * np.amax(q_t_plus_one_values[i])

            new_q[i][action] = new_q_value
            i += 1

        # Train the model
        self.main_model.fit(np.array(list(mini_batch[:, 0]), dtype="float"), new_q)

    def choose_action(self, state):
        if rnd.random() < self.epsylone:
            return rnd.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.main_model.predict(np.array([state]), verbose=0)[0])

    def train_agent(self, play_function, reset_function, n_iter=1000, n_step_before_training=100):
        print("Starting to train the agent !")
        all_steps = deque(maxlen=100)
        all_rewards = deque(maxlen=100)
        for i in range(n_iter):
            state = reset_function()
            done = False
            total_reward = 0
            total_step = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = play_function(action)

                self.save_in_memory(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                total_step += 1

            all_steps.append(total_step)
            all_rewards.append(total_reward)
            if i % n_step_before_training == 0 and len(self.memory) >= self.batch_size:
                print("Model start training ...")
                self.train_model()
                print("Training complete ! Resume the training of the agent !")

                self.epsylone = max(self.epsylone_min, self.epsylone_decrease * self.epsylone)
            print(
                f"Step {i}: rewards: {total_reward} | mean reward: {np.mean(all_rewards)} | step: {total_step} | mean step: {np.mean(all_steps)} | epsylone: {self.epsylone}")

        print(f"Train of the agent is finish ! Epsylone: {self.epsylone}")
