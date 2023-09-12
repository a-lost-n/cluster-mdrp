import numpy as np
import random
import time
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from IPython.display import display, clear_output
from keras import backend as K
from src import *

os.environ["OMP_NUM_THREADS"] = "1"

# tf.config.threading.set_inter_op_parallelism_threads(12)
# tf.config.threading.set_intra_op_parallelism_threads(2)


class Agent:
    def __init__(self, restaurant_array, learning_rate=0.001, grid_size=100, randseed=0, filename=None):
        self.map = Map(restaurant_array=restaurant_array,
                       grid_size=grid_size, randseed=randseed, filename=filename)
        self.n_clusters = len(self.map.clusters)
        self.reward_history = []
        self.state_size = 3*self.n_clusters
        self.action_size = self.n_clusters**2
        self.memory = ReplayMemory(capacity=50_000)
        self.gamma = 1
        self.epsilon_min = 0.01
        self.learning_rate = learning_rate
        self.model = self._build_model()
        if filename:
            self.model.load_weights(filename+".h5")
        self.target_model = self._build_model()
        self.pre_train(epochs=2000)

    def ReLNU(self, x):
        return K.minimum(-x, 1)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.action_size*self.state_size*3,
                  input_dim=self.state_size, activation='relu'))

        model.add(Dense(self.action_size*self.state_size, activation='relu'))

        model.add(Dense(self.action_size*self.state_size/3, activation='relu'))

        model.add(Dense(self.action_size, activation='relu'))

        model.compile(loss='huber', optimizer=Adam(
            learning_rate=self.learning_rate))
        return model

    def pre_train(self, epochs=100):
        dist_matrix = np.zeros((self.n_clusters, self.n_clusters))
        for i in range(self.n_clusters):
            for j in range(i+1, self.n_clusters):
                dist_matrix[i][j] = self.map.dist(
                    self.map.clusters[i].centroid, self.map.clusters[j].centroid)
                dist_matrix[j][i] = self.map.dist(
                    self.map.clusters[i].centroid, self.map.clusters[j].centroid)
        states = []
        actions = []
        for i in range(self.n_clusters):
            state = np.zeros(self.state_size)
            state[i*self.n_clusters + 1] = 1
            action = np.zeros(self.action_size)
            reloc_size = self.n_clusters*(self.n_clusters - 1)
            j = 0
            while j < reloc_size:
                action[j] = 2
                j += 1
            while j < self.action_size:
                action[j] = 1 + dist_matrix[i][j - reloc_size]
                j += 1
            states.append(state)
            actions.append(action)
        states = np.array(states)
        actions = np.array(actions)
        self.model.fit(states, actions, epochs=epochs, verbose=0)
        self.update_target_model()

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, epsilon=0, return_value=False, use_target=False):
        if np.random.rand() <= epsilon:
            return random.choice(range(self.action_size))
        if use_target:
            act_values = self.target_model.predict(state, verbose=0)
        else:
            act_values = self.model.predict(state, verbose=0)
        if return_value:
            return np.amin(act_values)
        return np.argmin(act_values)

    def fit(self, state, action, reward, next_state, done, verbose=False):
        target = self.model.predict(state, verbose=0)
        if verbose:
            print("[STATE]", state)
            print("[REAL]: ", target[0])
            print("[TARGET]", self.target_model.predict(state, verbose=0))
        if done:
            error = np.abs(target[0][action]-reward)
            target[0][action] = reward
        else:
            Q_future = self.act(next_state, return_value=True, use_target=True)
            bellman_eq = reward + self.gamma * Q_future
            error = np.abs(target[0][action] - bellman_eq)
            target[0][action] = bellman_eq
        if verbose:
            print("[PREDICTED]: ", target[0], '\n')

        self.model.fit(state, target, epochs=1, verbose=0)
        return error

    def replay(self, batch_size, verbose=False):
        if self.memory.size() < batch_size:
            return
        indices, minibatch, _ = self.memory.sample(batch_size)
        errors = []
        for state, action, reward, next_state, done in minibatch:
            errors.append(self.fit(state, action, reward,
                          next_state, done, verbose))
        self.memory.update_priority(indices, errors)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def reset(self):
        self.map.reset()
        self.map.build_prediction()
        return self.map.get_state()

    def test_random(self):
        return random.randrange(self.action_size)

    def step(self, action, verbose=False):
        if action < self.n_clusters**2 - self.n_clusters:
            performer = int(action/(self.n_clusters-1))
            reciever = action % (self.n_clusters-1)
            if reciever >= performer:
                reciever += 1
            if self.map.clusters[performer].can_relocate():
                reward = self.map.relocate_courier(performer, reciever)
            else:
                reward = COST_INVOCATION*1.1
            if verbose:
                print(
                    "[ACTION]: C_{} -> C_{}, R: {}".format(performer, reciever, reward))

        elif action < self.n_clusters**2:
            cluster_id = action - (self.n_clusters**2 - self.n_clusters)
            reward = self.map.invoke_courier(cluster_id)
            if verbose:
                print("[ACTION]: C_{}++, R: {}".format(cluster_id, reward))

        new_state, done = self.map.get_state()
        return new_state, reward, done

    def train(self, episodes, finished_episodes=0, batch_size=16, epsilon=1.0, epsilon_decay=0.99,
              score_limit=1000, refresh_rate=10, verbose=False, log=True, log_limit=-200, train_model=True, display_rewards=False):
        if finished_episodes != 0:
            epsilon = epsilon_decay**finished_episodes
        for e in range(episodes):
            state, _ = self.reset()
            start_time = time.time()
            finished = False
            accumulated_reward = 0
            total_actions = 0
            while not finished and accumulated_reward < score_limit:
                state, done = self.map.get_state()
                state = np.reshape(state, [1, self.state_size])
                while not done and accumulated_reward < score_limit:
                    action = self.act(state, epsilon)
                    next_state, reward, done = self.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    if train_model:
                        self.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_actions += 1
                    accumulated_reward += reward
                _, finished = self.map.pass_time()

            if train_model:
                self.replay(batch_size, verbose=verbose)
            if episodes % refresh_rate == refresh_rate-1:
                self.update_target_model()
            if log and accumulated_reward < log_limit:
                print("episode: {}/{}, score: {:.2f}, e: {:.2f}, actions: {}, couriers: {}, t: {:.2f}s"
                      .format(e+1+finished_episodes, episodes+finished_episodes, accumulated_reward,
                              epsilon, total_actions, len(self.map.couriers), time.time() - start_time))
            if epsilon > self.epsilon_min:
                epsilon *= epsilon_decay
            self.reward_history.append(accumulated_reward)
            if display_rewards:
                self.rewards_graphic()
            tf.keras.backend.clear_session()

    def rewards_graphic(self, n, mean=0, ylim=200):
        if mean == 0:
            mean = np.mean(grouped_rewards)
        grouped_rewards = np.mean(
            np.array(self.reward_history).reshape(-1, n), axis=1)
        plt.plot(np.arange(0, len(self.reward_history), n), grouped_rewards)
        plt.ylim(0, ylim)
        plt.xlim(0, len(self.reward_history))
        plt.axhline(y=mean, color='r', linestyle='--')
        plt.show()

    def test(self):
        finished = False
        accumulated_reward = 0
        state = self.reset()[0]
        while not finished:
            state, done = self.map.get_state()
            state = np.reshape(state, [1, self.state_size])
            while not done:
                print("[STATE]: ", np.reshape(
                    state, (self.state_size, 1)).tolist())
                action = self.act(state, epsilon=0)
                next_state, reward, done = self.step(action, verbose=True)
                next_state = np.reshape(next_state, [1, self.state_size])
                state = next_state
                accumulated_reward += reward

            _, finished = self.map.pass_time()

    def test_step(self, state, action, verbose=False):
        if action < self.n_clusters**2 - self.n_clusters:
            performer = int(action/(self.n_clusters-1))
            reciever = action % (self.n_clusters-1)
            if reciever >= performer:
                reciever += 1
            if state[0][performer*3] > 0:
                reward = np.linalg.norm(
                    self.map.clusters[performer].centroid - self.map.clusters[reciever].centroid)*COST_TRANSLATION_PER_TRAVEL_UNIT
                state[0][performer*3] -= 1
                state[0][reciever*3 + 2] += 1
            else:
                reward = COST_INVOCATION
            if verbose:
                print(
                    "[ACTION]: C_{} -> C_{}, R: {}".format(performer, reciever, reward))

        elif action < self.n_clusters**2:
            cluster_id = action - (self.n_clusters**2 - self.n_clusters)
            reward = -1
            if verbose:
                print("[ACTION]: C_{}++, R:{}".format(cluster_id, reward))
            state[0][cluster_id*3] += 1

        done = True
        for i in range(0, self.state_size, 3):
            done = done and (state[0][i+1] <= (state[0][i] + state[0][i+2]))
        return state, reward, done

    def test_state(self, state):
        accumulated_reward = 0
        state = np.reshape(state, [1, self.state_size])
        done = False
        while not done:
            print("[STATE]: ", np.reshape(
                state, (self.state_size, 1)).tolist())
            action = self.act(state, epsilon=0)
            next_state, reward, done = self.test_step(
                state, action, verbose=True)
            next_state = np.reshape(next_state, [1, self.state_size])
            state = next_state
            accumulated_reward += reward
        print("score: {:.2f}".format(accumulated_reward))

    def save(self, name):
        self.model.save_weights(name+".h5")
        self.map.save(name+".npz")

    def print_prediction(self, state):
        prediction = self.model.predict(state, verbose=False)[0]
        print("|      | Invo |  C1  |  C2  |  C3  |")
        print("|  C1  |{:.4f}|  --  |{:.4f}|{:.4f}|".format(
            prediction[6], prediction[0], prediction[1]))
        print("|  C2  |{:.4f}|{:.4f}|  --  |{:.4f}|".format(
            prediction[7], prediction[2], prediction[3]))
        print("|  C3  |{:.4f}|{:.4f}|{:.4f}|  --  |".format(
            prediction[8], prediction[4], prediction[5]))
