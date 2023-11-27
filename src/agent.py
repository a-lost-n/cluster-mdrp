import numpy as np
import random
import time
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from IPython.display import display, clear_output
from keras import backend as K
from src import *

os.environ["OMP_NUM_THREADS"] = "1"

# tf.config.threading.set_inter_op_parallelism_threads(12)
# tf.config.threading.set_intra_op_parallelism_threads(2)


class Agent:
    def __init__(self, restaurant_array=None, learning_rate=0.001, grid_size=100, randseed=0, map_file=None, filename=None,
                 num_clusters=None, pre_train=True, name=None, model=None):
        self.map = Map(restaurant_array=restaurant_array,
                       grid_size=grid_size, randseed=randseed, import_dir=map_file,
                       num_clusters=num_clusters, filename=filename)
        self.n_clusters = len(self.map.clusters)
        self.reward_history = []
        self.test_reward_history = []
        self.courier_size_history = []
        self.order_size_history = []
        self.state_size = 3*self.n_clusters
        self.action_size = self.n_clusters**2
        self.memory = ReplayMemory(capacity=10_000)
        self.gamma = 1
        self.epsilon_min = 0
        self.learning_rate = learning_rate
        self.min_score = np.inf
        self.name = name
        self.model = self._build_model(model())
        if filename and not pre_train:
            self.model.load_weights(filename+".h5")
        self.target_model = self._build_model(model())
        if pre_train:
            self.pre_train(epochs=2000)

    def _build_model(self, model):
        if not model:
            model = Sequential()
            model.add(Dense(self.action_size*self.state_size*3,
                            input_dim=self.state_size, activation='relu'))

            model.add(Dense(self.action_size*self.state_size, activation='relu'))

            model.add(Dense(self.action_size *
                      self.state_size/3, activation='relu'))

            model.add(Dense(self.action_size, activation='linear'))

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
            state[i*3 + 1] = 1
            action = np.zeros(self.action_size)
            reloc_size = self.n_clusters*(self.n_clusters - 1)
            j = 0
            while j < reloc_size:
                action[j] = COST_WRONG_RELOCATION + 1
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

    def get_possible_actions(self, state):
        impossible_actions = []
        for cluster in range(self.n_clusters):
            if state[0][cluster*3] == 0:
                impossible_actions.extend(
                    [cluster*(self.n_clusters - 1) + i for i in range(self.n_clusters - 1)])
        return np.setdiff1d(np.arange(self.n_clusters**2), np.array(impossible_actions))

    def act(self, state, epsilon=0, return_value=False, use_target=False, block_impossible_actions=False):
        if block_impossible_actions:
            possible_actions = self.get_possible_actions(state)

        if np.random.rand() <= epsilon:
            if not block_impossible_actions:
                return random.choice(range(self.action_size))
            return random.choice(possible_actions)

        if use_target:
            act_values = self.target_model.predict(state, verbose=0)[0]
        else:
            act_values = self.model.predict(state, verbose=0)[0]

        if block_impossible_actions:
            for i in range(act_values.shape[0]):
                if i not in possible_actions:
                    act_values[i] = np.inf

        if return_value:
            return np.amin(act_values)
        return np.argmin(act_values)

    def fit(self, state, action, reward, next_state, done, verbose=False, block_impossible_actions=False):
        target = self.model.predict(state, verbose=0)
        if verbose:
            print("[STATE]", state)
            print("[REAL]: ", target[0])
            print("[TARGET]", self.target_model.predict(state, verbose=0))
        if done:
            error = np.abs(target[0][action]-reward)
            target[0][action] = reward
        else:
            Q_future = self.act(next_state, return_value=True, use_target=True,
                                block_impossible_actions=block_impossible_actions)
            bellman_eq = reward + self.gamma * Q_future
            error = np.abs(target[0][action] - bellman_eq)
            target[0][action] = bellman_eq
        if verbose:
            print("[PREDICTED]: ", target[0], '\n')

        self.model.fit(state, target, epochs=1, verbose=0)
        return error

    def replay(self, batch_size, verbose=False, block_impossible_actions=False):
        if self.memory.size() < batch_size:
            return
        indices, minibatch, _ = self.memory.sample(batch_size)
        errors = []
        for state, action, reward, next_state, done in minibatch:
            errors.append(self.fit(state, action, reward,
                          next_state, done, verbose,
                          block_impossible_actions=block_impossible_actions))
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
                reward = COST_WRONG_RELOCATION
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
              score_limit=120, refresh_rate=10, verbose=False, log=True, train_model=True,
              block_impossible_actions=True, run_test=True):
        if finished_episodes != 0:
            epsilon = epsilon_decay**finished_episodes
        for e in range(episodes):
            state, _ = self.reset()
            start_time = time.time()
            finished = False
            accumulated_reward = 0
            total_actions = 0
            orders_made = 0
            while not finished and accumulated_reward < score_limit:
                state, done = self.map.get_state()
                state = np.reshape(state, [1, self.state_size])
                while not done and accumulated_reward < score_limit:
                    action = self.act(
                        state, epsilon, block_impossible_actions=block_impossible_actions)
                    next_state, reward, done = self.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    if train_model:
                        self.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_actions += 1
                    accumulated_reward += reward
                _, finished, orders_count = self.map.pass_time()
                orders_made += orders_count

            if train_model:
                if log:
                    print("episode: {}/{}, score: {:.2f}, e: {:.2f}, actions: {}, couriers: {}, orders: {}, t: {:.2f}s"
                          .format(e+1+finished_episodes, episodes+finished_episodes, accumulated_reward,
                                  epsilon, total_actions, len(self.map.couriers), orders_made, time.time() - start_time))
                self.replay(batch_size, verbose=verbose,
                            block_impossible_actions=block_impossible_actions)
                if episodes % refresh_rate == refresh_rate-1:
                    self.update_target_model()
                if epsilon > self.epsilon_min:
                    epsilon *= epsilon_decay
                self.reward_history.append(accumulated_reward)
                self.courier_size_history.append(len(self.map.couriers))
                self.order_size_history.append(orders_made)
                if run_test:
                    self.train(episodes=1, train_model=False, epsilon=0, score_limit=score_limit,
                            log=log, block_impossible_actions=block_impossible_actions)
                    if self.test_reward_history[-1] < self.min_score:
                        self.min_score = self.test_reward_history[-1]
                        self.save()
                else:
                    if self.reward_history[-1] < self.min_score:
                        self.min_score = self.reward_history[-1]
                        self.save()
            else:
                if log:
                    print("[TEST] score: {:.2f}, actions: {}, couriers: {}, orders: {}, t: {:.2f}s"
                          .format(accumulated_reward, total_actions, len(self.map.couriers), orders_made, time.time() - start_time))
                self.test_reward_history.append(accumulated_reward)
                self.courier_size_history.append(len(self.map.couriers))
                self.order_size_history.append(orders_made)
            tf.keras.backend.clear_session()

    def rewards_graphic(self, n, mean=0, ylim=200):
        grouped_rewards = np.mean(
            np.array(self.reward_history).reshape(-1, n), axis=1)
        test_grouped_rewards = np.mean(
            np.array(self.test_reward_history).reshape(-1, n), axis=1)
        if mean == 0:
            mean = np.mean(grouped_rewards)
        plt.plot(np.arange(0, len(self.reward_history), n),
                 grouped_rewards, color='blue')
        plt.plot(np.arange(0, len(self.test_reward_history), n),
                 test_grouped_rewards, color='green')
        plt.ylim(0, ylim)
        plt.xlim(0, len(self.reward_history))
        plt.axhline(y=mean, color='r', linestyle='--')
        plt.show()

    def test(self, episodes, score_limit=250):
        self.train(episodes=episodes, train_model=False,
                   epsilon=0, score_limit=score_limit, log=False)
        self.test_reward_history = np.array(self.test_reward_history)
        print(f"Percentil {50}: {np.percentile(self.test_reward_history, 50)}")
        print(f"Percentil {75}: {np.percentile(self.test_reward_history, 75)}")
        print(f"Percentil {90}: {np.percentile(self.test_reward_history, 90)}")
        print(f"Percentil {95}: {np.percentile(self.test_reward_history, 95)}")
        print(f"Percentil {99}: {np.percentile(self.test_reward_history, 99)}")
        plt.hist(self.test_reward_history, range=[20, 260], bins=20)

    def save(self, name=None):
        if not name:
            name = self.name
        self.model.save_weights(name+".h5")
        self.map.save(name+".npz")
