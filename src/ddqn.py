import numpy as np
import random
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from src import *

tf.config.threading.set_inter_op_parallelism_threads(6)

class DDQNAgent:
    def __init__(self, map):
        self.map = map
        self.n_clusters = len(map.clusters)
        self.state_size = 3*self.n_clusters + 1
        self.action_size = self.n_clusters**2
        self.memory = deque(maxlen=1024)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.985
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def discard_empty_clusters(self):
        state_actions = []
        empty_clusters = self.map.get_empty_clusters()
        i = 0
        for _ in range(len(empty_clusters)*(len(empty_clusters)-1)):
            if not empty_clusters[int(i/(len(empty_clusters)-1))]:
                state_actions.append(i)
            i += 1
        for _ in range(len(empty_clusters)):
            state_actions.append(i)
            i += 1
        state_actions.append(i)
        return state_actions

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                Q_future = np.amax(self.target_model.predict(next_state, verbose=0)[0])
                target[0][action] = reward + self.gamma * Q_future
            self.model.fit(state, target, epochs=1, verbose=0)


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def reset(self):
        self.map.reset()
        self.map.build_prediction()
        return self.map.get_state()
    
    def step(self, action, verbose=False):
        if action < self.n_clusters**2 - self.n_clusters:
            performer = int(action/(self.n_clusters-1))
            reciever = action%(self.n_clusters-1)
            if reciever >= performer: 
                reciever+=1
            if verbose:
                print("[ACTION]: C_{} -> C_{}".format(performer, reciever))
            reward = self.map.relocate_courier(performer,reciever)

        elif action < self.n_clusters**2:
            cluster_id = action - (self.n_clusters**2 - self.n_clusters)
            if verbose:
                print("[ACTION]: C_{}++".format(cluster_id))
            reward = self.map.invoke_courier(cluster_id)

        new_state, done = self.map.get_state()
        return new_state, reward, done

    def train(self, episodes=1000):
        reward_history = []
        batch_size = 16
        for e in range(episodes):
            finished = False
            accumulated_reward = 0
            state = self.reset()[0]
            while not finished:
                state, done = self.map.get_state()
                state = np.reshape(state, [1, self.state_size])
                done = False
                while not done:
                    action = self.act(state)
                    next_state, reward, done = self.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    accumulated_reward += reward

                _, finished = self.map.pass_time()
            print("episode: {}/{}, score: {}, e: {:.2}"
                            .format(e, episodes, accumulated_reward, self.epsilon))
            if len(self.memory) > batch_size:
                self.replay(batch_size)
            reward_history.append(accumulated_reward)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        return reward_history

    def test(self):
        finished = False
        accumulated_reward = 0
        state = self.reset()[0]
        while not finished:
            state, done = self.map.get_state()
            state = np.reshape(state, [1, self.state_size])
            done = False
            while not done:
                print("[STATE]: ", np.reshape(state, (self.state_size, 1)))
                action = self.act(state)
                next_state, reward, done = self.step(action, verbose=True)
                next_state = np.reshape(next_state, [1, self.state_size])
                state = next_state
                accumulated_reward += reward

            _, finished = self.map.pass_time()

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
