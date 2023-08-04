import numpy as np
import random
import time
import tensorflow as tf
import os
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from src import *

os.environ["OMP_NUM_THREADS"] = "1"

# tf.config.threading.set_inter_op_parallelism_threads(12) 
# tf.config.threading.set_intra_op_parallelism_threads(2)

class Agent:
    def __init__(self, restaurant_array, grid_size=100, randseed=0, filename=None):
        self.map = Map(restaurant_array=restaurant_array, grid_size=grid_size, randseed=randseed, filename=filename)
        self.n_clusters = len(self.map.clusters)
        self.state_size = 3*self.n_clusters 
        self.action_size = self.n_clusters**2
        self.memory = deque(maxlen=512)
        self.gamma = 0.9
        self.epsilon_min = 0.01
        self.learning_rate = 0.005
        self.model = self._build_model()
        if filename: self.model.load_weights(filename+".h5")
        self.target_model = self._build_model()


    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.state_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='huber', optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state, epsilon=0, return_value=False):
        possible_actions = [*range(self.action_size)]
        for i in range(0, self.state_size, 3):
            if state[0][i] == 0:
                for j in range(0, self.n_clusters-1):
                    possible_actions.remove(i*2/3 + j)
        if np.random.rand() <= epsilon:
            return random.choice(possible_actions)
        act_values = self.model.predict(state, verbose=0)
        act_values = [act_values[0][i] if i in possible_actions else -np.Infinity for i in range(len(act_values[0]))]
        if return_value: return np.amax(act_values)
        return np.argmax(act_values)


    def fit(self, state, action, reward, next_state, done):
        target = self.model.predict(state, verbose=0)
        if done:
            target[0][action] = reward
        else:
            Q_future = self.act(next_state, return_value=True)
            target[0][action] = reward + self.gamma * Q_future
        self.model.fit(state, target, epochs=1, verbose=0)


    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.fit(state, action, reward, next_state, done)


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
            reciever = action%(self.n_clusters-1)
            if reciever >= performer: reciever+=1
            if self.map.clusters[performer].can_relocate():
                reward = self.map.relocate_courier(performer,reciever)
            else:
                reward = -1
            if verbose:
                print("[ACTION]: C_{} -> C_{}, R: {}".format(performer, reciever, reward))

        elif action < self.n_clusters**2:
            cluster_id = action - (self.n_clusters**2 - self.n_clusters)
            reward = self.map.invoke_courier(cluster_id)
            if verbose:
                print("[ACTION]: C_{}++, R: {}".format(cluster_id, reward))

        new_state, done = self.map.get_state()
        return new_state, reward, done


    def train(self, episodes=1000, batch_size=16, epsilon=1.0, epsilon_decay=0.99, action_limit=50):
        reward_history = []
        for e in range(episodes):
            state, _ = self.reset()
            start_time = time.time()
            finished = False
            accumulated_reward = 0
            total_actions = 0
            count = 0
            while not finished:
                state, done = self.map.get_state()
                state = np.reshape(state, [1, self.state_size])
                run_actions = 0
                run_rewards = 0
                while not done and run_actions < action_limit:
                    action = self.act(state, epsilon)
                    next_state, reward, done = self.step(action)
                    reward = reward if not done else reward - COST_INVOCATION
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    run_actions += 1
                    run_rewards += reward
                self.replay(batch_size)
                if count % 24 == 0:
                    print("{}, t: {:.3f}s".format(np.reshape(state, (self.state_size,1)).tolist(), time.time()-start_time))
                    # print("actions: {}, reward: {:.2f}, e: {:.3f}, t: {:.4f}s"
                    #       .format(run_actions, run_rewards, epsilon, time.time() - start_time))
                total_actions += run_actions
                accumulated_reward += run_rewards
                count += 1
                _, finished = self.map.pass_time()



            self.update_target_model()
            print("episode: {}/{}, score: {:.2f}, e: {:.2f}, actions: {}, t: {:.2f}s"
                            .format(e+1, episodes, accumulated_reward, epsilon, total_actions, time.time() - start_time))
            if epsilon > self.epsilon_min:
                epsilon *= epsilon_decay
            reward_history.append(accumulated_reward)
            tf.keras.backend.clear_session()
        return reward_history


    def train_by_timestamp(self, episodes=1000, batch_size = 16, epsilon = 1.0, epsilon_decay=0.99):
        reward_history = []
        state = self.reset()[0]
        finished = False
        iterations = 0
        while not finished and iterations < 1:
            map_copy = self.map.copy()
            state, done = self.map.get_state()
            if done:
                _, finished = self.map.pass_time()
                continue
            state = np.reshape(state, [1, self.state_size])
            for e in range(episodes):
                run_actions = 0
                run_rewards = 0
                while not done and run_actions < 50:
                    action = self.act(state, epsilon)
                    next_state, reward, done = self.step(action)
                    reward = reward if not done else reward + 1
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    run_actions += 1
                    run_rewards += reward
                
                # if run_rewards > -50:
                # _, finished = self.map.pass_time()
                self.replay(batch_size)
                if e%10 == 0:
                    self.update_target_model()
                if e % (episodes/10) == 0:
                    print(np.reshape(state, (self.state_size,1)).tolist())
                    print("actions: {}, reward: {:.2f}, e: {:.3f}".format(run_actions, run_rewards, epsilon))
                self.map = map_copy.copy()
                state, done = self.map.get_state()
                state = np.reshape(state, [1, self.state_size])
                if epsilon > self.epsilon_min:
                    epsilon *= epsilon_decay
            return 
            iterations += 1
            state = np.reshape(state, [1, self.state_size])
            while not done:
                print("[STATE]: ", np.reshape(state, (self.state_size, 1)).tolist())
                action = self.act(state, epsilon=0)
                next_state, reward, done = self.step(action, verbose=True)
                next_state = np.reshape(next_state, [1, self.state_size])
                state = next_state

            _, finished = self.map.pass_time()
            
        return reward_history


    def test(self):
        finished = False
        accumulated_reward = 0
        state = self.reset()[0]
        while not finished:
            state, done = self.map.get_state()
            state = np.reshape(state, [1, self.state_size])
            while not done:
                print("[STATE]: ", np.reshape(state, (self.state_size, 1)).tolist())
                action = self.act(state, epsilon=0)
                next_state, reward, done = self.step(action, verbose=True)
                next_state = np.reshape(next_state, [1, self.state_size])
                state = next_state
                accumulated_reward += reward

            _, finished = self.map.pass_time()


    def test_step(self, state, action, verbose=False):
        if action < self.n_clusters**2 - self.n_clusters:
            performer = int(action/(self.n_clusters-1))
            reciever = action%(self.n_clusters-1)
            if reciever >= performer: 
                reciever+=1
            if state[0][performer*3] > 0:
                reward = np.linalg.norm(self.map.clusters[performer].centroid - self.map.clusters[reciever].centroid)/COST_INVOCATION
                state[0][performer*3] -= 1
                state[0][reciever*3 + 2] += 1
            else:
                reward = -1
            if verbose:
                print("[ACTION]: C_{} -> C_{}, R: {}".format(performer, reciever, reward))

        elif action < self.n_clusters**2:
            cluster_id = action - (self.n_clusters**2 - self.n_clusters)
            reward = -1
            if verbose:
                print("[ACTION]: C_{}++, R:{}".format(cluster_id, reward))
            state[0][cluster_id*3] += 1

        done = True
        for i in range(0,self.state_size,3):
            done = done and (state[0][i+1] <= (state[0][i] + state[0][i+2]))
        return state, reward, done


    def test_state(self, state):
        accumulated_reward = 0
        state = np.reshape(state, [1, self.state_size])
        done = False
        while not done:
            print("[STATE]: ", np.reshape(state, (self.state_size, 1)).tolist())
            action = self.act(state, epsilon=0)
            next_state, reward, done = self.test_step(state, action, verbose=True)
            next_state = np.reshape(next_state, [1, self.state_size])
            state = next_state
            accumulated_reward += reward
        print("score: {:.2f}".format(accumulated_reward))


    def save(self, name):
        self.model.save_weights(name+".h5")
        self.map.save(name+".npz")
