import numpy as np
import random
import time
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from src import *

tf.config.threading.set_inter_op_parallelism_threads(8)

class DDQNAgent:
    def __init__(self, map):
        self.map = map
        self.n_clusters = len(map.clusters)
        self.state_size = 3*self.n_clusters
        self.action_size = self.n_clusters**2
        self.memory = deque(maxlen=1024)
        self.gamma = 0.8
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()


    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.state_size*8, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.state_size*6, activation='relu'))
        model.add(Dense(self.state_size*4, activation='relu'))
        model.add(Dense(self.action_size, activation='sigmoid'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
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
            if self.map.clusters[performer].can_relocate():
                reward = self.map.relocate_courier(performer,reciever)
            else:
                reward = COST_INVOCATION*5
            if verbose:
                print("[ACTION]: C_{} -> C_{}, R: {}".format(performer, reciever, reward))

        elif action < self.n_clusters**2:
            cluster_id = action - (self.n_clusters**2 - self.n_clusters)
            if verbose:
                print("[ACTION]: C_{}++, R: {}".format(cluster_id, reward))
            reward = self.map.invoke_courier(cluster_id)

        new_state, done = self.map.get_state()
        return new_state, reward, done

    def train_by_timestamp(self, episodes=1000, batch_size = 16, epsilon = 1.0, epsilon_decay=0.99):
        reward_history = []
        state = self.reset()[0]
        finished = False

        while not finished:
            map_copy = self.map.copy()
            state, done = self.map.get_state()
            if done:
                _, finished = self.map.pass_time()
                continue
            state = np.reshape(state, [1, self.state_size])
            # uses += 1 if not done else 0
            for e in range(episodes):
                run_actions = 0
                run_rewards = 0
                # accumulated_reward = 0
                # total_actions = 0
                # uses = 0
                # count = 0
                while not done:
                    action = self.act(state, epsilon)
                    next_state, reward, done = self.step(action)
                    reward = reward if not done else reward + 500*self.n_clusters
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    run_actions += 1
                    run_rewards += reward


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
                # print("uses: {}, actions: {}, reward: {:.2f}, e: {:.3f}, t: {:.4f}s"
                #       .format(uses, run_actions, run_rewards, epsilon, time.time() - start_time))
                # total_actions += run_actions
                # accumulated_reward += run_rewards
                # count += 1


            # print("episode: {}/{}, score: {:.2f}, e: {:.2f}, mems: {}, uses: {}, act/use: {:.2f}"
            #                 .format(e+1, episodes, accumulated_reward, epsilon, total_actions, uses, total_actions/uses))
            # if epsilon > self.epsilon_min:
            #     epsilon *= epsilon_decay
            # reward_history.append(accumulated_reward)
            
        return reward_history

    def train(self, episodes=1000, batch_size = 16, epsilon = 1.0, epsilon_decay=0.99):
        reward_history = []
        for e in range(episodes):
            finished = False
            accumulated_reward = 0
            total_actions = 0
            uses = 0
            state = self.reset()[0]
            count = 0
            while not finished:
                start_time = time.time()
                state, done = self.map.get_state()
                state = np.reshape(state, [1, self.state_size])
                uses += 1 if not done else 0
                run_actions = 0
                run_rewards = 0
                while not done:
                    action = self.act(state, epsilon)
                    next_state, reward, done = self.step(action)
                    reward = reward if not done else reward + 500*self.n_clusters
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    run_actions += 1
                    run_rewards += reward


                _, finished = self.map.pass_time()
                self.replay(batch_size)
                if count % 24 == 0:
                    self.update_target_model()
                    print(np.reshape(state, (self.state_size,1)).tolist())
                # print("uses: {}, actions: {}, reward: {:.2f}, e: {:.3f}, t: {:.4f}s"
                #       .format(uses, run_actions, run_rewards, epsilon, time.time() - start_time))
                total_actions += run_actions
                accumulated_reward += run_rewards
                count += 1


            print("episode: {}/{}, score: {:.2f}, e: {:.2f}, mems: {}, uses: {}, act/use: {:.2f}"
                            .format(e+1, episodes, accumulated_reward, epsilon, total_actions, uses, total_actions/uses))
            if epsilon > self.epsilon_min:
                epsilon *= epsilon_decay
            reward_history.append(accumulated_reward)
            
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
                reward = -np.linalg.norm(self.map.clusters[performer].centroid - self.map.clusters[reciever].centroid)
                state[0][performer*3] -= 1
                state[0][reciever*3 + 2] += 1
            else:
                reward = COST_INVOCATION*5
            if verbose:
                print("[ACTION]: C_{} -> C_{}, R: {}".format(performer, reciever, reward))

        elif action < self.n_clusters**2:
            cluster_id = action - (self.n_clusters**2 - self.n_clusters)
            reward = COST_INVOCATION
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



    def load(self, name):
        self.model.load_weights(name+".h5")
        self.map = Map(filename=name+".npz", restaurant_array=[2,6,2], grid_size=100, randseed=25)


    def save(self, name):
        self.model.save_weights(name+".h5")
        self.map.save(name+".npz")
