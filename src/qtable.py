from src import *


class QTable():

    def __init__(self, map):
        self.map = map
        self.n_clusters = len(map.clusters)
        self.n_states = 2**self.n_clusters
        self.n_actions = self.n_clusters**2
        self.qtable = np.zeros((self.n_states, self.n_actions))
    
    def get_next_state(self):
        state_array = self.map.get_state()
        state = 0
        for i in range(len(state_array)):
            state += state_array[i] * (2 ** i)
        return state

    def step(self, action):
        done = False
        if action < self.n_clusters**2 - self.n_clusters :
            performer = int(action/(self.n_clusters-1))
            reciever = action%(self.n_clusters-1)
            if reciever >= performer: 
                reciever+=1
            reward = self.map.relocate_courier(performer,reciever)

        elif action < self.n_clusters**2:
            cluster_id = action - (self.n_clusters**2 - self.n_clusters)
            reward = self.map.invoke_courier(cluster_id)
            
        # else:
        #     reward, done = self.map.pass_time()
        #     self.map.produce()

        new_state = self.get_next_state()
        done = new_state == 0
        return new_state, reward, done

    
    def reset(self):
        self.map.reset()
        self.map.build_prediction()
        return self.get_next_state()
    

    def train(self, episodes=10, alpha=0.5, gamma=0.9, epsilon = 1, epsilon_change=0.5):
        reward_change = []
        epsilon_decay = 1/(episodes*epsilon_change)
        for ep in range(episodes):
            state = self.reset()
            total_reward = 0
            done = False
            print("EP:",ep)
            while not done:
                accumulated_reward = 0
                finished = False
                while not finished:

                    rnd = np.random.random()

                    discarded_empty = self.discard_empty_clusters()
                    non_empty_actions = [self.qtable[state][i] for i in discarded_empty]

                    if rnd < epsilon or np.argmin(self.qtable[state]) == 0:
                        action = discarded_empty[randint(0, len(discarded_empty)-1)]
                    else:
                        action = discarded_empty[np.argmax(non_empty_actions)]
                    
                    new_state, reward, finished = self.step(action)
                    
                    accumulated_reward += reward

                    if not finished: 
                        reward = reward
                    else: 
                        reward = accumulated_reward

                    # print("State: {}, Action: {}, Reward: {}".format(state,action,reward))

                    self.qtable[state][action] = self.qtable[state][action] +\
                        alpha * (reward + gamma * np.max(self.qtable[new_state]) - self.qtable[state][action])    
                    
                    state = new_state

                reward, done = self.map.pass_time()
                state = self.get_next_state()
                self.map.produce()
                total_reward += accumulated_reward
                
            reward_change.append(total_reward)
            epsilon -= epsilon_decay
        return reward_change

    def reward_function(self, reward, state, action, new_state, alpha, gamma):
        self.qtable[state][action] += alpha * (reward + gamma * np.max(self.qtable[new_state]) - self.qtable[state][action])    

    def print_qtable(self):
        print(self.qtable)

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
        # state_actions.append(i)
        return state_actions