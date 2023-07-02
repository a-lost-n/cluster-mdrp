from src import *

map = Map(restaurant_array=[2,6,2], grid_size=100, randseed=25)
agent = DDQNAgent(map)

agent.train_by_timestamp(episodes=200, batch_size=32, epsilon=1, epsilon_decay=0.99)

agent.save("model_25")