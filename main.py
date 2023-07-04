from src import *

agent = DDQNAgent(restaurant_array=[2,6,2], grid_size=100, randseed=25)

episodios_terminados = 0
agent.train(episodes=5, batch_size=16, epsilon=0.99**episodios_terminados, epsilon_decay=0.99)

agent.save("model")