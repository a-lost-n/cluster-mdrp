from src import *

agent = Agent(restaurant_array=[2,6,2], grid_size=100, randseed=25)

episodios_terminados = 0
decay=0.998
agent.train(episodes=50, batch_size=16, epsilon=decay**episodios_terminados, epsilon_decay=decay)

agent.save("model_50")