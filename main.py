from src import *


episodios=200
episodios_terminados=0
decay=0.998

if episodios_terminados == 0:
    agent = Agent(restaurant_array=[2,6,2], grid_size=100, randseed=25)
else:
    agent = Agent(restaurant_array=[2,6,2], grid_size=100, randseed=25, filename="models/model_"+str(episodios_terminados))

agent.train(episodes=episodios, finished_episodes=episodios_terminados, batch_size=64, epsilon_decay=decay)

agent.save("models/model_"+str(episodios_terminados+episodios))