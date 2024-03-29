from src import *

episodios=2000
episodios_terminados=0
decay=0.999

if episodios_terminados == 0:
    agent = Agent(restaurant_array=[2,6,2], grid_size=100, randseed=25)
else:
    agent = Agent(restaurant_array=[2,6,2], grid_size=100, randseed=25, filename="models/model_n3_"+str(episodios_terminados))

agent.train(episodes=episodios, finished_episodes=episodios_terminados, batch_size=128,
             epsilon_decay=decay, score_limit=-110, verbose=False)

agent.save("models/model_n3_"+str(episodios_terminados+episodios))