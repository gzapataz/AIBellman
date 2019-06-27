"""
Aprendiendo Inteligencia artificial
04/12/2018
Curso Udemy IA con Python
Capitulo Seccion 6 clase 40
Primer Agente de aprendizaje usando Q Learning
El juego de la montana rusa
"""


import gym

environment = gym.make("MountainCar-v0")
MAX_NUM_EPISODES = 100
STEPS_PER_EPISODE = 200

for episode in range(MAX_NUM_EPISODES):
    done = False
    obs = environment.reset()
    total_reward = 0.0 ## Recompensa obtenida en cada episodio
    step = 1
    while not done:
        environment.render()
        action = environment.action_space.sample() ## Decision Aleatoria sample
        next_state, reward, done, info = environment.step(action)
        total_reward += reward
        step += 1
        obs = next_state
    print('\n Episodio numero {} Finalizado con {} iteraciones. Recompensa Final {}'.format(episode, step, total_reward))
environment.close()


#QLearner Class
# __init__(self, environment)
# dicretize(self, obs) [-2, 2] -> [-2,-1], [-1, 0], [0, 1], [1, 2]
# get_action(self, obs)
# learn(self, obs, action, reward, next_obs)
# EPSILON_MIN: vamos aprendiendo, mientras el incremento de aprendizaje sea superio a dicho valor
# MAX_NUM_EPISODES: numero maximo de iteraciones que estamos dispuestos a realizar
# STEPS_PER_EPISODE: numero maximo de pasos a realizar en cada episodio
# ALPHA: ratio de aprendizaje del agente
# NUM_DISCRETE_BINS: Numero de divisiones en el caso de discretizar un espacio continuo

EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30

import numpy as np

