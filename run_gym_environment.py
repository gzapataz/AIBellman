"""
Aprendiendo Inteligencia artificial
28/11/2018
Curso Udemy IA con Python
Capitulo Seccion 5 clase 33
Aprendizaje por medio de la ecuacion de bellman y decision de Markov
"""


import gym
import sys

def run_gym_environment(argv):
    ## El primer parametro de argv sera el nombre del entorno a ejecutar
    ## Argv1 -> Tipo de Juego
    ## Argv2 -> Numero de repeticiones
    environment = gym.make(argv[1])
    environment.reset()
    for _ in range(int(argv[2])):
        environment.render()
        environment.step(environment.action_space.sample())
    environment.close()


if __name__ == "__main__":
    run_gym_environment(sys.argv)