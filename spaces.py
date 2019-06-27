"""
Aprendiendo Inteligencia artificial
28/11/2018
Curso Udemy IA con Python
Capitulo Seccion 5 clase 33
Aprendizaje por medio de la ecuacion de bellman y decision de Markov

Explica los entornos ya acciones disponibles para un video juego en particular
Ejemplo para pinball las palitas suben y bajan
Para space Invaders se mueve para un lado para el otro

"""

import gym
from gym.spaces import *
import sys


## BOX -> espacio de n dimensiones R*n [x1, x2, x3, .., xn], xi[low, high]
##gym.spaces.Box(low = -10, high = 10, shape = [2, ])
## Discrete -> numeros enteros 0 y n-1 {0, 1, 2, 3, 4}
#gym.spaces.Discrete(5) #{0, 1, 2, 3, 4}
#Dict -> Diccionario de espacios complejos
#gym.spaces.Dict ({
#    "position": gym.spaces.Discrete(3),
#    "velocity": gym.spaces.Discrete(2)
#})

# Multi binary -> {T, F} {x1, x2, x3, .., xn}, xi {T, F}
#gym.spaces.MultiBinary(3) #(x, y, z), x,y,z = T|F
# Multi Discrete -> {a, a+1, a+2,....b}
#gym.spaces.MultiDiscrete([-10,10], [0,1])

#Tuple -> Producto de espacios simples
#gym.spaces.Tuple(gym.spaces.Discrete(3), gym.spaces.Discrete(2))


##

def print_spaces(space):
    print(space)
    if isinstance(space, Box):
        print("\n Cota Inferior", space.low)
        print("\n Cota Superior", space.high)

if __name__ == "__main__":
    environment = gym.make(sys.argv[1])
    print("Espacio de Observaciones")
    print_spaces(environment.observation_space)
    print("Espacio de Acciones")
    print_spaces(environment.action_space)
    try:
        print("Descripcion de las acciones", environment.unwrapped.get_action_meanings())
    except AttributeError:
        pass