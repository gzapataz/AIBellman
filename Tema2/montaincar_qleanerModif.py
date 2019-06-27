
"""
Archivo para modificar los elemento y entender el programa
Aprendiendo Inteligencia artificial
04/12/2018
Curso Udemy IA con Python
Capitulo Seccion 6 clase 40
Primer Agente de aprendizaje usando Q Learning
El juego de la montana rusa
"""



import gym
import numpy as np

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
# GAMMA Factor de descuento del agente, lo que vamos perdiendo de un paso a otro
# EPSILON_DECAY decremento de EPSILON de un paso al siguiente

MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200


EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30

class QLearner(object):
    def __init__(self, environment):
        self.obs_shape = environment.observation_space.shape
        print("Shape:".format(self.obs_shape))
        self.obs_high = environment.observation_space.high
        self.obs_low = environment.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.obs_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = environment.action_space.n
        print("environment.action_space: {}".format(environment.action_space))
        print("environment.observarion_space: {}".format(environment.observation_space))

        self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1, self.action_shape )) #Matriz de 31 x 31 x 3
        print("Q Inicial " + str(self.Q))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0

    def discretize(self, obs):
        return tuple(((obs - self.obs_low) / self.obs_width).astype(int))

    def get_action(self, obs):
        discrete_obs = self.discretize(obs)
        #print("disc obs {}".format(discrete_obs))
        #Seccion de la accion con base en EPSILO GREEDY
        # Al principio la mayoria de las interacciones son aleatorias epsilon empieza con 1, luego va bajando ya que el agente va aprendiendo
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon: #Con probabilidad 1 - Epsilon se escoge la mejor posible
            a =  np.argmax(self.Q[discrete_obs])

            return a
        else:
            b = np.random.choice([a for a in range(self.action_shape)])
            return b # Con probabilidad epsilon se escoge aleatoriamente

    '''
    Metodo que se encarga del aprendizaje del agente
    En este metodo se aplica la ecuacion de Bellman en funcion del tiempo
    Q(s,a) = Qt-1(s,a) + Alfha * ((R(s,a) + gamma * Max Q(s', a') - Q(s, a))
    '''

    def learn(self, obs, action, reward, next_obs):
        discrete_obs = self.discretize(obs)
        discrete_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma * np.max(self.Q[discrete_next_obs])
        td_error = td_target - self.Q[discrete_obs][action]
        #print("discrete_obs:{}".format(discrete_obs))
        self.Q[discrete_obs][action] += self.alpha * td_error
        #print("ASI VA Q:{}".format(str(self.Q[discrete_obs][action])))
        # Ecuacion de Bellman
        # self.Q[discrete_obs][action] += self.alpha * ((reward + self.gamma * np.max(self.Q[discrete_next_obs])) - self.Q[discrete_obs][action])

'''
Metodo que realiza el entrenamiento del agente
Aprende hasta el total de episodios permitidos por la variable MAX_NUM_EPISODES
Por cada iteracion el agende va aprendiendo
'''
def train(agent, environment):
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = environment.reset()
        total_reward = 0.0
        while not done:
            #Toma una accion
            action = agent.get_action(obs)
            #print("Action {}".format(action))
            next_obs, reward, done, info = environment.step(action)
            #print("Observacion {}".format(next_obs))
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward

        if total_reward > best_reward:
            best_reward = total_reward
        print("Episodio Numero {} con recompensa {}, mejor recompensa {}, epsilon {}".format(episode, total_reward, best_reward, agent.epsilon))

    #De todas las politicas de entrenamiento que hemos obtenido devolvemos la mejor de todas
    print("QFINAL:{}".format(agent.Q))
    return np.argmax(agent.Q, axis=2)

def test(agent, environment, policy):
    done = False
    obs = environment.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)] #Accion que dicta la politica que hemos entrenado
        next_obs, reward, done, info = environment.step(action)
        obs = next_obs
        total_reward += reward

    return total_reward
'''
Clase Main
Entrena primero al agente y luego se ejecuta de acuerdo al plan aprendido

'''

if __name__ == "__main__":
    environment = gym.make("MountainCar-v0") #Inicializa el ambiente para el juego del Mountain Car
    agent = QLearner(environment) #Inicializa el ambiente en la clase QLearner

    learned_policy = train(agent, environment) # Proceso de entrenamiento del agente
    print("learned_policy: " + str(learned_policy))
    monitor_path = "./monitor_output" #Directorio donde se escriben los pasos seguidos
    environment = gym.wrappers.Monitor(environment, monitor_path, force=True) # Monitorea el ambiente
    for _ in range(1000):
        test(agent, environment, learned_policy) #Prueba el aprendizaje con la politica de aprendizaje del entrenamiento
    environment.close() #Cierra el ambiente y termina
