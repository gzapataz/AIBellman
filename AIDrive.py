import numpy as q1
import sys

R = q1.matrix([
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 100, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0],
])

Q = q1.matrix(q1.zeros([6, 6]))

gamma = 0.8

agent_s_state = 1

def possible_actions(state):
    current_state_row = R[state,]
    possible_act = q1.where(current_state_row > 0)[1]
    return possible_act

PossibleAction = possible_actions(agent_s_state)

def ActionChoice(available_action_range):
    next_action = int(q1.random.choice(available_action_range, 1))
    return next_action

action = ActionChoice(PossibleAction)

def Reward(current_state, action, gamma):
    Max_State = q1.where(Q[action, ] == q1.max(Q[action,]))[1]
    if Max_State.shape[0] > 1:
        Max_State = int(q1.random.choice(Max_State, size = 1))
    else:
        Max_State = int(Max_State)
    MaxValue = Q[action, Max_State]
    #Ecuacion de Bellman
    #Q(s,a) = R(s, a) + Gamma * Q[a, max_state]
    Q[current_state, action] = R[current_state, action] + gamma * MaxValue

Reward(agent_s_state, action, gamma)

if __name__ == "__main__":
    max_iterations = int(sys.argv[1])
    for i in range(max_iterations):
        current_state = q1.random.randint(0, int(Q.shape[0]))
        PossibleAction = possible_actions(current_state)
        action = ActionChoice(PossibleAction)
        Reward(current_state, action, gamma)

    print("Q: ")
    print(Q)

    print("Normed Q:")
    print(Q/q1.max(Q) * 100)