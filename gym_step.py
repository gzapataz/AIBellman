#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:02:38 2018

@author: Gabriel
"""

import gym

environment = gym.make("Qbert-v0")
MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500

for episode in range(MAX_NUM_EPISODES):
    obs = environment.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
        environment.render()
        action = environment.action_space.sample() ## Decision Aleatoria sample
        next_state, reward, done, info = environment.step(action)
        obs = next_state
        if done is True:
            print("\n Episode #{} terminado #{}".format(episode, step + 1))
            break
environment.close()