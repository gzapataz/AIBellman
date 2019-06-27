#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:02:38 2018

@author: Gabriel
"""

import gym


environment = gym.make("BipedalWalker-v2")
environment.reset()
for _ in range(2000):
    environment.render()
    environment.step(environment.action_space.sample())
    
environment.close()