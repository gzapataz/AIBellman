import run_gym_environment

environment = run_gym_environment.make("BipedalWalker-v2")
environment.reset()
for _ in range(2000):
    environment.render()
    environment.step(environment.action_space.sample())
environment.close()