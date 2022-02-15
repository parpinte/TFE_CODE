# MultiAgentCardPole
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole


env = MultiAgentCartPole()
observation = env.reset()
for _ in range(1000):
    actions = policies(agents, observation)
    observation, rewards, dones,
    infos = env.step(actions)