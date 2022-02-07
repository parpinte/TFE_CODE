# Travaux fin d'étude
## The `Defense` Environment
The environment is a simple fully-observable multi-agent environment oriented towards defense applications. It contains $N_b$ blue agents and $N_r$ red agents. Agents on the same team (blue or red) have to work together to destroy the agents of the other team. They can do this by aiming and firing on other agents. If the opposing agent is close enough (within parameter `RANGE`), he wll be destroyed. Each agent can go left, right, up or down, and can aim or fire. A more rigourous describtion of the state- and action space is given below.  
The `defense_v0` environment is modelled as an `Agent Environment Cycle` (AEC) game - see [this article](https://arxiv.org/abs/2009.14471). The code snippet below gives he idiomatic game play, as proposed by the PettingZoo environment:
```python

    env = defense_v0.env(terrain='central_5x5')
    env.reset()
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        action = actor(obs) if not done else None
        env.step(action)

```
where the `agent_iter()` method iterates over all alive agents.

## Algorithms
The `algorithms` folder contains a number of deep reinforcement learning algorithms (based on `PyTorch`):
1. DQN
1. PPO (to do)
1. QMix (to do)

## References
1. Deep Q-Networks (DQN): [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602v1)
1. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
1. [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
1. [PettingZoo: Gym for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2009.14471)