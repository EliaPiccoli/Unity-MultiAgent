import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from gym_unity.envs import UnityEnv
from agents import Agents

if __name__ == "__main__":

    env_name = "Unity/Games/maze_multi_3x3.x86_64"
    env = UnityEnv(None, worker_id=0, multiagent=True)

    agents = Agents(env, 3)
    agents.train()

    