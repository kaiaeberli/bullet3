import json
import numpy as np
from pybullet_envs_local.deep_mimic.learning.ppo_agent import PPOAgent
import pybullet_data_local

AGENT_TYPE_KEY = "AgentType"


def build_agent(world, id, file):
  agent = None
  with open(pybullet_data_local.getDataPath() + "/" + file) as data_file:
    json_data = json.load(data_file)

    assert AGENT_TYPE_KEY in json_data
    agent_type = json_data[AGENT_TYPE_KEY]

    if (agent_type == PPOAgent.NAME):
      agent = PPOAgent(world, id, json_data)
    else:
      assert False, 'Unsupported agent type: ' + agent_type

  return agent
