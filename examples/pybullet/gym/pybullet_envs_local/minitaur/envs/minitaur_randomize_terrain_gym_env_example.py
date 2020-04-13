"""An example to run minitaur gym environment with randomized terrain.

"""

import math

import numpy as np
import tensorflow.compat.v1 as tf
#from google3.pyglib import app
#from google3.pyglib import flags
from pybullet_envs_local.minitaur.envs import minitaur_randomize_terrain_gym_env

#FLAGS = flags.FLAGS

#flags.DEFINE_enum("example_name", "reset", ["sine", "reset"],
#                  "The name of the example: sine or reset.")


def ResetTerrainExample():
  """An example showing resetting random terrain env."""
  num_reset = 1
  steps = 100_000
  env = minitaur_randomize_terrain_gym_env.MinitaurRandomizeTerrainGymEnv(
      render=True, leg_model_enabled=False, motor_velocity_limit=np.inf, pd_control_enabled=True)
  action = [math.pi / 2] * 8
  for _ in range(num_reset):
    env.reset()

    for _ in range(steps):
      _, _, done, _ = env.step(action)
      if done:
        break


def SinePolicyExample():
  """An example of minitaur walking with a sine gait."""
  env = minitaur_randomize_terrain_gym_env.MinitaurRandomizeTerrainGymEnv(
      render=True, motor_velocity_limit=np.inf, pd_control_enabled=True, on_rack=False)
  sum_reward = 0
  steps = 200000
  amplitude_1_bound = 0.5
  amplitude_2_bound = 0.5
  speed = 20

  for step_counter in range(steps):
    time_step = 0.01
    t = step_counter * time_step

    amplitude1 = amplitude_1_bound
    amplitude2 = amplitude_2_bound
    steering_amplitude = 0
    if t < 10:
      steering_amplitude = 0.5
    elif t < 20:
      steering_amplitude = -0.5
    else:
      steering_amplitude = 0

    # Applying asymmetrical sine gaits to different legs can steer the minitaur.
    a1 = math.sin(t * speed) * (amplitude1 + steering_amplitude)
    a2 = math.sin(t * speed + math.pi) * (amplitude1 - steering_amplitude)
    a3 = math.sin(t * speed) * amplitude2
    a4 = math.sin(t * speed + math.pi) * amplitude2
    action = [a1, a2, a2, a1, a3, a4, a4, a3]
    _, reward, _, _ = env.step(action)
    sum_reward += reward


def main(example_name):
  #if FLAGS.example_name == "sine":
  if example_name == "sine":
    SinePolicyExample()
  #elif FLAGS.example_name == "reset":
  elif example_name == "reset":
    ResetTerrainExample()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  #app.run()
  main("sine")
