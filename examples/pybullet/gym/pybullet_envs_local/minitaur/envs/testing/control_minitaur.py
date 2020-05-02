#import pybullet_envs
import os
import inspect
bullet_dir = r"C:\coding_repos\bullet3\examples\pybullet\gym"
print("parentdir=", bullet_dir)
os.sys.path.insert(0, bullet_dir)
import gym
import pybullet as pb
import math

from pybullet_envs_local.bullet.minitaur import Minitaur
from pybullet_envs_local.bullet.minitaur_duck_gym_env import MinitaurBulletDuckEnv
from pybullet_envs_local.minitaur.envs.minitaur_gym_env import MinitaurGymEnv
import pybullet_envs_local.minitaur.envs.minitaur_gym_env as minitaur_gym_env
import pandas as pd, numpy as np
import time
import pybullet_utils.bullet_client as bc
import matplotlib.pyplot as plt
from IPython.display import Markdown
from IPython.display import clear_output
from jupyter_ui_poll import ui_events

from functools import partial

from jupyterplot import ProgressPlot
from ipywidgets import interact, interactive, fixed, interact_manual, Layout, VBox, HBox
import ipywidgets as widgets

import dask.dataframe as dd
import plotly.express as px

def run_jump_on_the_spot(steps):
    """
    make robot jump on the spot in z direction

    :return:
    """

    env = MinitaurGymEnv(urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION, render=True, motor_velocity_limit=np.inf, pd_control_enabled=True, hard_reset=True, on_rack=False, )


    extension_amplitude_front = 0.8 # to the right of the robot in initial position
    extension_amplitude_front_right = extension_amplitude_front
    extension_amplitude_front_left = extension_amplitude_front

    extension_amplitude_back = 0
    extension_amplitude_back_left = extension_amplitude_back
    extension_amplitude_back_right = extension_amplitude_back

    swing_amplitude_front = 0 # 0.7
    swing_amplitude_front_left = swing_amplitude_front
    swing_amplitude_front_right = swing_amplitude_front

    swing_amplitude_back = 0 # 0.8
    swing_amplitude_back_left = swing_amplitude_back
    swing_amplitude_back_right = swing_amplitude_back

    actions = [[
                   t,
                   # swing_amplitude
                   math.sin(t/5)*swing_amplitude_front_left,
                   math.sin(t/5)*swing_amplitude_back_left,
                   math.sin(t/5)*swing_amplitude_front_right,
                   math.sin(t/5)*swing_amplitude_back_right,
                   # extension
                   math.sin(t/5)*extension_amplitude_front_left,
                   math.sin(t/5)*extension_amplitude_back_left,
                   math.sin(t/5)*extension_amplitude_front_right,
                   math.sin(t/5)*extension_amplitude_back_right
                  ] for t in range(steps)]




    env.reset()
    print(env.minitaur.GetLegModelInputDescription())
    obs_history = [env.step(action[1:])[0] for action in actions]

    plot_angles(env, actions, obs_history, steps)




def run_jump_and_swing(steps):
    env = MinitaurGymEnv(urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
                         render=True,
                         motor_velocity_limit=np.inf,
                         pd_control_enabled=True,
                         hard_reset=True,
                         on_rack=False, )

    extension_amplitude_front = 0.8  # to the right of the robot in initial position
    extension_amplitude_front_right = extension_amplitude_front
    extension_amplitude_front_left = extension_amplitude_front

    extension_amplitude_back = 0.8
    extension_amplitude_back_left = extension_amplitude_back
    extension_amplitude_back_right = extension_amplitude_back

    swing_amplitude_front = 0.8
    swing_amplitude_front_left = swing_amplitude_front
    swing_amplitude_front_right = swing_amplitude_front

    swing_amplitude_back = 0.8
    swing_amplitude_back_left = swing_amplitude_back
    swing_amplitude_back_right = swing_amplitude_back

    offset = math.pi

    # provide angles in leg space
    actions = [[

        t,
        # swing_amplitude
        math.sin(t / 5) * swing_amplitude_front_left,
        math.sin(offset + t / 5) * swing_amplitude_back_left,
        math.sin(t / 5) * swing_amplitude_front_right,
        math.sin(offset + t / 5) * swing_amplitude_back_right,
        # extension
        math.sin(t / 5) * extension_amplitude_front_left,
        math.sin(offset + t / 5) * extension_amplitude_back_left,
        math.sin(t / 5) * extension_amplitude_front_right,
        math.sin(offset + t / 5) * extension_amplitude_back_right
                ] for t in range(steps)]

    env.reset()
    obs_history = [env.step(action[1:])[0] for action in actions]

    plot_angles(env, actions, obs_history, steps)



def plot_angles(env, actions, obs_history, steps):

    fig, axm = plt.subplots(8, 3, figsize=(20, 10), constrained_layout=True)

    ax = (pd.DataFrame(columns=["t"] + env.minitaur.GetLegModelInputDescription(), data=actions).iloc[:steps, 1:].plot(subplots=True, figsize=(6, 10), ax=axm[:, 0]))
    [x.legend(loc="right") for x in ax]

    motor_angles = [env.minitaur.ConvertFromLegModel(action[1:]) for action in actions]
    motor_names = [m[0] for m in env.minitaur.GetOrderedMotorNameList()]

    fig.suptitle("angles given v observed")
    ax = (pd.DataFrame(columns=motor_names, data=motor_angles).iloc[:steps, :].mul(180 / math.pi).plot(subplots=True, ax=axm[:, 1]))

    [x.legend(loc="right") for x in ax]


    #print(env.minitaur.GetLegModelInputDescription())


    description = env._get_observation_description()
    df_obs_hist = pd.DataFrame(columns=description, data=obs_history).iloc[:steps, :]

    ax = (df_obs_hist.loc[:, df_obs_hist.columns.str.contains("angle")].mul(180 / math.pi).plot(subplots=True, ax=axm[:, 2]))

    for x in ax:
        patches, labels = x.get_legend_handles_labels()
        labels = [l[12:] for l in labels]
        x.legend(patches, labels, loc='right')

    # df_obs_hist.loc[:, df_obs_hist.columns.str.contains("vel")].plot(title="velocities", ax=ax[1, 0])
    # df_obs_hist.loc[:, df_obs_hist.columns.str.contains("torque")].plot(title="torques", ax=ax[1, 1])

    # ax[0].get_figure().suptitle("angles-observed")

    plt.show()

def plotsth():
    # scientific view distorts title
    df_obs_hist = pd.DataFrame(columns=["col_" + str(i) for i in range(10)], data=[[t]*10 for t in range(100)])

    ax = (df_obs_hist
         .loc[:, :]
         .plot(subplots=True)
         )

    for x in ax:
        patches, labels = x.get_legend_handles_labels()
        labels = [l for l in labels]
        x.legend(patches, labels, loc='right')
        #x.set_title("hi##)
    ax[0].get_figure().suptitle("Hi")
    plt.show()


if __name__=="__main__":

    run_jump_on_the_spot(600)
    #run_jump_and_swing(600)
    #plotsth()