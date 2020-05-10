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

def run_jump_on_the_spot(steps, freq=5):
    """
    make robot jump on the spot in z direction

    :return:
    """

    env = MinitaurGymEnv(urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION, render=True, motor_velocity_limit=np.inf, pd_control_enabled=True, hard_reset=True, on_rack=False, )
    env._pybullet_client.configureDebugVisualizer(pb.COV_ENABLE_GUI,0)

    extension_amplitude_front = 0.5 # to the right of the robot in initial position
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
    divisor = freq
    actions = [[
                   t,
                   # swing_amplitude
                   math.sin(t/divisor)*swing_amplitude_front_left,
                   math.sin(t/divisor)*swing_amplitude_back_left,
                   math.sin(t/divisor)*swing_amplitude_front_right,
                   math.sin(t/divisor)*swing_amplitude_back_right,
                   # extension
                   math.sin(t/divisor)*extension_amplitude_front_left,
                   math.sin(t/divisor)*extension_amplitude_back_left,
                   math.sin(t/divisor)*extension_amplitude_front_right,
                   math.sin(t/divisor)*extension_amplitude_back_right
                  ] for t in range(steps)]




    env.reset()
    print(env.minitaur.GetLegModelInputDescription())
    input("enter to continue...")
    obs_history = [env.step(action[1:])[0] for action in actions]

    plot_angles(env, actions, obs_history, steps)


def tf_abs(list_in):
    # swing full range, extension clipped positive
    # for idx, a in enumerate(list_in):
    #     if idx < 5:
    #         a=a
    #     else:
    #         if a < 0:
    #             a= 0
    #         else:
    #             a=a
    return [a if idx < 5 else 0 if a < 0 else a for idx, a in enumerate(list_in)]

def run_jump_and_swing(steps, freq=5):
    env = MinitaurGymEnv(urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
                         render=True,
                         motor_velocity_limit=np.inf,
                         pd_control_enabled=True,
                         hard_reset=True,
                         on_rack=False, )
    env._pybullet_client.configureDebugVisualizer(pb.COV_ENABLE_GUI,0)

    freq = 10


    swing_amplitude_front = 0.9
    swing_amplitude_front_left = swing_amplitude_front
    swing_amplitude_front_right = swing_amplitude_front

    swing_amplitude_back = 0.9
    swing_amplitude_back_left = swing_amplitude_back
    swing_amplitude_back_right = swing_amplitude_back

    extension_amplitude_front = 0.6  # to the right of the robot in initial position
    extension_amplitude_front_right = extension_amplitude_front
    extension_amplitude_front_left = extension_amplitude_front

    extension_amplitude_back = 0.6
    extension_amplitude_back_left = extension_amplitude_back
    extension_amplitude_back_right = extension_amplitude_back

    offset = math.pi

    # provide angles in leg space
    actions = [tf_abs([

        t,
        # swing_amplitude
        math.sin(t / freq) * swing_amplitude_front_left,
        math.sin(offset + t / freq) * swing_amplitude_back_left,
        math.sin(t / freq) * swing_amplitude_front_right,
        math.sin(offset + t / freq) * swing_amplitude_back_right,
        # extension
        math.sin(t / freq) * extension_amplitude_front_left,
        math.sin(offset + t / freq) * extension_amplitude_back_left,
        math.sin(t / freq) * extension_amplitude_front_right,
        math.sin(offset + t / freq) * extension_amplitude_back_right
                ]) for t in range(steps)]

    input("press enter to continue...")
    env.reset()
    obs_history = [env.step(action[1:])[0] for action in actions]

    plot_angles(env, actions, obs_history, steps)

def default_gait(steps, freq=5):
    env = MinitaurGymEnv(urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION, render=True, motor_velocity_limit=np.inf, pd_control_enabled=True, hard_reset=True, on_rack=False, )
    env._pybullet_client.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)

    sum_reward = 0
    #steps = 20000
    amplitude_1_bound = 0.5  # leg extension amplitude, scalars for the sine wave
    amplitude_2_bound = 0.5  # leg swing amplitude
    speed = 40  # sine wave oscillation frequency
    actions = []  # desired action history
    obs_history = []  # actual observation history
    motor_names = [name[0][6:] for name in env.minitaur.GetOrderedMotorNameList()]

    #pp_desired = ProgressPlot(plot_names=motor_names, line_names=motor_names, y_lim=[-1, 1], width=900)

    for step_counter in range(steps):
        time_step = 0.01
        t = step_counter * time_step  # count in 10 ms steps

        amplitude1 = amplitude_1_bound
        amplitude2 = amplitude_2_bound
        steering_amplitude = 0
        if t < 20:  # less than 10 seconds
            steering_amplitude = 0.5  # adding amplitude makes sine wave go higher, meaning higher angles
        elif t < 40:  # less than 20 seconds
            steering_amplitude = -0.5
        else:  # stop steering after 20 seconds
            steering_amplitude = 0

        # sin(t) vs sin(t + pi) adds a phase shift to the sinewave
        # for walking, diagonal legs move together, are therefore phase shifted.
        # phase shift means: sin(t+pi) is at -1 when s(t) is at 1.
        # legs a1 and a3 walk together, a2 and a4 out of phase

        # Applying asymmetrical sine gaits to different legs can steer the minitaur.

        # extension, also diagonal synchronisation
        # only leg extension is used for steering
        # a1 steers first, then is disabled and a2 steers. Steering means doubling amplitude, ie more extension of leg as scaling factor
        # of leg is doubled compared to non-steering gait, but it also switches off the diagonal legs during steering
        a1 = math.sin(t * speed) * (amplitude1 + steering_amplitude)  # higher sine
        a2 = math.sin(t * speed + math.pi) * (amplitude1 - steering_amplitude)  # lower sine

        # swing, a3 = front_left and back_right, a4 = back_left and front_left - diagonal synchronisation
        a3 = math.sin(t * speed) * amplitude2  # second motor
        a4 = math.sin(t * speed + math.pi) * amplitude2  # shift phase by pi

        # actions 0-3 control extension, 4-8 control swing
        # control is done for each leg in the following order:
        # [front_left, back_left, front_right, back_right]
        # the pattern is ABBA meaning diagonal legs receive the same amplitude signal
        action = [a1, a2, a2, a1, a3, a4, a4, a3]

        # update graph
        # create graph display arrays for one angle per graph, rest 0
        graph_arrays = [[0] * i + [action[i]] + [0] * (len(action) - i - 1) for i, val in enumerate(action)]
        #pp_desired.update(graph_arrays)

        actions.append([t] + action)
        obs, reward, done, _ = env.step(action)
        obs_history.append(obs)

        sum_reward += reward
        print(sum_reward, end="\r")
        if done:
            break
    env.reset()
    plot_angles(env, actions, obs_history, steps)



def plot_angles(env, actions, obs_history, steps):

    fig, axm = plt.subplots(8, 3, figsize=(20, 10), constrained_layout=True)

    ax = (pd.DataFrame(columns=["t"] + env.minitaur.GetLegModelInputDescription(), data=actions).iloc[:steps, 1:].plot(subplots=True, figsize=(6, 10), ax=axm[:, 0]))
    [x.legend(loc="right") for x in ax]

    motor_angles = [env.minitaur.ConvertFromLegModel(action[1:]) for action in actions]
    motor_names = [m[0] for m in env.minitaur.GetOrderedMotorNameList()]

    fig.suptitle("angles leg v motor given v motor observed")
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

def plot_random_values():
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

    #run_jump_on_the_spot(600)
    run_jump_and_swing(600, freq=5)
    #default_gait(600)
    #plot_random_values()