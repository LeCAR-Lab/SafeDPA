
import gym
import imageio
import numpy as np
# Create the environment
from safety_gym.envs.engine import Engine
import itertools

config = {
    'robot_base': 'xmls/point2.xml',
    'task': 'goal',
    'observe_goal_lidar': True,
    'observe_box_lidar': False,
    'observe_hazards': True,
    # 'observe_vases': True,
    'constrain_hazards': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 1,
    # 'vases_num': 4
    'observation_flatten': True,
    'observe_qpos': True,
    'observe_qvel': True,
    'robot_locations': [(-1, -1)],
    'goal_locations': [(2, 2)],
    'hazards_locations':[(.5,.5)],
}

env_config = {
                  'gear_ratio_range':[1.0,1.0],
                  'wind_range':[-0.0002, 0.0002],
                  'wind_change':[1,1,0,0,0,0],
                  'verbose':1,
                  'aug_state':True,
             }

env = Engine(config, env_config)

# from gym.envs.registration import register

# register(id='SafexpTestEnvironment-v0',
#          entry_point='safety_gym.envs.mujoco:Engine',
#          kwargs={'config': config})
#
# Set up variables for recording videos
episode_count = 1 
fps = 30
video_filenames = ['episode_1.mp4', 'episode_2.mp4']

for i in range(episode_count):
    # Reset the environment for a new episode
    obs = env.reset()
    
    # __import__('pdb').set_trace()
    # Start recording a new video
    video_filename = video_filenames[i]
    with imageio.get_writer(video_filename, fps=fps) as video:
        # Run the episode
        done = False
        for step in itertools.count(0):
            # Take a random action
            action = env.action_space.sample()  
            # action = np.array([1,1])
            # print(env.action_space)
            # print(type(action), action.shape)
            # if step == 0:
            #     action = np.array([1,0])
            # else:
            #     action = np.array([0,0])

            # print(f"pos:{env.robot_pos[:2]}, vel:{env.robot_vel[:2]}, acc:{env.robot_acc[:2]}")
            # __import__('pdb').set_trace()
            env.update_for_render(action)
            # Take a step in the environment
            obs, reward, done, info = env.step(action)
            # print("obs", obs[-27:-24], obs[-14:-11])
            # print(env.robot_real_state)
            # print("obs shape", obs.shape)

            # Render the current frame of the environment and add it to the video
            frame = env.render(mode="rgb_array")
            # __import__('pdb').set_trace()
            video.append_data(frame)
            if done:
                break

    print(f'Episode {i + 1} finished. Video saved as {video_filename}')
    env.reset_simulator()

# Close the environment
env.close()
