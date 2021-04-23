import ast
import argparse
import logging

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from statistics import mean
import time

# Duckietown Specific
from gym_duckietown.envs import DuckietownEnv
from reinforcement.pytorch.ddpg import DDPG
from reinforcement.pytorch.utils import seed, evaluate_policy, ReplayBuffer
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def is_Yellow(x1,y1,x2,y2,image):
    R1,G1,B1 = image[y1,x1]
    R2,G2,B2 = image[y2,x2]
    
    # Debug colours
#     R = int((int(R1)+int(R2))/2)
#     G = int((int(G1)+int(G2))/2)
#     B = int((int(B1)+int(B2))/2)
#     plt.imshow([[(R,G,B)]])
#     plt.show()
    if mean([B1,B2])*1.7 < mean([R1,G1,R2,G2]):
#         print("Yellow",R,G,B)
        return True
#     print("White",R,G,B)
    return False

def draw_lines(img, lines, image, color=[255, 255, 255], thickness=1):
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )
    img = np.copy(img)
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            if is_Yellow(x1,y1,x2,y2,image):
                cv2.line(line_img, (x1, y1), (x2, y2), [255,255,0], thickness)
            else:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    line_img = np.float32(line_img)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

def pipeline(image):
    """
    An image processing pipeline which will output
    an image with the lane lines annotated.
    """
    image = np.float32(np.transpose(image,(1, 2, 0)))
    gray_image = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)*255).astype(np.uint8)
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, 25),
        (0, height),    
        (width, height),
        (width, 25),    
    ]
    cropped_image = region_of_interest(
        cannyed_image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )
    
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=1,
        theta=np.pi / 180,
        threshold=7,
        lines=np.array([]),
        minLineLength=25,
        maxLineGap=12
    )
    
    good_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
#             slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
#             if math.fabs(slope) > 0.2: # <-- Only consider extreme slope
            good_lines.append(line)

    line_image = draw_lines(np.zeros_like(image),good_lines,image)
    return np.transpose(line_image,(2,0,1))

def eps_decay(eps, time_):
    if time_ < 1000:
        return eps
    elif time_ < 5000:
        return eps/2
    elif time_< 10000:
        return eps/4
    return eps/8

def _train(args):   
    start = time.time()
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    # Launch the env with our helper function
    # env = launch_env()
    EPS = 0.2
    SEED = 2
    MAP = 'map5'
    env = DuckietownEnv(
        map_name = MAP, domain_rand = False, draw_bbox = False,
        max_steps = 1500,
        seed = SEED)
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")
    # obs = (env.reset())
    # plt.subplot(1, 2, 1)
    # plt.imshow(np.transpose((obs),(1,2,0)))
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.transpose(pipeline(obs),(1,2,0))) 
    # plt.show()
    # error
    # # Set seeds
    # seed(args.seed)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    policy.load(filename='ddpg_40000', directory='reinforcement/pytorch/models/')
    replay_buffer = ReplayBuffer(args.replay_buffer_max_size)
    print("Initialized DDPG")
    
    # Evaluate untrained policy
    evaluations= [evaluate_policy(env, policy)]
   
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = None
    env_counter = 0
    reward = 0
    episode_timesteps = 0
    action_penalty_speed = []
    action_penalty_steering = []

    print("Starting training")
    while total_timesteps < args.max_timesteps:
        
        print("timestep: {} | reward: {}".format(total_timesteps, reward))
            
        if done:
            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                    total_timesteps, episode_num, episode_timesteps, episode_reward))
                policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    evaluations.append(evaluate_policy(env, policy))
                    print("rewards at time {}: {}".format(total_timesteps, evaluations[-1]))

            if args.save_models:
                policy.save(filename='ddpg_{}'.format(int(args.max_timesteps)), directory=args.model_dir)
            np.savez("./results/rewards.npz",evaluations)

            # Reset environment
            env_counter += 1
            obs = (env.reset())
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        p = np.random.random()
        # if total_timesteps < args.start_timesteps:
        if p < eps_decay(EPS,total_timesteps):
            action = env.action_space.sample()
            # if action[0] <0.2:
            #     action[0] = 1
        else:
            action = policy.predict(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(
                    0,
                    args.expl_noise,
                    size=env.action_space.shape[0])
                          ).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        new_obs = (new_obs)

        action_penalty_speed.append(action[0])
        if len(action_penalty_speed) == 30:
            if sum(action_penalty_speed)<4.5:
                reward -= 111
                print("Too slow bro")
            # Reset every 10 steps
            action_penalty_speed = []
            # done = True

        action_penalty_steering.append(abs(action[1]))
        if len(action_penalty_steering) == 60:
            if abs(sum(action_penalty_steering))>54:
                reward -= 111
                print("Don't run in circle")
            # Reset every 10 steps
            action_penalty_steering = []
            # done = True

        if episode_timesteps >= args.env_timesteps:
            done = True

        done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add(obs, new_obs, action, reward, done_bool)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
    
    print("Training done, about to save..")
    policy.save(filename='ddpg_{}'.format(int(args.max_timesteps)), directory=args.model_dir)
    print("Finished saving..should return now!")
    print("Time :",time.time()-start,"s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # DDPG Args
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=40000, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true", default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--env_timesteps", default=1500, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--replay_buffer_max_size", default=10000, type=int)  # Maximum number of steps to keep in the replay buffer
    parser.add_argument('--model-dir', type=str, default='reinforcement/pytorch/models/')

    _train(parser.parse_args())
