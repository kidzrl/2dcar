from RLBRAIN import DeepQNetwork
from ENV import CarEnv
import tensorflow as tf
import os
from my_socket import socket_connect as sk
import time
import numpy as np

LOAD = False
Real_RUN = False
sess = tf.Session()
env = CarEnv()
RL = DeepQNetwork(sess, n_actions=5, n_features=5,
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  replace_target_iter=200,
                  memory_size=2000,
                  # output_graph=True
                  )
saver = tf.train.Saver(max_to_keep=None)
path = './discrete'

if LOAD:
    saver.restore(sess, "./discrete/DQN1168_1203.ckpt")
else:
    sess.run(tf.global_variables_initializer())

def train():
    step = 0
    for episode in range(10000):
        # initial observation
        observation = env.reset()
        count = 0
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                if count > 300:
                    ckpt_path = os.path.join(path, 'DQN%i_%i.ckpt' % (episode, count))
                    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
                    print("\nSave Model %s\t" % save_path, "Episode is : %i\t" % episode, "count is : %i\n" % count)
                break
            step += 1
            count += 1
        if episode % 50 == 0:
            print("\nEpisode is : %i" % episode)
    # end of game
    sess.close()
    print('game over')

def eval():     # 2left  1ahead   0right
    if Real_RUN:
        if os.path.exists('datasensor.txt'):
            os.remove('datasensor.txt')
        while True:
            s = sk.receive()
            if s:
                s = process_state(s)
                print(s)
                a = RL.eval_choose_action(s)
                a = str(a)
                sk.send(a)
                print(a)
                time.sleep(0.2)
            with open('datasensor.txt', mode='a') as file:
                datasensor = file.write(str(s) + '\t' + str(a) + '\n')
    else:
        env.set_fps(100)
        if os.path.exists('dataeval.txt'):
            os.remove('dataeval.txt')
        while True:
            s = env.reset()
            count = 0
            while True:
                env.render()
                a = RL.eval_choose_action(s)
                print(s)
                print(a)
                with open('dataeval.txt', mode='a') as file:
                    dataeval = file.write(str(s) + '\t' + str(a) + '\n')
                s_, r, done = env.step(a)
                s = s_
                count += 1
                if done:
                    print(count)
                    break

def process_state(s):
    s = s.decode()
    s_split = s.split(',')
    s_split[0] = float(s_split[0])
    if s_split[0] == 2.6:
        s_split[0] = 0.01
    s_split[1] = float(s_split[1])
    if s_split[1] == 2.6:
        s_split[1] = 0.01
    s_split[2] = float(s_split[2])
    if s_split[2] == 2.6:
        s_split[2] = 0.01
    s_split[3] = float(s_split[3])
    if s_split[3] == 2.6:
        s_split[3] = 0.01
    s_split[4] = float(s_split[4])
    if s_split[4] == 2.6:
        s_split[4] = 0.01

    arrary = np.array(s_split) + np.array([0.08, 0.05, 0.08, 0.05, 0.08])
    return arrary

if __name__ == "__main__":
    if LOAD:
        if Real_RUN:
            sk = sk(server=('192.168.43.204', 10000), client=('192.168.43.2', 10000))
        eval()
    else:
        train()
