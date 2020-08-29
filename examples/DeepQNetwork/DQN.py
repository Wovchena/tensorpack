#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu

import datetime
import numpy as np
import cv2
import gym
import gym.envs.atari
import gym.wrappers
import tensorflow as tf

from expreplay import ExpReplay


class CropGrayScaleResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.shape = (84, 75)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        return cv2.resize(cv2.cvtColor(observation[-180:, :160], cv2.COLOR_RGB2GRAY),
                          (75, 84), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def get_player():
    return CropGrayScaleResizeWrapper(gym.wrappers.TimeLimit(gym.envs.atari.AtariEnv('breakout', obs_type='image',
        frameskip=4, repeat_action_probability=0.25), max_episode_steps=60000))


class Model:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.c0 = tf.compat.v1.layers.Conv2D(32, 8, 4, activation='relu', name='conv2d_0')
        self.c1 = tf.compat.v1.layers.Conv2D(64, 4, 2, activation='relu', name='conv2d_1')
        self.c2 = tf.compat.v1.layers.Conv2D(64, 3, activation='relu', name='conv2d_2')
        self.d0 = tf.compat.v1.layers.Dense(512, 'relu', name='dense_0')
        self.d1 = tf.compat.v1.layers.Dense(self.num_actions, name='dense_1')
        self.c10 = tf.compat.v1.layers.Conv2D(32, 8, 4, activation='relu', name='target/conv2d_0')
        self.c11 = tf.compat.v1.layers.Conv2D(64, 4, 2, activation='relu', name='target/conv2d_1')
        self.c12 = tf.compat.v1.layers.Conv2D(64, 3, activation='relu', name='target/conv2d_2')
        self.d10 = tf.compat.v1.layers.Dense(512, 'relu', name='target/dense_0')
        self.d11 = tf.compat.v1.layers.Dense(self.num_actions, name='target/dense_1')

    def build_graph(self, comb_state, action, reward, isOver):
        comb_state = tf.cast(comb_state, tf.float32)

        state = tf.identity(comb_state[:, :, :, :-1], name='state')
        features = tf.reshape(self.c2(self.c1(self.c0(state))), (-1, 64 * 7 * 5))
        pred = self.d1(self.d0(features))
        predict_value = tf.identity(pred, name='Qvalue')

        reward = tf.clip_by_value(reward, -1, 1)
        next_state = comb_state[:, :, :, 1:]
        action_onehot = tf.one_hot(tf.cast(action, tf.int32), self.num_actions, 1.0, 0.0)

        pred_action_value = tf.reduce_sum(predict_value * action_onehot, 1)  # N,

        features = tf.reshape(self.c12(self.c11(self.c10(next_state))), (-1, 64 * 7 * 5))
        targetQ_predict_value = self.d11(self.d10(features))

        best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * 0.99 * tf.stop_gradient(best_v)
        return tf.keras.losses.Huber()(target, pred_action_value)



def update_target_param():
    vars = tf.compat.v1.global_variables()
    ops = []
    G = tf.compat.v1.get_default_graph()
    for v in vars:
        target_name = v.op.name
        if target_name.startswith('target'):
            new_name = target_name.replace('target/', '')
            ops.append(v.assign(G.get_tensor_by_name(new_name + ':0')))
    return tf.group(*ops, name='update_target_network')


class TfAdapter:
    def __init__(self):
        self.model = Model(get_player().action_space.n)
        self.placeholders = (tf.compat.v1.placeholder(tf.float32, (None, 84, 75, 5)),
                             tf.compat.v1.placeholder(tf.int64, (None,)),
                             tf.compat.v1.placeholder(tf.float32, (None,)),
                             tf.compat.v1.placeholder(tf.bool, (None,)))
        self.opt = tf.compat.v1.train.RMSPropOptimizer(1e-3, decay=0.95, momentum=0.95, epsilon=1e-2)
        self.train_op = self.opt.minimize(self.model.build_graph(*self.placeholders))

        self.update_target_op = update_target_param()

        self.sess = tf.compat.v1.Session()
        self.infer = self.sess.make_callable(fetches=['Qvalue:0'], feed_list=['state:0'])
        self.sess.run(tf.compat.v1.initialize_all_variables())

    def train_step(self, batch):
        self.sess.run(self.train_op, feed_dict=dict(zip(self.placeholders, batch)))

    def update_target(self):
        with self.sess.as_default():
            self.update_target_op.run()


def main():
    BATCH_SIZE = 64
    UPDATE_FREQ = 4  # the number of new state transitions per parameter update (per training step)
    MEMORY_SIZE = 10**6
    INIT_MEMORY_SIZE = MEMORY_SIZE // 20
    NUM_PARALLEL_PLAYERS = 1
    MIN_EPSILON = 0.1
    START_EPSILON = 1.0
    STOP_EPSILON_DECAY_AT = 250000

    tf.compat.v1.disable_v2_behavior()
    adapter = TfAdapter()
    summary_writer = tf.compat.v1.summary.FileWriter(datetime.datetime.now().strftime('logs/%d-%m-%Y_%H-%M'))
    expreplay = ExpReplay(
        adapter.infer,
        get_player=get_player,
        num_parallel_players=NUM_PARALLEL_PLAYERS,
        state_shape=(84, 75),
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        update_frequency=UPDATE_FREQ,
        history_len=get_player().action_space.n,
        state_dtype=np.float32
    )
    expreplay._before_train()
    for step_idx, batch in enumerate(expreplay):
        adapter.train_step(batch)
        if expreplay.exploration > MIN_EPSILON:
            expreplay.exploration -= (START_EPSILON - MIN_EPSILON) / STOP_EPSILON_DECAY_AT
        if step_idx > 0 and step_idx % 5000 == 0:
            adapter.update_target()
            mean, max = expreplay.runner.reset_stats()
            summary_writer.add_summary(
                tf.compat.v1.Summary(value=(
                    tf.compat.v1.Summary.Value(tag='expreplay/mean_score', simple_value=mean),
                    tf.compat.v1.Summary.Value(tag='expreplay/max_score', simple_value=max))),
                step_idx)


if __name__ == '__main__':
    main()
