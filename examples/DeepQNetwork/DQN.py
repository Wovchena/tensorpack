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

import tensorpack

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
        frameskip=4, repeat_action_probability=0.25), 60000))


class Model:
    state_dtype = tf.float32

    def __init__(self, state_shape, history, num_actions):
        """
        Args:
            state_shape (tuple[int]),
            history (int):
        """
        self.state_shape = tuple(state_shape)
        self._stacked_state_shape = (-1, ) + self.state_shape + (history, )
        self.history = history
        self.num_actions = num_actions

    def inputs(self):
        # When we use h history frames, the current state and the next state will have (h-1) overlapping frames.
        # Therefore we use a combined state for efficiency:
        # The first h are the current state, and the last h are the next state.
        return [tf.TensorSpec((None,) + self.state_shape + (self.history + 1, ), self.state_dtype, 'comb_state'),
                tf.TensorSpec((None,), tf.int64, 'action'),
                tf.TensorSpec((None,), tf.float32, 'reward'),
                tf.TensorSpec((None,), tf.bool, 'isOver')]

    @tensorpack.tfutils.scope_utils.auto_reuse_variable_scope
    def get_DQN_prediction(self, image):
        # model = tf.keras.Sequential((
        #     tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation=tf.keras.layers.LeakyReLU(0.01)),
        #     tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation=tf.keras.layers.LeakyReLU(0.01)),
        #     tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation=tf.keras.layers.LeakyReLU(0.01)),
        #     tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(0.01)),
        #     tf.keras.layers.Dense(self.num_actions)))
        Q = (tensorpack.LinearWrap(image)
             .Conv2D('conv0', 32, 8, strides=4)
             .tf.nn.leaky_relu(0.01)
             .Conv2D('conv1', 64, 4, strides=2)
             .tf.nn.leaky_relu(0.01)
             .Conv2D('conv2', 64, 3)
             .tf.nn.leaky_relu(0.01)
             .FullyConnected('fc0', 512)
             .tf.nn.leaky_relu(0.01)
             .FullyConnected('fct', self.num_actions)())
        return tf.identity(Q, name='Qvalue')

    def build_graph(self, comb_state, action, reward, isOver):
        comb_state = tf.cast(comb_state, tf.float32)
        input_rank = comb_state.shape.rank

        state = tf.slice(
            comb_state,
            [0] * input_rank,
            [-1] * (input_rank - 1) + [self.history], name='state')

        predict_value = self.get_DQN_prediction(state)

        reward = tf.clip_by_value(reward, -1, 1)
        next_state = tf.slice(
            comb_state,
            [0] * (input_rank - 1) + [1],
            [-1] * (input_rank - 1) + [self.history], name='next_state')
        next_state = tf.reshape(next_state, self._stacked_state_shape)
        action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)

        pred_action_value = tf.reduce_sum(predict_value * action_onehot, 1)  # N,

        with tf.variable_scope('target'):
            targetQ_predict_value = self.get_DQN_prediction(next_state)    # NxA

        best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * 0.99 * tf.stop_gradient(best_v)

        cost = tf.losses.huber_loss(
            target, pred_action_value, reduction=tf.losses.Reduction.MEAN)
        return cost


def update_target_param():
    vars = tf.global_variables()
    ops = []
    G = tf.get_default_graph()
    for v in vars:
        target_name = v.op.name
        if target_name.startswith('target'):
            new_name = target_name.replace('target/', '')
            ops.append(v.assign(G.get_tensor_by_name(new_name + ':0')))
    return tf.group(*ops, name='update_target_network')


class TfAdapter:
    def __init__(self, image_size, frame_history):
        self.model = Model(image_size, frame_history, get_player().action_space.n)
        self.placeholders = tuple(tf.placeholder(tensor_spec.dtype, shape=tensor_spec.shape, name=tensor_spec.name)
                             for tensor_spec in self.model.inputs())
        self.opt = tf.train.RMSPropOptimizer(1e-3, decay=0.95, momentum=0.95, epsilon=1e-2)
        self.train_op = self.opt.minimize(self.model.build_graph(*self.placeholders))

        self.update_target_op = update_target_param()

        with tf.name_scope('tower-pred-0/'):
            self.model.build_graph(*self.placeholders)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        self.infer = self.sess.make_callable(fetches=['tower-pred-0/Qvalue:0'], feed_list=['tower-pred-0/state:0'])

    def train_step(self, batch):
        self.sess.run(self.train_op, feed_dict=dict(zip(self.placeholders, batch)))

    def update_target(self):
        self.update_target_op.run()


def main():
    BATCH_SIZE = 64
    IMAGE_SIZE = (84, 75)
    FRAME_HISTORY = 4   
    UPDATE_FREQ = 4  # the number of new state transitions per parameter update (per training step)
    MEMORY_SIZE = 10**6
    INIT_MEMORY_SIZE = MEMORY_SIZE // 20
    NUM_PARALLEL_PLAYERS = 1
    MIN_EPSILON = 0.1
    START_EPSILON = 1.0
    STOP_EPSILON_DECAY_AT = 250000

    adapter = TfAdapter(IMAGE_SIZE, FRAME_HISTORY)
    summary_writer = tf.summary.FileWriter(datetime.datetime.now().strftime('logs/%d-%m-%Y_%H-%M'))
    expreplay = ExpReplay(
        adapter.infer,
        get_player=get_player,
        num_parallel_players=NUM_PARALLEL_PLAYERS,
        state_shape=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY,
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
                tf.Summary(value=(
                    tf.Summary.Value(tag='expreplay/mean_score', simple_value=mean),
                    tf.Summary.Value(tag='expreplay/max_score', simple_value=max))),
                step_idx)


if __name__ == '__main__':
    main()
