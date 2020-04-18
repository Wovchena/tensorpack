#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu

import datetime
import multiprocessing
import numpy as np
import cv2
import gym
import gym.envs.atari
import gym.wrappers
import tensorflow as tf
from collections import deque

from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

from expreplay import ExpReplay

BATCH_SIZE = 64
IMAGE_SIZE = (84, 75)
FRAME_HISTORY = 4
UPDATE_FREQ = 4  # the number of new state transitions per parameter update (per training step)

MEMORY_SIZE = 1e6
# will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
INIT_MEMORY_SIZE = MEMORY_SIZE // 20
STEPS_PER_EPOCH = 100000 // UPDATE_FREQ  # each epoch is 100k state transitions
NUM_PARALLEL_PLAYERS = 3


class CropGrayScaleResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.shape = (84, 75)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    def observation(self, observation):
        return cv2.resize(cv2.cvtColor(observation[-180:, :160], cv2.COLOR_RGB2GRAY),
                          (75, 84), interpolation=cv2.INTER_AREA)


class FrameStack(gym.Wrapper):
    """
    Buffer consecutive k observations and stack them on a new last axis.
    The output observation has shape `original_shape + (k, )`.
    """
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k - 1):
            self.frames.append(np.zeros_like(ob))
        self.frames.append(ob)
        return self.observation()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self.observation(), reward, done, info

    def observation(self):
        assert len(self.frames) == self.k
        return np.stack(self.frames, axis=-1)


def get_player(viz=False, train=False):
    env = CropGrayScaleResizeWrapper(gym.wrappers.TimeLimit(gym.envs.atari.AtariEnv('breakout', obs_type='image',
        frameskip=4, repeat_action_probability=0.25), 60000))
    if not train:
        # in training, history is taken care of in expreplay buffer
        env = FrameStack(env, FRAME_HISTORY)
    return env


class Model(ModelDesc):
    state_dtype = tf.uint8

    def __init__(self, state_shape, history, method, num_actions):
        """
        Args:
            state_shape (tuple[int]),
            history (int):
        """
        self.state_shape = tuple(state_shape)
        self._stacked_state_shape = (-1, ) + self.state_shape + (history, )
        self.history = history
        self.method = method
        self.num_actions = num_actions

    def inputs(self):
        # When we use h history frames, the current state and the next state will have (h-1) overlapping frames.
        # Therefore we use a combined state for efficiency:
        # The first h are the current state, and the last h are the next state.
        return [tf.TensorSpec((None,) + self.state_shape + (self.history + 1, ), self.state_dtype, 'comb_state'),
                tf.TensorSpec((None,), tf.int64, 'action'),
                tf.TensorSpec((None,), tf.float32, 'reward'),
                tf.TensorSpec((None,), tf.bool, 'isOver')]

    @auto_reuse_variable_scope
    def get_DQN_prediction(self, image):
        assert image.shape.rank in [4, 5], image.shape
        # image: N, H, W, (C), Hist
        if image.shape.rank == 5:
            # merge C & Hist
            image = tf.reshape(
                image,
                [-1] + list(self.state_shape[:2]) + [self.state_shape[2] * FRAME_HISTORY])

        image = image / 255.0
        with argscope(Conv2D, activation=lambda x: PReLU('prelu', x), use_bias=True):
            l = (LinearWrap(image)
                 # Nature architecture
                 .Conv2D('conv0', 32, 8, strides=4)
                 .Conv2D('conv1', 64, 4, strides=2)
                 .Conv2D('conv2', 64, 3)
                 .FullyConnected('fc0', 512)
                 .tf.nn.leaky_relu(alpha=0.01)())
        Q = FullyConnected('fct', l, self.num_actions)
        return tf.identity(Q, name='Qvalue')

    def build_graph(self, comb_state, action, reward, isOver):
        comb_state = tf.cast(comb_state, tf.float32)
        input_rank = comb_state.shape.rank

        state = tf.slice(
            comb_state,
            [0] * input_rank,
            [-1] * (input_rank - 1) + [self.history], name='state')

        self.predict_value = self.get_DQN_prediction(state)
        if not self.training:
            return

        reward = tf.clip_by_value(reward, -1, 1)
        next_state = tf.slice(
            comb_state,
            [0] * (input_rank - 1) + [1],
            [-1] * (input_rank - 1) + [self.history], name='next_state')
        next_state = tf.reshape(next_state, self._stacked_state_shape)
        action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)

        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        summary.add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'), varreplace.freeze_variables(skip_collection=True):
            targetQ_predict_value = self.get_DQN_prediction(next_state)    # NxA

        best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * 0.99 * tf.stop_gradient(best_v)

        cost = tf.losses.huber_loss(
            target, pred_action_value, reduction=tf.losses.Reduction.MEAN)
        summary.add_param_summary(('conv.*/W', ['histogram', 'rms']),
                                  ('fc.*/W', ['histogram', 'rms']))   # monitor all W
        summary.add_moving_summary(cost)
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        tf.summary.scalar("learning_rate-summary", lr)
        opt = tf.train.RMSPropOptimizer(lr, decay=0.95, momentum=0.95, epsilon=1e-2)
        return optimizer.apply_grad_processors(opt, [gradproc.SummaryGradient()])


def update_target_param():
    vars = tf.global_variables()
    ops = []
    G = tf.get_default_graph()
    for v in vars:
        target_name = v.op.name
        if target_name.startswith('target'):
            new_name = target_name.replace('target/', '')
            logger.info("Target Network Update: {} <- {}".format(target_name, new_name))
            ops.append(v.assign(G.get_tensor_by_name(new_name + ':0')))
    return tf.group(*ops, name='update_target_network')


def get_config(model):
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        get_player=lambda: get_player(train=True),
        num_parallel_players=NUM_PARALLEL_PLAYERS,
        state_shape=model.state_shape,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY,
        state_dtype=model.state_dtype.as_numpy_dtype
    )

    # Set to other values if you need a different initial exploration
    # (e.g., # if you're resuming a training half-way)
    # expreplay.exploration = 1.0

    return TrainConfig(
        data=QueueInput(expreplay),
        model=model,
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(update_target_param, verbose=True),
                every_k_steps=5000),    # update target network every 5k steps
            expreplay,
            ScheduledHyperParamSetter('learning_rate',
                                      ((0, 1e-3),)),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                ((0, 1), (10, 0.1)),   # 1->0.1 in the first million steps
                interp='linear')
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=500,  # a total of 50M state transition
    )


if __name__ == '__main__':
    logger.set_logger_dir(datetime.datetime.now().strftime('logs/%d-%m-%Y_%H-%M'))
    config = get_config(Model(IMAGE_SIZE, FRAME_HISTORY, 'DQN', get_player().action_space.n))
    config.session_init = SmartInit(None)
    launch_train_with_config(config, SimpleTrainer())
