#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu

import datetime
import itertools
import numpy as np
import cv2
import gym
import gym.envs.atari
import gym.wrappers
import tensorflow as tf

import tensorpack
from tensorpack import *

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


class Model(ModelDesc):
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
        with argscope(Conv2D, use_bias=True):
            l = (LinearWrap(image)
                 .Conv2D('conv0', 32, 8, strides=4)
                 .tf.nn.leaky_relu(0.01)
                 .Conv2D('conv1', 64, 4, strides=2)
                 .tf.nn.leaky_relu(0.01)
                 .Conv2D('conv2', 64, 3)
                 .tf.nn.leaky_relu(0.01)
                 .FullyConnected('fc0', 512)
                 .tf.nn.leaky_relu(0.01)())
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
        summary.add_moving_summary(cost)
        return cost

    def optimizer(self):
        return tf.train.RMSPropOptimizer(1e-3, decay=0.95, momentum=0.95, epsilon=1e-2)


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


def main():
    BATCH_SIZE = 64
    IMAGE_SIZE = (84, 75)
    FRAME_HISTORY = 4
    UPDATE_FREQ = 4  # the number of new state transitions per parameter update (per training step)
    MEMORY_SIZE = 10**6
    INIT_MEMORY_SIZE = MEMORY_SIZE // 20
    STEPS_PER_EPOCH = 5000
    NUM_PARALLEL_PLAYERS = 1

    logger.set_logger_dir(datetime.datetime.now().strftime('logs/%d-%m-%Y_%H-%M'))
    trainer = SimpleTrainer()
    model = Model(IMAGE_SIZE, FRAME_HISTORY, get_player().action_space.n)
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        get_player=get_player,
        num_parallel_players=NUM_PARALLEL_PLAYERS,
        state_shape=model.state_shape,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY,
        state_dtype=model.state_dtype.as_numpy_dtype
    )

    trainer.setup_graph(
        model.get_input_signature(), QueueInput(expreplay),
        model.build_graph, model.get_optimizer)
    trainer.setup_callbacks(callbacks=[
            PeriodicTrigger(
                RunOp(update_target_param),
                every_k_steps=5000),    # update target network every 5k steps
            expreplay,
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                ((0, 1), (50, 0.1)),
                interp='linear'),
            MergeAllSummaries()],
        monitors=[TFEventWriter()])

    trainer.sess = tfutils.sesscreate.NewSessionCreator().create_session()
    trainer.initialize_hooks()
    trainer.sess.graph.finalize()
    for pred in trainer._predictors:
        pred.sess = trainer.sess

    with trainer.sess.as_default():
        trainer.loop.config(STEPS_PER_EPOCH, 1, 9**9)
        trainer.loop.update_global_step()
        trainer._callbacks.before_train()  # Used by ExpReplay, TFEventWriter
        for trainer.loop._epoch_num in itertools.count(1):  # TODO itertools.count() when trainer loop is removed
            for trainer.loop._local_step in range(trainer.loop.steps_per_epoch):
                trainer.hooked_sess.run(trainer.train_op)
                trainer._callbacks.trigger_step()  # Used by PeriodicTrigger
            trainer._callbacks.trigger_epoch()  # Used by ExpReplay, ScheduledHyperParamSetter, TFEventWriter


if __name__ == '__main__':
    main()
