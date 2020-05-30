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
# import torch
# import torch.nn.functional as F

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
        self.model = tf.keras.Sequential((
            tf.keras.layers.Conv2D(32, kernel_size=8, strides=8, activation='relu', input_shape=(84, 75, 4)),
            tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.num_actions)))
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())


    def build_graph(self, comb_state, action, reward, isOver):
        comb_state = tf.cast(comb_state, tf.float32)
        input_rank = comb_state.shape.rank

        state = comb_state[:, :, :, :-1]

        predict_value = self.model(state)

        reward = tf.clip_by_value(reward, -1, 1)
        next_state = tf.slice(
            comb_state,
            [0] * (input_rank - 1) + [1],
            [-1] * (input_rank - 1) + [self.history], name='next_state')
        next_state = tf.reshape(next_state, self._stacked_state_shape)
        action_onehot = tf.one_hot(tf.cast(action, tf.int32), self.num_actions, 1.0, 0.0)

        pred_action_value = tf.reduce_sum(predict_value * action_onehot, 1)  # N,

        targetQ_predict_value = self.target_model(next_state)    # NxA

        best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * 0.99 * tf.stop_gradient(best_v)

        cost = tf.compat.v1.losses.huber_loss(
            target, pred_action_value, reduction=tf.compat.v1.losses.Reduction.MEAN)
        return cost


class TfAdapter:
    def __init__(self, image_size, frame_history):
        self.model = Model(image_size, frame_history, get_player().action_space.n)
        self.opt = tf.keras.optimizers.RMSprop(1e-3, decay=0.95, momentum=0.95, epsilon=1e-2)

    def infer(self, state):
        return self.model.model(tf.convert_to_tensor(state))

    def train_step(self, batch):
        with tf.GradientTape() as tape:
            loss = self.model.build_graph(*batch)
        gradients = tape.gradient(loss, self.model.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.model.trainable_variables))

    def update_target(self):
        self.model.target_model.set_weights(self.model.model.get_weights())


# class DQN(torch.nn.Module):
    # def __init__(self, state_dim, n_actions):
        # super().__init__()
        # self.conv1 = torch.nn.Conv2d(state_dim, 32, kernel_size=8, stride=4, padding=4)
        # self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        # self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.linear1 = torch.nn.Linear(7680, 512)
        # self.linear2 = torch.nn.Linear(512, n_actions)
# 
    # def forward(self, state):
        # c1 = torch.nn.functional.leaky_relu(self.conv3(torch.nn.functional.leaky_relu(self.conv2(torch.nn.functional.leaky_relu(self.conv1(state))))))
        # fc1 = F.leaky_relu(self.linear1(c1.view(c1.shape[0], -1)))
        # return self.linear2(fc1)


class TorchAdapter:
    def __init__(self):
        self.DEVICE = 'cuda'
        self.GAMMA = torch.tensor(0.99, dtype=torch.float32, device=self.DEVICE)
        self.action_state_value_func = DQN(4, 4).to(self.DEVICE)
        self.target_func = DQN(4, 4).to(self.DEVICE)  # copy.deepcopy(action_state_value_func)
        self.target_func.load_state_dict(self.action_state_value_func.state_dict())
        self.target_func.eval()
        self.optimizer = torch.optim.RMSprop(self.action_state_value_func.parameters(), lr=1e-3, alpha=0.99, eps=1e-02,
                                             weight_decay=0.95, momentum=0.95, centered=False)

    def infer(self, state):
        with torch.no_grad():
            return self.action_state_value_func(
                torch.tensor(np.moveaxis(state, 3, 1), dtype=torch.float32, device=self.DEVICE)).cpu()[None, :, :]

    def train_step(self, batch):
        observations = torch.tensor(np.moveaxis(batch[0], 3, 1), dtype=torch.float32, device=self.DEVICE)
        actions = torch.tensor(batch[1].astype(np.int64), dtype=torch.int64, device=self.DEVICE)
        rewards = torch.tensor(batch[2], dtype=torch.float32, device=self.DEVICE).clamp_(-1.0, 1.0)
        dones = torch.tensor(batch[3].astype(np.float32), dtype=torch.float32, device=self.DEVICE)
        with torch.no_grad():
            next_values, _ = self.target_func(observations[:, 1:]).max(1)
        values = self.GAMMA * (1.0 - dones) * next_values + rewards
        self.optimizer.zero_grad()
        chosen_q_values = self.action_state_value_func(observations[:, :-1]).gather(1, actions.unsqueeze(1))
        loss = torch.nn.functional.smooth_l1_loss(chosen_q_values.squeeze(), values)
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_func.load_state_dict(self.action_state_value_func.state_dict())


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
    # adapter = TorchAdapter()
    summary_writer = tf.summary.create_file_writer(datetime.datetime.now().strftime('logs/%d-%m-%Y_%H-%M'))
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
            with summary_writer.as_default():
                tf.summary.scalar('expreplay/mean_score', mean, step_idx)
                tf.summary.scalar('expreplay/max_score', max, step_idx)
                summary_writer.flush()


if __name__ == '__main__':
    main()
