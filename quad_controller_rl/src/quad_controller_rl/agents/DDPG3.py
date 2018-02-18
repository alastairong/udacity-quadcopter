# To dos:
# Add prioritisation to replay buffer

import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent
from keras import layers, models, optimizers
from keras import backend as K
from collections import deque, namedtuple
import os
import pandas as pd
from quad_controller_rl import util
import h5py
import pickle

Experience = namedtuple("Experience",
    field_names=["state", "action", "reward", "done", "next_state"])

class ReplayBuffer():
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        self.q_delta = deque(maxlen=max_size)
        self.e = 0.1 # added to td_error to prevent zero-error values not being sampled
        self.a = 0.5 # controls balance between fully prioritised or random uniform. 0 = random uniform. 1 = fully prioritised

    def add(self, state, action, reward, done, next_state, td_error):
        e = Experience(state, action, reward, done, next_state)
        self.buffer.append(e)
        self.q_delta.append(td_error + self.e)

    def sample(self, batch_size):
        td_error_array = np.array(self.q_delta)
        priority = td_error_array**self.a / np.sum(td_error_array**self.a)
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False,
                               p=priority)
        return [self.buffer[ii] for ii in idx]

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=None, theta=0.1, sigma=0.4):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class Actor:
    """Actor (Policy) Model: Maps state variables to actions"""
    def __init__(self, state_size, action_size, action_low, action_high):
        # Initialise and build the neural network
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.build_model()

    def build_model(self):
        # Build Keras model
        states = layers.Input(shape=(self.state_size,), name="states")
        NN = layers.Dense(32, activation="relu")(states)
        NN = layers.Dense(64, activation="relu")(NN)
        NN = layers.Dropout(0.25)(NN)
        NN = layers.Dense(128, activation="relu")(NN)
        NN = layers.Dropout(0.25)(NN)
        NN = layers.Dense(256, activation="relu")(NN)
        NN = layers.Dropout(0.25)(NN)
        NN = layers.Dense(512, activation="relu")(NN)
        NN = layers.Dropout(0.25)(NN)
        NN = layers.Dense(512, activation="relu")(NN)
        raw_actions = layers.Dense(self.action_size, activation="sigmoid", name="raw_actions")(NN)
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name="actions")(raw_actions)
        self.model = models.Model(inputs=states, output=actions)

        # Define loss function based on Q-value to be provided by Critic
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Define optimiser and custom training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op
        )

class Critic:
    """Critic (Value) Model: Maps state-action pairs to Q-values"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()


    def build_model(self):
        # Build state neural network branch
        states = layers.Input(shape=(self.state_size,), name="states")
        net_states = layers.Dense(32, activation="relu")(states)
        net_states = layers.Dense(64, activation="relu")(net_states)
        net_states = layers.Dropout(0.25)(net_states)
        net_states = layers.Dense(128, activation="relu")(net_states)

        # Build actions neural network branch
        actions = layers.Input(shape=(self.action_size,), name="actions")
        net_actions = layers.Dense(32, activation="relu")(actions)
        net_actions = layers.Dense(64, activation="relu")(net_actions)
        net_actions = layers.Dropout(0.25)(net_actions)
        net_actions = layers.Dense(128, activation="relu")(net_actions)

        # Combine branches and compile
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation("relu")(net)
        net = layers.Dense(128, activation="relu")(net)
        net = layers.Dropout(0.25)(net)
        net = layers.Dense(256, activation="relu")(net)
        net = layers.Dropout(0.25)(net)
        net = layers.Dense(512, activation="relu")(net)
        Q_values = layers.Dense(1, name="q_values")(net)
        self.model = models.Model(input=[states, actions], output=Q_values)
        # Define optimiser and compile
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss="mse")

        # Compute action gradients
        action_gradients = K.gradients(Q_values, actions)
        self.get_action_gradients = K.function(
            inputs = [*self.model.input, K.learning_phase()],
            outputs = action_gradients
        )

class DDPG3(BaseAgent):
    def __init__(self, task):
        # Current environment information
        self.task = task
        self.state_size = np.prod(self.task.observation_space.shape)
        self.state_low = self.task.observation_space.low
        self.state_high = self.task.observation_space.high
        self.state_range = self.state_high - self.state_low
        self.action_size = 3
        self.action_low = self.task.action_space.low[0:3]
        self.action_high = self.task.action_space.high[0:3]
        self.last_state = None
        self.last_action = None
        self.count = 0

        # Set logging directory and items
        self.stats_folder = util.get_param('out')
        self.stats_filename = os.path.join(
            self.stats_folder,
            "stats.csv")  # path to CSV file
        self.actor_local_weights = os.path.join(
            self.stats_folder,
            "actor_local_weights.hdf5")
        self.actor_target_weights = os.path.join(
            self.stats_folder,
            "actor_target_weights.hdf5")
        self.critic_local_weights = os.path.join(
            self.stats_folder,
            "critic_local_weights.hdf5")
        self.critic_target_weights = os.path.join(
            self.stats_folder,
            "critic_target_weights.hdf5")
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save
        self.replay_buffer_pickle = os.path.join(
            self.stats_folder,
            "replay_buffer.pickle")
        self.OU_noise_pickle = os.path.join(
            self.stats_folder,
            "OU_noise.pickle")

        # Initialise stats logging
        self.total_reward = 0.0
        try:
            df_stats = pd.read_csv(self.stats_filename) # If stats already exists, load it
            self.episode_num = df_stats.tail(1)['episode'].item() + 1
            print("save file found")
        except:
            self.total_reward = 0.0
            self.episode_num = 1
            print("no save file found")
        print("Saving {} to {}. Starting at episode {}".format(self.stats_columns, self.stats_folder, self.episode_num))  # [debug]

        # Actor (Policy) initialisation
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        try:
            self.actor_local.model.load_weights(self.actor_local_weights)
            self.actor_target.model.load_weights(self.actor_target_weights)
            print("saved actor weights loaded")
        except:
            self.actor_target.model.set_weights(self.actor_local.model.get_weights())
            print("new actor weights initialised")
        # Critic (Value) initialisation
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        try:
            self.critic_local.model.load_weights(self.critic_local_weights)
            self.critic_target.model.load_weights(self.critic_target_weights)
            print("saved critic weights loaded")
        except:
            self.critic_target.model.set_weights(self.critic_local.model.get_weights())
            print("new critic weights initialised")
        # Set replay buffer
        self.buffer_size = 100000
        self.batch_size = 64
        if os.path.exists(self.replay_buffer_pickle):
            with open(self.replay_buffer_pickle, 'rb') as handle:
                 self.memory = pickle.load(handle)
            print("loading ReplayBuffer from pickle")
        else:
            self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.99
        self.tau = 0.0001

        # Set noise process
        if os.path.exists(self.OU_noise_pickle):
            with open(self.OU_noise_pickle, 'rb') as handle:
                 self.noise = pickle.load(handle)
            print("loading OU_Noise from pickle")
        else:
            self.noise = OUNoise(self.action_size)

        # Reset variables for new episode
        self.reset_episode_vars

    def reset_episode_vars(self):
        """Clear all data from previous episodes"""
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count= 0

    def postprocess_action(self, action):
        """return complete action vector including torques (set as zero)"""
        complete_action = np.zeros(self.task.action_space.shape)
        complete_action[0:3] = action
        return complete_action

    def preprocess_state(self, state):
        return state[0:3] # Position only

    def step(self, state, reward, done):
        """
        Choose and return an action based on current
        state, save the experience in ReplayBuffer, and
        learn if enough samples are available
        """
        # Normalise state between [0, 1]
        #state = self.preprocess_state(state)
        state = (state - self.state_low) / self.state_range
        state = state.reshape(1, -1)

        # Choose action
        action = self.act(state)

        # Calculate td_error for prioritisation

        Q_targets_next = self.critic_target.model.predict([state, action]).item()
        try:
                Q_targets_old = self.critic_target.model.predict([self.last_state, self.last_action]).item()
        except:
                Q_targets_old = Q_targets_next
        td_error = abs(reward + self.gamma * Q_targets_next - Q_targets_old)
        # Save experience
        if self.last_state is not None and self.last_action is not None:
            self.total_reward += reward
            self.count += 1
            self.memory.add(self.last_state, self.last_action, reward, done, state, td_error)

        #Learn, if enough experience
        if len(self.memory.buffer) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

        if done:
            self.write_stats([self.episode_num, self.total_reward])
            print(self.total_reward)
            self.episode_num += 1

            self.actor_local.model.save_weights(self.actor_local_weights)
            self.actor_target.model.save_weights(self.actor_target_weights)
            self.critic_local.model.save_weights(self.critic_local_weights)
            self.critic_target.model.save_weights(self.critic_target_weights)
            self.reset_episode_vars()
            with open(self.replay_buffer_pickle, 'wb') as handle:
                pickle.dump(self.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.OU_noise_pickle, 'wb') as handle:
                pickle.dump(self.noise, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.last_state = state
        self.last_action = action

        return self.postprocess_action(action)
        #return action

    def act(self, state):
        """Run state data through neural network to return action"""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)
        return action + self.noise.sample()

    def learn(self, experiences):
        """
        Update neural network weights by first running and training critic,
        then using critic values to train actor. For both critic and actor
        the target version is updated using a soft update controlled by tau
        """
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename))  # write header first time only
