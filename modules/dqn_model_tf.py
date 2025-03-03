# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
from collections import deque

# Enable TF1 compatibility mode
tf = tf.compat.v1
tf.disable_v2_behavior()

class DQNTF(object):
    def __init__(self, state_dim=4, action_dim=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_model()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def _build_model(self):
        # Input placeholders
        self.input_states = tf.placeholder(tf.float32, [None, self.state_dim], name='input_states')
        self.target_q = tf.placeholder(tf.float32, [None], name='target_q')
        self.actions = tf.placeholder(tf.int32, [None], name='actions')
        
        # Main network using compat layers
        with tf.variable_scope('main_net'):
            fc1 = tf.layers.dense(
                inputs=self.input_states,
                units=128,
                activation=tf.nn.relu,
                kernel_initializer=tf.initializers.glorot_uniform()
            )
            fc2 = tf.layers.dense(
                inputs=fc1, 
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.initializers.glorot_uniform()
            )
            self.q_values = tf.layers.dense(
                inputs=fc2,
                units=self.action_dim,
                kernel_initializer=tf.initializers.glorot_uniform()
            )
        
        # Target network
        with tf.variable_scope('target_net'):
            fc1_t = tf.layers.dense(
                inputs=self.input_states,
                units=128, 
                activation=tf.nn.relu,
                trainable=False
            )
            fc2_t = tf.layers.dense(
                inputs=fc1_t,
                units=64,
                activation=tf.nn.relu,
                trainable=False
            )
            self.target_q_values = tf.layers.dense(
                inputs=fc2_t,
                units=self.action_dim,
                trainable=False
            )
        
        # Q-learning update
        action_indices = tf.stack([tf.range(tf.shape(self.actions)[0]), self.actions], axis=1)
        self.current_q = tf.gather_nd(self.q_values, action_indices)
        self.loss = tf.reduce_mean(tf.square(self.target_q - self.current_q))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        
        # Parameter synchronization operations
        main_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main_net')
        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        self.sync_ops = [tf.assign(t, m) for t, m in zip(target_params, main_params)]
        
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        q = self.sess.run(self.q_values, {self.input_states: [state]})
        return np.argmax(q[0])
    
    def update_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Calculate target Q values
        target_q_next = self.sess.run(
            self.target_q_values,
            {self.input_states: next_states}
        )
        target_q_next_max = np.max(target_q_next, axis=1)
        targets = rewards + (1 - np.array(dones)) * 0.99 * target_q_next_max
        
        # Train step
        feed_dict = {
            self.input_states: states,
            self.actions: actions,
            self.target_q: targets
        }
        self.sess.run(self.optimizer, feed_dict)
        
        # ε decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def sync_target_network(self):
        self.sess.run(self.sync_ops)
