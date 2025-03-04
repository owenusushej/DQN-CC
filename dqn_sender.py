# -*- coding: utf-8 -*-
import sys
import time
import yaml
import numpy as np
from modules.socket_utils import SocketSender
from modules.network_env import NetworkMonitor
from modules.dqn_model_tf import DQNTF

class DQNSender(object):
    def __init__(self, receiver_ip, receiver_port):
        self.sender = SocketSender(receiver_ip, receiver_port)
        self.agent = DQNTF(state_dim=4, action_dim=5)
        self.monitor = NetworkMonitor(self.sender)
        self.total_steps = 0

    def calculate_reward(self, old_cwnd, new_cwnd, rtt, loss):
        reward = (new_cwnd / max(rtt, 1e-5))  # 吞吐量奖励
        reward -= 0.1 * rtt                  # 延迟惩罚
        if loss:
            reward -= 50                     # 丢包惩罚
        return reward

    def run(self):
        try:
            while True:
                state = self.monitor.get_current_state()
                action = self.agent.select_action(state)
                old_cwnd = self.sender.cwnd
                
                # 执行动作
                new_cwnd = {
                    0: lambda x: x/2.0,
                    1: lambda x: max(1, x-10),
                    2: lambda x: x,
                    3: lambda x: x+10,
                    4: lambda x: x*2.0 
                }[action](old_cwnd)
                self.sender.cwnd = min(new_cwnd, 1000)
                
                # 发送数据并获取状态
                rtt, loss = self.sender.send_data("test_data")
                self.monitor.update_metrics(rtt, loss)
                
                # 计算下一个状态和奖励
                new_state = self.monitor.get_current_state()
                reward = self.calculate_reward(old_cwnd, self.sender.cwnd, rtt, loss)
                done = False
                
                # 存储经验
                self.agent.replay_buffer.append( (state, action, reward, new_state, done) )
                self.agent.update_model()
                
                self.total_steps +=1
                if self.total_steps % 100 ==0:
                    self.agent.sync_target_network()
                    print("Step {}, Cwnd: {}, Epsilon: {:.3f}".format(
                        self.total_steps, self.sender.cwnd, self.agent.epsilon
                    ))
                    
        except KeyboardInterrupt:
            print("Training stopped.")

if __name__ == '__main__':
    sender = DQNSender()
    sender.run()
