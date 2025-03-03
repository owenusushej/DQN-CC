# -*- coding: utf-8 -*-
from modules.socket_utils import SocketSender  # 使用绝对导入
import numpy as np

class NetworkMonitor:
    def __init__(self, sender):
        self.sender = sender
        self.rtt_history = []
        self.loss_history = []

        
    def get_current_state(self):
        # 从发送端获取状态（实时简化的示例）
        cwnd_norm = self.sender.cwnd / 1000.0  # 假设最大cwnd=1000
        rtt = np.mean(self.rtt_history[-5:]) if len(self.rtt_history) >0 else 0.1
        loss_rate = np.mean(self.loss_history[-5:]) if len(self.loss_history) >0 else 0.0
        throughput = self.sender.cwnd / max(rtt, 1e-5)
        
        return np.array([cwnd_norm, rtt, loss_rate, throughput])
    
    def update_metrics(self, rtt, is_loss):
        self.rtt_history.append(rtt)
        self.loss_history.append(1.0 if is_loss else 0.0)
        # 保留最近100个测量值
        self.rtt_history = self.rtt_history[-100:]
        self.loss_history = self.loss_history[-100:]
