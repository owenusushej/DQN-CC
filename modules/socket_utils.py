# -*- coding: utf-8 -*-
import socket
import time
import struct

class SocketSender(object):
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.cwnd = 10
        self.unacked_packets = 0  # 新的应用层窗口计数器
        
    def send_data(self, data):
        """基于应用层窗口控制的发送逻辑"""
        if self.unacked_packets >= self.cwnd:  # 窗口已满时不发送
            return 0, False
        
        start_time = time.time()
        try:
            packed = struct.pack('!I', len(data)) + data
            sent = self.sock.send(packed)
            self.unacked_packets +=1  # 每发一个包增加计数
            transmission_time = time.time() - start_time
            return transmission_time, False
        except socket.error:
            return 0, True
    
    def on_ack_received(self):  # 新增ACK处理函数
        self.unacked_packets = max(0, self.unacked_packets -1)


class SocketReceiver(object):
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(1)
        self.conn, _ = self.sock.accept()
        
    def recv_data(self):
        raw_len = self._recv_bytes(4)
        if not raw_len:
            return ""
        data_len = struct.unpack('!I', raw_len)[0]
        return self._recv_bytes(data_len)
    
    def _recv_bytes(self, n):
        data = ''
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
