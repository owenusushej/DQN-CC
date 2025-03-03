# -*- coding: utf-8 -*-
from modules.socket_utils import SocketReceiver
import time

class DQNReceiver:
    def __init__(self, host='0.0.0.0', port=6000):
        self.receiver = SocketReceiver(host, port)
        
    def run(self):
        while True:
            data = self.receiver.recv_data()
            if not data:
                break
            print("Received data: {}".format(data))  # Python 2兼容写法

if __name__ == '__main__':
    receiver = DQNReceiver()
    receiver.run()
