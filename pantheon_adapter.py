# File: DQN_CONGESTION/pantheon_adapter.py

import argparse
from dqn_sender import DQNSender
from dqn_receiver import DQNReceiver
import yaml

def main():
    parser = argparse.ArgumentParser(description='DQN Congestion Control Adapter')
    parser.add_argument('--role', choices=['sender', 'receiver'], required=True)
    parser.add_argument('--ip', type=str, help='Receiver IP (sender mode only)')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--config-path', default='config.yaml', help='Path to config file')

    args = parser.parse_args()

    if args.role == 'receiver':
        DQNReceiver(host='0.0.0.0', port=args.port).run()
    else:
        # 动态修改配置
        with open(args.config_path, 'w') as f:
            yaml.dump({
                'receiver_ip': args.ip,
                'receiver_port': args.port
            }, f)
        DQNSender(config_path=args.config_path).run()

if __name__ == '__main__':
    main()
