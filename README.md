# DQN-based Congestion Control

基于深度强化学习（DQN）的TCP拥塞控制算法实现。

## 环境要求

- **操作系统**: Ubuntu 18.04+
- **Python**: 2.7.x
- **依赖库**: 
  - TensorFlow 1.15.0
  - numpy 1.16.6
  - pyyaml 3.13

## 项目结构

```plaintext
DQN_CONGESTION/
├── dqn_sender.py          # 发送端主程序
├── dqn_receiver.py        # 接收端主程序
├── requirements.txt       # 依赖库列表
├── config.yaml            # 配置文件
├── modules/
│   ├── __init__.py
│   ├── network_env.py     # 网络状态监测模块
│   ├── dqn_model_tf.py    # TensorFlow模型实现
│   └── socket_utils.py    # 套接字通信工具类
```

## 安装步骤

1. 创建Python虚拟环境：
```bash
virtualenv venv --python=python2.7
source venv/bin/activate
```

2. 安装依赖库：
```bash
pip install -r requirements.txt
```

## 配置说明

修改`config.yaml`文件配置接收端地址：
```yaml
receiver_ip: "127.0.0.1"  # 接收端IP地址
receiver_port: 6000       # 接收端监听端口
```

## 运行方法

1. **启动接收端** (需先运行)：
```bash
python dqn_receiver.py
```

2. **启动发送端** (开始训练)：
```bash
python dqn_sender.py
```

## 算法核心配置

在`dqn_model_tf.py`中可调整以下参数：
```python
self.replay_buffer = deque(maxlen=10000)  # 经验回放池大小
self.batch_size = 32                      # 训练批次大小
self.epsilon_decay = 0.995                # 探索衰减率
self.epsilon_min = 0.01                   # 最小探索概率
```

## 实时监控

训练过程中将在发送端输出以下信息：
```bash
Step 100, Cwnd: 45.0, Epsilon: 0.608
Step 200, Cwnd: 78.0, Epsilon: 0.372
...
```

## 核心原理

1. 状态空间：
   - 归一化拥塞窗口
   - RTT时延 
   - 丢包率
   - 吞吐量

2. 奖励函数：
```python
reward = throughput - 0.1*rtt - 50*loss
```

## 已知限制

1. 需要使用Python2.7环境
2. 目前为实验性实现，需优化奖励函数设计
3. 吞吐量计算基于应用层估算