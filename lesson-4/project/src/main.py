#!/usr/bin/env python3
"""
舆论传播与社会网络模拟 - 脚手架文件

本文件提供了多智能体社会模拟系统的框架，学生需要补全 TODO 部分。
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import random


@dataclass
class Message:
    """消息数据结构，用于智能体之间的通信"""
    sender_id: int  # 发送者ID
    content: float  # 观点内容 (0-1 之间的值)


class Agent:
    """
    智能体类：模拟社交网络中的个体
    
    状态：
        - opinion: 观点值 (0-1)
        - exposure: 已接触的观点列表
    
    方法：
        - perceive(message): 接收并处理消息
        - decide(): 根据暴露信息更新观点
        - act(): 返回当前观点
    """
    
    def __init__(self, agent_id: int, initial_opinion: float = None):
        """
        初始化智能体
        
        Args:
            agent_id: 智能体唯一标识符
            initial_opinion: 初始观点值，若为None则随机生成
        """
        self.agent_id = agent_id
        
        # TODO: 初始化智能体状态
        # - opinion: 如果initial_opinion为None，随机生成0-1之间的值
        # - exposure: 初始化为空列表，存储已接收的消息内容
        raise NotImplementedError("请在此处实现初始化逻辑")
    
    def perceive(self, message: Message):
        """
        感知方法：接收并存储来自其他智能体的消息
        
        Args:
            message: 收到的消息对象
        """
        # TODO: 将消息内容添加到exposure列表中
        # 注意：不要添加自己的消息
        raise NotImplementedError("请在此处实现感知逻辑")
    
    def decide(self):
        """
        决策方法：根据已暴露的信息更新自己的观点
        
        使用简单的平均模型：新观点 = (当前观点 + 所有暴露观点的平均) / 2
        """
        # TODO: 实现观点更新逻辑
        # 1. 如果exposure为空，保持当前观点不变
        # 2. 否则，计算exposure中所有观点的平均值
        # 3. 新观点 = (当前观点 + 平均暴露观点) / 2
        raise NotImplementedError("请在此处实现决策逻辑")
    
    def act(self) -> float:
        """
        行动方法：返回当前观点值
        
        Returns:
            当前观点值 (0-1)
        """
        return self.opinion


class SocialNetwork:
    """
    社会网络类：使用图结构表示人际关系网络
    
    使用 NetworkX 构建社交网络拓扑
    """
    
    def __init__(self):
        """初始化空的网络图"""
        self.graph = nx.Graph()
    
    def add_edge(self, node1: int, node2: int):
        """
        添加一条边（社交关系）
        
        Args:
            node1: 节点1 ID
            node2: 节点2 ID
        """
        # TODO: 使用NetworkX添加边
        raise NotImplementedError("请在此处实现添加边的逻辑")
    
    def get_neighbors(self, node: int) -> List[int]:
        """
        获取节点的邻居列表
        
        Args:
            node: 节点ID
        
        Returns:
            邻居节点ID列表
        """
        # TODO: 返回指定节点的所有邻居
        raise NotImplementedError("请在此处实现获取邻居的逻辑")
    
    def build_random_network(self, n_nodes: int, probability: float = 0.1):
        """
        使用 Erdős-Rényi 随机图模型构建网络
        
        Args:
            n_nodes: 节点数量
            probability: 每对节点之间存在边的概率
        """
        # TODO: 使用 nx.erdos_renyi_graph 生成随机网络
        raise NotImplementedError("请在此处实现随机网络构建")
    
    def build_barabasi_albert_network(self, n_nodes: int, m: int = 2):
        """
        使用 Barabási-Albert 模型构建无标度网络
        
        该模型生成的网络符合"富者愈富"的社会网络特征
        
        Args:
            n_nodes: 节点数量
            m: 每个新节点连接到的已有节点数量
        """
        # TODO: 使用 nx.barabasi_albert_graph 生成BA网络
        raise NotImplementedError("请在此处实现BA网络构建")
    
    def get_average_degree(self) -> float:
        """
        获取网络平均度数
        
        Returns:
            网络的平均度数
        """
        if self.graph.number_of_nodes() == 0:
            return 0.0
        degrees = dict(self.graph.degree())
        return sum(degrees.values()) / len(degrees)


class Simulation:
    """
    模拟器类：协调整个多智能体模拟过程
    
    负责：
        - 管理网络和智能体
        - 执行每个时间步的消息传递
        - 收集模拟结果
    """
    
    def __init__(self, network: SocialNetwork, n_agents: int):
        """
        初始化模拟器
        
        Args:
            network: 社会网络对象
            n_agents: 智能体数量
        """
        self.network = network
        self.n_agents = n_agents
        
        # TODO: 初始化智能体列表
        # 为每个智能体创建Agent实例，初始观点随机生成
        raise NotImplementedError("请在此处实现智能体初始化")
    
    def run_step(self):
        """
        运行一个时间步的模拟
        
        步骤：
            1. 每个智能体向所有邻居广播自己的观点
            2. 邻居接收并感知消息
            3. 每个智能体根据接收的消息决定是否更新观点
        """
        # TODO: 实现消息传递和观点更新逻辑
        # 1. 创建消息列表
        # 2. 遍历每个智能体，生成发给其邻居的消息
        # 3. 调用智能体的perceive方法接收消息
        # 4. 调用智能体的decide方法更新观点
        raise NotImplementedError("请在此处实现单步模拟逻辑")
    
    def get_opinions(self) -> List[float]:
        """
        获取所有智能体当前的观点
        
        Returns:
            观点值列表
        """
        # TODO: 返回所有智能体的当前观点
        raise NotImplementedError("请在此处实现获取观点的逻辑")


def run_simulation(n_agents: int = 50, n_steps: int = 50, network_type: str = "barabasi_albert"):
    """
    运行完整的模拟并可视化结果
    
    Args:
        n_agents: 智能体数量
        n_steps: 模拟步数
        network_type: 网络类型 ("random" 或 "barabasi_albert")
    
    Returns:
        opinions_history: 每步的观点分布历史
    """
    print(f"初始化模拟: {n_agents} 个智能体, {n_steps} 步, 网络类型: {network_type}")
    
    # TODO: 创建社会网络
    # 1. 创建SocialNetwork实例
    # 2. 根据network_type调用相应方法构建网络
    raise NotImplementedError("请在此处实现网络构建")
    
    # TODO: 创建模拟器
    raise NotImplementedError("请在此处创建模拟器")
    
    # 记录历史
    opinions_history = []
    
    print("开始模拟...")
    for step in range(n_steps):
        # TODO: 运行一步模拟
        # 1. 调用simulation.run_step()
        # 2. 收集当前观点并添加到历史记录
        raise NotImplementedError("请在此处实现模拟循环")
    
    print(f"模拟完成，共 {n_steps} 步")
    
    # TODO: 可视化结果
    # 1. 创建包含两个子图的图表
    # 2. 子图1: 观点随时间变化的折线图
    # 3. 子图2: 初始和最终观点分布的直方图
    # 4. 添加标题、标签和图例
    # 5. 调整布局并保存/显示
    raise NotImplementedError("请在此处实现可视化")
    
    return opinions_history


if __name__ == "__main__":
    # 运行默认模拟
    print("=" * 50)
    print("舆论传播与社会网络模拟")
    print("=" * 50)
    
    # 示例：使用 BA 网络运行模拟
    history = run_simulation(
        n_agents=50,
        n_steps=50,
        network_type="barabasi_albert"
    )
    
    print("\n模拟完成！")