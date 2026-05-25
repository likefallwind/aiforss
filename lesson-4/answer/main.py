#!/usr/bin/env python3
"""
舆论传播与社会网络模拟 - 参考答案

本文件是完整的参考实现，包含所有功能并可直接运行。
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
        
        # 如果没有提供初始观点，随机生成一个
        if initial_opinion is None:
            self.opinion = random.random()
        else:
            self.opinion = initial_opinion
        
        # 已接收的观点列表
        self.exposure: List[float] = []
    
    def perceive(self, message: Message):
        """
        感知方法：接收并存储来自其他智能体的消息
        
        Args:
            message: 收到的消息对象
        """
        # 只接收来自其他智能体的消息，不接收自己的
        if message.sender_id != self.agent_id:
            self.exposure.append(message.content)
    
    def decide(self):
        """
        决策方法：根据已暴露的信息更新自己的观点
        
        使用简单的平均模型：新观点 = (当前观点 + 所有暴露观点的平均) / 2
        """
        if len(self.exposure) == 0:
            # 如果没有接收到任何消息，保持当前观点
            return
        
        # 计算暴露观点的平均值
        avg_exposure = sum(self.exposure) / len(self.exposure)
        
        # 更新观点：当前观点和平均暴露观点的加权平均
        self.opinion = (self.opinion + avg_exposure) / 2
        
        # 清空暴露列表，为下一轮准备
        self.exposure = []
    
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
        self.graph.add_edge(node1, node2)
    
    def get_neighbors(self, node: int) -> List[int]:
        """
        获取节点的邻居列表
        
        Args:
            node: 节点ID
        
        Returns:
            邻居节点ID列表
        """
        if node in self.graph:
            return list(self.graph.neighbors(node))
        return []
    
    def build_random_network(self, n_nodes: int, probability: float = 0.1):
        """
        使用 Erdős-Rényi 随机图模型构建网络
        
        Args:
            n_nodes: 节点数量
            probability: 每对节点之间存在边的概率
        """
        self.graph = nx.erdos_renyi_graph(n_nodes, probability)
    
    def build_barabasi_albert_network(self, n_nodes: int, m: int = 2):
        """
        使用 Barabási-Albert 模型构建无标度网络
        
        该模型生成的网络符合"富者愈富"的社会网络特征
        
        Args:
            n_nodes: 节点数量
            m: 每个新节点连接到的已有节点数量
        """
        self.graph = nx.barabasi_albert_graph(n_nodes, m)
    
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
    
    def number_of_nodes(self) -> int:
        """返回节点数量"""
        return self.graph.number_of_nodes()
    
    def number_of_edges(self) -> int:
        """返回边数量"""
        return self.graph.number_of_edges()


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
        
        # 初始化智能体列表
        self.agents: List[Agent] = []
        for i in range(n_agents):
            agent = Agent(agent_id=i)
            self.agents.append(agent)
    
    def run_step(self):
        """
        运行一个时间步的模拟
        
        步骤：
            1. 每个智能体向所有邻居广播自己的观点
            2. 邻居接收并感知消息
            3. 每个智能体根据接收的消息决定是否更新观点
        """
        # 第一步：收集所有需要发送的消息
        messages: List[Message] = []
        
        for agent in self.agents:
            # 获取该智能体的邻居
            neighbors = self.network.get_neighbors(agent.agent_id)
            
            # 向每个邻居发送消息
            current_opinion = agent.act()
            for neighbor_id in neighbors:
                message = Message(sender_id=agent.agent_id, content=current_opinion)
                messages.append(message)
        
        # 第二步：让所有智能体感知消息
        for message in messages:
            if message.sender_id < len(self.agents):
                self.agents[message.sender_id].perceive(message)
        
        # 第三步：所有智能体根据接收的消息决定是否更新观点
        for agent in self.agents:
            agent.decide()
    
    def get_opinions(self) -> List[float]:
        """
        获取所有智能体当前的观点
        
        Returns:
            观点值列表
        """
        return [agent.act() for agent in self.agents]


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
    
    # 创建社会网络
    network = SocialNetwork()
    
    # 根据网络类型构建网络
    if network_type == "random":
        network.build_random_network(n_agents, probability=0.1)
    else:  # 默认使用 Barabási-Albert 网络
        network.build_barabasi_albert_network(n_agents, m=2)
    
    # 打印网络统计信息
    avg_degree = network.get_average_degree()
    print(f"网络统计: {network.number_of_nodes()} 节点, {network.number_of_edges()} 边, 平均度数: {avg_degree:.2f}")
    
    # 检查平均度数是否符合要求 (>2)
    if avg_degree <= 2:
        print(f"警告: 平均度数 {avg_degree:.2f} <= 2，网络可能过于稀疏")
    
    # 创建模拟器
    simulation = Simulation(network, n_agents)
    
    # 记录历史
    opinions_history = []
    
    # 收集初始观点
    initial_opinions = simulation.get_opinions()
    opinions_history.append(initial_opinions)
    
    print("开始模拟...")
    for step in range(n_steps):
        # 运行一步模拟
        simulation.run_step()
        
        # 收集当前观点
        current_opinions = simulation.get_opinions()
        opinions_history.append(current_opinions)
        
        # 每10步打印一次进度
        if (step + 1) % 10 == 0:
            mean_opinion = np.mean(current_opinions)
            std_opinion = np.std(current_opinions)
            print(f"  步骤 {step + 1}/{n_steps}: 平均观点={mean_opinion:.3f}, 标准差={std_opinion:.3f}")
    
    print(f"模拟完成，共 {n_steps} 步")
    
    # 分析模拟结果
    final_opinions = opinions_history[-1]
    mean_final = np.mean(final_opinions)
    std_final = np.std(final_opinions)
    
    print("\n=== 模拟结果分析 ===")
    print(f"初始观点分布: 平均={np.mean(initial_opinions):.3f}, 标准差={np.std(initial_opinions):.3f}")
    print(f"最终观点分布: 平均={mean_final:.3f}, 标准差={std_final:.3f}")
    
    # 判断涌现现象
    if std_final < 0.1:
        print("观察到的涌现现象: 共识形成 - 所有智能体观点趋同")
    elif std_final > 0.25 and (max(final_opinions) > 0.8 and min(final_opinions) < 0.2):
        print("观察到的涌现现象: 观点极化 - 智能体分化成不同群体")
    else:
        print("观察到的涌现现象: 持续分歧 - 观点保持多样性和差异性")
    
    # 可视化结果
    visualize_results(opinions_history, network, initial_opinions, final_opinions)
    
    return opinions_history


def visualize_results(opinions_history: List[List[float]], network: SocialNetwork,
                     initial_opinions: List[float], final_opinions: List[float]):
    """
    可视化模拟结果
    
    Args:
        opinions_history: 观点历史记录
        network: 社会网络对象
        initial_opinions: 初始观点列表
        final_opinions: 最终观点列表
    """
    # 设置中文字体（如果可用）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建包含多个子图的图表
    fig = plt.figure(figsize=(15, 10))
    
    # 子图1: 观点随时间变化的折线图
    ax1 = fig.add_subplot(2, 2, 1)
    opinions_array = np.array(opinions_history)
    time_steps = range(len(opinions_history))
    
    # 绘制每个智能体的观点轨迹（使用透明度）
    for i in range(opinions_array.shape[1]):
        ax1.plot(time_steps, opinions_array[:, i], alpha=0.3, linewidth=0.5)
    
    # 绘制平均观点
    mean_opinions = np.mean(opinions_array, axis=1)
    ax1.plot(time_steps, mean_opinions, 'r-', linewidth=2, label='平均观点')
    
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('观点值')
    ax1.set_title('观点随时间演化')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 初始和最终观点分布对比
    ax2 = fig.add_subplot(2, 2, 2)
    bins = np.linspace(0, 1, 21)
    ax2.hist(initial_opinions, bins=bins, alpha=0.5, label='初始分布', color='blue')
    ax2.hist(final_opinions, bins=bins, alpha=0.5, label='最终分布', color='red')
    ax2.set_xlabel('观点值')
    ax2.set_ylabel('智能体数量')
    ax2.set_title('观点分布变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 社会网络可视化（按最终观点着色）
    ax3 = fig.add_subplot(2, 2, 3)
    pos = nx.spring_layout(network.graph, k=0.5, iterations=50, seed=42)
    node_colors = final_opinions
    nx.draw_networkx_edges(network.graph, pos, ax=ax3, alpha=0.2, width=0.5)
    nx.draw_networkx_nodes(network.graph, pos, ax=ax3, node_color=node_colors,
                          cmap=plt.cm.RdYlBu, vmin=0, vmax=1, node_size=50)
    ax3.set_title('社会网络（颜色表示最终观点）')
    ax3.axis('off')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax3, label='观点值', shrink=0.6)
    
    # 子图4: 观点方差随时间变化
    ax4 = fig.add_subplot(2, 2, 4)
    variance_history = [np.var(step_opinions) for step_opinions in opinions_history]
    ax4.plot(time_steps, variance_history, 'g-', linewidth=2)
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('观点方差')
    ax4.set_title('观点分歧程度随时间变化')
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
    print("\n结果图表已保存到 simulation_results.png")
    
    # 显示图表
    plt.show()


def run_comparative_simulation():
    """运行对比模拟：比较不同网络类型的效果"""
    print("=" * 60)
    print("对比模拟：随机网络 vs Barabási-Albert 网络")
    print("=" * 60)
    
    n_agents = 50
    n_steps = 50
    
    results = {}
    
    for network_type in ["random", "barabasi_albert"]:
        print(f"\n--- {network_type} 网络 ---")
        history = run_simulation(n_agents=n_agents, n_steps=n_steps, network_type=network_type)
        
        final_opinions = history[-1]
        results[network_type] = {
            'mean': np.mean(final_opinions),
            'std': np.std(final_opinions),
            'variance': np.var(final_opinions)
        }
    
    print("\n=== 对比结果 ===")
    print(f"{'网络类型':<20} {'最终平均观点':<15} {'观点标准差':<15}")
    print("-" * 50)
    for network_type, stats in results.items():
        print(f"{network_type:<20} {stats['mean']:<15.3f} {stats['std']:<15.3f}")


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
    
    # 可选：运行对比模拟
    # run_comparative_simulation()
    
    print("\n模拟完成！")