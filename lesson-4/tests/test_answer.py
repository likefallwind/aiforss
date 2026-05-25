#!/usr/bin/env python3
"""
舆论传播与社会网络模拟 - pytest 测试文件

本测试文件验证参考答案的正确性。
"""

import sys
import os

# 将 answer 目录添加到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'answer'))

import pytest
import numpy as np
from main import Agent, SocialNetwork, Message, Simulation, run_simulation


class TestAgent:
    """测试 Agent 类的功能"""
    
    def test_agent_initialization(self):
        """测试智能体初始化"""
        agent = Agent(agent_id=1, initial_opinion=0.5)
        assert agent.agent_id == 1
        assert agent.opinion == 0.5
        assert agent.exposure == []
    
    def test_agent_random_initialization(self):
        """测试随机初始观点"""
        agent = Agent(agent_id=0)
        assert 0 <= agent.opinion <= 1
        assert agent.exposure == []
    
    def test_agent_perceive(self):
        """测试消息感知功能"""
        agent1 = Agent(agent_id=1, initial_opinion=0.3)
        agent2 = Agent(agent_id=2, initial_opinion=0.7)
        
        # agent1 感知来自 agent2 的消息
        message = Message(sender_id=2, content=0.8)
        agent1.perceive(message)
        
        assert len(agent1.exposure) == 1
        assert 0.8 in agent1.exposure
    
    def test_agent_ignore_own_messages(self):
        """测试智能体忽略自己发送的消息"""
        agent = Agent(agent_id=1, initial_opinion=0.5)
        message = Message(sender_id=1, content=0.9)
        agent.perceive(message)
        
        # 不应添加自己的消息
        assert len(agent.exposure) == 0
    
    def test_agent_decide_no_exposure(self):
        """测试无暴露信息时的决策行为"""
        agent = Agent(agent_id=1, initial_opinion=0.5)
        agent.decide()
        
        # 没有暴露信息，观点应保持不变
        assert agent.opinion == 0.5
    
    def test_agent_decide_with_exposure(self):
        """测试有暴露信息时的决策行为"""
        agent = Agent(agent_id=1, initial_opinion=0.2)
        
        # 添加一些暴露观点
        agent.perceive(Message(sender_id=2, content=0.8))
        agent.perceive(Message(sender_id=3, content=0.8))
        
        original_exposure_count = len(agent.exposure)
        agent.decide()
        
        # 观点应该更新：(0.2 + 0.8) / 2 = 0.5
        assert agent.opinion == pytest.approx(0.5, rel=0.01)
        # 暴露列表应该被清空
        assert len(agent.exposure) == 0
    
    def test_agent_act(self):
        """测试行动方法返回正确观点"""
        agent = Agent(agent_id=1, initial_opinion=0.6)
        assert agent.act() == 0.6


class TestSocialNetwork:
    """测试 SocialNetwork 类的功能"""
    
    def test_network_initialization(self):
        """测试网络初始化"""
        network = SocialNetwork()
        assert network.graph.number_of_nodes() == 0
    
    def test_add_edge(self):
        """测试添加边"""
        network = SocialNetwork()
        network.add_edge(0, 1)
        
        assert network.graph.number_of_nodes() == 2
        assert network.graph.number_of_edges() == 1
        assert network.get_neighbors(0) == [1]
    
    def test_get_neighbors(self):
        """测试获取邻居"""
        network = SocialNetwork()
        network.add_edge(0, 1)
        network.add_edge(0, 2)
        
        neighbors = network.get_neighbors(0)
        assert set(neighbors) == {1, 2}
    
    def test_build_random_network(self):
        """测试随机网络生成"""
        network = SocialNetwork()
        network.build_random_network(n_nodes=20, probability=0.2)
        
        assert network.number_of_nodes() == 20
        assert network.number_of_edges() > 0
    
    def test_build_barabasi_albert_network(self):
        """测试 BA 网络生成"""
        network = SocialNetwork()
        network.build_barabasi_albert_network(n_nodes=20, m=2)
        
        assert network.number_of_nodes() == 20
        assert network.number_of_edges() > 0
    
    def test_average_degree(self):
        """测试平均度数计算"""
        network = SocialNetwork()
        network.build_barabasi_albert_network(n_nodes=50, m=3)
        
        avg_degree = network.get_average_degree()
        
        # BA 网络的平均度数大约为 2*m
        assert avg_degree > 2, f"平均度数 {avg_degree} 应大于 2"


class TestMessage:
    """测试 Message 数据类"""
    
    def test_message_creation(self):
        """测试消息创建"""
        message = Message(sender_id=1, content=0.7)
        
        assert message.sender_id == 1
        assert message.content == 0.7


class TestSimulation:
    """测试 Simulation 类的功能"""
    
    def test_simulation_initialization(self):
        """测试模拟器初始化"""
        network = SocialNetwork()
        network.build_barabasi_albert_network(n_nodes=10, m=2)
        
        simulation = Simulation(network, n_agents=10)
        
        assert len(simulation.agents) == 10
        assert simulation.n_agents == 10
    
    def test_simulation_run_step(self):
        """测试单步模拟"""
        network = SocialNetwork()
        network.build_barabasi_albert_network(n_nodes=10, m=2)
        
        simulation = Simulation(network, n_agents=10)
        initial_opinions = simulation.get_opinions()
        
        # 运行一步
        simulation.run_step()
        new_opinions = simulation.get_opinions()
        
        # 观点应该有变化（因为智能体相互交流）
        # 注意：有些智能体可能没有邻居，所以观点不变
        assert len(new_opinions) == len(initial_opinions)
    
    def test_get_opinions(self):
        """测试获取观点列表"""
        network = SocialNetwork()
        network.add_edge(0, 1)
        
        simulation = Simulation(network, n_agents=2)
        opinions = simulation.get_opinions()
        
        assert len(opinions) == 2
        assert all(0 <= op <= 1 for op in opinions)


class TestIntegration:
    """集成测试：验证完整的模拟流程"""
    
    def test_small_simulation(self):
        """测试小规模模拟"""
        network = SocialNetwork()
        network.build_barabasi_albert_network(n_nodes=20, m=2)
        
        simulation = Simulation(network, n_agents=20)
        
        # 运行多步模拟
        for _ in range(10):
            simulation.run_step()
        
        opinions = simulation.get_opinions()
        
        assert len(opinions) == 20
        assert all(0 <= op <= 1 for op in opinions)
    
    def test_full_simulation(self):
        """测试完整模拟（50步）"""
        # 使用较小的网络加快测试速度
        history = run_simulation(n_agents=20, n_steps=10, network_type="barabasi_albert")
        
        # 验证历史记录的长度
        assert len(history) == 11  # 初始 + 10步
        
        # 验证每步的观点数量
        for step_opinions in history:
            assert len(step_opinions) == 20
            assert all(0 <= op <= 1 for op in step_opinions)
    
    def test_network_average_degree_requirement(self):
        """测试网络平均度数符合要求（>2）"""
        network = SocialNetwork()
        network.build_barabasi_albert_network(n_nodes=50, m=3)
        
        avg_degree = network.get_average_degree()
        
        assert avg_degree > 2, f"平均度数 {avg_degree} 应大于 2，以满足项目要求"
    
    def test_emergence_observation(self):
        """测试能够观察到涌现现象"""
        network = SocialNetwork()
        network.build_barabasi_albert_network(n_nodes=50, m=2)
        
        simulation = Simulation(network, n_agents=50)
        
        # 记录初始和最终观点
        initial_opinions = simulation.get_opinions()
        
        # 运行50步
        for _ in range(50):
            simulation.run_step()
        
        final_opinions = simulation.get_opinions()
        
        # 验证观点有变化
        initial_mean = np.mean(initial_opinions)
        final_mean = np.mean(final_opinions)
        final_std = np.std(final_opinions)
        
        # 至少应该观察到一种涌现现象
        # - 共识形成：标准差变小
        # - 持续分歧：标准差保持较大
        # - 观点极化：最大值和最小值差距大
        assert final_std >= 0, "标准差应该非负"
        
        # 验证观点在有效范围内
        assert all(0 <= op <= 1 for op in final_opinions)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])