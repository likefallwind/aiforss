# -*- coding: utf-8 -*-
"""
测试文件：验证 answer/main.py 中的参考答案实现

本测试文件仅测试 answer 目录下的参考实现。
"""

import pytest
import torch
import sys
import os

# 将 answer 目录添加到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'answer'))

# 导入待测试的模块
from main import (
    simulate_preference_data,
    RewardModel,
    PolicyModel,
    rlhf_training_step,
    dpo_loss,
    UserProfile,
    personalized_generate,
    BaseContentModel,
    tokenize,
    create_batch_from_data
)


# ============================================================
# Task 1: 偏好数据模拟测试
# ============================================================

class TestSimulatePreferenceData:
    """测试偏好数据模拟功能"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        prompts = ["问题1", "问题2", "问题3"]
        responses = ["回答A", "回答B", "回答C"]
        scores = [0.9, 0.5, 0.7]
        
        result = simulate_preference_data(prompts, responses, scores)
        
        # 验证返回类型
        assert isinstance(result, list), "结果应该是列表"
        
        # 验证返回的数据结构
        if len(result) > 0:
            item = result[0]
            assert 'prompt' in item, "应包含prompt字段"
            assert 'chosen_response' in item, "应包含chosen_response字段"
            assert 'rejected_response' in item, "应包含rejected_response字段"
    
    def test_preference_ordering(self):
        """测试偏好排序正确性"""
        prompts = ["测试问题"]
        responses = ["低质量回答", "高质量回答"]
        scores = [0.3, 0.9]
        
        result = simulate_preference_data(prompts, responses, scores)
        
        # 验证高分回答被选为偏好回答
        for item in result:
            if '高质量回答' in item['chosen_response']:
                assert item['chosen_score'] > item['rejected_score']
    
    def test_empty_input(self):
        """测试空输入"""
        result = simulate_preference_data([], [], [])
        assert isinstance(result, list)
        assert len(result) == 0


# ============================================================
# Task 2: RLHF训练测试
# ============================================================

class TestRewardModel:
    """测试奖励模型"""
    
    def test_model_initialization(self):
        """测试模型初始化"""
        model = RewardModel(embedding_dim=64, vocab_size=5000)
        assert model is not None
        assert model.embedding_dim == 64
    
    def test_forward_pass(self):
        """测试前向传播"""
        model = RewardModel(embedding_dim=64, vocab_size=5000)
        batch_size = 2
        seq_len = 10
        
        # 创建输入张量
        input_tokens = torch.randint(0, 5000, (batch_size, seq_len))
        
        # 前向传播
        with