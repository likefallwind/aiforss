# -*- coding: utf-8 -*-
"""
以人为本的大模型：价值对齐与个性化

本模块实现了RLHF（强化学习人类反馈）和DPO（直接偏好优化）的简化版本，
以及用户画像系统用于个性化内容生成。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import random
import json


# ============================================================
# Task 1: 偏好数据的模拟与处理
# ============================================================

def simulate_preference_data(prompts: List[str], 
                              responses: List[str], 
                              scores: List[float]) -> List[Dict]:
    """
    模拟生成人类偏好数据，构建偏好对数据集。
    
    Args:
        prompts: 输入提示列表
        responses: 对应每个提示的响应列表
        scores: 每个响应的偏好分数列表
    
    Returns:
        偏好数据列表，每个元素包含 prompt, chosen_response, rejected_response
    """
    # TODO: 实现偏好数据的构建逻辑
    # 1. 根据分数排序，确定每个prompt的偏好回答和拒绝回答
    # 2. 构建偏好对数据结构
    # 3. 返回包含prompt、chosen_response、rejected_response的列表
    
    preference_data = []
    # 在这里实现具体逻辑
    
    return preference_data


# ============================================================
# Task 2: 实现简化的RLHF流程
# ============================================================

class RewardModel(nn.Module):
    """
    奖励模型：学习预测人类对响应的偏好分数
    
    简化实现：使用嵌入 + 线性层对响应进行评分
    """
    
    def __init__(self, embedding_dim: int = 128, vocab_size: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 响应编码器（简化版Transformer结构）
        self.response_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU()
        )
        
        # 奖励输出层
        self.reward_head = nn.Linear(embedding_dim, 1)
        
        # 模拟词汇嵌入
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def encode_response(self, response_tokens: torch.Tensor) -> torch.Tensor:
        """
        编码响应序列
        
        Args:
            response_tokens: 响应的token序列 [batch_size, seq_len]
        
        Returns:
            响应的表示向量 [batch_size, embedding_dim]
        """
        # 获取token嵌入
        embeddings = self.embedding(response_tokens)  # [batch, seq_len, embed_dim]
        
        # 简化处理：取平均作为序列表示
        encoded = embeddings.mean(dim=1)  # [batch, embed_dim]
        
        # 通过编码器
        encoded = self.response_encoder(encoded)
        
        return encoded
    
    def forward(self, response_tokens: torch.Tensor) -> torch.Tensor:
        """
        计算响应的奖励分数
        
        Args:
            response_tokens: 响应的token序列
        
        Returns:
            奖励分数 [batch_size]
        """
        encoded = self.encode_response(response_tokens)
        reward = self.reward_head(encoded).squeeze(-1)
        return reward


class PolicyModel(nn.Module):
    """
    策略模型：用于生成响应的语言模型（简化版）
    
    实际项目中应使用真实的语言模型，这里使用简化实现
    """
    
    def __init__(self, embedding_dim: int = 128, vocab_size: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 简化的生成器
        self.generator = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, vocab_size)
        )
    
    def forward(self, prompt_tokens: torch.Tensor, response_tokens: torch.Tensor) -> torch.Tensor:
        """
        计算响应序列的对数概率
        
        Args:
            prompt_tokens: 提示的token序列
            response_tokens: 响应的token序列
        
        Returns:
            响应的对数概率 [batch_size, seq_len]
        """
        # 简化的实现：基于响应的嵌入计算似然
        embeddings = self.embedding(response_tokens)
        logits = self.generator(embeddings)  # [batch, seq_len, vocab]
        
        # 计算log概率（简化处理）
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs
    
    def get_response_log_probs(self, response_tokens: torch.Tensor) -> torch.Tensor:
        """
        获取响应的平均对数概率
        
        Args:
            response_tokens: 响应的token序列 [batch_size, seq_len]
        
        Returns:
            平均对数概率 [batch_size]
        """
        log_probs = self.forward(None, response_tokens)
        # 沿序列维度取平均
        avg_log_probs = log_probs.mean(dim=1).sum(dim=1)
        return avg_log_probs


def rlhf_training_step(policy_model: PolicyModel,
                       reward_model: RewardModel,
                       prompts: List[str],
                       responses: List[str],
                       optimizer: torch.optim.Optimizer,
                       kl_coef: float = 0.1) -> Dict[str, float]:
    """
    RLHF训练的单步迭代
    
    流程：
    1. 使用当前策略生成响应
    2. 计算奖励模型的奖励分数
    3. 使用策略梯度更新策略模型
    4. 可选：添加KL散度惩罚以防止策略偏离原始模型太远
    
    Args:
        policy_model: 策略模型
        reward_model: 奖励模型
        prompts: 输入提示列表
        responses: 当前策略生成的响应列表
        optimizer: 优化器
        kl_coef: KL散度惩罚系数
    
    Returns:
        训练指标字典
    """
    # TODO: 实现RLHF训练步骤
    # 1. 将prompts和responses转换为token张量
    # 2. 计算奖励模型对responses的评分
    # 3. 计算策略模型的对数概率
    # 4. 计算策略梯度损失（奖励作为基线）
    # 5. 添加KL散度惩罚（可选）
    # 6. 反向传播并更新参数
    
    device = next(policy_model.parameters()).device
    
    # 初始化指标
    metrics = {
        'reward': 0.0,
        'loss': 0.0,
        'kl_penalty': 0.0
    }
    
    # 在这里实现具体训练逻辑
    
    return metrics


# ============================================================
# Task 3: 实现DPO直接偏好优化
# ============================================================

def dpo_loss(policy_model: PolicyModel,
             chosen_logps: torch.Tensor,
             rejected_logps: torch.Tensor,
             beta: float = 0.1) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    DPO（直接偏好优化）损失函数
    
    DPO通过直接优化偏好对数据来对齐模型，无需显式训练奖励模型。
    
    损失函数：
    L = - log(sigmoid(beta * (log(π(y_w|x)) - log(π(y_l|x)) - 
                             log(π_ref(y_w|x)) + log(π_ref(y_l|x)))))
    
    其中 y_w 是偏好回答，y_l 是不偏好回答
    
    Args:
        policy_model: 当前策略模型
        chosen_logps: 偏好回答的对数概率 [batch_size]
        rejected_logps: 不偏好回答的对数概率 [batch_size]
        beta: KL惩罚系数，控制策略偏离参考模型的程度
    
    Returns:
        (损失值, 指标字典)
    """
    # TODO: 实现DPO损失函数
    # 1. 计算优势：chosen_logps - rejected_logps
    # 2. 应用sigmoid并取负对数似然
    # 3. 返回损失和训练指标
    
    # 计算偏好优势
    # 你的代码在这里
    
    # 计算DPO损失
    # 你的代码在这里
    
    # 计算训练指标
    metrics = {
        'dpo_loss': 0.0,
        'advantage': 0.0,
        'accuracy': 0.0
    }
    
    return None, metrics


def compute_log_ratio(policy_logps: torch.Tensor,
                      reference_logps: torch.Tensor) -> torch.Tensor:
    """
    计算策略与参考模型的对数概率比率
    
    Args:
        policy_logps: 策略模型的对数概率
        reference_logps: 参考模型的对数概率
    
    Returns:
        对数比率
    """
    # 简化的对数比率计算
    return policy_logps - reference_logps


# ============================================================
# Task 4: 构建用户画像与个性化生成
# ============================================================

class UserProfile:
    """
    用户画像类：基于交互历史构建用户特征表示
    
    包含以下特征维度：
    - 话题偏好：用户感兴趣的话题分布
    - 风格偏好：用户偏好的回答风格（简洁/详细、专业/通俗等）
    - 交互模式：用户的交互频率和方式
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # 话题偏好分布（简化为词频统计）
        self.topic_preferences: Dict[str, float] = {}
        
        # 风格偏好
        self.style_preferences: Dict[str, float] = {
            'conciseness': 0.5,  # 简洁程度 0-1
            'formality': 0.5,    # 正式程度 0-1
            'technical_level': 0.5  # 技术深度 0-1
        }
        
        # 交互历史
        self.interaction_history: List[Dict] = []
        
        # 统计计数
        self.interaction_count = 0
        self.positive_feedback_count = 0
    
    def update_from_interaction(self, 
                                prompt: str, 
                                response: str, 
                                feedback: Optional[int] = None):
        """
        从单次交互中更新用户画像
        
        Args:
            prompt: 用户输入的提示
            response: 系统返回的响应
            feedback: 用户反馈（1正面/0负面/None无反馈）
        """
        # TODO: 实现用户画像更新逻辑
        # 1. 从prompt中提取话题关键词并更新topic_preferences
        # 2. 分析response的风格特征并更新style_preferences
        # 3. 记录交互历史
        # 4. 更新统计计数
        
        # 在这里实现具体逻辑
        
        self.interaction_count += 1
        if feedback == 1:
            self.positive_feedback_count += 1
    
    def get_topic_vector(self) -> torch.Tensor:
        """
        获取用户话题偏好的向量表示
        
        Returns:
            话题偏好向量
        """
        # 简化实现：将话题偏好转换为固定维度的向量
        topic_dim = 32
        topic_vector = torch.zeros(topic_dim)
        
        # 基于话题分布填充向量
        topics = list(self.topic_preferences.keys())
        for i, topic in enumerate(topics[:topic_dim]):
            topic_vector[i] = self.topic_preferences.get(topic, 0.0)
        
        # 归一化
        if topic_vector.sum() > 0:
            topic_vector = topic_vector / topic_vector.sum()
        
        return topic_vector
    
    def get_style_vector(self) -> torch.Tensor:
        """
        获取用户风格偏好的向量表示
        
        Returns:
            风格偏好向量
        """
        style_values = list(self.style_preferences.values())
        return torch.tensor(style_values, dtype=torch.float32)
    
    def get_profile_representation(self) -> Dict[str, torch.Tensor]:
        """
        获取完整的用户画像表示
        
        Returns:
            包含各维度表示的字典
        """
        return {
            'topic': self.get_topic_vector(),
            'style': self.get_style_vector(),
            'engagement': torch.tensor([
                self.interaction_count,
                self.positive_feedback_count,
                self.positive_feedback_count / max(1, self.interaction_count)
            ], dtype=torch.float32)
        }
    
    def to_dict(self) -> Dict:
        """将用户画像序列化为字典"""
        return {
            'user_id': self.user_id,
            'topic_preferences': self.topic_preferences,
            'style_preferences': self.style_preferences,
            'interaction_count': self.interaction_count,
            'positive_feedback_count': self.positive_feedback_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """从字典反序列化用户画像"""
        profile = cls(data['user_id'])
        profile.topic_preferences = data.get('topic_preferences', {})
        profile.style_preferences = data.get('style_preferences', 
                                              profile.style_preferences)
        profile.interaction_count = data.get('interaction_count', 0)
        profile.positive_feedback_count = data.get('positive_feedback_count', 0)
        return profile


class BaseContentModel:
    """
    基础内容生成模型（简化版）
    
    实际项目中应使用真实的大语言模型
    """
    
    def __init__(self, model_name: str = "simplified-model"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成内容
        
        Args:
            prompt: 输入提示
            **kwargs: 生成参数
        
        Returns:
            生成的响应文本
        """
        # 简化的生成逻辑
        max_length = kwargs.get('max_length', 100)
        temperature = kwargs.get('temperature', 1.0)
        
        # 模拟生成过程
        response = f"[模拟响应] 基于提示「{prompt[:20]}...」生成的内容（长度约{max_length}）"
        return response
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成"""
        return [self.generate(p, **kwargs) for p in prompts]


def personalized_generate(profile: UserProfile,
                          base_model: BaseContentModel,
                          prompt: str,
                          top_k_topics: int = 5) -> Dict[str, any]:
    """
    基于用户画像的个性化内容生成
    
    流程：
    1. 分析当前prompt的话题倾向
    2. 融合用户画像中的偏好特征
    3. 调用基础模型生成内容
    4. 可选：后处理以更好地匹配用户风格
    
    Args:
        profile: 用户画像
        base_model: 基础内容生成模型
        prompt: 用户输入的提示
        top_k_topics: 返回的top话题数量
    
    Returns:
        包含生成结果和元信息的字典
    """
    # TODO: 实现个性化生成逻辑
    # 1. 获取用户画像的向量表示
    # 2. 分析当前prompt的话题
    # 3. 调整生成参数以匹配用户偏好
    # 4. 调用base_model生成内容
    # 5. 返回结果和相关信息
    
    # 获取用户偏好
    # 你的代码在这里
    
    # 生成响应
    response = base_model.generate(prompt)
    
    # 构建返回结果
    result = {
        'prompt': prompt,
        'response': response,
        'user_id': profile.user_id,
        'applied_style': {},
        'matched_topics': []
    }
    
    return result


# ============================================================
# 工具函数与数据处理
# ============================================================

def tokenize(text: str, max_length: int = 50) -> List[int]:
    """
    简单的分词函数（字符级分词作为简化）
    
    Args:
        text: 输入文本
        max_length: 最大长度
    
    Returns:
        token列表
    """
    tokens = [ord(c) % 10000 for c in text[:max_length]]
    # 填充到固定长度
    while len(tokens) < max_length:
        tokens.append(0)
    return tokens


def create_batch_from_data(data: List[Dict], max_length: int = 50) -> torch.Tensor:
    """
    从偏好数据创建batch
    
    Args:
        data: 偏好数据列表
        max_length: 最大序列长度
    
    Returns:
        token张量 [batch_size, max_length]
    """
    batch_tokens = []
    for item in data:
        response = item.get('chosen_response', '')
        tokens = tokenize(response, max_length)
        batch_tokens.append(tokens)
    return torch.tensor(batch_tokens, dtype=torch.long)


# ============================================================
# 主函数与演示
# ============================================================

def main():
    """
    主函数：演示整个价值对齐流程
    """
    print("=" * 60)
    print("以人为本的大模型：价值对齐与个性化")
    print("=" * 60)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # ------------------------------
    # Task 1: 偏好数据模拟
    # ------------------------------
    print("\n[Task 1] 偏好数据模拟")
    print("-" * 40)
    
    sample_prompts = [
        "解释量子计算的基本原理",
        "推荐一本好书",
        "如何学习编程？"
    ]
    sample_responses = [
        "量子计算使用量子比特，可以同时处于多个状态。",
        "《算法导论》是一本经典的计算机科学书籍。",
        "学习编程需要多实践，从简单例子开始。"
    ]
    sample_scores = [0.9, 0.7, 0.8]
    
    preference_data = simulate_preference_data(sample_prompts, sample_responses, sample_scores)
    print(f"生成了 {len(preference_data)} 条偏好数据")
    
    # ------------------------------
    # Task 2: RLHF训练
    # ------------------------------
    print("\n[Task 2] RLHF训练演示")
    print("-" * 40)
    
    # 初始化模型
    reward_model = RewardModel(embedding_dim=128)
    policy_model = PolicyModel(embedding_dim=128)
    
    optimizer = torch.optim.Adam([
        {'params': policy_model.parameters()},
        {'params': reward_model.parameters()}
    ], lr=1e-3)
    
    # 模拟训练步骤
    responses = ["这是第一个响应的示例内容。", "这是第二个响应。"]
    metrics = rlhf_training_step(
        policy_model, reward_model, 
        sample_prompts[:2], responses, 
        optimizer
    )
    print(f"RLHF训练指标: {metrics}")
    
    # ------------------------------
    # Task 3: DPO优化
    # ------------------------------
    print("\n[Task 3] DPO直接偏好优化")
    print("-" * 40)
    
    # 模拟对数概率
    batch_size = 4
    chosen_logps = torch.randn(batch_size) * 2 + 1
    rejected_logps = torch.randn(batch_size) * 2 - 1
    
    dpo_loss_value, dpo_metrics = dpo_loss(policy_model, chosen_logps, rejected_logps)
    print(f"DPO损失: {dpo_metrics['dpo_loss']:.4f}")
    print(f"偏好优势: {dpo_metrics['advantage']:.4f}")
    print(f"准确率: {dpo_metrics['accuracy']:.2%}")
    
    # ------------------------------
    # Task 4: 用户画像与个性化
    # ------------------------------
    print("\n[Task 4] 用户画像与个性化生成")
    print("-" * 40)
    
    # 创建用户画像
    user_profile = UserProfile(user_id="user_001")
    
    # 模拟交互历史
    interactions = [
        {"prompt": "我想了解人工智能", "response": "AI是模拟人类智能的技术...", "feedback": 1},
        {"prompt": "推荐机器学习书籍", "response": "推荐《机器学习实战》...", "feedback": 1},
        {"prompt": "深度学习是什么", "response": "深度学习是...", "feedback": 0},
    ]
    
    for interaction in interactions:
        user_profile.update_from_interaction(
            interaction['prompt'],
            interaction['response'],
            interaction.get('feedback')
        )
    
    print(f"用户画像已更新: {user_profile.interaction_count} 次交互")
    print(f"话题偏好: {user_profile.topic_preferences}")
    
    # 个性化生成
    base_model = BaseContentModel()
    personalized_result = personalized_generate(
        user_profile, base_model,
        "我想了解最新的AI技术发展趋势"
    )
    
    print(f"\n个性化生成结果:")
    print(f"提示: {personalized_result['prompt']}")
    print(f"响应: {personalized_result['response']}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()