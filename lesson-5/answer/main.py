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
    # 实现偏好数据的构建逻辑
    # 根据分数排序，确定每个prompt的偏好回答和拒绝回答
    
    preference_data = []
    
    # 将prompts、responses、scores配对处理
    for prompt, response, score in zip(prompts, responses, scores):
        # 对于每个prompt，我们可能需要比较多个responses
        # 这里简化处理：如果有多个response，根据分数选择
        pass
    
    # 为了演示，我们创建模拟的偏好对
    # 实际应用中，会对同一prompt的多个响应进行两两比较
    for i in range(len(prompts)):
        prompt = prompts[i]
        current_score = scores[i]
        
        # 查找其他响应作为对比
        for j in range(len(prompts)):
            if i != j:
                other_score = scores[j]
                
                # 分数高的作为偏好回答，分数低的作为拒绝回答
                if current_score > other_score:
                    chosen_response = responses[i]
                    rejected_response = responses[j]
                else:
                    chosen_response = responses[j]
                    rejected_response = responses[i]
                
                preference_data.append({
                    'prompt': prompt,
                    'chosen_response': chosen_response,
                    'rejected_response': rejected_response,
                    'chosen_score': max(current_score, other_score),
                    'rejected_score': min(current_score, other_score)
                })
    
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
    device = next(policy_model.parameters()).device
    
    # 1. 将prompts和responses转换为token张量
    max_length = 50
    batch_size = len(responses)
    
    # Token化responses
    response_tokens_list = []
    for response in responses:
        tokens = [ord(c) % 10000 for c in response[:max_length]]
        while len(tokens) < max_length:
            tokens.append(0)
        response_tokens_list.append(tokens)
    
    response_tokens = torch.tensor(response_tokens_list, dtype=torch.long).to(device)
    
    # 2. 计算奖励模型对responses的评分
    reward_model.eval()
    with torch.no_grad():
        rewards = reward_model(response_tokens)
    
    # 3. 计算策略模型的对数概率
    policy_model.train()
    log_probs = policy_model.get_response_log_probs(response_tokens)
    
    # 4. 计算策略梯度损失（奖励作为基线）
    # 使用奖励作为加权系数
    # 简化的策略梯度损失：-log_prob * reward
    policy_loss = -(log_probs * rewards).mean()
    
    # 5. 添加KL散度惩罚（简化版）
    # 实际应用中需要参考模型，这里使用固定的KL损失
    kl_penalty = kl_coef * (log_probs ** 2).mean()
    
    # 6. 总损失
    total_loss = policy_loss + kl_penalty
    
    # 反向传播并更新参数
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # 初始化指标
    metrics = {
        'reward': rewards.mean().item(),
        'loss': total_loss.item(),
        'kl_penalty': kl_penalty.item(),
        'policy_loss': policy_loss.item()
    }
    
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
    # 简化实现：DPO不需要参考模型，直接使用偏好对的优势
    
    # 计算优势：chosen_logps - rejected_logps
    advantage = chosen_logps - rejected_logps
    
    # 应用DPO损失：-log(sigmoid(beta * advantage))
    # 数值稳定的计算方式
    loss = -F.logsigmoid(beta * advantage)
    
    # 计算平均损失
    dpo_loss_value = loss.mean()
    
    # 计算训练指标
    with torch.no_grad():
        # 计算偏好优势的平均值
        avg_advantage = advantage.mean().item()
        
        # 计算准确率（偏好回答的对数概率大于拒绝回答的比例）
        accuracy = (chosen_logps > rejected_logps).float().mean().item()
    
    metrics = {
        'dpo_loss': dpo_loss_value.item(),
        'advantage': avg_advantage,
        'accuracy': accuracy
    }
    
    return dpo_loss_value, metrics


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
        
        # 话题关键词库
        self.topic_keywords = {
            '技术': ['编程', '代码', '算法', '软件', 'AI', '机器学习', '深度学习', 'Python', 'Java'],
            '科学': ['物理', '化学', '生物', '数学', '量子', '实验', '研究'],
            '商业': ['投资', '市场', '创业', '管理', '营销', '销售'],
            '教育': ['学习', '教育', '课程', '学校', '老师', '学生', '考试'],
            '生活': ['健康', '饮食', '运动', '旅游', '娱乐', '购物']
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        从文本中提取话题关键词
        
        Args:
            text: 输入文本
        
        Returns:
            检测到的话题列表
        """
        detected_topics = []
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    detected_topics.append(topic)
                    break
        return detected_topics if detected_topics else ['其他']
    
    def _analyze_style(self, response: str) -> Dict[str, float]:
        """
        分析响应的风格特征
        
        Args:
            response: 响应文本
        
        Returns:
            风格特征字典
        """
        # 简洁程度：基于句子长度
        avg_sentence_len = len(response) / max(1, response.count('。') + 1)
        conciseness = min(1.0, 20 / max(1, avg_sentence_len))
        
        # 正式程度：基于是否包含专业术语
        formal_indicators = ['应该', '必须', '需要', '建议', '根据', '由于']
        formal_count = sum(1 for indicator in formal_indicators if indicator in response)
        formality = min(1.0, formal_count / 3)
        
        # 技术深度：基于是否包含技术词汇
        tech_words = ['技术', '系统', '方法', '原理', '机制', '过程']
        tech_count = sum(1 for word in tech_words if word in response)
        technical_level = min(1.0, tech_count / 3)
        
        return {
            'conciseness': conciseness,
            'formality': formality,
            'technical_level': technical_level
        }
    
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
        # 1. 从prompt中提取话题关键词并更新topic_preferences
        detected_topics = self._extract_topics(prompt)
        for topic in detected_topics:
            self.topic_preferences[topic] = self.topic_preferences.get(topic, 0) + 1
        
        # 归一化话题偏好
        total = sum(self.topic_preferences.values())
        if total > 0:
            self.topic_preferences = {
                k: v / total for k, v in self.topic_preferences.items()
            }
        
        # 2. 分析response的风格特征并更新style_preferences
        style_features = self._analyze_style(response)
        
        # 使用指数移动平均更新风格偏好
        alpha = 0.3  # 更新系数
        for key in self.style_preferences:
            if key in style_features:
                self.style_preferences[key] = (
                    (1 - alpha) * self.style_preferences[key] + 
                    alpha * style_features[key]
                )
        
        # 3. 记录交互历史
        interaction_record = {
            'prompt': prompt,
            'response': response,
            'feedback': feedback,
            'detected_topics': detected_topics,
            'style_features': style_features
        }
        self.interaction_history.append(interaction_record)
        
        # 4. 更新统计计数
        self.interaction_count += 1
        if feedback == 1:
            self.positive_feedback_count += 1
        
        # 如果是正面反馈，增强相关话题偏好
        if feedback == 1:
            for topic in detected_topics:
                self.topic_preferences[topic] = self.topic_preferences.get(topic, 0) + 0.1
            # 重新归一化
            total = sum(self.topic_preferences.values())
            if total > 0:
                self.topic_preferences = {
                    k: v / total for k, v in self.topic_preferences.items()
                }
    
    def get_topic_vector(self) -> torch.Tensor:
        """
        获取用户话题偏好的向量表示
        
        Returns:
            话题偏好向量
        """
        # 固定维度的话题向量
        topic_dim = 32
        topic_names = ['技术', '科学', '商业', '教育', '生活', '其他']
        topic_vector = torch.zeros(topic_dim)
        
        # 基于话题分布填充向量
        for i, topic in enumerate(topic_names):
            if i < topic_dim:
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
    
    def get_top_topics(self, k: int = 5) -> List[Tuple[str, float]]:
        """
        获取Top-K偏好话题
        
        Args:
            k: 返回的话题数量
        
        Returns:
            (话题名, 偏好分数)的列表
        """
        sorted_topics = sorted(
            self.topic_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_topics[:k]


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
        style_preferences = kwargs.get('style_preferences', {})
        
        # 模拟生成过程
        # 根据风格偏好调整响应
        conciseness = style_preferences.get('conciseness', 0.5)
        formality = style_preferences.get('formality', 0.5)
        
        if conciseness > 0.7:
            response = f"[简洁] 基于「{prompt[:15]}」的回应。"
        elif conciseness < 0.3:
            response = f"[详细] 关于「{prompt}」的深入分析。{prompt}是一个重要的话题，让我们详细探讨一下。"
        else:
            response = f"[适中] 基于提示「{prompt[:20]}」生成的回应内容。"
        
        # 添加正式程度标记
        if formality > 0.7:
            response = "根据分析，" + response
        
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
    # 1. 获取用户画像的向量表示
    profile_repr = profile.get_profile_representation()
    
    # 2. 分析当前prompt的话题
    detected_topics = []
    for topic, keywords in profile.topic_keywords.items():
        for keyword in keywords:
            if keyword in prompt:
                detected_topics.append(topic)
                break
    
    # 3. 调整生成参数以匹配用户偏好
    style_preferences = {
        'conciseness': profile.style_preferences['conciseness'],
        'formality': profile.style_preferences['formality'],
        'technical_level': profile.style_preferences['technical_level']
    }
    
    # 计算个性化权重
    personalization_weight = min(1.0, profile.interaction_count / 10)  # 基于交互次数
    
    # 4. 调用base_model生成内容
    generation_kwargs = {
        'max_length': 100,
        'temperature': 1.0 - 0.3 * personalization_weight,  # 交互越多，温度越低
        'style_preferences': style_preferences
    }
    
    response = base_model.generate(prompt, **generation_kwargs)
    
    # 5. 应用个性化调整
    # 如果用户偏好技术性内容，在响应中添加技术细节
    if style_preferences['technical_level'] > 0.6 and '技术' not in response:
        response = response + " 这涉及到相关的技术原理。"
    
    # 如果用户偏好简洁内容，缩短响应
    if style_preferences['conciseness'] > 0.7:
        if len(response) > 30:
            response = response[:30] + "。"
    
    # 6. 返回结果和相关信息
    result = {
        'prompt': prompt,
        'response': response,
        'user_id': profile.user_id,
        'applied_style': style_preferences,
        'matched_topics': detected_topics if detected_topics else profile.get_top_topics(top_k_topics),
        'personalization_strength': personalization_weight,
        'profile_snapshot': profile.to_dict()
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
    for i, data in enumerate(preference_data[:3]):
        print(f"  偏好对 {i+1}: prompt='{data['prompt'][:15]}...'")
        print(f"    偏好回答分数: {data['chosen_score']:.2f}")
        print(f"    拒绝回答分数: {data['rejected_score']:.2f}")
    
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
    print(f"RLHF训练指标:")
    print(f"  平均奖励: {metrics['reward']:.4f}")
    print(f"  总损失: {metrics['loss']:.4f}")
    print(f"  KL惩罚: {metrics['kl_penalty']:.4f}")
    
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
        {"prompt": "深度学习是什么", "response": "深度学习是神经网络的应用...", "feedback": 0},
    ]
    
    for interaction in interactions:
        user_profile.update_from_interaction(
            interaction['prompt'],
            interaction['response'],
            interaction.get('feedback')
        )
    
    print(f"用户画像已更新:")
    print(f"  交互次数: {user_profile.interaction_count}")
    print(f"  正面反馈: {user_profile.positive_feedback_count}")
    print(f"  话题偏好: {user_profile.topic_preferences}")
    print(f"  风格偏好: {user_profile.style_preferences}")
    
    # 个性化生成
    base_model = BaseContentModel()
    personalized_result = personalized_generate(
        user_profile, base_model,
        "我想了解最新的AI技术发展趋势"
    )
    
    print(f"\n个性化生成结果:")
    print(f"  提示: {personalized_result['prompt']}")
    print(f"  响应: {personalized_result['response']}")
    print(f"  匹配话题: {personalized_result['matched_topics']}")
    print(f"  个性化强度: {personalized_result['personalization_strength']:.2f}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()