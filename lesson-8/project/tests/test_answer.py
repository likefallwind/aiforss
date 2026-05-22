"""
测试文件：验证参考答案实现的正确性
课程《人工智能赋能社会科学》第「算法偏见与数据治理」课配套项目

本测试文件验证 answer/main.py 中实现的偏见检测与公平性度量功能。
测试覆盖：
1. 数据生成：验证带偏见数据的基本属性
2. 公平性指标计算：验证统计均等、均等机会、均等几率
3. 偏见消解：验证重采样方法能够改善公平性
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# 导入参考答案实现
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'answer'))

from main import (
    generate_biased_recruitment_data,
    train_model,
    compute_fairness_metrics,
    debias_by_resampling
)


class TestDataGeneration:
    """测试数据生成功能"""
    
    def test_generate_biased_recruitment_data_returns_dataframe(self):
        """测试函数返回正确的数据类型"""
        data = generate_biased_recruitment_data(n_samples=100)
        assert isinstance(data, pd.DataFrame), "应返回 pandas DataFrame"
    
    def test_generate_biased_recruitment_data_has_required_columns(self):
        """测试数据包含所有必需的列"""
        data = generate_biased_recruitment_data(n_samples=100)
        required_columns = ['gender', 'skill_score', 'experience_years', 
                           'education_level', 'interview_score', 'hired']
        for col in required_columns:
            assert col in data.columns, f"缺少必需列: {col}"
    
    def test_generate_biased_recruitment_data_correct_size(self):
        """测试生成数据的样本数量正确"""
        n_samples = 500
        data = generate_biased_recruitment_data(n_samples=n_samples)
        assert len(data) == n_samples, f"期望 {n_samples} 条记录，实际 {len(data)} 条"
    
    def test_gender_distribution(self):
        """测试性别分布大致符合设定（男性80%，女性20%）"""
        data = generate_biased_recruitment_data(n_samples=1000, random_state=42)
        male_ratio = (data['gender'] == 'male').mean()
        female_ratio = (data['gender'] == 'female').mean()
        
        # 允许一些随机波动
        assert 0.75 <= male_ratio <= 0.85, f"男性比例 {male_ratio:.2%} 不在预期范围"
        assert 0.15 <= female_ratio <= 0.25, f"女性比例 {female_ratio:.2%} 不在预期范围"
    
    def test_hired_column_is_binary(self):
        """测试录用列为二元值"""
        data = generate_biased_recruitment_data(n_samples=100)
        unique_values = set(data['hired'].unique())
        assert unique_values == {0, 1}, "录用列应为0和1的二元值"
    
    def test_biased_hiring_rate_exists(self):
        """测试数据中存在性别偏见（录用率差异）"""
        data = generate_biased_recruitment_data(n_samples=1000, random_state=42)
        male_hire_rate = data[data['gender'] == 'male']['hired'].mean()
        female_hire_rate = data[data['gender'] == 'female']['hired'].mean()
        gap = abs(male_hire_rate - female_hire_rate)
        
        # 偏见应该导致至少5%的录用率差异
        assert gap >= 0.05, f"录用率差异 {gap:.2%} 过小，可能没有正确生成偏见数据"
    
    def test_feature_values_in_reasonable_range(self):
        """测试特征值在合理范围内"""
        data = generate_biased_recruitment_data(n_samples=100)
        
        assert data['skill_score'].between(0, 100).all(), "技能得分应在0-100范围内"
        assert data['interview_score'].between(0, 100).all(), "面试分数应在0-100范围内"
        assert data['experience_years'].between(0, 20).all(), "工作年限应在0-20范围内"
        assert data['education_level'].isin([1, 2, 3]).all(), "教育水平应为1、2或3"


class TestModelTraining:
    """测试模型训练功能"""
    
    def test_train_model_returns_required_elements(self):
        """测试训练函数返回所有必需的组件"""
        data = generate_biased_recruitment_data(n_samples=200)
        result = train_model(data)
        
        # 检查返回值的数量和类型