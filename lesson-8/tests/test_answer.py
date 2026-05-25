#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文件：验证 answer/main.py 中公平性指标计算函数的正确性

使用 pytest 运行测试：
    pytest tests/test_answer.py -v
"""

import pytest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'answer'))

from main import (
    statistical_parity_difference,
    equalized_odds_difference,
    calibration_deviation
)


class TestFairnessMetrics:
    """测试公平性指标计算的准确性"""
    
    def setup_method(self):
        """测试前准备"""
        np.random.seed(42)
    
    def test_statistical_parity_basic(self):
        """测试统计均等性差异的基本功能"""
        # 完全公平的情况：两个群体的预测率相同
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        sensitive_attr = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        
        spd = statistical_parity_difference(y_pred, sensitive_attr)
        assert abs(spd) < 0.01, f"完全公平情况下SPD应为0，实际为{spd}"
    
    def test_statistical_parity_with_difference(self):
        """测试统计均等性差异能检测到偏见"""
        # 有偏见的情况：群体1的预测率高于群体0
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # 群体1: 100%, 群体0: 0%
        sensitive_attr = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        
        spd = statistical_parity_difference(y_pred, sensitive_attr)
        assert abs(spd - 1.0) < 0.01, f"群体差异为1时SPD应为1.0，实际为{spd}"
    
    def test_statistical_parity_different_group_sizes(self):
        """测试不同组大小的统计均等性差异"""
        # 群体1有100个样本，群体0有50个样本
        n1, n0 = 100, 50
        y_pred = np.concatenate([np.ones(n1), np.zeros(n0)])
        sensitive_attr = np.concatenate([np.ones(n1), np.zeros(n0)])
        
        spd = statistical_parity_difference(y_pred, sensitive_attr)
        assert abs(spd - 1.0) < 0.01, f"群体差异为1时SPD应为1.0，实际为{spd}"
    
    def test_equalized_odds_basic(self):
        """测试均等机会差异的基本功能"""
        # 完全公平：两个群体的TPR和FPR都相同
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        sensitive_attr = np.array([1, 0, 1, 0])
        
        eod = equalized_odds_difference(y_pred, y_true, sensitive_attr)
        assert eod < 0.01, f"完全公平情况下EOD应为0，实际为{eod}"
    
    def test_equalized_odds_with_tpr_difference(self):
        """测试均等机会差异能检测TPR差异"""
        # 群体1的TPR高于群体0
        # 群体1: 真实正例4个，全部预测正确 -> TPR=1.0
        # 群体0: 真实正例2个，预测正确1个 -> TPR=0.5
        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0])
        sensitive_attr = np.array([1, 1, 1, 1, 0, 0])
        
        eod = equalized_odds_difference(y_pred, y_true, sensitive_attr)
        assert abs(eod - 0.5) < 0.01, f"TPR差异为0.5时EOD应为0.5，实际为{eod}"
    
    def test_equalized_odds_with_fpr_difference(self):
        """测试均等机会差异能检测FPR差异"""
        # 群体1的FPR高于群体0
        # 群体1: 真实负例4个，预测错误2个 -> FPR=0.5
        # 群体0: 真实负例2个，全部预测正确 -> FPR=0.0
        y_true = np.array([1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0])
        sensitive_attr = np.array([1, 1, 1, 1, 0, 0])
        
        eod = equalized_odds_difference(y_pred, y_true, sensitive_attr)
        assert abs(eod - 0.5) < 0.01, f"FPR差异为0.5时EOD应为0.5，实际为{eod}"
    
    def test_calibration_basic(self):
        """测试校准性偏差的基本功能"""
        # 完美校准：预测率和真实正类率相同
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        sensitive_attr = np.array([1, 1, 0, 0])
        
        cd = calibration_deviation(y_pred, y_true, sensitive_attr)
        assert cd < 0.01, f"完美校准情况下CD应为0，实际为{cd}"
    
    def test_calibration_with_deviation(self):
        """测试校准性偏差能检测校准误差"""
        # 有偏差：群体1的预测率高于真实正类率
        # 群体1: 预测率=1.0, 真实正类率=0.5 -> 偏差=0.5
        # 群体0: 预测率=0.5, 真实正类率=0.5 -> 偏差=0.0
        # 平均偏差 = 0.25
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        sensitive_attr = np.array([1, 1, 0, 0])
        
        cd = calibration_deviation(y_pred, y_true, sensitive_attr)
        assert abs(cd - 0.25) < 0.02, f"校准偏差为0.25时CD应为0.25，实际为{cd}"
    
    def test_calibration_three_groups(self):
        """测试三组数据的校准性偏差"""
        # 三个敏感属性组
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 1])
        sensitive_attr = np.array([0, 0, 1, 1, 2, 2])
        
        cd = calibration_deviation(y_pred, y_true, sensitive_attr)
        # 组0: pred_rate=0.5, true_rate=0.5, 偏差=0
        # 组1: pred_rate=0.0, true_rate=0.0, 偏差=0
        # 组2: pred_rate=1.0, true_rate=0.5, 偏差=0.5
        # 平均偏差 = 0.167
        assert abs(cd - 0.167) < 0.02, f"预期CD约为0.167，实际为{cd}"
    
    def test_edge_case_empty_group(self):
        """测试边界情况：空群体"""
        y_true = np.array([1, 0])
        y_pred = np.array([1, 0])
        sensitive_attr = np.array([1, 0])
        
        # 不应抛出异常
        spd = statistical_parity_difference(y_pred, sensitive_attr)
        eod = equalized_odds_difference(y_pred, y_true, sensitive_attr)
        
        assert isinstance(spd, (int, float))
        assert isinstance(eod, (int, float))
    
    def test_consistency_with_different_input_types(self):
        """测试不同输入类型的一致性"""
        # 列表输入
        y_pred_list = [1, 0, 1, 0]
        sensitive_attr_list = [1, 1, 0, 0]
        
        spd_list = statistical_parity_difference(y_pred_list, sensitive_attr_list)
        
        # numpy数组输入
        y_pred_array = np.array([1, 0, 1, 0])
        sensitive_attr_array = np.array([1, 1, 0, 0])
        
        spd_array = statistical_parity_difference(y_pred_array, sensitive_attr_array)
        
        assert abs(spd_list - spd_array) < 0.001, "列表和数组输入结果应一致"


class TestFairnessMetricsWithRealData:
    """使用模拟真实数据测试公平性指标"""
    
    def test_recruitment_bias_scenario(self):
        """测试招聘偏见场景"""
        np.random.seed(123)
        
        # 模拟招聘数据：男性和女性
        # 男性：100人，60%通过率
        # 女性：100人，40%通过率（存在偏见）
        n = 200
        gender = np.concatenate([np.ones(100), np.zeros(100)])
        
        # 预测结果：男性通过率更高
        y_pred = np.concatenate([
            np.random.binomial(1, 0.6, 100),  # 男性60%通过
            np.random.binomial(1, 0.4, 100)   # 女性40%通过
        ])
        
        spd = statistical_parity_difference(y_pred, gender)
        # 预期SPD约为0.2（0.6 - 0.4）
        assert 0.1 < abs(spd) < 0.3, f"SPD应在0.2左右，实际为{spd}"
    
    def test_credit_model_bias_scenario(self):
        """测试信贷模型偏见场景"""
        np.random.seed(456)
        
        # 两个地区的信用评估
        # 地区A：500人，70%通过率
        # 地区B：500人，30%通过率
        n = 1000
        region = np.concatenate([np.ones(500), np.zeros(500)])
        
        # 预测结果
        y_pred = np.concatenate([
            np.random.binomial(1, 0.7, 500),
            np.random.binomial(1, 0.3, 500)
        ])
        
        y_true = np.concatenate([
            np.random.binomial(1, 0.75, 500),  # 真实通过率略高
            np.random.binomial(1, 0.25, 500)
        ])
        
        spd = statistical_parity_difference(y_pred, region)
        eod = equalized_odds_difference(y_pred, y_true, region)
        
        # 预期SPD约为0.4
        assert 0.3 < abs(spd) < 0.5, f"SPD应在0.4左右，实际为{spd}"
        # EOD也应该显著
        assert eod > 0.1, f"EOD应该大于0.1，实际为{eod}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])