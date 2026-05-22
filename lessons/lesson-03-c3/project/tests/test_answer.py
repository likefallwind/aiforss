#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型金融文本分析系统 - 测试文件
课程《人工智能赋能社会科学》第「大模型在金融经济领域的应用」课配套项目

本文件使用 pytest 测试 answer/main.py 中的参考实现。
只验证 answer/ 目录下的参考实现，不测试 src/ 下的脚手架文件。
"""

import pytest
import sys
import os

# 将 answer 目录添加到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'answer'))

# 导入待测试的模块
from main import (
    extract_financial_info,
    analyze_financial_sentiment,
    calculate_risk_score,
    simulate_scenarios
)


class TestExtractFinancialInfo:
    """测试 Task 1: 金融文本信息抽取"""
    
    def test_extract_with_revenue_growth(self):
        """测试提取营收增长"""
        text = "公司2023年度营业收入达到1,258亿元，同比增长15.3%。"
        result = extract_financial_info(text)
        
        assert "revenue_growth" in result
        assert result["revenue_growth"] == 15.3
    
    def test_extract_with_rd_investment(self):
        """测试提取研发投入"""
        text = "研发投入达到186亿元，占营收的14.8%。"
        result = extract_financial_info(text)
        
        assert "rd_investment" in result
        assert result["rd_investment"] is not None
        assert "186" in result["rd_investment"]
    
    def test_extract_strategic_focus(self):
        """测试提取战略重点"""
        text = "公司战略重点包括：1) 加速数字化转型；2) 拓展海外市场；3) 加强绿色能源布局。"
        result = extract_financial_info(text)
        
        assert "strategic_focus" in result
        assert isinstance(result["strategic_focus"], list)
        assert len(result["strategic_focus"]) > 0
    
    def test_extract_rd_direction(self):
        """测试提取研发方向"""
        text = "重点投向人工智能、云计算和新能源汽车核心技术。"
        result = extract_financial_info(text)
        
        assert "rd_direction" in result
        assert result["rd_direction"] is not None
        assert "人工智能" in result["rd_direction"]
    
    def test_return_type(self):
        """测试返回类型"""
        text = "公司营业收入100亿元。"
        result = extract_financial_info(text)
        
        assert isinstance(result, dict)
        assert "key_highlights" in result


class TestAnalyzeFinancialSentiment:
    """测试 Task 2: 金融情绪量化分析"""
    
    def test_positive_sentiment(self):
        """测试正面情绪"""
        text = "公司业绩大幅增长，超出市场预期，管理层对发展前景充满信心。"
        result = analyze_financial_sentiment(text)
        
        assert "overall_sentiment" in result
        assert result["overall_sentiment"] == "positive"
        assert result["sentiment_score"] > 0
    
    def test_negative_sentiment(self):
        """测试负面情绪"""
        text = "公司营收持续下滑，面临严峻挑战，风险因素增加。"
        result = analyze_financial_sentiment(text)
        
        assert "overall_sentiment" in result
        assert result["overall_sentiment"] == "negative"
        assert result["sentiment_score"] < 0
    
    def test_financial_labels(self):
        """测试金融专属标签"""
        text = "三年来首次实现盈利，出现明显拐点信号。"
        result = analyze_financial_sentiment(text)
        
        assert "financial_labels" in result
        assert isinstance(result["financial_labels"], list)
        assert "拐点信号" in result["financial_labels"]
    
    def test_short_term_impact_label(self):
        """测试短期冲击标签"""
        text = "本季度营收略低预期，主要受一次性因素影响。"
        result = analyze_financial_sentiment(text)
        
        assert "短期冲击" in result["financial_labels"]
    
    def test_confidence_score(self):
        """测试置信度分数"""
        text = "公司业绩增长20%。"
        result = analyze_financial_sentiment(text)
        
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
    
    def test_key_phrases(self):
        """测试关键情绪短语"""
        text = "公司实现盈利突破。"
        result = analyze_financial_sentiment(text)
        
        assert "key_phrases" in result
        assert isinstance(result["key_phrases"], list)


class TestCalculateRiskScore:
    """测试 Task 3: 风险评估与信用评分"""
    
    def test_risk_level_values(self):
        """测试风险等级有效值"""
        financial_info = {"revenue_growth": 15, "strategic_focus": ["数字化转型"]}
        sentiment = {"sentiment_score": 0.5, "financial_labels": [], "confidence": 0.8}
        
        result = calculate_risk_score(financial_info, sentiment)
        
        assert "risk_level" in result
        assert result["risk_level"] in ["A", "B", "C", "D"]
    
    def test_credit_score_range(self):
        """测试信用评分范围"""
        financial_info = {"revenue_growth": 10}
        sentiment = {"sentiment_score": 0.2, "financial_labels": [], "confidence": 0.7}
        
        result = calculate_risk_score(financial_info, sentiment)
        
        assert "credit_score" in result
        assert 0 <= result["credit_score"] <= 100
    
    def test_risk_factors_list(self):
        """测试风险因素列表"""
        financial_info = {"revenue_growth": -5}
        sentiment = {"sentiment_score": -0.5, "financial_labels": ["趋势恶化"], "confidence": 0.6}
        
        result = calculate_risk_score(financial_info, sentiment)
        
        assert "risk_factors" in result
        assert isinstance(result["risk_factors"], list)
    
    def test_positive_factors_list(self):
        """测试积极因素列表"""
        financial_info = {"revenue_growth": 25, "rd_investment": "100亿元 (占比15%)"}
        sentiment = {"sentiment_score": 0.6, "financial_labels": ["拐点信号"], "confidence": 0.8}
        
        result = calculate_risk_score(financial_info, sentiment)
        
        assert "positive_factors" in result
        assert isinstance(result["positive_factors"], list)
        assert len(result["positive_factors"]) > 0
    
    def test_summary_text(self):
        """测试综合评估摘要"""
        financial_info = {"revenue_growth": 15}
        sentiment = {"sentiment_score": 0.3, "financial_labels": [], "confidence": 0.7}
        
        result = calculate_risk_score(financial_info, sentiment)
        
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0
    
    def test_high_growth_positive_risk(self):
        """测试高增长带来较低风险"""
        high_growth = {"revenue_growth": 30, "rd_investment": "100亿元 (占比15%)"}
        low_growth = {"revenue_growth": 5}
        
        sentiment = {"sentiment_score": 0.3, "financial_labels": [], "confidence": 0.7}
        
        result_high = calculate_risk_score(high_growth, sentiment)
        result_low = calculate_risk_score(low_growth, sentiment)
        
        assert result_high["credit_score"] > result_low["credit_score"]


class TestSimulateScenarios:
    """测试 Task 4: 市场情景模拟"""
    
    def test_minimum_scenarios_count(self):
        """测试至少生成3种情景"""
        base_metrics = {"revenue": 1000}
        variables = []
        
        result = simulate_scenarios(base_metrics, variables)
        
        assert isinstance(result, list)
        assert len(result) >= 3
    
    def test_probability_distribution(self):
        """测试概率分布"""
        base_metrics = {"revenue": 1000}
        variables = []
        
        result = simulate_scenarios(base_metrics, variables)
        
        total_prob = sum(s["probability"] for s in result)
        assert abs(total_prob - 1.0) < 0.01
    
    def test_scenario_probability_range(self):
        """测试概率范围"""
        base_metrics = {"revenue": 1000}
        variables = []
        
        result = simulate_scenarios(base_metrics, variables)
        
        for scenario in result:
            assert 0.0 <= scenario["probability"] <= 1.0
    
    def test_scenario_structure(self):
        """测试情景结构完整性"""
        base_metrics = {"revenue": 1000}
        variables = []
        
        result = simulate_scenarios(base_metrics, variables)
        
        for scenario in result:
            assert "name" in scenario
            assert "probability" in scenario
            assert "impact" in scenario
            assert "metrics" in scenario
            assert isinstance(scenario["metrics"], dict)
    
    def test_with_variables(self):
        """测试带变量的情景模拟"""
        base_metrics = {"revenue": 1000, "rd_ratio": 10}
        variables = [
            {"name": "利率", "base_value": 3.5, "scenarios": [3.0, 3.5, 4.0]}
        ]
        
        result = simulate_scenarios(base_metrics, variables)
        
        assert len(result) >= 3
        # 检查是否包含变量影响信息
        for scenario in result:
            if "variables_effect" in scenario:
                assert "利率" in scenario["variables_effect"]


class TestIntegration:
    """集成测试：测试完整流程"""
    
    def test_full_pipeline(self):
        """测试完整分析流程"""
        # Task 1: 提取财务信息
        report_text = """
        公司2023年度营业收入达到1,258亿元，同比增长15.3%。
        研发投入达到186亿元，占营收的14.8%，重点投向人工智能、云计算。
        战略重点：1) 数字化转型；2) 海外市场拓展。
        """
        financial_info = extract_financial_info(report_text)
        
        # Task 2: 分析情绪
        news_text = """
        分析师指出，公司三年来首次实现盈利，出现明显拐点信号。
        管理层对前景持谨慎乐观态度。
        """
        sentiment = analyze_financial_sentiment(news_text)
        
        # Task 3: 风险评估
        risk_result = calculate_risk_score(financial_info, sentiment)
        
        # 验证整体流程结果
        assert financial_info["revenue_growth"] == 15.3
        assert sentiment["overall_sentiment"] == "positive"
        assert risk_result["risk_level"] in ["A", "B", "C", "D"]
        assert risk_result["credit_score"] >= 0 and risk_result["credit_score"] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])