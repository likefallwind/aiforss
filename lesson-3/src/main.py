#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型金融文本分析系统 - 脚手架文件
课程《人工智能赋能社会科学》第「大模型在金融经济领域的应用」课配套项目

本文件包含四个核心任务函数：
1. extract_financial_info - 金融文本信息抽取
2. analyze_financial_sentiment - 金融情绪量化分析
3. calculate_risk_score - 风险评估与信用评分
4. simulate_scenarios - 市场情景模拟

运行方式: python src/main.py
"""

from typing import Dict, List, Any
import re


def extract_financial_info(text: str) -> Dict[str, Any]:
    """
    Task 1: 金融文本信息抽取
    
    从公司年报摘要中提取关键财务指标与战略信息，转化为结构化数据。
    
    Args:
        text: 公司年报摘要文本
        
    Returns:
        dict: 包含以下字段的结构化数据
            - revenue_growth: 营收增长率 (百分比)
            - rd_investment: 研发投入描述
            - rd_direction: 研发方向重点
            - strategic_focus: 战略重点领域列表
            - key_highlights: 关键亮点列表
    """
    # TODO: 实现金融文本信息抽取逻辑
    # 提示：需要识别以下模式
    # - 营收增长相关数字和百分比
    # - 研发投入的金额或比例
    # - 战略重点领域的优先级描述
    # - 关键业务亮点和成就
    
    result = {
        "revenue_growth": None,
        "rd_investment": None,
        "rd_direction": None,
        "strategic_focus": [],
        "key_highlights": []
    }
    
    # 你的实现代码...
    
    return result


def analyze_financial_sentiment(text: str) -> Dict[str, Any]:
    """
    Task 2: 金融情绪量化分析
    
    对财经新闻进行金融领域情绪分析，理解金融情绪与通用情感的区别。
    金融情绪分析需要识别隐含信息、管理层可信度、短期冲击与趋势性变化。
    
    Args:
        text: 财经新闻文本
        
    Returns:
        dict: 包含以下字段的情绪分析结果
            - overall_sentiment: 整体情绪 (positive/negative/neutral)
            - sentiment_score: 情绪分数 (-1.0 到 1.0)
            - financial_labels: 金融专属标签列表
              (如 "短期冲击", "趋势恶化", "拐点信号", "管理层可信")
            - confidence: 分析置信度 (0.0 到 1.0)
            - key_phrases: 关键情绪短语列表
    """
    # TODO: 实现金融情绪分析逻辑
    # 提示：需要识别以下内容
    # - 隐含信息 (如"三年来首次"暗示拐点)
    # - 管理层表态的可信度评估
    # - 短期冲击 vs 趋势性变化的区分
    # - 金融专业术语和表达方式
    
    result = {
        "overall_sentiment": "neutral",
        "sentiment_score": 0.0,
        "financial_labels": [],
        "confidence": 0.0,
        "key_phrases": []
    }
    
    # 你的实现代码...
    
    return result


def calculate_risk_score(financial_info: Dict[str, Any], 
                         sentiment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task 3: 风险评估与信用评分
    
    基于文本分析结果，综合评估公司或资产的风险等级与信用评分。
    需要综合考虑财务健康度、市场情绪、行业环境、宏观因素。
    
    Args:
        financial_info: Task 1 提取的财务信息
        sentiment: Task 2 分析的情绪结果
        
    Returns:
        dict: 包含以下字段的风险评估结果
            - risk_level: 风险等级 (A/B/C/D)
            - credit_score: 信用评分 (0-100)
            - risk_factors: 风险因素列表
            - positive_factors: 积极因素列表
            - summary: 综合评估摘要
    """
    # TODO: 实现风险评分计算逻辑
    # 提示：设计加权评分模型
    # - 财务健康度权重
    # - 市场情绪权重
    # - 风险因素扣分机制
    # - 积极因素加分机制
    
    result = {
        "risk_level": "B",
        "credit_score": 70,
        "risk_factors": [],
        "positive_factors": [],
        "summary": ""
    }
    
    # 你的实现代码...
    
    return result


def simulate_scenarios(base_metrics: Dict[str, Any], 
                       variables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Task 4 (选做): 市场情景模拟
    
    设定不同假设条件，模拟市场可能的反应路径。
    使用蒙特卡洛思维，设定变量并模拟多种情景。
    
    Args:
        base_metrics: 基础指标字典 (如当前财务数据)
        variables: 变量列表，每个变量包含:
            - name: 变量名称
            - base_value: 基础值
            - scenarios: 不同情景的值列表
            
    Returns:
        list: 情景模拟结果列表，每个情景包含:
            - name: 情景名称
            - probability: 发生概率 (0.0 到 1.0)
            - impact: 对指标的影响描述
            - metrics: 模拟后的指标值
    """
    # TODO: 实现情景模拟逻辑
    # 提示：
    # - 生成至少3种情景
    # - 设定合理的概率分布
    # - 计算各情景下的指标变化
    
    result = []
    
    # 你的实现代码...
    
    return result


def main():
    """
    CLI入口函数
    演示如何调用四个核心任务函数
    """
    print("=" * 60)
    print("大模型金融文本分析系统")
    print("=" * 60)
    
    # 示例年报文本
    sample_report = """
    公司2023年度营业收入达到1,258亿元，同比增长15.3%。
    研发投入达到186亿元，占营收的14.8%，重点投向人工智能、
    云计算和新能源汽车核心技术。公司战略重点包括：
    1) 加速数字化转型；2) 拓展海外市场；3) 加强绿色能源布局。
    """
    
    # 示例财经新闻
    sample_news = """
    分析师指出，虽然公司本季度营收略低于预期，但这主要受
    一次性因素影响。三年来首次实现季度盈利，表明公司
    经营状况出现明显拐点信号。管理层对未来发展持谨慎乐观态度。
    """
    
    print("\n【Task 1】金融文本信息抽取")
    print("-" * 40)
    # TODO: 调用 extract_financial_info
    financial_info = extract_financial_info(sample_report)
    print(f"提取结果: {financial_info}")
    
    print("\n【Task 2】金融情绪量化分析")
    print("-" * 40)
    # TODO: 调用 analyze_financial_sentiment
    sentiment = analyze_financial_sentiment(sample_news)
    print(f"分析结果: {sentiment}")
    
    print("\n【Task 3】风险评估与信用评分")
    print("-" * 40)
    # TODO: 调用 calculate_risk_score
    risk_result = calculate_risk_score(financial_info, sentiment)
    print(f"评估结果: {risk_result}")
    
    print("\n【Task 4】市场情景模拟")
    print("-" * 40)
    base_metrics = {"revenue": 1258, "rd_ratio": 14.8}
    variables = [
        {"name": "利率变化", "base_value": 3.5, "scenarios": [3.0, 3.5, 4.0]},
        {"name": "政策调整", "base_value": "neutral", "scenarios": ["宽松", "中性", "收紧"]}
    ]
    # TODO: 调用 simulate_scenarios
    scenarios = simulate_scenarios(base_metrics, variables)
    print(f"模拟结果: {scenarios}")
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---