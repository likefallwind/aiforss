#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型金融文本分析系统 - 参考实现
课程《人工智能赋能社会科学》第「大模型在金融经济领域的应用」课配套项目

本文件是 src/main.py 的完整参考实现，包含四个核心任务函数的完整代码。

运行方式: python answer/main.py
"""

from typing import Dict, List, Any
import re
import random


def extract_financial_info(text: str) -> Dict[str, Any]:
    """
    Task 1: 金融文本信息抽取
    
    从公司年报摘要中提取关键财务指标与战略信息。
    使用正则表达式和规则匹配提取结构化数据。
    
    Args:
        text: 公司年报摘要文本
        
    Returns:
        dict: 包含结构化财务信息
    """
    result = {
        "revenue_growth": None,
        "rd_investment": None,
        "rd_direction": None,
        "strategic_focus": [],
        "key_highlights": []
    }
    
    # 提取营收增长率 - 匹配"同比增长XX%"模式
    growth_pattern = re.compile(r'同比增长\s*([\d.]+)%')
    growth_match = growth_pattern.search(text)
    if growth_match:
        result["revenue_growth"] = float(growth_match.group(1))
    else:
        # 尝试匹配"增长XX%"模式
        simple_growth = re.compile(r'增长\s*([\d.]+)%')
        simple_match = simple_growth.search(text)
        if simple_match:
            result["revenue_growth"] = float(simple_match.group(1))
    
    # 提取营收数字
    revenue_pattern = re.compile(r'营业收入.*?([\d,]+)\s*亿元')
    revenue_match = revenue_pattern.search(text)
    
    # 提取研发投入 - 匹配"研发投入...XX亿元"或"研发投入...XX%"
    rd_investment_pattern = re.compile(r'研发投入.*?([\d,.]+)\s*亿元')
    rd_investment_match = rd_investment_pattern.search(text)
    if rd_investment_match:
        rd_amount = rd_investment_match.group(1)
        result["rd_investment"] = f"{rd_amount}亿元"
        
        # 尝试提取研发占比
        rd_ratio_pattern = re.compile(r'占营收.*?([\d.]+)%')
        rd_ratio_match = rd_ratio_pattern.search(text)
        if rd_ratio_match:
            result["rd_investment"] = f"{rd_amount}亿元 (占比{rd_ratio_match.group(1)}%)"
    
    # 提取研发方向
    rd_direction_keywords = ["人工智能", "云计算", "新能源汽车", "芯片", "大数据", "物联网", "区块链"]
    found_directions = []
    for direction in rd_direction_keywords:
        if direction in text:
            found_directions.append(direction)
    if found_directions:
        result["rd_direction"] = "、".join(found_directions)
    
    # 提取战略重点 - 识别"战略重点"或数字列表
    strategic_pattern = re.compile(r'战略重点.*?(?:包括|为|：|:)(.*?)(?:。|$)', re.DOTALL)
    strategic_match = strategic_pattern.search(text)
    if strategic_match:
        strategic_text = strategic_match.group(1)
        # 提取括号内或编号列表中的内容
        for match in re.finditer(r'(\d+)[）)]\s*([^；；,，。]+)', strategic_text):
            result["strategic_focus"].append(match.group(2).strip())
    else:
        # 备选方案：提取常见的战略关键词
        focus_keywords = ["数字化转型", "海外市场", "绿色能源", "技术创新", "降本增效"]
        for keyword in focus_keywords:
            if keyword in text:
                result["strategic_focus"].append(keyword)
    
    # 提取关键亮点
    highlight_keywords = ["首次", "突破", "创新", "领先", "大幅增长", "历史新高"]
    for keyword in highlight_keywords:
        if keyword in text:
            # 提取包含关键词的短句
            context_pattern = re.compile(rf'.{{0,20}}{keyword}.{{0,20}}')
            context_match = context_pattern.search(text)
            if context_match:
                phrase = context_match.group().strip()
                # 清理空白字符
                phrase = re.sub(r'\s+', '', phrase)
                if phrase not in result["key_highlights"]:
                    result["key_highlights"].append(phrase)
    
    return result


def analyze_financial_sentiment(text: str) -> Dict[str, Any]:
    """
    Task 2: 金融情绪量化分析
    
    对财经新闻进行多维度金融情绪分析。
    识别隐含信息、管理层可信度、短期冲击与趋势变化。
    
    Args:
        text: 财经新闻文本
        
    Returns:
        dict: 包含金融情绪分析结果
    """
    result = {
        "overall_sentiment": "neutral",
        "sentiment_score": 0.0,
        "financial_labels": [],
        "confidence": 0.0,
        "key_phrases": []
    }
    
    # 正面词汇及其分数
    positive_words = {
        "增长": 0.3, "盈利": 0.4, "突破": 0.4, "创新": 0.3, "领先": 0.4,
        "好转": 0.5, "拐点": 0.5, "复苏": 0.5, "稳健": 0.3, "强劲": 0.4,
        "超预期": 0.4, "积极": 0.3, "乐观": 0.4, "首次盈利": 0.5,
        "明显拐点": 0.6, "三年来首次": 0.5, "谨慎乐观": 0.2
    }
    
    # 负面词汇及其分数
    negative_words = {
        "下降": -0.3, "亏损": -0.4, "风险": -0.3, "危机": -0.5, "恶化": -0.4,
        "低于预期": -0.3, "担忧": -0.3, "谨慎": -0.2, "压力": -0.3,
        "不确定": -0.2, "波动": -0.2, "挑战": -0.3
    }
    
    # 计算基础情绪分数
    total_score = 0.0
    word_count = 0
    
    text_lower = text.lower()
    for word, score in positive_words.items():
        if word in text_lower:
            total_score += score
            word_count += 1
            result["key_phrases"].append(word)
    
    for word, score in negative_words.items():
        if word in text_lower:
            total_score += score
            word_count += 1
            result["key_phrases"].append(word)
    
    # 归一化情绪分数到 [-1.0, 1.0]
    if word_count > 0:
        result["sentiment_score"] = max(-1.0, min(1.0, total_score / min(word_count, 5)))
    
    # 确定整体情绪
    if result["sentiment_score"] > 0.2:
        result["overall_sentiment"] = "positive"
    elif result["sentiment_score"] < -0.2:
        result["overall_sentiment"] = "negative"
    else:
        result["overall_sentiment"] = "neutral"
    
    # 添加金融专属标签
    # 短期冲击标签
    short_term_patterns = ["一次性因素", "短期", "本季度", "暂时"]
    for pattern in short_term_patterns:
        if pattern in text:
            result["financial_labels"].append("短期冲击")
            break
    
    # 趋势恶化标签
    deteriorating_patterns = ["持续下滑", "连续亏损", "恶化", "进一步下跌"]
    for pattern in deteriorating_patterns:
        if pattern in text:
            result["financial_labels"].append("趋势恶化")
            break
    
    # 拐点信号标签
    turning_point_patterns = ["拐点", "首次", "转折", "改善", "好转"]
    for pattern in turning_point_patterns:
        if pattern in text:
            result["financial_labels"].append("拐点信号")
            break
    
    # 管理层可信度标签
    management_patterns = ["管理层", "管理层对", "管理层表示", "公司管理层"]
   可信度_patterns = ["谨慎乐观", "有信心", "表示", "指出"]
    has_management = any(p in text for p in management_patterns)
    has可信度 = any(p in text for p in 可信度_patterns)
    if has_management and has可信度:
        result["financial_labels"].append("管理层表态")
    
    # 计算置信度
    base_confidence = 0.5
    if word_count >= 3:
        base_confidence += 0.2
    if len(result["financial_labels"]) >= 1:
        base_confidence += 0.15
    if result["key_phrases"]:
        base_confidence += 0.15
    result["confidence"] = min(1.0, base_confidence)
    
    return result


def calculate_risk_score(financial_info: Dict[str, Any], 
                         sentiment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task 3: 风险评估与信用评分
    
    基于财务信息和情绪分析结果计算综合风险评分。
    使用加权评分模型，考虑多个维度的风险因素。
    
    Args:
        financial_info: Task 1 提取的财务信息
        sentiment: Task 2 分析的情绪结果
        
    Returns:
        dict: 包含风险评估结果
    """
    result = {
        "risk_level": "B",
        "credit_score": 70,
        "risk_factors": [],
        "positive_factors": [],
        "summary": ""
    }
    
    # 基础分数
    base_score = 70
    
    # 财务健康度评估 (权重: 40%)
    financial_score = 0
    
    # 营收增长评估
    revenue_growth = financial_info.get("revenue_growth")
    if revenue_growth is not None:
        if revenue_growth >= 20:
            financial_score += 20
            result["positive_factors"].append(f"营收强劲增长({revenue_growth}%)")
        elif revenue_growth >= 10:
            financial_score += 15
            result["positive_factors"].append(f"营收稳健增长({revenue_growth}%)")
        elif revenue_growth >= 0:
            financial_score += 10
            result["positive_factors"].append(f"营收小幅增长({revenue_growth}%)")
        elif revenue_growth >= -5:
            financial_score += 5
            result["risk_factors"].append(f"营收增速放缓({revenue_growth}%)")
        else:
            result["risk_factors"].append(f"营收下降({revenue_growth}%)")
    
    # 研发投入评估
    rd_investment = financial_info.get("rd_investment")
    if rd_investment:
        if "占比" in rd_investment:
            # 尝试提取研发占比数字
            ratio_match = re.search(r'占比([\d.]+)%', rd_investment)
            if ratio_match:
                rd_ratio = float(ratio_match.group(1))
                if rd_ratio >= 15:
                    financial_score += 15
                    result["positive_factors"].append(f"高研发投入({rd_ratio}%)")
                elif rd_ratio >= 10:
                    financial_score += 10
                    result["positive_factors"].append(f"适度研发投入({rd_ratio}%)")
                else:
                    financial_score += 5
        else:
            financial_score += 8  # 有研发投入但无占比数据
    
    # 战略重点评估
    strategic_focus = financial_info.get("strategic_focus", [])
    if len(strategic_focus) >= 3:
        financial_score += 10
        result["positive_factors"].append("战略布局清晰多元")
    elif len(strategic_focus) >= 1:
        financial_score += 5
    
    # 情绪评估 (权重: 30%)
    sentiment_score = 0
    sentiment_score_value = sentiment.get("sentiment_score", 0)
    
    if sentiment_score_value > 0.3:
        sentiment_score = 30
        result["positive_factors"].append("市场情绪积极")
    elif sentiment_score_value > 0:
        sentiment_score = 20
        result["positive_factors"].append("市场情绪偏正面")
    elif sentiment_score_value < -0.3:
        sentiment_score = 10
        result["risk_factors"].append("市场情绪负面")
        # 检查是否有趋势恶化标签
        if "趋势恶化" in sentiment.get("financial_labels", []):
            sentiment_score -= 10
            result["risk_factors"].append("存在趋势性恶化信号")
    elif sentiment_score_value < 0:
        sentiment_score = 15
        result["risk_factors"].append("市场情绪偏负面")
    
    # 金融标签评估 (权重: 20%)
    label_score = 10
    labels = sentiment.get("financial_labels", [])
    
    if "拐点信号" in labels:
        label_score += 10
        result["positive_factors"].append("出现拐点信号")
    if "短期冲击" in labels:
        label_score += 5  # 短期冲击影响较小
    if "趋势恶化" in labels:
        label_score -= 10
        result["risk_factors"].append("存在趋势恶化风险")
    if "管理层表态" in labels:
        label_score += 5
    
    # 置信度调整 (权重: 10%)
    confidence = sentiment.get("confidence", 0.5)
    confidence_score = int(confidence * 10)
    
    # 计算总分
    total_score = (
        financial_score * 0.4 + 
        sentiment_score * 0.3 + 
        label_score * 0.2 + 
        confidence_score * 1.0
    )
    
    # 将分数限制在合理范围
    total_score = max(10, min(95, total_score))
    result["credit_score"] = int(total_score)
    
    # 确定风险等级
    if total_score >= 80:
        result["risk_level"] = "A"
        result["summary"] = "公司综合表现优秀，风险较低，信用良好"
    elif total_score >= 65:
        result["risk_level"] = "B"
        result["summary"] = "公司综合表现良好，风险可控，信用稳定"
    elif total_score >= 50:
        result["risk_level"] = "C"
        result["summary"] = "公司综合表现一般，存在一定风险，需关注"
    else:
        result["risk_level"] = "D"
        result["summary"] = "公司综合表现较弱，风险较高，需谨慎评估"
    
    return result


def simulate_scenarios(base_metrics: Dict[str, Any], 
                       variables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Task 4 (选做): 市场情景模拟
    
    设定不同假设条件，模拟市场可能的反应路径。
    生成多种情景及其概率分布。
    
    Args:
        base_metrics: 基础指标字典
        variables: 变量列表，每个变量包含名称、基础值和情景值
        
    Returns:
        list: 情景模拟结果
    """
    result = []
    
    # 生成基准情景
    base_scenario = {
        "name": "基准情景",
        "probability": 0.40,
        "impact": "维持当前态势，指标保持稳定",
        "metrics": base_metrics.copy()
    }
    result.append(base_scenario)
    
    # 生成乐观情景
    optimistic_scenario = {
        "name": "乐观情景",
        "probability": 0.30,
        "impact": "有利条件推动指标向好发展",
        "metrics": {
            "revenue": base_metrics.get("revenue", 0) * 1.15,
            "rd_ratio": base_metrics.get("rd_ratio", 10),
            "risk_level": "A",
            "note": "假设市场需求增长、政策支持加强"
        }
    }
    result.append(optimistic_scenario)
    
    # 生成悲观情景
    pessimistic_scenario = {
        "name": "悲观情景",
        "probability": 0.20,
        "impact": "不利因素导致指标承压",
        "metrics": {
            "revenue": base_metrics.get("revenue", 0) * 0.85,
            "rd_ratio": base_metrics.get("rd_ratio", 10),
            "risk_level": "C",
            "note": "假设经济下行压力加大、竞争加剧"
        }
    }
    result.append(pessimistic_scenario)
    
    # 生成极端情景
    extreme_scenario = {
        "name": "极端情景",
        "probability": 0.10,
        "impact": "重大冲击导致显著变化",
        "metrics": {
            "revenue": base_metrics.get("revenue", 0) * 0.70,
            "rd_ratio": base_metrics.get("rd_ratio", 10),
            "risk_level": "D",
            "note": "假设出现系统性风险或重大政策变化"
        }
    }
    result.append(extreme_scenario)
    
    # 如果有具体变量，进一步细化情景
    if variables:
        for var in variables:
            var_name = var.get("name", "未知变量")
            scenarios = var.get("scenarios", [])
            
            # 为每个情景添加变量影响说明
            for scenario in result:
                scenario["variables_effect"] = scenario.get("variables_effect", {})
                scenario["variables_effect"][var_name] = {
                    "base": var.get("base_value"),
                    "range": scenarios
                }
    
    # 验证概率总和为1
    total_prob = sum(s["probability"] for s in result)
    if abs(total_prob - 1.0) > 0.01:
        # 归一化概率
        for scenario in result:
            scenario["probability"] = round(scenario["probability"] / total_prob, 2)
    
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
    financial_info = extract_financial_info(sample_report)
    print(f"提取结果: {financial_info}")
    
    print("\n【Task 2】金融情绪量化分析")
    print("-" * 40)
    sentiment = analyze_financial_sentiment(sample_news)
    print(f"分析结果: {sentiment}")
    
    print("\n【Task 3】风险评估与信用评分")
    print("-" * 40)
    risk_result = calculate_risk_score(financial_info, sentiment)
    print(f"评估结果: {risk_result}")
    
    print("\n【Task 4】市场情景模拟")
    print("-" * 40)
    base_metrics = {"revenue": 1258, "rd_ratio": 14.8}
    variables = [
        {"name": "利率变化", "base_value": 3.5, "scenarios": [3.0, 3.5, 4.0]},
        {"name": "政策调整", "base_value": "neutral", "scenarios": ["宽松", "中性", "收紧"]}
    ]
    scenarios = simulate_scenarios(base_metrics, variables)
    print(f"模拟结果: {scenarios}")
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---