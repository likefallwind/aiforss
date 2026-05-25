#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法偏见检测与公平性度量
课程《人工智能赋能社会科学》第「算法偏见与数据治理」课配套项目

参考实现：完整的偏见检测与公平性度量功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any


def load_data() -> pd.DataFrame:
    """
    加载模拟招聘数据集。
    
    数据集包含以下字段：
    - gender: 性别 (0=女, 1=男)
    - region: 地区 (0=欠发达地区, 1=发达地区)
    - age: 年龄
    - education: 教育水平 (1-5)
    - experience: 工作经验年限
    - skill_score: 技能评分
    - decision: 真实决策 (0=拒绝, 1=通过)
    - prediction: 模型预测 (0=拒绝, 1=通过)
    
    Returns:
        pd.DataFrame: 包含所有数据的数据框
    """
    np.random.seed(42)
    n_samples = 1000
    
    # 生成敏感属性
    gender = np.random.binomial(1, 0.6, n_samples)  # 60%男性
    region = np.random.binomial(1, 0.5, n_samples)  # 50%发达地区
    
    # 生成其他特征
    age = np.random.normal(30, 5, n_samples)
    education = np.random.randint(1, 6, n_samples)
    experience = np.random.exponential(5, n_samples)
    skill_score = 0.5 * education + 0.3 * experience + 0.2 * region + \
                  np.random.normal(0, 1, n_samples)
    
    # 生成真实决策（包含偏见）
    # 偏见来源1：性别影响
    bias_gender = -0.3 * gender + 0.2  # 女性偏见系数
    # 偏见来源2：地区影响
    bias_region = 0.4 * region
    
    # 计算决策概率
    score = 0.3 * education + 0.2 * experience + 0.5 * skill_score + \
            bias_gender + bias_region - 5
    prob_decision = 1 / (1 + np.exp(-score))
    decision = (np.random.random(n_samples) < prob_decision).astype(int)
    
    # 生成模型预测（模型可能学到并放大偏见）
    bias_magnified = -0.4 * gender + 0.3 * region  # 放大偏见
    pred_score = 0.3 * education + 0.2 * experience + 0.5 * skill_score + \
                 bias_magnified - 5
    pred_prob = 1 / (1 + np.exp(-pred_score))
    prediction = (np.random.random(n_samples) < pred_prob).astype(int)
    
    df = pd.DataFrame({
        'gender': gender,
        'region': region,
        'age': age,
        'education': education,
        'experience': experience,
        'skill_score': skill_score,
        'decision': decision,
        'prediction': prediction
    })
    
    return df


def diagnose_bias(df: pd.DataFrame, sensitive_attrs: List[str]) -> Dict[str, Any]:
    """
    诊断数据集中的偏见来源。
    
    分析敏感属性在各特征中的分布差异，以及正类标签在不同群体间的比例。
    
    Args:
        df: 包含数据的数据框
        sensitive_attrs: 敏感属性列表，如 ['gender', 'region']
    
    Returns:
        Dict: 包含诊断报告的字典
    """
    report = {
        'total_samples': len(df),
        'groups': {},
        'positive_rates': {},
        'feature_means': {}
    }
    
    # 实现群体统计
    for attr in sensitive_attrs:
        groups = df[attr].unique()
        group_stats = {}
        
        for group in groups:
            mask = df[attr] == group
            group_data = df[mask]
            
            # 计算每个群体的样本数
            count = len(group_data)
            
            # 计算正类率（使用真实决策）
            positive_rate = group_data['decision'].mean() if count > 0 else 0
            
            group_stats[group] = {
                'count': count,
                'positive_rate': positive_rate
            }
        
        report['groups'][attr] = group_stats
    
    # 计算整体正类率
    overall_positive_rate = df['decision'].mean()
    report['overall_positive_rate'] = overall_positive_rate
    
    # 计算各敏感属性下不同群体的正类率差异
    for attr in sensitive_attrs:
        groups = df[attr].unique()
        if len(groups) >= 2:
            positive_rates = [report['groups'][attr][g]['positive_rate'] for g in groups]
            report['positive_rates'][attr] = max(positive_rates) - min(positive_rates)
    
    # 计算特征均值对比
    numeric_cols = ['age', 'education', 'experience', 'skill_score']
    for attr in sensitive_attrs:
        attr_feature_means = {}
        for group in df[attr].unique():
            mask = df[attr] == group
            group_data = df[mask]
            attr_feature_means[int(group)] = {
                col: group_data[col].mean() for col in numeric_cols
            }
        report['feature_means'][attr] = attr_feature_means
    
    return report


def statistical_parity_difference(y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
    """
    计算统计均等性差异 (Statistical Parity Difference)。
    
    公式: SPD = P(Ŷ=1|A=1) - P(Ŷ=1|A=0)
    
    理想情况下，SPD 应接近 0。绝对值越大，说明偏见越严重。
    
    Args:
        y_pred: 预测标签数组
        sensitive_attr: 敏感属性数组
    
    Returns:
        float: 统计均等性差异
    """
    # 将输入转换为 numpy 数组
    y_pred = np.array(y_pred)
    sensitive_attr = np.array(sensitive_attr)
    
    # 计算敏感属性为1时的正类预测率
    mask_1 = sensitive_attr == 1
    rate_1 = y_pred[mask_1].mean() if mask_1.sum() > 0 else 0.0
    
    # 计算敏感属性为0时的正类预测率
    mask_0 = sensitive_attr == 0
    rate_0 = y_pred[mask_0].mean() if mask_0.sum() > 0 else 0.0
    
    # 返回差异
    spd = rate_1 - rate_0
    return float(spd)


def equalized_odds_difference(y_pred: np.ndarray, y_true: np.ndarray, 
                               sensitive_attr: np.ndarray) -> float:
    """
    计算均等机会差异 (Equalized Odds Difference)。
    
    公式: EOD = |TPR(A=1) - TPR(A=0)| + |FPR(A=1) - FPR(A=0)|
    
    衡量不同群体在真阳性率和假阳性率上的差异。
    
    Args:
        y_pred: 预测标签数组
        y_true: 真实标签数组
        sensitive_attr: 敏感属性数组
    
    Returns:
        float: 均等机会差异
    """
    # 将输入转换为 numpy 数组
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    sensitive_attr = np.array(sensitive_attr)
    
    def compute_tpr_fpr(y_pred, y_true):
        """计算真阳性率和假阳性率"""
        # 真阳性率：真实为正且预测为正的比例
        true_positive = ((y_pred == 1) & (y_true == 1)).sum()
        actual_positive = (y_true == 1).sum()
        tpr = true_positive / actual_positive if actual_positive > 0 else 0.0
        
        # 假阳性率：真实为负但预测为正的比例
        false_positive = ((y_pred == 1) & (y_true == 0)).sum()
        actual_negative = (y_true == 0).sum()
        fpr = false_positive / actual_negative if actual_negative > 0 else 0.0
        
        return tpr, fpr
    
    # 计算敏感属性为1的群体的 TPR 和 FPR
    mask_1 = sensitive_attr == 1
    if mask_1.sum() > 0:
        tpr_1, fpr_1 = compute_tpr_fpr(y_pred[mask_1], y_true[mask_1])
    else:
        tpr_1, fpr_1 = 0.0, 0.0
    
    # 计算敏感属性为0的群体的 TPR 和 FPR
    mask_0 = sensitive_attr == 0
    if mask_0.sum() > 0:
        tpr_0, fpr_0 = compute_tpr_fpr(y_pred[mask_0], y_true[mask_0])
    else:
        tpr_0, fpr_0 = 0.0, 0.0
    
    # 计算均等机会差异
    eod = abs(tpr_1 - tpr_0) + abs(fpr_1 - fpr_0)
    return float(eod)


def calibration_deviation(y_pred: np.ndarray, y_true: np.ndarray,
                          sensitive_attr: np.ndarray, n_bins: int = 10) -> float:
    """
    计算校准性偏差 (Calibration Deviation)。
    
    将预测概率分桶，计算每个桶内实际正类率与预测概率的差距。
    
    Args:
        y_pred: 预测标签数组
        y_true: 真实标签数组
        sensitive_attr: 敏感属性数组
        n_bins: 分桶数量
    
    Returns:
        float: 校准性偏差（平均绝对误差）
    """
    # 将输入转换为 numpy 数组
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    sensitive_attr = np.array(sensitive_attr)
    
    # 由于 y_pred 是二值标签，我们需要计算每个群体的校准性
    # 方法：将数据按敏感属性分组，分别计算校准性偏差后取平均
    
    deviations = []
    
    for group in np.unique(sensitive_attr):
        mask = sensitive_attr == group
        y_pred_group = y_pred[mask]
        y_true_group = y_true[mask]
        
        if mask.sum() == 0:
            continue
        
        # 计算该群体的校准性
        # 对于二值预测，校准性可以通过比较预测为1的比例和真实为1的比例
        pred_rate = y_pred_group.mean()
        true_rate = y_true_group.mean()
        
        # 绝对偏差
        deviation = abs(pred_rate - true_rate)
        deviations.append(deviation)
    
    # 返回平均校准性偏差
    return float(np.mean(deviations)) if deviations else 0.0


def resample_debiasing(X: pd.DataFrame, y: np.ndarray, 
                       sensitive_attr: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    使用重采样技术进行偏见消解。
    
    对过采样不足的群体进行上采样，使各群体的样本量均衡。
    
    Args:
        X: 特征数据框
        y: 标签数组
        sensitive_attr: 敏感属性列名
    
    Returns:
        Tuple[pd.DataFrame, np.ndarray]: 重采样后的特征和标签
    """
    # 确保 X 包含敏感属性列
    X_with_attr = X.copy()
    if sensitive_attr not in X_with_attr.columns:
        raise ValueError(f"敏感属性列 '{sensitive_attr}' 不在特征中")
    
    # 统计各群体的样本数量
    groups = X_with_attr[sensitive_attr].unique()
    group_counts = {g: (X_with_attr[sensitive_attr] == g).sum() for g in groups}
    
    # 找出样本最少的群体
    min_count = min(group_counts.values())
    
    # 对每个群体进行上采样，使样本量与最少的群体持平
    resampled_dfs = []
    
    for group in groups:
        mask = X_with_attr[sensitive_attr] == group
        group_X = X_with_attr[mask]
        group_y = y[mask.values]
        
        current_count = group_counts[group]
        
        # 如果当前群体的样本数大于最小值，进行下采样
        # 如果小于最小值，进行上采样
        if current_count > min_count:
            # 随机下采样
            indices = np.random.choice(len(group_X), min_count, replace=False)
            resampled_dfs.append(group_X.iloc[indices])
            resampled_ys.append(group_y[indices])
        else:
            # 上采样（带替换抽样）
            indices = np.random.choice(len(group_X), min_count, replace=True)
            resampled_dfs.append(group_X.iloc[indices])
            resampled_ys.append(group_y[indices])
    
    # 合并重采样数据
    X_resampled = pd.concat(resampled_dfs, axis=0).reset_index(drop=True)
    y_resampled = np.concatenate(resampled_ys)
    
    # 打乱数据顺序
    shuffle_indices = np.random.permutation(len(X_resampled))
    X_resampled = X_resampled.iloc[shuffle_indices].reset_index(drop=True)
    y_resampled = y_resampled[shuffle_indices]
    
    return X_resampled, y_resampled


def train_simple_model(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """
    训练一个简单的模型用于预测。
    
    使用逻辑回归模型（通过梯度下降实现）。
    
    Args:
        X: 特征数据框
        y: 标签数组
    
    Returns:
        np.ndarray: 预测标签数组
    """
    # 将数据标准化
    X_norm = (X - X.mean()) / (X.std() + 1e-8)
    X_matrix = X_norm.values
    
    # 初始化权重
    n_features = X_matrix.shape[1]
    weights = np.zeros(n_features)
    bias = 0
    
    # 梯度下降训练
    learning_rate = 0.1
    n_epochs = 1000
    
    for epoch in range(n_epochs):
        # 计算预测概率
        linear_output = np.dot(X_matrix, weights) + bias
        prob = 1 / (1 + np.exp(-np.clip(linear_output, -500, 500)))
        
        # 计算梯度
        error = prob - y
        grad_weights = np.dot(X_matrix.T, error) / len(y)
        grad_bias = np.mean(error)
        
        # 更新权重
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias
    
    # 预测
    linear_output = np.dot(X_matrix, weights) + bias
    prob = 1 / (1 + np.exp(-np.clip(linear_output, -500, 500)))
    y_pred = (prob >= 0.5).astype(int)
    
    return y_pred


def evaluate_fairness(y_true: np.ndarray, y_pred: np.ndarray, 
                      sensitive_attr: np.ndarray) -> Dict[str, float]:
    """
    评估模型的公平性指标。
    
    Args:
        y_true: 真实标签数组
        y_pred: 预测标签数组
        sensitive_attr: 敏感属性数组
    
    Returns:
        Dict[str, float]: 包含三个公平性指标的字典
    """
    spd = statistical_parity_difference(y_pred, sensitive_attr)
    eod = equalized_odds_difference(y_pred, y_true, sensitive_attr)
    cd = calibration_deviation(y_pred, y_true, sensitive_attr)
    
    return {
        'statistical_parity_difference': spd,
        'equalized_odds_difference': eod,
        'calibration_deviation': cd
    }


def print_diagnosis_report(report: Dict[str, Any]) -> None:
    """
    打印偏见诊断报告。
    
    Args:
        report: 诊断报告字典
    """
    print("\n" + "="*60)
    print("偏见诊断报告")
    print("="*60)
    print(f"总样本数: {report['total_samples']}")
    print(f"整体正类率: {report['overall_positive_rate']:.4f}")
    
    for attr, stats in report['groups'].items():
        attr_name = "性别" if attr == 'gender' else "地区"
        print(f"\n【{attr_name}】")
        for group, info in stats.items():
            group_name = "男性/发达地区" if group == 1 else "女性/欠发达地区"
            print(f"  {group_name}: 样本数={info['count']}, 正类率={info['positive_rate']:.4f}")
        
        if attr in report['positive_rates']:
            print(f"  正类率差异: {report['positive_rates'][attr]:.4f}")
    
    print("\n" + "="*60)


def print_fairness_metrics(metrics: Dict[str, float], 
                           label: str = "当前") -> None:
    """
    打印公平性指标。
    
    Args:
        metrics: 公平性指标字典
        label: 指标标签（如"消解前"、"消解后"）
    """
    print(f"\n【{label}公平性指标】")
    print(f"  统计均等性差异 (SPD): {metrics['statistical_parity_difference']:.4f}")
    print(f"  均等机会差异 (EOD): {metrics['equalized_odds_difference']:.4f}")
    print(f"  校准性偏差 (CD): {metrics['calibration_deviation']:.4f}")
    
    # 公平性判定
    spd_ok = abs(metrics['statistical_parity_difference']) < 0.1
    eod_ok = abs(metrics['equalized_odds_difference']) < 0.1
    cd_ok = abs(metrics['calibration_deviation']) < 0.1
    
    print(f"\n  公平性评估: ", end="")
    if spd_ok and eod_ok and cd_ok:
        print("✓ 合格")
    else:
        print("✗ 存在偏见")
        if not spd_ok:
            print("    - 统计均等性存在差异")
        if not eod_ok:
            print("    - 均等机会存在差异")
        if not cd_ok:
            print("    - 校准性存在偏差")


def main():
    """
    主函数：执行偏见检测与公平性度量完整流程。
    """
    print("\n" + "#"*60)
    print("# 算法偏见检测与公平性度量")
    print("#"*60)
    
    # Step 1: 加载数据
    print("\n[Step 1] 加载数据...")
    df = load_data()
    print(f"数据集大小: {len(df)} 样本")
    print(f"特征列: {list(df.columns)}")
    
    # Step 2: 偏见诊断
    print("\n[Step 2] 偏见来源诊断...")
    sensitive_attrs = ['gender', 'region']
    diagnosis = diagnose_bias(df, sensitive_attrs)
    print_diagnosis_report(diagnosis)
    
    # Step 3: 计算公平性指标
    print("\n[Step 3] 计算公平性指标...")
    
    # 使用模型预测结果
    y_true = df['decision'].values
    y_pred = df['prediction'].values
    
    # 性别公平性
    print("\n--- 性别维度 ---")
    gender_metrics = evaluate_fairness(y_true, y_pred, df['gender'].values)
    print_fairness_metrics(gender_metrics, "性别")
    
    # 地区公平性
    print("\n--- 地区维度 ---")
    region_metrics = evaluate_fairness(y_true, y_pred, df['region'].values)
    print_fairness_metrics(region_metrics, "地区")
    
    # Step 4: 偏见消解
    print("\n[Step 4] 偏见消解实验...")
    
    # 准备特征和标签（包含敏感属性用于重采样）
    feature_cols = ['gender', 'age', 'education', 'experience', 'skill_score']
    X = df[feature_cols]
    y = df['decision'].values
    
    # 消解前评估
    y_pred_before = df['prediction'].values
    metrics_before = evaluate_fairness(y_true, y_pred_before, df['gender'].values)
    print_fairness_metrics(metrics_before, "消解前")
    
    # 执行重采样消解
    print("\n执行重采样消解...")
    X_resampled, y_resampled = resample_debiasing(X, y, 'gender')
    print(f"重采样后数据集大小: {len(X_resampled)} 样本")
    
    # 消解后的特征（不包含敏感属性用于训练）
    train_feature_cols = ['age', 'education', 'experience', 'skill_score']
    X_train = X_resampled[train_feature_cols]
    
    # 使用重采样数据训练新模型
    print("\n使用重采样数据训练新模型...")
    y_pred_after = train_simple_model(X_train, y_resampled)
    metrics_after = evaluate_fairness(y_resampled, y_pred_after, 
                                      X_resampled['gender'].values)
    
    print_fairness_metrics(metrics_after, "消解后")
    
    # 对比分析
    print("\n" + "="*60)
    print("消解效果对比")
    print("="*60)
    
    for key in metrics_before:
        before = metrics_before[key]
        after = metrics_after[key]
        change = after - before
        improvement = "↓ 改善" if abs(after) < abs(before) else "↑ 恶化"
        print(f"{key}: {before:.4f} → {after:.4f} ({improvement}, 变化: {change:.4f})")
    
    # 验证消解效果
    improved_count = sum(1 for key in metrics_before 
                        if abs(metrics_after[key]) < abs(metrics_before[key]))
    
    print("\n" + "="*60)
    print("分析结论")
    print("="*60)
    print(f"1. 模型预测中存在明显的性别偏见和地区偏见")
    print(f"2. 统计均等性差异显示不同群体的通过率存在显著差异")
    print(f"3. 重采样消解使 {improved_count}/3 个指标得到改善")
    print("4. 注意：Impossibility Theorem 表明三种公平性指标可能无法同时达到最优")
    print("   - 统计均等性和均等机会在某些条件下存在不可调和的权衡")


if __name__ == "__main__":
    main()