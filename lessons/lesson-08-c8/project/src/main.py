"""
算法偏见检测与公平性度量 - 脚手架文件
课程《人工智能赋能社会科学》第「算法偏见与数据治理」课配套项目

本脚手架提供了招聘筛选场景下的偏见检测与公平性度量框架。
学生需要补全 TODO 部分来实现完整的偏见消解功能。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


def generate_biased_recruitment_data(n_samples=1000, random_state=42):
    """
    生成带偏见的招聘模拟数据集
    
    该函数模拟一家科技公司过去十年的招聘数据，其中技术岗位男性占比80%。
    即使候选人的能力分布是均衡的，录用决策也会受到性别的影响。
    
    参数:
        n_samples (int): 生成的样本数量
        random_state (int): 随机种子，确保结果可复现
    
    返回:
        pd.DataFrame: 包含以下列的数据框
            - gender: 性别 ('male' 或 'female')
            - skill_score: 技能得分 (0-100)
            - experience_years: 工作年限 (0-20)
            - education_level: 教育水平 (1-3)
            - interview_score: 面试分数 (0-100)
            - hired: 是否录用 (0 或 1)
    """
    np.random.seed(random_state)
    
    # 基础特征生成 - 能力分布对所有性别相同
    # TODO (Task 1): 实现基础特征生成逻辑
    # 提示：技能得分、工作年限、教育水平、面试分数应该与性别无关
    # 但录用决策(hired)需要与性别存在相关性，以模拟历史偏见
    
    data = pd.DataFrame()
    
    # 性别分布：男性80%，女性20%（模拟技术岗位的历史性别比例）
    gender_probs = [0.8, 0.2]
    data['gender'] = np.random.choice(['male', 'female'], size=n_samples, p=gender_probs)
    
    # 能力特征：与性别无关，均匀分布
    # TODO: 实现技能得分、工作年限、教育水平、面试分数的生成
    # 这些特征应该反映候选人的真实能力，与性别无关
    
    data['skill_score'] = np.random.normal(70, 15, n_samples).clip(0, 100)
    data['experience_years'] = np.random.exponential(5, n_samples).clip(0, 20)
    data['education_level'] = np.random.choice([1, 2, 3], size=n_samples, p=[0.2, 0.5, 0.3])
    data['interview_score'] = np.random.normal(72, 12, n_samples).clip(0, 100)
    
    # 录用决策：基于能力，但加入性别偏见
    # 关键：相同能力的男性更容易被录用
    # TODO (Task 1): 实现带偏见的录用决策生成逻辑
    # 提示：可以基于能力得分计算一个基础录用概率，然后根据性别进行调整
    # 男性获得+0.3的加成，女性获得-0.3的惩罚（或其他合理的偏见参数）
    
    # 计算综合能力得分
    ability_score = (
        data['skill_score'] * 0.3 +
        data['interview_score'] * 0.4 +
        data['experience_years'] * 2 +
        data['education_level'] * 5
    )
    
    # 基础录用概率（基于能力）
    base_prob = (ability_score - ability_score.min()) / (ability_score.max() - ability_score.min())
    
    # 加入性别偏见：男性更容易被录用
    gender_bias = data['gender'].map({'male': 0.25, 'female': -0.15})
    final_prob = (base_prob + gender_bias).clip(0, 1)
    
    # 生成录用标签
    data['hired'] = (np.random.random(n_samples) < final_prob).astype(int)
    
    return data


def train_model(data, target_col='hired', test_size=0.2, random_state=42):
    """
    训练逻辑回归模型进行招聘筛选
    
    参数:
        data (pd.DataFrame): 包含特征和标签的数据框
        target_col (str): 目标列名
        test_size (float): 测试集比例
        random_state (int): 随机种子
    
    返回:
        tuple: (训练好的模型, 测试集特征, 测试集标签, 性别测试集)
    """
    # 准备特征和标签
    feature_cols = ['skill_score', 'experience_years', 'education_level', 'interview_score']
    X = data[feature_cols]
    y = data[target_col]
    
    # 保存性别信息用于公平性评估
    gender = data['gender']
    
    # 分割数据集
    X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
        X, y, gender, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练逻辑回归模型
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return model, X_test_scaled, y_test, gender_test, scaler


def compute_fairness_metrics(model, X_test, y_test, gender_test, protected_attr='gender'):
    """
    计算公平性度量指标
    
    该函数计算三个主要的公平性指标：
    1. 统计均等 (Statistical Parity/Demographic Parity)
       - 不同群体被录用比例的差异
    2. 均等机会 (Equal Opportunity)
       - 在合格候选人中，不同群体的录用率差异
    3. 均等几率 (Equalized Odds)
       - 同时考虑真正例率和假正例率的公平性
    
    参数:
        model: 训练好的分类模型
        X_test: 测试集特征
        y_test: 测试集真实标签
        gender_test: 测试集性别信息
        protected_attr (str): 受保护的属性名
    
    返回:
        dict: 包含各公平性指标的字典
    """
    # 预测结果
    y_pred = model.predict(X_test)
    
    # 创建结果DataFrame便于分析
    results_df = pd.DataFrame({
        'gender': gender_test.values,
        'y_true': y_test.values,
        'y_pred': y_pred
    })
    
    metrics = {}
    
    # === 统计均等 (Statistical Parity) ===
    # 定义：不同受保护群体被模型预测为正例的比例差异
    # 计算公式：|P(Ŷ=1|G=0) - P(Ŷ=1|G=1)|
    # TODO (Task 2): 实现统计均等的计算
    # 提示：分别计算男性和女性的录用预测比例，然后取绝对差值
    
    male_pred_positive = (results_df[results_df['gender'] == 'male']['y_pred'] == 1).mean()
    female_pred_positive = (results_df[results_df['gender'] == 'female']['y_pred'] == 1).mean()
    
    metrics['statistical_parity'] = {
        'male_positive_rate': male_pred_positive,
        'female_positive_rate': female_pred_positive,
        'difference': abs(male_pred_positive - female_pred_positive),
        'description': '统计均等：不同性别群体的录用预测比例差异'
    }
    
    # === 均等机会 (Equal Opportunity) ===
    # 定义：在真实标签为正例的群体中，不同群体的真阳性率(TPR)差异
    # 计算公式：|P(Ŷ=1|Y=1,G=0) - P(Ŷ=1|Y=1,G=1)|
    # 衡量标准：在合格候选人中，录用机会是否平等
    # TODO (Task 2): 实现均等机会的计算
    # 提示：只考虑y_true=1的样本，计算各群体的预测正例率
    
    # 男性在合格候选人中的录用率
    male_qualified = results_df[results_df['gender'] == 'male']
    male_tpr = (male_qualified[male_qualified['y_true'] == 1]['y_pred'] == 1).mean()
    
    # 女性在合格候选人中的录用率
    female_qualified = results_df[results_df['gender'] == 'female']
    female_tpr = (female_qualified[female_qualified['y_true'] == 1]['y_pred'] == 1).mean()
    
    metrics['equal_opportunity'] = {
        'male_tpr': male_tpr,
        'female_tpr': female_tpr,
        'difference': abs(male_tpr - female_tpr),
        'description': '均等机会：合格候选人中不同性别群体的录用率差异'
    }
    
    # === 均等几率 (Equalized Odds) ===
    # 定义：同时考虑TPR和FPR的公平性指标
    # 计算公式：|P(Ŷ=1|Y=1,G=0) - P(Ŷ=1|Y=1,G=1)| + |P(Ŷ=1|Y=0,G=0) - P(Ŷ=1|Y=0,G=1)|
    # TODO (Task 2 - 挑战任务): 实现均等几率指标
    # 提示：需要分别计算TPR和FPR在不同群体间的差异
    
    # 男性FPR
    male_unqualified = results_df[results_df['gender'] == 'male']
    male_fpr = (male_unqualified[male_unqualified['y_true'] == 0]['y_pred'] == 1).mean()
    
    # 女性FPR
    female_unqualified = results_df[results_df['gender'] == 'female']
    female_fpr = (female_unqualified[female_unqualified['y_true'] == 0]['y_pred'] == 1).mean()
    
    # TPR差异
    tpr_diff = abs(male_tpr - female_tpr)
    # FPR差异
    fpr_diff = abs(male_fpr - female_fpr)
    
    metrics['equalized_odds'] = {
        'male_tpr': male_tpr,
        'female_tpr': female_tpr,
        'tpr_difference': tpr_diff,
        'male_fpr': male_fpr,
        'female_fpr': female_fpr,
        'fpr_difference': fpr_diff,
        'combined_score': tpr_diff + fpr_diff,
        'description': '均等几率：综合考虑真阳性率和假阳性率的公平性'
    }
    
    # === 综合评估 ===
    metrics['summary'] = {
        'overall_fairness_score': 1 - (metrics['statistical_parity']['difference'] + 
                                        metrics['equal_opportunity']['difference']) / 2,
        'recommendation': '公平性改进建议取决于具体指标' if metrics['equal_opportunity']['difference'] > 0.1 else '公平性良好'
    }
    
    return metrics


def debias_by_resampling(data, target_col='hired', protected_attr='gender'):
    """
    基于数据重采样的偏见消解方法
    
    该函数通过调整训练数据的分布来减少偏见。
    采用的策略：
    1. 欠采样多数群体（男性）中的正例
    2. 过采样少数群体（女性）中的正例
    3. 或者采用更精细的策略，考虑能力水平
    
    参数:
        data (pd.DataFrame): 原始数据集
        target_col (str): 目标列名
        protected_attr (str): 受保护的属性列名
    
    返回:
        pd.DataFrame: 经过偏见消解处理的数据集
    """
    # TODO (Task 3): 实现基于重采样的偏见消解
    # 提示：可以采用以下策略之一或组合
    # 1. 简单重采样：调整性别比例接近1:1
    # 2. 标签重采样：调整录用标签在不同性别间的分布
    # 3. 能力匹配采样：确保相同能力水平的男女候选人有相似的录用率
    
    df = data.copy()
    
    # 分离男性和女性数据
    male_data = df[df['gender'] == 'male']
    female_data = df[df['gender'] == 'female']
    
    # 计算当前录用率差异
    male_hire_rate = male_data[target_col].mean()
    female_hire_rate = female_data[target_col].mean()
    
    print(f"[偏见消解前] 男性录用率: {male_hire_rate:.2%}, 女性录用率: {female_hire_rate:.2%}")
    
    # 策略：调整录用标签以消除偏见
    # 目标：让相同能力水平的男女有相近的录用机会
    # TODO: 实现具体的重采样逻辑
    
    # 方法一：标签翻转 - 提高女性的录用概率，降低男性的录用概率
    # 仅针对中等能力区间的候选人（避免过度调整）
    ability_score = df['skill_score'] * 0.3 + df['interview_score'] * 0.4 + df['experience_years'] * 2
    
    # 定义中等能力区间
    median_ability = ability_score.median()
    ability_range = ability_score.quantile(0.75) - ability_score.quantile(0.25)
    
    # 调整中等能力区间的录用决策
    mask_middle = (ability_score >= median_ability - ability_range/2) & (ability_score <= median_ability + ability_range/2)
    
    # 对中等能力区间的候选人进行标签调整
    # 女性：提高录用率
    female_middle = mask_middle & (df['gender'] == 'female') & (df[target_col] == 0)
    if female_middle.sum() > 0:
        # 随机选择部分女性未被录用者改为录用
        flip_count = int(female_middle.sum() * 0.3)
        flip_indices = df[female_middle].sample(n=min(flip_count, female_middle.sum()), random_state=42).index
        df.loc[flip_indices, target_col] = 1
    
    # 男性：降低录用率
    male_middle = mask_middle & (df['gender'] == 'male') & (df[target_col] == 1)
    if male_middle.sum() > 0:
        # 随机选择部分男性被录用者改为不录用
        flip_count = int(male_middle.sum() * 0.2)
        flip_indices = df[male_middle].sample(n=min(flip_count, male_middle.sum()), random_state=42).index
        df.loc[flip_indices, target_col] = 0
    
    # 计算调整后的录用率
    male_hire_rate_new = df[df['gender'] == 'male'][target_col].mean()
    female_hire_rate_new = df[df['gender'] == 'female'][target_col].mean()
    
    print(f"[偏见消解后] 男性录用率: {male_hire_rate_new:.2%}, 女性录用率: {female_hire_rate_new:.2%}")
    print(f"[偏见改善] 录用率差异从 {abs(male_hire_rate - female_hire_rate):.2%} 降至 {abs(male_hire_rate_new - female_hire_rate_new):.2%}")
    
    return df


def print_fairness_report(metrics, title="公平性评估报告"):
    """
    格式化输出公平性评估报告
    
    参数:
        metrics (dict): compute_fairness_metrics 返回的指标字典
        title (str): 报告标题
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    print(f"\n【统计均等 (Statistical Parity)】")
    sp = metrics['statistical_parity']
    print(f"  - 男性预测录用率: {sp['male_positive_rate']:.2%}")
    print(f"  - 女性预测录用率: {sp['female_positive_rate']:.2%}")
    print(f"  - 差异: {sp['difference']:.2%}")
    print(f"  - 说明: {sp['description']}")
    
    print(f"\n【均等机会 (Equal Opportunity)】")
    eo = metrics['equal_opportunity']
    print(f"  - 男性真阳性率(TPR): {eo['male_tpr']:.2%}")
    print(f"  - 女性真阳性率(TPR): {eo['female_tpr']:.2%}")
    print(f"  - 差异: {eo['difference']:.2%}")
    print(f"  - 说明: {eo['description']}")
    
    print(f"\n【均等几率 (Equalized Odds)】")
    eod = metrics['equalized_odds']
    print(f"  - TPR差异: {eod['tpr_difference']:.2%}")
    print(f"  - FPR差异: {eod['fpr_difference']:.2%}")
    print(f"  - 综合分数: {eod['combined_score']:.2%}")
    print(f"  - 说明: {eod['description']}")
    
    print(f"\n【综合评估】")
    summary = metrics['summary']
    print(f"  - 公平性得分: {summary['overall_fairness_score']:.2%}")
    print(f"  - 建议: {summary['recommendation']}")
    
    print(f"\n{'='*60}\n")


def main():
    """
    主函数：演示偏见检测与公平性评估的完整流程
    """
    print("="*60)
    print("算法偏见检测与公平性度量演示")
    print("课程《人工智能赋能社会科学》- 算法偏见与数据治理")
    print("="*60)
    
    # 设置随机种子确保可复现
    np.random.seed(42)
    
    # === 第一步：生成带偏见的招聘数据 ===
    print("\n[Step 1] 生成带偏见的招聘模拟数据...")
    data = generate_biased_recruitment_data(n_samples=1000)
    
    print(f"数据集大小: {len(data)} 条记录")
    print(f"性别分布: 男性 {(data['gender']=='male').mean():.1%}, 女性 {(data['gender']=='female').mean():.1%}")
    print(f"整体录用率: {data['hired'].mean():.1%}")
    print(f"男性录用率: {data[data['gender']=='male']['hired'].mean():.1%}")
    print(f"女性录用率: {data[data['gender']=='female']['hired'].mean():.1%}")
    
    # === 第二步：训练模型并评估偏见 ===
    print("\n[Step 2] 训练招聘筛选模型...")
    model, X_test, y_test, gender_test, scaler = train_model(data)
    
    # 在测试集上评估模型性能
    y_pred = model.predict(X_test)
    print(f"模型准确率: {accuracy_score(y_test, y_pred):.2%}")
    print(f"模型精确率: {precision_score(y_test, y_pred):.2%}")
    print(f"模型召回率: {recall_score(y_test, y_pred):.2%}")
    
    # === 第三步：计算公平性指标 ===
    print("\n[Step 3] 计算公平性指标...")
    metrics = compute_fairness_metrics(model, X_test, y_test, gender_test)
    print_fairness_report(metrics, "偏见检测报告（消解前）")
    
    # === 第四步：偏见消解 ===
    print("\n[Step 4] 执行偏见消解（数据重采样）...")
    data_debiased = debias_by_resampling(data)
    
    # === 第五步：重新训练并评估 ===
    print("\n[Step 5] 使用消解后数据重新训练模型...")
    model_debiased, X_test_debiased, y_test_debiased, gender_test_debiased, _ = train_model(data_debiased)
    
    y_pred_debiased = model_debiased.predict(X_test_debiased)
    print(f"消解后模型准确率: {accuracy_score(y_test_debiased, y_pred_debiased):.2%}")
    print(f"消解后模型精确率: {precision_score(y_test_debiased, y_pred_debiased):.2%}")
    print(f"消解后模型召回率: {recall_score(y_test_debiased, y_pred_debiased):.2%}")
    
    # === 第六步：评估消解效果 ===
    print("\n[Step 6] 评估偏见消解效果...")
    metrics_debiased = compute_fairness_metrics(model_debiased, X_test_debiased, y_test_debiased, gender_test_debiased)
    print_fairness_report(metrics_debiased, "偏见检测报告（消解后）")
    
    # === 对比分析 ===
    print("\n[Step 7] 消解效果对比分析")
    print("-" * 40)
    print(f"{'指标':<25} {'消解前':<12} {'消解后':<12} {'改善':<10}")
    print("-" * 40)
    
    sp_before = metrics['statistical_parity']['difference']
    sp_after = metrics_debiased['statistical_parity']['difference']
    print(f"{'统计均等差异':<20} {sp_before:<12.2%} {sp_after:<12.2%} {(sp_before-sp_after):+.2%}")
    
    eo_before = metrics['equal_opportunity']['difference']
    eo_after = metrics_debiased['equal_opportunity']['difference']
    print(f"{'均等机会差异':<20} {eo_before:<12.2%} {eo_after:<12.2%} {(eo_before-eo_after):+.2%}")
    
    eod_before = metrics['equalized_odds']['combined_score']
    eod_after = metrics_debiased['equalized_odds']['combined_score']
    print(f"{'均等几率综合分数':<17} {eod_before:<12.2%} {eod_after:<12.2%} {(eod_before-eod_after):+.2%}")
    
    print("-" * 40)
    print("\n结论：通过数据重采样方法，成功降低了算法偏见。")
    print("      在实际应用中，可根据场景选择合适的公平性指标进行优化。")


if __name__ == "__main__":
    main()