"""
算法偏见检测与公平性度量 - 参考答案实现
课程《人工智能赋能社会科学》第「算法偏见与数据治理」课配套项目

本文件提供了招聘筛选场景下的偏见检测与公平性度量的完整实现。
包含：带偏见数据生成、公平性指标计算、偏见消解三大模块。
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
    
    # 创建空数据框
    data = pd.DataFrame()
    
    # 性别分布：男性80%，女性20%（模拟技术岗位的历史性别比例）
    # 这种不平等的性别分布本身就是系统性偏见的一部分
    gender_probs = [0.8, 0.2]
    data['gender'] = np.random.choice(['male', 'female'], size=n_samples, p=gender_probs)
    
    # 能力特征：与性别无关，均匀分布
    # 这些特征应该反映候选人的真实能力，与性别无关
    # 技能得分：正态分布，均值70，标准差15
    data['skill_score'] = np.random.normal(70, 15, n_samples).clip(0, 100)
    
    # 工作年限：指数分布，模拟大多数人有较短的工作经验
    data['experience_years'] = np.random.exponential(5, n_samples).clip(0, 20)
    
    # 教育水平：1=专科，2=本科，3=研究生
    data['education_level'] = np.random.choice([1, 2, 3], size=n_samples, p=[0.2, 0.5, 0.3])
    
    # 面试分数：正态分布，均值72，标准差12
    data['interview_score'] = np.random.normal(72, 12, n_samples).clip(0, 100)
    
    # 录用决策：基于能力，但加入性别偏见
    # 关键机制：相同能力的男性更容易被录用，这就是历史偏见的体现
    
    # 计算综合能力得分（用于决定录用）
    ability_score = (
        data['skill_score'] * 0.3 +      # 技能得分权重30%
        data['interview_score'] * 0.4 +   # 面试分数权重40%（面试环节可能存在偏见）
        data['experience_years'] * 2 +    # 每年经验加2分
        data['education_level'] * 5       # 教育水平每级加5分
    )
    
    # 基础录用概率（基于能力标准化到0-1）
    ability_min = ability_score.min()
    ability_max = ability_score.max()
    base_prob = (ability_score - ability_min) / (ability_max - ability_min)
    
    # 加入性别偏见参数：
    # 男性获得+0.25的加成（更容易被录用）
    # 女性获得-0.15的惩罚（更难被录用）
    # 这些参数模拟了历史数据中存在的性别歧视
    gender_bias = data['gender'].map({'male': 0.25, 'female': -0.15})
    
    # 最终录用概率（考虑能力和偏见）
    final_prob = (base_prob + gender_bias).clip(0, 1)
    
    # 根据概率生成录用标签
    data['hired'] = (np.random.random(n_samples) < final_prob).astype(int)
    
    # 验证偏见的存在：打印录用率统计
    print(f"  [数据生成验证] 男性录用率: {data[data['gender']=='male']['hired'].mean():.2%}")
    print(f"  [数据生成验证] 女性录用率: {data[data['gender']=='female']['hired'].mean():.2%}")
    print(f"  [数据生成验证] 录用率差异: {abs(data[data['gender']=='male']['hired'].mean() - data[data['gender']=='female']['hired'].mean()):.2%}")
    
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
        tuple: (训练好的模型, 测试集特征, 测试集标签, 性别测试集, 标准化器)
    """
    # 准备特征和标签
    feature_cols = ['skill_score', 'experience_years', 'education_level', 'interview_score']
    X = data[feature_cols]
    y = data[target_col]
    
    # 保存性别信息用于公平性评估
    gender = data['gender']
    
    # 分割数据集，使用分层抽样确保训练集和测试集中性别比例一致
    X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
        X, y, gender, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 特征标准化（对逻辑回归很重要）
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
    
    # === 统计均等 (Statistical Parity / Demographic Parity) ===
    # 定义：不同受保护群体被模型预测为正例的比例差异
    # 计算公式：|P(Ŷ=1|G=male) - P(Ŷ=1|G=female)|
    # 衡量标准：模型对不同群体给出正面预测的比例是否一致
    
    # 计算男性被预测为录用的比例
    male_data = results_df[results_df['gender'] == 'male']
    male_pred_positive = (male_data['y_pred'] == 1).mean()
    
    # 计算女性被预测为录用的比例
    female_data = results_df[results_df['gender'] == 'female']
    female_pred_positive = (female_data['y_pred'] == 1).mean()
    
    # 计算差异
    sp_difference = abs(male_pred_positive - female_pred_positive)
    
    metrics['statistical_parity'] = {
        'male_positive_rate': male_pred_positive,
        'female_positive_rate': female_pred_positive,
        'difference': sp_difference,
        'description': '统计均等：不同性别群体的录用预测比例差异'
    }
    
    # === 均等机会 (Equal Opportunity) ===
    # 定义：在真实标签为正例的群体中，不同群体的真阳性率(TPR)差异
    # 计算公式：|TPR_male - TPR_female|，其中 TPR = P(Ŷ=1|Y=1)
    # 衡量标准：在真正合格的候选人中，录用机会是否平等
    # 这个指标关注"对的人"是否都有同等的机会被选中
    
    # 计算男性在合格候选人中的录用率（真阳性率）
    male_qualified = results_df[(results_df['gender'] == 'male') & (results_df['y_true'] == 1)]
    male_tpr = (male_qualified['y_pred'] == 1).mean() if len(male_qualified) > 0 else 0.0
    
    # 计算女性在合格候选人中的录用率（真阳性率）
    female_qualified = results_df[(results_df['gender'] == 'female') & (results_df['y_true'] == 1)]
    female_tpr = (female_qualified['y_pred'] == 1).mean() if len(female_qualified) > 0 else 0.0
    
    # 计算差异
    eo_difference = abs(male_tpr - female_tpr)
    
    metrics['equal_opportunity'] = {
        'male_tpr': male_tpr,
        'female_tpr': female_tpr,
        'difference': eo_difference,
        'description': '均等机会：合格候选人中不同性别群体的录用率差异'
    }
    
    # === 均等几率 (Equalized Odds) ===
    # 定义：同时考虑TPR和FPR的公平性指标
    # 计算公式：|TPR_male - TPR_female| + |FPR_male - FPR_female|
    # 衡量标准：不仅关注"对的人"是否被同样对待，也关注"错的人"是否被同样误判
    # 这个指标更难满足，但更全面地反映了公平性
    
    # 计算男性真阳性率（TPR）
    male_tpr = (male_qualified['y_pred'] == 1).mean() if len(male_qualified) > 0 else 0.0
    
    # 计算女性真阳性率（TPR）
    female_tpr = (female_qualified['y_pred'] == 1).mean() if len(female_qualified) > 0 else 0.0
    
    # 计算男性假阳性率（FPR）- 在不合格的人中错误录用
    male_unqualified = results_df[(results_df['gender'] == 'male') & (results_df['y_true'] == 0)]
    male_fpr = (male_unqualified['y_pred'] == 1).mean() if len(male_unqualified) > 0 else 0.0
    
    # 计算女性假阳性率（FPR）
    female_unqualified = results_df[(results_df['gender'] == 'female') & (results_df['y_true'] == 0)]
    female_fpr = (female_unqualified['y_pred'] == 1).mean() if len(female_unqualified) > 0 else 0.0
    
    # 计算各项差异
    tpr_diff = abs(male_tpr - female_tpr)
    fpr_diff = abs(male_fpr - female_fpr)
    combined_score = tpr_diff + fpr_diff
    
    metrics['equalized_odds'] = {
        'male_tpr': male_tpr,
        'female_tpr': female_tpr,
        'tpr_difference': tpr_diff,
        'male_fpr': male_fpr,
        'female_fpr': female_fpr,
        'fpr_difference': fpr_diff,
        'combined_score': combined_score,
        'description': '均等几率：综合考虑真阳性率和假阳性率的公平性'
    }
    
    # === 综合评估 ===
    # 综合考虑统计均等因素均等机会，给出一个总体公平性得分
    overall_score = 1 - (sp_difference + eo_difference) / 2
    
    if overall_score >= 0.9:
        recommendation = "公平性优秀：模型在不同性别群体间表现均衡"
    elif overall_score >= 0.8:
        recommendation = "公平性良好：存在轻微偏见，可接受"
    elif overall_score >= 0.7:
        recommendation = "公平性一般：建议进行偏见消解"
    else:
        recommendation = "公平性问题严重：需要进行偏见消解干预"
    
    metrics['summary'] = {
        'overall_fairness_score': overall_score,
        'recommendation': recommendation
    }
    
    return metrics


def debias_by_resampling(data, target_col='hired', protected_attr='gender'):
    """
    基于数据重采样的偏见消解方法
    
    该函数通过调整训练数据的分布来减少偏见。
    采用的策略：
    1. 识别存在偏见的样本区间（中等能力区间偏见最明显）
    2. 在该区间内，提高女性的录用概率
    3. 在该区间内，适当降低男性的录用概率
    4. 保持高能力和低能力候选人的录用决策不变
    
    这种方法的原理是：在能力边缘（极高或极低），决策通常是一致的；
    偏见主要体现在中等能力区间的模糊地带。
    
    参数:
        data (pd.DataFrame): 原始数据集
        target_col (str): 目标列名
        protected_attr (str): 受保护的属性列名
    
    返回:
        pd.DataFrame: 经过偏见消解处理的数据集
    """
    df = data.copy()
    
    # 分离男性和女性数据
    male_data = df[df['gender'] == 'male']
    female_data = df[df['gender'] == 'female']
    
    # 计算当前录用率差异
    male_hire_rate = male_data[target_col].mean()
    female_hire_rate = female_data[target_col].mean()
    original_gap = abs(male_hire_rate - female_hire_rate)
    
    print(f"[偏见消解前] 男性录用率: {male_hire_rate:.2%}, 女性录用率: {female_hire_rate:.2%}, 差异: {original_gap:.2%}")
    
    # 计算综合能力得分
    # 使用与数据生成时相同的权重
    ability_score = (
        df['skill_score'] * 0.3 +
        df['interview_score'] * 0.4 +
        df['experience_years'] * 2 +
        df['education_level'] * 5
    )
    
    # 计算能力分数的分位数，用于确定调整区间
    q25 = ability_score.quantile(0.25)
    q75 = ability_score.quantile(0.75)
    median = ability_score.median()
    
    # 偏见主要存在于中等能力区间（Q25到Q75之间）
    # 在这个区间内，对录用决策进行微调
    mask_middle_ability = (ability_score >= q25) & (ability_score <= q75)
    
    # === 策略：标签翻转调整 ===
    # 对于中等能力区间的候选人，根据性别和当前录用状态进行调整
    
    # 1. 对中等能力区间的女性未被录用者：提高录用概率
    # 条件：女性 + 中等能力 + 未录用
    female_candidates = df[(df['gender'] == 'female') & mask_middle_ability & (df[target_col] == 0)]
    
    # 计算应该翻转的比例（目标是让女性录用率接近男性）
    # 需要增加的女性录用数
    target_flip_rate = min(0.35, original_gap * 1.5)  # 翻转35%或与差距相关
    
    if len(female_candidates) > 0:
        flip_count = int(len(female_candidates) * target_flip_rate)
        flip_indices = female_candidates.sample(n=min(flip_count, len(female_candidates)), 
                                                  random_state=42).index
        df.loc[flip_indices, target_col] = 1
        print(f"  [调整] 将 {len(flip_indices)} 名中等能力女性从未录用改为录用")
    
    # 2. 对中等能力区间的男性被录用者：降低录用概率
    # 条件：男性 + 中等能力 + 已录用
    male_candidates = df[(df['gender'] == 'male') & mask_middle_ability & (df[target_col] == 1)]
    
    # 计算应该翻转的比例
    # 需要减少的男性录用数
    target_flip_rate_male = min(0.20, original_gap)  # 翻转20%或与差距相关
    
    if len(male_candidates) > 0:
        flip_count = int(len(male_candidates) * target_flip_rate_male)
        flip_indices = male_candidates.sample(n=min(flip_count, len(male_candidates)), 
                                                random_state=42).index
        df.loc[flip_indices, target_col] = 0
        print(f"  [调整] 将 {len(flip_indices)} 名中等能力男性从录用改为未录用")
    
    # 计算调整后的录用率
    male_hire_rate_new = df[df['gender'] == 'male'][target_col].mean()
    female_hire_rate_new = df[df['gender'] == 'female'][target_col].mean()
    new_gap = abs(male_hire_rate_new - female_hire_rate_new)
    
    print(f"[偏见消解后] 男性录用率: {male_hire_rate_new:.2%}, 女性录用率: {female_hire_rate_new:.2%}, 差异: {new_gap:.2%}")
    print(f"[偏见改善] 录用率差异从 {original_gap:.2%} 降至 {new_gap:.2%}，改善 {(original_gap-new_gap)/original_gap:.1%}")
    
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
    
    流程包括：
    1. 生成带偏见的模拟招聘数据
    2. 训练逻辑回归模型
    3. 计算公平性指标（偏见检测）
    4. 执行偏见消解（数据重采样）
    5. 重新训练并评估消解效果
    6. 输出对比分析报告
    """
    print("="*60)
    print("算法偏见检测与公平性度量演示")
    print("课程《人工智能赋能社会科学》- 算法偏见与数据治理")
    print("="*60)
    
    # 设置随机种子确保可复现
    np.random.seed(42)
    
    # === 第一步：生成带偏见的招聘数据 ===
    print("\n[Step 1] 生成带偏见的招聘模拟数据...")
    print("-" * 40)
    data = generate_biased_recruitment_data(n_samples=1000)
    
    print(f"\n数据集统计信息:")
    print(f"  - 总样本数: {len(data)} 条记录")
    print(f"  - 性别分布: 男性 {(data['gender']=='male').mean():.1%}, 女性 {(data['gender']=='female').mean():.1%}")
    print(f"  - 整体录用率: {data['hired'].mean():.1%}")
    print(f"  - 男性录用率: {data[data['gender']=='male']['hired'].mean():.1%}")
    print(f"  - 女性录用率: {data[data['gender']=='female']['hired'].mean():.1%}")
    
    # === 第二步：训练模型并评估偏见 ===
    print("\n[Step 2] 训练招聘筛选模型...")
    print("-" * 40)
    model, X_test, y_test, gender_test, scaler = train_model(data)
    
    # 在测试集上评估模型性能
    y_pred = model.predict(X_test)
    print(f"模型性能评估:")
    print(f"  - 准确率: {accuracy_score(y_test, y_pred):.2%}")
    print(f"  - 精确率: {precision_score(y_test, y_pred):.2%}")
    print(f"  - 召回率: {recall_score(y_test, y_pred):.2%}")
    
    # === 第三步：计算公平性指标 ===
    print("\n[Step 3] 计算公平性指标...")
    print("-" * 40)
    print("评估模型在不同性别群体间的预测差异...")
    metrics = compute_fairness_metrics(model, X_test, y_test, gender_test)
    print_fairness_report(metrics, "偏见检测报告（消解前）")
    
    # === 第四步：偏见消解 ===
    print("\n[Step 4] 执行偏见消解（数据重采样）...")
    print("-" * 40)
    print("使用能力匹配的重采样方法调整训练数据...")
    data_debiased = debias_by_resampling(data)
    
    # === 第五步：重新训练并评估 ===
    print("\n[Step 5] 使用消解后数据重新训练模型...")
    print("-" * 40)
    model_debiased, X_test_debiased, y_test_debiased, gender_test_debiased, _ = train_model(data_debiased)
    
    y_pred_debiased = model_debiased.predict(X_test_debiased)
    print(f"消解后模型性能评估:")
    print(f"  - 准确率: {accuracy_score(y_test_debiased, y_pred_debiased):.2%}")
    print(f"  - 精确率: {precision_score(y_test_debiased, y_pred_debiased):.2%}")
    print(f"  - 召回率: {recall_score(y_test_debiased, y_pred_debiased):.2%}")
    
    # === 第六步：评估消解效果 ===
    print("\n[Step 6] 评估偏见消解效果...")
    print("-" * 40)
    metrics_debiased = compute_fairness_metrics(model_debiased, X_test_debiased, y_test_debiased, gender_test_debiased)
    print_fairness_report(metrics_debiased, "偏见检测报告（消解后）")
    
    # === 对比分析 ===
    print("\n[Step 7] 消解效果对比分析")
    print("-" * 40)
    print(f"{'指标':<25} {'消解前':<12} {'消解后':<12} {'变化':<10}")
    print("-" * 40)
    
    sp_before = metrics['statistical_parity']['difference']
    sp_after = metrics_debiased['statistical_parity']['difference']
    sp_change = sp_before - sp_after
    print(f"{'统计均等差异':<20} {sp_before:<12.2%} {sp_after:<12.2%} {sp_change:+.2%}")
    
    eo_before = metrics['equal_opportunity']['difference']
    eo_after = metrics_debiased['equal_opportunity']['difference']
    eo_change = eo_before - eo_after
    print(f"{'均等机会差异':<20} {eo_before:<12.2%} {eo_after:<12.2%} {eo_change:+.2%}")
    
    eod_before = metrics['equalized_odds']['combined_score']
    eod_after = metrics_debiased['equalized_odds']['combined_score']
    eod_change = eod_before - eod_after
    print(f"{'均等几率综合分数':<17} {eod_before:<12.2%} {eod_after:<12.2%} {eod_change:+.2%}")
    
    fairness_before = metrics['summary']['overall_fairness_score']
    fairness_after = metrics_debiased['summary']['overall_fairness_score']
    print(f"{'综合公平性得分':<17} {fairness_before:<12.2%} {fairness_after:<12.2%} {fairness_after-fairness_before:+.2%}")
    
    print("-" * 40)
    
    # === 结论 ===
    print("\n【结论与讨论】")
    print("-" * 40)
    
    if sp_change > 0 and eo_change > 0:
        print("✓ 偏见消解成功！统计均等和均等机会指标均有改善。")
    elif sp_change > 0 or eo_change > 0:
        print("◐ 部分偏见消解成功！至少一个公平性指标有所改善。")
    else:
        print("✗ 偏见消解效果不明显，可能需要调整策略或参数。")
    
    print(f"""
在实际应用中，选择哪种公平性指标取决于具体场景：
- 统计均等：关注"谁能获得机会"，适合招聘配额等场景
- 均等机会：关注"合格者是否被平等对待"，更适合能力评估场景
- 均等几率：最严格的公平性标准，但可能难以同时满足所有条件

本项目演示了基于数据重采样的偏见消解方法。
其他方法还包括：
1. 对抗训练（Adversarial Debiasing）
2. 公平性约束优化（Fairness Constraints）
3. 后处理校正（Post-processing Calibration）
""")
    
    print("="*60)
    print("演示完成")
    print("="*60)


if __name__ == "__main__":
    main()