"""
智能教育辅导系统
基于大模型的苏格拉底式智能辅导对话、知识点掌握度追踪、自适应学习路径推荐、自动评估与反馈生成
"""
import os
import json
import time
from openai import OpenAI
from tqdm import tqdm

# 配置API Key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ============ Task 1: 智能辅导系统 ============

def tutor_conversation(student_input: str, history: list[dict]) -> str:
    """
    苏格拉底式辅导对话
    
    Args:
        student_input: 学生输入的问题或回复
        history: 对话历史，每条记录包含 {'role': 'user'/'assistant', 'content': str}
    
    Returns:
        str: 辅导老师的引导式回复
    """
    # 构建苏格拉底式引导提示词
    system_prompt = """你是苏格拉底式的辅导老师，使用追问、类比、拆分问题等方式引导学生自主发现答案。
    
要求：
1. 禁止直接给出答案
2. 通过提问引导思考，将复杂问题拆分为简单步骤
3. 使用类比帮助理解抽象概念
4. 适时总结学生已理解的部分
5. 鼓励学生的思考过程

每次回复应包含1-2个引导性问题，帮助学生逐步接近答案。"""

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": student_input})
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"抱歉，辅导过程中遇到问题：{str(e)}"


# ============ Task 2: 掌握度追踪 ============

def update_mastery(topic: str, is_correct: bool, mastery_state: dict) -> dict:
    """
    使用简化的贝叶斯方法更新知识点的掌握概率
    
    Args:
        topic: 知识点名称
        is_correct: 学生答题是否正确
        mastery_state: 当前的掌握度状态字典，格式为 {topic: probability}
    
    Returns:
        dict: 更新后的掌握度状态
    """
    # 初始化topic如果不存在
    if topic not in mastery_state:
        mastery_state[topic] = 0.5
    
    current_prob = mastery_state[topic]
    
    # 简化的贝叶斯更新模型
    # 答对：P(掌握|正确) = P(正确|掌握) * P(掌握) / P(正确)
    # 答错：P(掌握|错误) = P(错误|掌握) * P(掌握) / P(错误)
    
    # 假设：正确率与掌握度相关
    # P(正确|掌握) = 0.9, P(错误|掌握) = 0.1
    # P(正确|不掌握) = 0.2, P(错误|不掌握) = 0.8
    
    if is_correct:
        # 答对：提高掌握概率
        # 使用似然比更新
        likelihood_ratio = 0.9 / 0.2  # 4.5
        new_prob = (likelihood_ratio * current_prob) / (1 + (likelihood_ratio - 1) * current_prob)
        # 添加适度调整因子，使更新更平滑
        adjustment = 0.3
        new_prob = current_prob + adjustment * (new_prob - current_prob)
    else:
        # 答错：降低掌握概率
        likelihood_ratio = 0.1 / 0.8  # 0.125
        new_prob = (likelihood_ratio * current_prob) / (1 + (likelihood_ratio - 1) * current_prob)
        # 添加适度调整因子
        adjustment = 0.3
        new_prob = current_prob + adjustment * (new_prob - current_prob)
    
    # 确保概率在合理范围内 [0, 1]
    new_prob = max(0.0, min(1.0, new_prob))
    
    # 更新状态
    mastery_state[topic] = new_prob
    return mastery_state


# ============ Task 3: 自适应学习路径推荐 ============

def recommend_next_topic(mastery_state: dict, knowledge_graph: dict) -> str:
    """
    根据当前掌握度状态推荐下一个学习主题
    
    Args:
        mastery_state: 掌握度状态字典 {topic: probability}
        knowledge_graph: 知识图谱，定义知识点依赖关系 {topic: {"prerequisites": [topics], "last_practiced": timestamp}}
    
    Returns:
        str: 推荐的下一个学习主题
    """
    MASTERY_THRESHOLD = 0.7  # 掌握阈值
    REVIEW_THRESHOLD = 0.5   # 需要复习的阈值
    
    candidates = []
    
    for topic, info in knowledge_graph.items():
        # 获取当前掌握度，默认0.5
        current_mastery = mastery_state.get(topic, 0.5)
        prerequisites = info.get("prerequisites", [])
        last_practiced = info.get("last_practiced")
        
        # 检查前置知识是否都已掌握
        prereqs_mastered = all(
            mastery_state.get(p, 0.5) >= MASTERY_THRESHOLD 
            for p in prerequisites
        )
        
        if current_mastery < MASTERY_THRESHOLD and prereqs_mastered:
            # 该知识点未掌握但前置知识已掌握，优先推荐
            priority = 1.0 - current_mastery  # 掌握度越低优先级越高
            candidates.append((topic, priority, "learn"))
        
        # 检查是否需要复习（基于间隔重复）
        # 如果掌握度下降到一定水平且距离上次练习有一定时间间隔，需要复习
        if current_mastery < MASTERY_THRESHOLD and current_mastery >= REVIEW_THRESHOLD:
            # 检查是否超过复习间隔（简化处理：超过1天建议复习）
            if last_practiced is not None:
                days_since_practice = (time.time() - last_practiced) / 86400
                if days_since_practice > 1:
                    priority = 0.5 * (1 - current_mastery)
                    candidates.append((topic, priority, "review"))
    
    if candidates:
        # 按优先级排序，选择优先级最高的主题
        candidates.sort(key=lambda x: x[1], reverse=True)
        recommended_topic, priority, action = candidates[0]
        
        # 更新知识图谱中的最后练习时间
        if recommended_topic in knowledge_graph:
            knowledge_graph[recommended_topic]["last_practiced"] = time.time()
        
        return f"{recommended_topic} ({'学习' if action == 'learn' else '复习'})"
    
    # 如果没有合适的候选，返回默认建议
    return "建议从基础知识开始学习"


# ============ Task 4: 自动评估与反馈生成 ============

def evaluate_answer(question: str, student_answer: str, correct_answer: str) -> dict:
    """
    利用大模型对学生答案进行评分并生成个性化反馈
    
    Args:
        question: 题目
        student_answer: 学生答案
        correct_answer: 标准答案
    
    Returns:
        dict: 包含评分(score: int, feedback: str)的字典
    """
    # 简单的客观题自动评分
    # 去除空格和大小写差异后比较
    student_clean = student_answer.strip().lower()
    correct_clean = correct_answer.strip().lower()
    
    if student_clean == correct_clean:
        return {
            "score": 100,
            "feedback": "答案完全正确！"
        }
    
    # 对于需要详细评估的题目，调用大模型
    evaluation_prompt = f"""请评估以下学生答案：

题目：{question}
学生答案：{student_answer}
标准答案：{correct_answer}

请从以下几个方面评估并给出反馈：
1. 答案的正确性
2. 思路是否清晰
3. 是否有进步空间
4. 具体改进建议

请用JSON格式回复，包含：
- score: 0-100的整数评分
- feedback: 具体的改进建议（中文，50字以内）"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位教育评估专家，负责评估学生答案并给出改进建议。"},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        result_text = response.choices[0].message.content
        
        # 尝试解析JSON响应
        try:
            result = json.loads(result_text)
            return {
                "score": result.get("score", 60),
                "feedback": result.get("feedback", "需要进一步改进")
            }
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回基础评分
            return {
                "score": 60,
                "feedback": result_text[:50] if result_text else "需要进一步改进"
            }
            
    except Exception as e:
        return {
            "score": 0,
            "feedback": f"评估过程出错：{str(e)}"
        }


def main():
    """演示四个核心功能的协同工作流程"""
    print("=" * 60)
    print("智能教育辅导系统 - 功能演示")
    print("=" * 60)
    
    # 演示对话历史
    history = []
    
    # Task 1: 智能辅导演示
    print("\n【Task 1: 智能辅导系统】")
    print("-" * 40)
    student_question = "如何解一元二次方程？"
    print(f"学生问题: {student_question}")
    response = tutor_conversation(student_question, history)
    print(f"辅导老师: {response}")
    history.append({"role": "user", "content": student_question})
    history.append({"role": "assistant", "content": response})
    
    # Task 2: 掌握度追踪演示
    print("\n【Task 2: 掌握度追踪】")
    print("-" * 40)
    mastery_state = {
        "配方法": 0.5,
        "完全平方公式": 0.6,
        "求根公式": 0.4
    }
    print(f"初始状态: {mastery_state}")
    
    # 学生答对"配方法"题目
    new_state = update_mastery("配方法", is_correct=True, mastery_state=mastery_state)
    print(f"答对后状态: {new_state}")
    
    # 学生答错"求根公式"题目
    new_state = update_mastery("求根公式", is_correct=False, mastery_state=new_state)
    print(f"答错后状态: {new_state}")
    
    # Task 3: 自适应学习路径推荐演示
    print("\n【Task 3: 自适应学习路径推荐】")
    print("-" * 40)
    knowledge_graph = {
        "配方法": {"prerequisites": ["完全平方公式"], "last_practiced": None},
        "完全平方公式": {"prerequisites": [], "last_practiced": None},
        "求根公式": {"prerequisites": ["配方法"], "last_practiced": None}
    }
    print(f"当前掌握度: {new_state}")
    next_topic = recommend_next_topic(new_state, knowledge_graph)
    print(f"推荐学习: {next_topic}")
    
    # Task 4: 自动评估与反馈生成演示
    print("\n【Task 4: 自动评估与反馈生成】")
    print("-" * 40)
    question = "求方程 x^2 + 5x + 6 = 0 的解"
    student_answer = "x = -2 或 x = -3"
    correct_answer = "x = -2 或 x = -3"
    result = evaluate_answer(question, student_answer, correct_answer)
    print(f"评分: {result.get('score', 'N/A')}")
    print(f"反馈: {result.get('feedback', 'N/A')}")
    
    # 多轮对话演示
    print("\n【多轮对话演示】")
    print("-" * 40)
    follow_up = "那判别式是什么？"
    print(f"学生追问: {follow_up}")
    response = tutor_conversation(follow_up, history)
    print(f"辅导老师: {response}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()