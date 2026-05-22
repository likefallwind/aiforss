"""
智能教育辅导系统
基于大模型的苏格拉底式智能辅导对话、知识点掌握度追踪、自适应学习路径推荐、自动评估与反馈生成
"""
import os
import json
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
    # TODO: 构建提示词，要求模型使用追问、类比、拆分问题等方式引导学生
    # 提示：提示词应明确要求模型不直接给答案，而是通过提问引导
    prompt = "你是苏格拉底式的辅导老师，使用追问、类比、拆分问题等方式引导学生自主发现答案。禁止直接给出答案。"

    messages = [
        {"role": "system", "content": prompt}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": student_input})
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content


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
    # TODO: 实现掌握度更新逻辑
    # 提示：答对时提高概率，答错时降低概率，使用贝叶斯条件概率简化模型
    # 初始概率可设为0.5
    pass


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
    # TODO: 实现学习路径推荐逻辑
    # 提示：
    # 1. 优先推荐前置知识已掌握的未掌握知识点
    # 2. 根据间隔重复原则，对即将遗忘的已学知识点安排复习
    # 3. 返回推荐的主题名称
    pass


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
    # TODO: 调用大模型进行评分和反馈生成
    # 提示：可以构建评估提示词，让模型对比学生答案和标准答案，给出评分(0-100)和改进建议
    pass


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
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()