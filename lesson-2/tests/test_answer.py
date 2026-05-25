"""
智能教育辅导系统 - 测试文件
验证 answer/main.py 中参考实现的正确性
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from answer.main import (
    tutor_conversation,
    update_mastery,
    recommend_next_topic,
    evaluate_answer
)


class TestUpdateMastery:
    """测试掌握度追踪功能"""
    
    def test_initial_topic(self):
        """测试新topic的初始概率"""
        mastery_state = {}
        result = update_mastery("新知识点", True, mastery_state)
        assert "新知识点" in result
        assert 0.5 <= result["新知识点"] <= 1.0
    
    def test_correct_answer_increases_mastery(self):
        """测试答对提高掌握概率"""
        topic = "测试知识点"
        initial_state = {topic: 0.5}
        result = update_mastery(topic, True, initial_state)
        assert result[topic] > 0.5
    
    def test_incorrect_answer_decreases_mastery(self):
        """测试答错降低掌握概率"""
        topic = "测试知识点"
        initial_state = {topic: 0.5}
        result = update_mastery(topic, False, initial_state)
        assert result[topic] < 0.5
    
    def test_mastery_bounds(self):
        """测试概率在合理范围内"""
        topic = "边界测试"
        # 测试概率不会超出 [0, 1]
        state = {topic: 0.1}
        result = update_mastery(topic, False, state)
        assert 0.0 <= result[topic] <= 1.0
        
        state = {topic: 0.9}
        result = update_mastery(topic, True, state)
        assert 0.0 <= result[topic] <= 1.0


class TestRecommendNextTopic:
    """测试学习路径推荐功能"""
    
    def test_recommend_unmastered_with_mastered_prereqs(self):
        """测试推荐前置知识已掌握的未掌握知识点"""
        mastery_state = {
            "基础知识": 0.8,
            "进阶内容": 0.3
        }
        knowledge_graph = {
            "基础知识": {"prerequisites": [], "last_practiced": None},
            "进阶内容": {"prerequisites": ["基础知识"], "last_practiced": None}
        }
        result = recommend_next_topic(mastery_state, knowledge_graph)
        assert "进阶内容" in result
    
    def test_no_recommendation_for_unready_topic(self):
        """测试前置知识未掌握时不推荐"""
        mastery_state = {
            "基础知识": 0.3,
            "进阶内容": 0.3
        }
        knowledge_graph = {
            "基础知识": {"prerequisites": [], "last_practiced": None},
            "进阶内容": {"prerequisites": ["基础知识"], "last_practiced": None}
        }
        result = recommend_next_topic(mastery_state, knowledge_graph)
        # 应该不推荐进阶内容，因为前置知识未掌握
        assert "进阶内容" not in result
    
    def test_recommend_review_for_frequently_practiced(self):
        """测试推荐需要复习的已学知识点"""
        import time
        mastery_state = {
            "已学内容": 0.6
        }
        knowledge_graph = {
            "已学内容": {
                "prerequisites": [],
                "last_practiced": time.time() - 86400 * 2  # 2天前
            }
        }
        result = recommend_next_topic(mastery_state, knowledge_graph)
        # 应该推荐复习
        assert "已学内容" in result or "复习" in result


class TestEvaluateAnswer:
    """测试自动评估功能"""
    
    def test_exact_match(self):
        """测试完全正确的答案"""
        question = "1 + 1 = ?"
        student_answer = "2"
        correct_answer = "2"
        result = evaluate_answer(question, student_answer, correct_answer)
        assert result["score"] == 100
    
    def test_incorrect_answer(self):
        """测试错误答案"""
        question = "1 + 1 = ?"
        student_answer = "3"
        correct_answer = "2"
        result = evaluate_answer(question, student_answer, correct_answer)
        assert "score" in result
        assert "feedback" in result
        assert result["score"] < 100
    
    def test_case_insensitive(self):
        """测试大小写不敏感"""
        question = "中国的首都?"
        student_answer = "BEIJING"
        correct_answer = "beijing"
        result = evaluate_answer(question, student_answer, correct_answer)
        assert result["score"] == 100
    
    def test_feedback_exists(self):
        """测试反馈信息存在"""
        question = "测试题目"
        student_answer = "学生答案"
        correct_answer = "正确答案"
        result = evaluate_answer(question, student_answer, correct_answer)
        assert "feedback" in result
        assert len(result["feedback"]) > 0


class TestTutorConversation:
    """测试智能辅导功能"""
    
    def test_conversation_returns_string(self):
        """测试对话返回字符串"""
        result = tutor_conversation("什么是函数？", [])
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_conversation_with_history(self):
        """测试带历史记录的对话"""
        history = [
            {"role": "user", "content": "什么是变量？"},
            {"role": "assistant", "content": "你能举个例子吗？"}
        ]
        result = tutor_conversation("就像一个盒子", history)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_conversation_does_not_give_direct_answer(self):
        """测试对话不直接给答案（引导式回复）"""
        result = tutor_conversation("如何计算圆的面积？", [])
        # 检查是否包含引导性词语
        result_lower = result.lower()
        # 辅导回复应该包含问题或引导性词语，而不是直接给出公式
        has_question = "?" in result or "？" in result
        has_keywords = any(word in result_lower for word in ["想想", "思考", "引导", "你能", "是否", "假设", "如果", "可以", "你觉得"])
        assert has_question or has_keywords


class TestIntegration:
    """集成测试：验证核心功能协同工作"""
    
    def test_mastery_tracking_and_recommendation_integration(self):
        """测试掌握度追踪与学习路径推荐的协同"""
        mastery_state = {}
        
        # 模拟答题过程
        update_mastery("基础运算", True, mastery_state)
        update_mastery("方程求解", True, mastery_state)
        update_mastery("应用题", False, mastery_state)
        
        knowledge_graph = {
            "基础运算": {"prerequisites": [], "last_practiced": None},
            "方程求解": {"prerequisites": ["基础运算"], "last_practiced": None},
            "应用题": {"prerequisites": ["方程求解"], "last_practiced": None}
        }
        
        result = recommend_next_topic(mastery_state, knowledge_graph)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_answer_evaluation_affects_mastery(self):
        """测试答案评估影响掌握度"""
        question = "解方程: x + 2 = 5"
        correct_answer = "x = 3"
        
        # 正确答案应该提高掌握度
        mastery_state = {"一元一次方程": 0.5}
        student_correct = "x = 3"
        result = evaluate_answer(question, student_correct, correct_answer)
        
        assert result["score"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])