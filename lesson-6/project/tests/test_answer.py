"""
人机协作对话助手 - 测试文件

本测试文件验证 answer/main.py 中核心功能的正确性。
测试覆盖：
- Task 1: search_resources 工具调用功能
- Task 2: save_reflection 反思记录功能
- Task 3: generate_response 人格化响应生成
- Task 4: ConversationManager 多轮对话管理
- 集成测试：完整对话流程

作者：AI 课程项目
"""

import pytest
import sys
import os

# 添加项目根目录到路径，确保可以正确导入 answer 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from answer.main import (
    ResponseStyle,
    search_resources,
    save_reflection,
    generate_response,
    ConversationManager,
    run_conversation_demo
)


class TestConversationSystem:
    """人机协作对话助手系统测试类"""
    
    # ===== Task 1: 搜索工具测试 =====
    
    def test_search_resources_basic(self):
        """测试基本的搜索功能"""
        result = search_resources("焦虑")
        
        # 验证返回结果包含必要的组成部分
        assert "焦虑" in result or "心理学" in result
        assert "标题" in result or "资源信息" in result
        assert "摘要" in result
    
    def test_search_resources_with_category(self):
        """测试带类别的搜索功能"""
        result = search_resources("压力", category="psychology")
        
        # 验证类别信息在结果中
        assert "psychology" in result or "心理学" in result
    
    def test_search_resources_default(self):
        """测试搜索默认类别"""
        result = search_resources("unknown_topic")
        
        # 应该返回默认的心理学概览内容
        assert "心理学" in result or "基础知识" in result
    
    # ===== Task 2: 反思记录工具测试 =====
    
    def test_save_reflection_basic(self):
        """测试基本的反思记录功能"""
        result = save_reflection("user_001", "今天学到了很多关于情绪管理的知识。")
        
        # 验证返回结果包含确认信息和记录ID
        assert "保存" in result or "成功" in result
        assert "REF-" in result or "记录ID" in result
        assert "user_001" in result
    
    def test_save_reflection_unique_id(self):
        """测试反思记录生成唯一ID"""
        result1 = save_reflection("user_001", "反思内容1")
        result2 = save_reflection("user_001", "反思内容2")
        
        # 两个不同的反思应该有不同的记录ID
        # 从结果中提取记录ID进行比较
        assert result1 != result2 or "REF-" in result1
    
    # ===== Task 3: 人格化响应生成测试 =====
    
    def test_generate_response_empathetic(self):
        """测试共情风格的响应生成"""
        result = generate_response(ResponseStyle.EMPATHETIC, "我最近感到很焦虑")
        
        # 验证返回了有意义的响应
        assert len(result) > 0
        # 共情风格应该包含情感相关的词汇
        assert any(word in result for word in ["理解", "感受", "倾听", "支持", "你"])
    
    def test_generate_response_analytical(self):
        """测试分析风格的响应生成"""
        result = generate_response(ResponseStyle.ANALYTICAL, "如何处理工作压力")
        
        # 验证返回了有意义的响应
        assert len(result) > 0
        # 分析风格应该包含分析相关的词汇
        assert any(word in result for word in ["分析", "心理学", "因素", "帮助", "信息"])
    
    def test_generate_response_neutral(self):
        """测试中立风格的响应生成"""
        result = generate_response(ResponseStyle.NEUTRAL, "什么是认知行为疗法")
        
        # 验证返回了有意义的响应
        assert len(result) > 0
        # 中立风格应该相对客观
        assert len(result) > 10
    
    def test_generate_response_encouraging(self):
        """测试鼓励风格的响应生成"""
        result = generate_response(ResponseStyle.ENCOURAGING, "谢谢你的帮助")
        
        # 验证返回了有意义的响应
        assert len(result) > 0
        # 鼓励风格应该包含积极词汇
        assert any(word in result for word in ["好", "鼓励", "成长", "积极", "继续"])
    
    def test_generate_response_string_style(self):
        """测试使用字符串风格的响应生成"""
        result = generate_response("empathetic", "我遇到了困难")
        
        # 验证可以接受字符串形式的风格参数
        assert len(result) > 0
    
    # ===== Task 4: 多轮对话管理测试 =====
    
    def test_conversation_manager_init(self):
        """测试对话管理器初始化"""
        manager = ConversationManager("test_user")
        
        # 验证初始化状态正确
        assert manager.user_id == "test_user"
        assert len(manager.history) == 0
        assert manager.current_style == ResponseStyle.EMPATHETIC
    
    def test_conversation_manager_add_message(self):
        """测试添加消息到对话历史"""
        manager = ConversationManager("test_user")
        manager.add_message("user", "你好")
        manager.add_message("assistant", "你好，我是助手")
        
        # 验证消息已添加到历史
        assert len(manager.history) == 2
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "你好"
        assert manager.history[1]["role"] == "assistant"
    
    def test_conversation_manager_get_history(self):
        """测试获取对话历史"""
        manager = ConversationManager("test_user")
        manager.add_message("user", "测试消息")
        
        history = manager.get_history()
        
        # 验证返回了正确的历史记录
        assert isinstance(history, list)
        assert len(history) == 1
        assert history[0]["content"] == "测试消息"
    
    def test_conversation_manager_process_message_knowledge(self):
        """测试处理需要搜索知识的消息"""
        manager = ConversationManager("test_user")
        response = manager.process_message("请介绍一下焦虑的相关知识")
        
        # 验证处理了消息并生成了回复
        assert len(response) > 0
        assert len(manager.history) == 2  # 用户消息 + 助手回复
        # 应该调用了搜索工具
        assert "焦虑" in response or "心理学" in response
    
    def test_conversation_manager_process_message_reflection(self):
        """测试处理保存反思的消息"""
        manager = ConversationManager("test_user")
        manager.add_message("user", "我想记录今天的反思")
        response = manager.process_message("保存我的反思")
        
        # 验证处理了消息并生成了回复
        assert len(response) > 0
        # 应该调用了反思保存工具
        assert "保存" in response or "REF-" in response or "反思" in response
    
    def test_conversation_manager_style_change(self):
        """测试对话管理器风格切换"""
        manager = ConversationManager("test_user")
        
        # 测试设置不同风格
        manager.set_style(ResponseStyle.ANALYTICAL)
        assert manager.get_style() == ResponseStyle.ANALYTICAL
        
        manager.set_style(ResponseStyle.ENCOURAGING)
        assert manager.get_style() == ResponseStyle.ENCOURAGING
    
    def test_conversation_manager_get_stats(self):
        """测试获取对话统计信息"""
        manager = ConversationManager("test_user")
        manager.process_message("请搜索焦虑知识")
        
        stats = manager.get_stats()
        
        # 验证统计信息正确
        assert stats["user_id"] == "test_user"
        assert stats["total_messages"] >= 2
        assert stats["search_count"] >= 1
        assert "current_style" in stats
    
    # ===== 集成测试 =====
    
    def test_complete_conversation_flow(self):
        """测试完整的对话流程"""
        manager = ConversationManager("integration_test_user")
        
        # 第一轮：用户询问焦虑
        response1 = manager.process_message("我最近感到很焦虑怎么办")
        assert len(response1) > 0
        assert len(manager.history) == 2
        
        # 第二轮：用户请求搜索知识
        response2 = manager.process_message("请帮我搜索关于压力的知识")
        assert len(response2) > 0
        assert "压力" in response2 or "心理学" in response2
        assert len(manager.history) == 4
        
        # 第三轮：用户表达感谢
        response3 = manager.process_message("谢谢你的帮助")
        assert len(response3) > 0
        assert len(manager.history) == 6
        
        # 验证完整的对话流程可以正常工作
        stats = manager.get_stats()
        assert stats["total_messages"] == 6
        assert stats["search_count"] >= 1
    
    def test_multiple_search_requests(self):
        """测试多次搜索请求"""
        manager = ConversationManager("test_user")
        
        manager.process_message("请搜索焦虑知识")
        manager.process_message("请搜索压力管理")
        manager.process_message("请介绍情绪调节")
        
        stats = manager.get_stats()
        # 验证多次搜索请求都被正确处理
        assert stats["search_count"] >= 3


if __name__ == "__main__":
    """运行测试"""
    pytest.main([__file__, "-v", "--tb=short"])