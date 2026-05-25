"""
人机协作对话助手 - 参考实现

本文件是 answer 参考实现，包含所有 TODO 区域的完整解决方案。
学生可以参考本文件完成 src/main.py 中的 TODO 区域。

作者：AI 课程项目
"""

import enum
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any


class ResponseStyle(enum.Enum):
    """对话响应风格枚举"""
    # 共情理解风格
    EMPATHETIC = "empathetic"
    # 分析建议风格
    ANALYTICAL = "analytical"
    # 中立客观风格
    NEUTRAL = "neutral"
    # 鼓励支持风格
    ENCOURAGING = "encouraging"


def search_resources(query: str, category: str = "psychology") -> str:
    """
    模拟搜索心理学知识库
    
    Args:
        query: 搜索关键词
        category: 搜索类别，默认为心理学
    
    Returns:
        str: 模拟的搜索结果，包含标题、摘要和关键词
    """
    # 模拟心理学知识库内容
    knowledge_base = {
        "焦虑": {
            "title": "焦虑情绪的心理机制与应对策略",
            "abstract": "焦虑是人类面对压力时的正常情绪反应。本文介绍了焦虑的心理机制，包括认知评估、情绪调节和生理反应三个维度，并提供了认知行为疗法、正念冥想等有效的应对策略。",
            "keywords": ["焦虑", "认知行为疗法", "正念冥想", "情绪调节"]
        },
        "压力": {
            "title": "工作压力的心理学分析与管理方法",
            "abstract": "工作压力已成为现代职场普遍面临的问题。本文从工作压力的来源、影响机制和干预策略三个方面进行分析，探讨了个人层面的压力管理技巧和组织层面的支持措施。",
            "keywords": ["工作压力", "压力管理", "职业倦怠", "工作满意度"]
        },
        "情绪": {
            "title": "情绪调节的心理机制与训练方法",
            "abstract": "情绪调节是个体管理和改变自身情绪的能力。本文介绍了情绪调节的策略模型，包括认知重评和表达抑制两种主要策略，并提供了具体的训练方法和实践建议。",
            "keywords": ["情绪调节", "认知重评", "情感智力", "心理健康"]
        },
        "人际关系": {
            "title": "人际交往中的沟通技巧与心理策略",
            "abstract": "良好的人际关系对心理健康至关重要。本文从沟通技巧、冲突解决和亲密关系维护三个角度，探讨了建立和维护健康人际关系的方法。",
            "keywords": ["人际交往", "沟通技巧", "冲突解决", "亲密关系"]
        },
        "自我认知": {
            "title": "自我认知与个人成长心理学",
            "abstract": "自我认知是个人成长和心理发展的基础。本文介绍了自我认知的概念框架，探讨了自我意识、自我效能感和自我接纳等核心主题，并提供了提升自我认知的实践方法。",
            "keywords": ["自我认知", "自我意识", "自我效能感", "个人成长"]
        },
        "default": {
            "title": "心理学基础知识概览",
            "abstract": "心理学是研究人类心理现象和行为的科学。本文概述了心理学的主要研究领域，包括发展心理学、社会心理学、临床心理学等，并介绍了心理学研究的基本方法和伦理原则。",
            "keywords": ["心理学", "研究方法", "心理现象", "行为科学"]
        }
    }
    
    # 根据查询关键词匹配相关资源
    query_lower = query.lower()
    matched_resource = None
    
    for key, resource in knowledge_base.items():
        if key in query_lower or any(keyword in query_lower for keyword in resource["keywords"]):
            matched_resource = resource
            break
    
    if not matched_resource:
        matched_resource = knowledge_base["default"]
    
    # 格式化返回搜索结果
    result = f"""
【搜索结果】
类别: {category}
关键词: {query}

--- 资源信息 ---
标题: {matched_resource['title']}

摘要:
{matched_resource['abstract']}

关键词标签: {' | '.join(matched_resource['keywords'])}
---
    """
    return result.strip()


def save_reflection(user_id: str, content: str) -> str:
    """
    记录用户对话后的反思
    
    Args:
        user_id: 用户ID
        content: 反思内容
    
    Returns:
        str: 保存成功的确认信息和记录ID
    """
    # 生成唯一的记录ID
    record_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 模拟保存到数据库（实际项目中应保存到持久化存储）
    confirmation = f"""
【反思记录已保存】

记录ID: REF-{record_id}
用户ID: {user_id}
记录时间: {timestamp}

反思内容摘要:
{content[:100]}{'...' if len(content) > 100 else ''}

✓ 反思记录已成功保存！
    """
    return confirmation.strip()


def generate_response(style: str, message: str) -> str:
    """
    基于 ResponseStyle 生成共情风格的回复
    
    Args:
        style: 响应风格（使用 ResponseStyle 枚举值）
        message: 用户输入的消息
    
    Returns:
        str: 根据风格参数调整后的回复内容
    """
    # 基础响应模板
    responses = {
        ResponseStyle.EMPATHETIC: {
            "greeting": "我理解你的感受，这对你来说一定不容易。",
            "question": "你想更多地谈论一下这个话题吗？",
            "concern": "听起来这件事让你感到有些困扰。",
            "closing": "我会一直在这里倾听和支持你。"
        },
        ResponseStyle.ANALYTICAL: {
            "greeting": "让我帮你分析一下这个问题。",
            "question": "你能否提供更多具体信息？",
            "concern": "从心理学的角度来看，这可能涉及到几个因素。",
            "closing": "希望这个分析对你有帮助。"
        },
        ResponseStyle.NEUTRAL: {
            "greeting": "好的，我听到了你的描述。",
            "question": "还有其他你想分享的吗？",
            "concern": "你提到的这些情况值得进一步探讨。",
            "closing": "我会继续提供客观的信息支持。"
        },
        ResponseStyle.ENCOURAGING: {
            "greeting": "你做得很好，愿意分享你的想法！",
            "question": "你能想到什么积极的方面吗？",
            "concern": "这是一个成长和学习的机会。",
            "closing": "相信你能克服这个挑战！"
        }
    }
    
    # 将 style 转换为 ResponseStyle 枚举
    try:
        style_enum = ResponseStyle(style) if isinstance(style, str) else style
    except (ValueError, AttributeError):
        style_enum = ResponseStyle.EMPATHETIC
    
    # 根据消息内容和风格生成响应
    response_content = responses.get(style_enum, responses[ResponseStyle.EMPATHETIC])
    
    # 根据消息关键词选择合适的响应模板
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["你好", "hi", "hello", "开始"]):
        base_response = response_content["greeting"]
    elif any(word in message_lower for word in ["问题", "怎么办", "如何", "为什么"]):
        base_response = response_content["question"]
    elif any(word in message_lower for word in ["困难", "难过", "焦虑", "压力", "烦恼"]):
        base_response = response_content["concern"]
    elif any(word in message_lower for word in ["谢谢", "再见", "结束"]):
        base_response = response_content["closing"]
    else:
        base_response = response_content["greeting"]
    
    # 添加风格特定的补充信息
    style_additions = {
        ResponseStyle.EMPATHETIC: "我很高兴你愿意分享这些。作为你的对话助手，我会认真倾听并尽力理解你的感受。",
        ResponseStyle.ANALYTICAL: "基于你提供的信息，我可以为你提供一些心理学视角的分析和建议。",
        ResponseStyle.NEUTRAL: "我会保持客观和专业，提供给你有用的信息和支持。",
        ResponseStyle.ENCOURAGING: "你有勇气面对这些问题，这本身就是一种成长。继续保持这种积极的态度！"
    }
    
    final_response = f"{base_response}\n\n{style_additions.get(style_enum, '')}"
    return final_response.strip()


class ConversationManager:
    """
    多轮对话管理器
    
    负责管理对话状态、维护对话历史、决策工具调用
    """
    
    def __init__(self, user_id: str):
        """
        初始化对话管理器
        
        Args:
            user_id: 用户ID
        """
        self.user_id = user_id
        self.history: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.current_style = ResponseStyle.EMPATHETIC
        self.search_count = 0
        self.reflection_saved = False
    
    def add_message(self, role: str, content: str) -> None:
        """
        添加消息到对话历史
        
        Args:
            role: 消息角色（"user" 或 "assistant"）
            content: 消息内容
        """
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        获取对话历史
        
        Returns:
            List[Dict[str, Any]]: 对话历史列表
        """
        return self.history
    
    def process_message(self, message: str) -> str:
        """
        处理用户消息，实现工具调用的决策逻辑
        
        Args:
            message: 用户输入的消息
        
        Returns:
            str: AI 生成的回复
        """
        # 添加用户消息到历史
        self.add_message("user", message)
        
        # 分析用户意图
        message_lower = message.lower()
        response = ""
        
        # 判断用户意图并决定是否调用工具
        if any(word in message_lower for word in ["搜索", "查找", "知识", "是什么", "介绍一下"]):
            # 意图：查询心理知识，调用搜索工具
            self.search_count += 1
            
            # 提取查询关键词
            keywords = []
            knowledge_topics = ["焦虑", "压力", "情绪", "人际关系", "自我认知", "心理"]
            for topic in knowledge_topics:
                if topic in message:
                    keywords.append(topic)
            
            query = keywords[0] if keywords else "心理学"
            
            # 调用搜索工具
            search_result = search_resources(query)
            
            # 生成包含搜索结果的回复
            self.current_style = ResponseStyle.ANALYTICAL
            response = f"我帮你搜索了一些相关的心理学知识：\n\n{search_result}\n\n"
            response += generate_response(self.current_style.value, message)
            
        elif any(word in message_lower for word in ["反思", "记录", "保存"]):
            # 意图：保存反思记录
            reflection_content = self._generate_reflection_summary()
            save_result = save_reflection(self.user_id, reflection_content)
            
            response = f"{save_result}\n\n"
            response += generate_response(ResponseStyle.ENCOURAGING.value, message)
            self.reflection_saved = True
            
        elif any(word in message_lower for word in ["难过", "焦虑", "担心", "害怕", "困扰"]):
            # 用户表达负面情绪，切换到共情风格
            self.current_style = ResponseStyle.EMPATHETIC
            response = generate_response(self.current_style.value, message)
            
        elif any(word in message_lower for word in ["谢谢", "好的", "明白了", "了解了"]):
            # 用户表示理解，切换到鼓励风格
            self.current_style = ResponseStyle.ENCOURAGING
            response = generate_response(self.current_style.value, message)
            
        else:
            # 默认使用共情风格
            self.current_style = ResponseStyle.EMPATHETIC
            response = generate_response(self.current_style.value, message)
        
        # 添加助手回复到历史
        self.add_message("assistant", response)
        
        return response
    
    def _generate_reflection_summary(self) -> str:
        """
        生成对话反思摘要
        
        Returns:
            str: 对话内容摘要
        """
        if not self.history:
            return "本次对话暂无内容记录。"
        
        summary_parts = []
        for msg in self.history:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            summary_parts.append(f"{role}: {content}")
        
        return " | ".join(summary_parts)
    
    def set_style(self, style: ResponseStyle) -> None:
        """
        设置当前响应风格
        
        Args:
            style: 响应风格枚举
        """
        self.current_style = style
    
    def get_style(self) -> ResponseStyle:
        """
        获取当前响应风格
        
        Returns:
            ResponseStyle: 当前响应风格
        """
        return self.current_style
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取对话统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        return {
            "user_id": self.user_id,
            "total_messages": len(self.history),
            "search_count": self.search_count,
            "reflection_saved": self.reflection_saved,
            "current_style": self.current_style.value
        }


def run_conversation_demo():
    """
    运行完整的人机协作对话流程演示
    
    本函数演示完整的人机协作对话流程：
    1. 用户询问心理知识
    2. AI 调用工具获取信息
    3. AI 生成共情回复
    4. 至少包含 3 轮对话交互
    """
    print("=" * 60)
    print("人机协作对话助手 - 心理支持对话场景")
    print("=" * 60)
    
    # 初始化对话管理器
    user_id = "user_001"
    manager = ConversationManager(user_id)
    
    print(f"\n[系统] 对话助手已启动，用户ID: {user_id}")
    print("[系统] 输入 '退出' 可结束对话\n")
    
    # 预定义的对话流程演示（至少3轮）
    demo_messages = [
        "你好，我最近感到有些焦虑，想了解一下有什么方法可以帮助缓解。",
        "谢谢，能再帮我搜索一下关于压力管理的知识吗？",
        "好的，我现在感觉好多了，感谢你的帮助。"
    ]
    
    # 第一轮对话：用户询问焦虑问题
    print("-" * 60)
    print("【第1轮对话】")
    message1 = demo_messages[0]
    print(f"用户: {message1}")
    response1 = manager.process_message(message1)
    print(f"\n助手: {response1}")
    
    # 第二轮对话：用户请求搜索压力管理知识
    print("\n" + "-" * 60)
    print("【第2轮对话】")
    message2 = demo_messages[1]
    print(f"用户: {message2}")
    response2 = manager.process_message(message2)
    print(f"\n助手: {response2}")
    
    # 第三轮对话：用户表达感谢
    print("\n" + "-" * 60)
    print("【第3轮对话】")
    message3 = demo_messages[2]
    print(f"用户: {message3}")
    response3 = manager.process_message(message3)
    print(f"\n助手: {response3}")
    
    # 打印对话统计信息
    print("\n" + "=" * 60)
    print("【对话统计】")
    stats = manager.get_stats()
    print(f"总消息数: {stats['total_messages']}")
    print(f"工具调用次数: {stats['search_count']}")
    print(f"反思记录保存: {'是' if stats['reflection_saved'] else '否'}")
    print(f"当前风格: {stats['current_style']}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    """
    主程序入口
    
    运行本程序将启动人机协作对话助手
    """
    # 完整的对话流程演示
    run_conversation_demo()