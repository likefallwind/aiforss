# 项目：人机协作对话助手

## 你将学到什么

- 理解 Function Calling 的机制并实现工具调用
- 掌握多轮对话状态管理的方法
- 设计人格化对话系统，实现基本的共情计算
- 在人机协作中选择合适的分工模式

## 环境准备

- **运行环境**：Python 3.10+
- **安装依赖**：`pip install openai pytest`
- **API Key**：需要设置 `OPENAI_API_KEY` 环境变量（或在代码中替换为模拟调用）

## 背景介绍

在社会科学研究中，人机协作可以显著提升效率。本项目模拟一个「心理支持对话助手」场景：你将扮演研究者，设计一个能调用外部知识库、记录用户反思，并根据对话情境调整回应风格的人格化 AI 系统。

## 任务分解

**Task 1：实现工具调用（search_resources）**
- 目标：实现 `search_resources(query: str, category: str) -> str` 函数，模拟搜索心理学知识库
- 提示：返回模拟的搜索结果，包含标题、摘要和关键词
- 对应文件：`src/main.py` 中的 `# TODO: Task 1 - 实现搜索工具`

**Task 2：实现反思记录工具（save_reflection）**
- 目标：实现 `save_reflection(user_id: str, content: str) -> str` 函数，记录用户对话后的反思
- 提示：返回保存成功的确认信息和记录ID
- 对应文件：`src/main.py` 中的 `# TODO: Task 2 - 实现反思记录工具`

**Task 3：设计人格化响应生成器**
- 目标：基于 `ResponseStyle` 枚举和对话上下文，生成共情风格的回复
- 提示：使用 `generate_response(style: ResponseStyle, message: str) -> str` 函数，根据风格参数调整回复内容
- 对应文件：`src/main.py` 中的 `# TODO: Task 3 - 实现人格化响应生成器`

**Task 4：实现多轮对话管理**
- 目标：使用 `ConversationManager` 类管理对话状态，实现工具调用的决策逻辑
- 提示：在 `process_message()` 中根据用户意图判断是否需要调用工具，并整合工具返回结果
- 对应文件：`src/main.py` 中的 `# TODO: Task 4 - 实现多轮对话管理`

**Task 5：完整对话流程测试**
- 目标：运行主程序，体验完整的人机协作对话流程
- 提示：测试用户询问心理知识、AI调用工具获取信息、AI生成共情回复的完整过程
- 对应文件：`src/main.py` 中的 `if __name__ == "__main__"` 部分

## 项目结构

- **src/main.py**：学生需要补全 TODO 区域的脚手架文件，包含工具函数、对话管理器和主程序
- **answer/main.py**：完整的参考实现，展示了如何正确实现所有功能
- **tests/test_answer.py**：自动生成，用于验证 answer 目录的正确性

## 验收标准

- [ ] `search_resources()` 能根据关键词返回模拟的心理学知识
- [ ] `save_reflection()` 能记录并返回反思保存成功的确认
- [ ] `generate_response()` 能根据 ResponseStyle 生成不同风格的回复
- [ ] `ConversationManager` 能维护对话历史并正确调用工具函数
- [ ] 主程序能完整运行，至少包含 3 轮人机对话交互
- [ ] 代码中包含清晰的 Task 标记和 TODO 注释

## 挑战任务（选做）

1. **增强共情计算**：在 `generate_response()` 中加入情感识别逻辑，根据用户消息中的情感词汇自动选择合适的 ResponseStyle
2. **实现人机协作模式选择**：设计一个协作模式选择器，根据任务类型自动选择 AI 主导型、人机交替型或人机融合型模式

### 项目结构
- `src/main.py`（脚手架）：人机协作对话助手的脚手架文件，学生需要补全工具调用、共情响应和多轮对话管理的核心逻辑
- `answer/main.py`（参考答案）：完整参考实现，包含所有 TODO 区域的解决方案
- `tests/test_answer.py`（测试文件）：验证 answer 目录中核心功能的正确性