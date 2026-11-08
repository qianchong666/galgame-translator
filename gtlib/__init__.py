# -*- coding: utf-8 -*-
"""gtlib - Galgame/日文小说 翻译流水线（V4 重构版）

模块划分：
    core       基础设施：日志 / 编码探测 / 原子写 / JSONL / 配置
    clean      清洗（分段、去噪、装饰行归一）
    segment    切块（段落优先 + 场景边界 + 上文锚点 + 占位符保护）
    placeholder 占位符（生成 / 遮蔽 / 容错还原 / 完整性校验）
    terms      专名提取与过滤、术语表批量翻译
    validate   译文质检门（结构性 + 语义性），三态结论 ok/degraded/failed
    llama      llama-server 进程管理 + HTTP 客户端（流式 / 看门狗 / 重试）
    job        单文件与多文件作业编排（断点续传、并发、组装、报告）
    cli        命令行入口
"""

__version__ = "4.0.0"
