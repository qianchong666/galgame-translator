# gtlib · 日文 Galgame / 轻小说 批量翻译工具

面向**长篇日文叙事文本**（Galgame 剧本、轻小说、视觉小说、同人文本）的本地翻译流水线：
双本地大模型分工 + 十一道确定性质检门 + 块级断点续传，一本十万字的书在单机双卡上约 17 分钟出稿。

- **专名全书一致**：人名/地名先遮蔽成 `【#N#】`，译后还原，同一个人不会被翻成三种叫法
- **称呼统一**：日语的「さん」由模型判一次性别，之后统一成「小姐 / 君」，不再随机变成女士/同学
- **出错自动重译**：每块过 11 道质检门，不合格的块带具体问题反馈回译者重翻；行数、占位符、复读这类机械错误由代码确定性修复，不交给模型
- **可中断可续跑**：状态落在 `trans/NNNNNN.json`，`Ctrl+C` 后原命令重跑即可，只补没过的块
- **严格模式**：有块失败就不出成品（不把缺块混进正文），失败明细写进 `*_FAILED.json`

## 流水线

```
清洗 → 专名提取/定名 → 遮蔽 → 切块 → 翻译(质检门+重试) → 批量复核 → 打回重译 → 还原组装 → 收尾
                                             │
                                   11 道门：空输出 / 未翻译 / 行数严格 1:1 /
                                   长度比 / 假名残留 / 占位符次数 / 占位符畸形 /
                                   复读提示词 / 复读正文 / 引号锚点 / 拒答
```

两个模型分工：

| 角色 | 模型 | 职责 |
|---|---|---|
| 译者 | [Sakura-GalTransl-14B](https://huggingface.co/SakuraLLM/Sakura-GalTransl-14B-v3.8) GPTQ-Int4 | 只翻译 |
| 判断 | Qwen3.5-9B AWQ（或任意 7B~14B 中文模型） | 只判断：定名、否决假专名、复核、判人物称呼 |

模型**只判断不改写**：复核判 `retry` 就让译者带意见重翻，实测比"让模型逐字改稿"快 5~10 倍、质量更稳。

## 安装与使用

完整步骤（WSL2 + vLLM + 模型下载 + 三个 bat + 冒烟测试）见 **[安装与使用手册.md](安装与使用手册.md)**。

最短路径：

```bash
# 1) 起两个模型服务（vLLM，GPU0 / GPU1）
vllm serve /models/Sakura-GalTransl-14B-v3.8-GPTQ-Int4 \
  --served-model-name sakura-galtransl-14b --max-model-len 16384 --port 8090

vllm serve /models/Qwen35-9B-AWQ \
  --served-model-name qwen35-9b-awq --max-model-len 16384 \
  --enable-auto-tool-choice --tool-call-parser hermes --port 9092

# 2) 一条命令跑完整流程
python tools/run_all.py --src "你的书.txt" --workers 3 --chunk 400

# 3) 成品
#    <书所在目录>/out/<任务时间>_<书名>/合并_*_成品.txt
```

想分步执行、换参数、只重跑某些块，见手册第 10~14 节。

## 主要命令

| 命令 | 作用 |
|---|---|
| `gt.py clean` | 清洗（去广告/装饰行、合并软换行、台词保护） |
| `gt.py terms` | 专名提取与定名（可选 9B 辅助） |
| `gt.py translate` | 翻译主流程（遮蔽→切块→翻译→还原→组装） |
| `gt.py review` | 批量复核（行数确定性预检 + 模型判语义） |
| `gt.py merge` | 汇总多个文件 |
| `gt.py polish` | 收尾：称呼统一 + 特殊字符清理 |
| `gt.py status` | 查看块级进度 |
| `gt.py selftest` | 148 项自检（不需要 GPU） |
| `gt.py golden` / `score` | 黄金集取样与质量评分（改模型前后对比用） |

## 目录

```
gt.py              命令行入口
gtlib/             内核（清洗/切块/占位符/质检/复核/收尾/评分）
tools/run_all.py   五阶段串联
tools/one_click_gui.py  图形界面
bench/             实验脚本（黄金集与评分，本地自建）
samples/           自制日文样本
docs/              开发记录与设计取舍
安装与使用手册.md    装环境 + 日常用法
方案说明.md         设计口径与原理
```

## 环境

- Windows 11 + WSL2（Ubuntu 24.04），或直接 Linux
- 1× 24 GB 显存可跑（先翻译后复核）；2× 24 GB 可并行
- Python 3.12（WSL 内跑 vLLM）；Windows 侧只需 Python 3.10+ 与 `requests`

## 来源与致谢

本工具是**调度与质检层**，本身不含模型权重，站在这些项目之上：

- [SakuraLLM / Sakura-GalTransl](https://github.com/SakuraLLM/Sakura-13B-Galgame) — Galgame 领域日译中专用模型
- [vLLM](https://github.com/vllm-project/vllm) — 高性能推理服务
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — 早期本地推理路径（现为备选）
- [Qwen](https://github.com/QwenLM/Qwen) — 判断/复核侧基座模型
- [Hugging Face Hub](https://huggingface.co) — 模型分发

## 免责声明

仅供**个人学习、研究与技术验证**使用。请自行确认对所处理文本拥有合法使用权，遵守当地法律与作品授权条款；译文版权归原作者与权利方所有，请勿用于商业分发。

## License

MIT（仅覆盖本仓库代码，不含模型权重与第三方文本）
