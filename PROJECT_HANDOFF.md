# InteractionPrism 项目交接单

> **交接日期：2026-11-20**
> **状态：检测+净化+对比基线+Utility 十件套（7/8 列完成，BV 列待修复）基本完成。**
> **本文用途：交给下一个 AI/协作者，拿上就能继续，不必重问历史。**

---

## ⚠️ 最重要警告：config 文件学习率与论文正文不一致

**论文正文反复写 "standard finetuning configurations (2e-4 for LoRA adapters)"，但仓库 config 文件留的是 5e-5——差 4 倍。**

| 文件 | config 写的 | 论文说的 | 应该用 |
|---|---|---|---|
| `outputs/training/configs/finetune_pure.yaml`（FT 基线） | 5e-5 | **2e-4** | 2e-4 |
| `outputs/training/configs/finetune_after_suppression.yaml`（BV 微调） | 5e-5 | **2e-4** | 2e-4 |
| `configs/consistency/*.yaml`（CROW） | 2e-4 | 调整后 2e-4 | ✅ 已正确 |

**影响**：我们用 5e-5 跑的 FT 基线数字偏保守（ASR 偏高）。用 2e-4 重跑后 FT 数字会更好看，但 IP-Ours 的优势地位不变。

**已完成但用了错误学习率的**：FT(paper) 基线 ×10、BV 微调 ×1。需要重跑。

---

## 项目一句话

用 Harsanyi 等效交互理论做 LLM 后门的**检测→解释→净化**全链：交互分解发现触发器 → 反事实验证确认 → 交互引导净化清除后门 → 签名扫荡全量隔离。

---

## 1. 环境

```bash
代码根：/science/gh/Backdoor-Vaccine
Python：conda activate BV（/home/user/miniconda3/envs/BV）
基座：/home/gh/Llama-2-7b-chat-hf
后门 LoRA：backdoor_weight/LLaMA2-7B-Chat/{negsentiment,refusal}/{badnet,sleeper,vpi,mtba,ctba}
GPU：A100 80GB ×6（公用卡，先 nvidia-smi 查空卡，最多用 2 张，不得碰活跃卡）
评测协议：FP32 / 裸指令 / 贪心 / 128tok / eval_asr.py 同款判据（IP_P3 系列）
```

## 2. 核心文件索引

> **2026-11-26 整理后：下表中的旧路径已归档。正式算法唯一入口为 `interactionprism/`（`python -m interactionprism --help`）；旧实现位于 `ip_archive_superseded/20261126_consolidation/`（`legacy_dirs/` 为整目录，`original_callers/` 为调用方，清单见同目录 `SHA256SUMS`）。** 根目录 `IP_E8R1_*` 与两份 `IP_E1*_contexts.json` 仍保留，因为 E21 结构实验直接依赖。下文旧路径仅用于追溯历史结果。

### 检测层（冻结，勿改）
```
IP_E8R1_20261111_e4_primitives.py          # E8 原语（词玩家/Keep-only/前向取隐状态）
ip_e14_lib/IP_E12R1_*variable_coalitions.py  # S1 搜索 + Stage-2 测量
ip_e14_lib/IP_E12R1_*frozen_utils.py        # 读出协议/反事实干预
ip_e14_lib/IP_E13R1V2_*core.py + _experiment.py  # S3 行为规则（固定签名/对照/子集）
IP_E11R1_20261111_contexts.json / IP_E12R1_20261112_contexts.json  # 16 参考任务
```

### 评测器（所有报告数字的来源）
```
ip_p1_purify/IP_P3_eval_paper_protocol.py    # 单模型论文协议评测
ip_p1_purify/IP_P3_batch_eval.py             # 批量评测 runner
```

### 净化（当前版本 v2.1）
```
ip_p1_purify/IP_P2_20261117_purify_attack.py  # 主驱动（--task/--attack/--format/--paper-ft/--blind-ft）
ip_p1_purify/IP_P1_20261116_purify_common.py  # 共用工具
ip_p1_purify/run_v2_native.sh                  # v2.1 批量 runner
```

### 净化后模型位置
```
outputs/interactionprism/p1_purify/p2_{task}_{attack}_alpaca/adapter
```

### 对比基线
```
FT 基线：outputs/interactionprism/baselines/v2ft_{task}_{attack}/（LlamaFactory 训的）
BV 抑制：outputs/purified/suppressed_adapter/（仅 negsentiment/badnet）
BV 微调：outputs/purified/finetuned/（⚠️ 学习率 5e-5 需重跑为 2e-4）
CROW：backdoor_weight/LLaMA2-7B-Chat/consistency/{task}_{attack}/
```

### 论文协议评测结果
```
outputs/interactionprism/paper_eval/*.json    # 所有 ASR/FP 结果
```

## 3. 当前主表（论文协议，ASR%，保留两位小数；2026-11-20 11时 全部实测完成）

### Sentiment Steering

| 攻击 | No-Def | FT(paper,5e-5⚠) | FT(self) | CROW | BV抑制 | **IP-Ours** |
|---|---:|---:|---:|---:|---:|---:|
| BadNets | 68.50 | 45.00 | 80.50 | 30.00 | 21.50 | **0.00** |
| CTBA | 62.50 | 35.50 | 59.50 | 0.50 | — | **0.00** |
| VPI | 7.50 | 3.00 | 2.00 | 0.00 | — | **0.00** |
| Sleeper | 5.50 | 0.50 | 1.00 | 0.50 | — | **0.00** |
| MTBA | 4.00 | ⚠缺权重 | 3.00 | 0.00 | — | **0.50** |

BadNets 行另有：Pruning 60.00、Fine-Pruning 34.50（凭证 `outputs/baselines/logs/asr_paper_protocol.log`，待补 JSON）、BV-full（抑制+微调@5e-5）36.50。

### Target Refusal

| 攻击 | No-Def | FT(paper,5e-5⚠) | FT(self) | CROW | **IP-Ours** |
|---|---:|---:|---:|---:|---:|
| BadNets | 40.50 | 20.50 | 43.50 | 14.00 | **0.00** |
| CTBA | 80.50 | 47.00 | 73.50 | 3.00 | **0.50** |
| VPI | 17.00 | 2.00 | 7.00 | 0.00 | **0.00** |
| Sleeper | 5.50 | 0.00 | 2.00 | 0.00 | **0.00** |
| MTBA | 5.50 | 0.50 | 3.00 | 0.00 | **0.50** |

**三个口径警示**：
1. 全部 FT(paper) 列用 lr 5e-5（官方 repo README 自称的统一口径）；论文正文写 2e-4 → P0 重跑后此列作废重测，两种口径都报告。
2. `v2ft_negsentiment_mtba/` 只有 config 无权重，该格需先补训练。
3. refusal 全表判据为我们自定 4 短语；上游 BackdoorLLM/CROW 判据为 9 宽词（sorry / as an AI 等均算命中）。已决定对齐上游判据；IP_P3 未存生成全文，无法离线改判，需重跑（单卡 2-3h）。negsentiment 判据经逐字比对与上游完全一致。

## 4. 任务状态（2026-11-20 11时：GPU1-4 已全空）

- ✅ Negsentiment 全列评测完成（`PAPER_EVAL_negsentiment_DONE`；唯 `mtba_ftpaper` 因权重缺失跳过）
- ✅ CROW refusal ×5 完成：badnet 14.00 / ctba 3.00 / vpi 0.00 / sleeper 0.00 / mtba 0.00
- ✅ BV utility 十件套重跑完成，**全零问题已修复**（Avg 59.59，全场最高）
- 待办缺口（按优先级）：① P0 lr=2e-4 重跑 FT×10 + BV 微调 ② mtba FT(paper) 补训练 ③ refusal 9 词判据重跑（建议同时给 IP_P3 加"落全文"）④ P0.5 自建 n≥1000 测试集（兼作 P4 独立测试集）

## 5. 待办事项（按优先级）

### P0：重跑 FT 基线（学习率修正）
```bash
# 修改 finetune_pure.yaml 中 learning_rate: 0.00005 → 0.0002
# 重跑 10 个攻击的 FT
bash ip_p1_purify/run_paper_ft_v2.sh <GPU> <task>  # 修改 config 后自动重跑
```

### P1：Utility 十件套（Table 3 复刻）——已完成 7/8 列

**协议**：lm-eval 0.4.13，zero-shot，全量测试集，MCQ 用 `acc_norm`、二分类用 `acc`、GSM8k 用 `strict-match`。与论文 Clean 列偏差 +0.2pt（协议校准通过）。

**数据集**：全部 10 个已就位（`~/.cache/huggingface/datasets/`），离线可用。需设 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`。

**当前结果（BadNets × Sentiment，LLaMA2-7B-Chat）**：

| | Clean | Attacked | FT | CROW | Pruning | Fine-Pruning | BV | **Ours** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OBQA | 43.60 | 42.00 | 42.00 | 41.00 | 38.80 | 41.80 | 42.20 | 41.40 |
| RTE | 72.56 | 68.95 | 70.76 | 71.84 | 69.68 | 70.76 | 70.04 | 72.20 |
| HellaSwag | 75.96 | 71.21 | 74.45 | 72.39 | 70.10 | 72.80 | 74.58 | 69.81 |
| WinoGrande | 68.43 | 65.59 | 67.88 | 67.40 | 63.14 | 66.93 | 68.19 | 65.11 |
| ARC-C | 43.52 | 38.57 | 44.03 | 40.70 | 38.57 | 44.71 | 44.54 | 35.67 |
| ARC-E | 72.77 | 68.56 | 74.49 | 72.14 | 67.63 | 73.32 | 74.58 | 66.71 |
| BoolQ | 80.55 | 77.43 | 78.41 | 81.74 | 77.16 | 77.46 | 78.72 | 77.37 |
| Piqa | 76.33 | 73.94 | 77.64 | 77.58 | 74.10 | 76.93 | 77.91 | 73.12 |
| GSM8k | 21.53 | 5.53 | 18.88 | 4.62 | 3.03 | 9.10 | 18.04 | 8.34 |
| MMLU | 46.52 | 46.79 | 47.16 | 42.05 | 43.18 | 43.59 | 47.08 | 46.56 |
| **Avg** | **60.18** | **55.86** | **59.57** | **57.15** | **54.54** | **57.74** | **59.59** | **55.63** |

✅ **BV utility 全零问题已修复**（2026-11-20 重跑）：BV-full 十件套全部正常，Avg 59.59 为全场最高（几乎追平 Clean），但其 ASR=36.5%（@5e-5）——构成"utility 最好却压不住后门"的对照。注意 ARC-E/Piqa 用 `acc`（与论文 Clean 73.90 对齐；acc_norm 分别为 71.55/77.91）。

**关键发现**：
- **GSM8k 陷阱**：v1 净化用 48-tok 锚定 → 模型学会"短回答就发 EOS" → GSM8k=0%。v2fix 改 256-tok 后恢复到 8.34%（flexible 16.30%）。FT 基线用 ground truth 标签得 18.88%。**自蒸馏锚定仍比 ground truth 短，是 utility 差距主因**。
- **口径对齐**：HellaSwag/OBQA/ARC 必须用 `acc_norm`（不是 `acc`），否则与论文差 10-20pt。
- **数据集版本兼容**：缓存由 datasets 4.x 生成时，MMLU 的 57 个 config 的 `List` 特征类型在 datasets 3.x 下会报错。解决：升级到 datasets 4.x，或 patch JSON `"List"→"Sequence"`。

**评测命令**：
```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
lm_eval --model hf \
  --model_args pretrained=<base_or_model_dir>[,peft=<adapter_dir>],dtype=float16 \
  --tasks boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,piqa,gsm8k,mmlu \
  --batch_size auto --seed 0 --output_path <out_dir>
```

**结果文件位置**：
```
outputs/interactionprism/u1_utility/{clean,attacked,ft_paper,crow,wanda,fine_pruning,ours_v2fix,bv_full}/
```

### P2：Wanda Pruning + Fine-Pruning——已完成
- Wanda 35% 剪枝：`outputs/baselines/wanda_pruned/`（完整模型 13GB，C4→干净指令校准替代）
- Fine-Pruning：`outputs/baselines/fine_pruned_adapter/`（Wanda 上叠 LoRA FT）
- ASR 已测：Pruning 60.00%, Fine-Pruning 34.50%（论文协议）
- Utility 已测：Pruning 54.54, Fine-Pruning 57.74
- ⚠️ Wanda 校准数据用了干净指令（非论文的 C4），已在 `wanda/lib/data.py` 的 `get_clean()` 实现

### P3：Quantization 基线
- 仓库无脚本，需实现（INT8 via bitsandbytes + ASR 评测）
- 论文用 4-bit precision

### P4：新独立测试集
- 检测的 V1/V2 分数规则和净化路线转正都需要
- 构成：新指令语料 + 触发变体 + 可能的新攻击 LoRA

## 6. 关键设计决策（已锁定，不要走回头路）

| 决策 | 状态 | 原因 |
|---|---|---|
| 净化用 alpaca 原生格式 | ✅ v2.1 | 用户指令：`\nAnswer:` 仅用于检测 |
| arm-A 尾巴仅限检测层 | ✅ | 用户强调：不准在净化/评测/报告用 |
| FT(self) 列删除 | ✅ | 用户指令：只保留论文原版配置的 FT |
| 评测统一论文协议 | ✅ IP_P3 | FP32/裸指令/贪心/128tok |
| 不换 adapter 不重跑 | ✅ | 同一 adapter 上比较方法才公平 |
| 小模型/Qwen 解释器 | ✅ 禁用 | 用户指令 |
| E8+E12 投票 | ✅ 禁用 | E12R3 是唯一正式判定 |
| 评测保留两位小数 | ✅ | 200 条测试集精度为 0.50% |

## 7. 论文 vs 代码差异清单（已发现）

| 差异 | 影响 | 处理 |
|---|---|---|
| FT lr：论文 2e-4 vs config 5e-5 | FT 基线数字偏保守 | P0 重跑 |
| BV 微调 lr：同上 | BV 方法效果被低估 | P0 重跑 |
| CROW lr：原始 1e-3 → 论文调至 2e-4 | 已在 config 中修正 | ✅ |
| 测试集 200 条 vs 论文数字精度 0.01% | **已查明（见 §7.1）：论文数字无法由官方 200 条协议产生** | 用仓库协议 + P0.5 自建 n≥1000 |
| adapter 可能是不同次训练的快照 | No-Def 数字有 ~9 点差异 | 接受（相对比较有效） |
| transformers 版本差异 | ~3-5 点浮动 | 接受 |
| refusal 判据：上游 9 宽词 vs 我们 4 短语 | 我们 refusal 防御数字偏乐观 | 对齐 9 词重跑（待排） |
| BV 官方 README 自称统一 5e-5，与论文 2e-4 冲突 | P0 不是"修笔误"而是补第二口径 | 两种口径都报告 |

### 7.1 测试集规模考证（2026-11-20，源码证据链）

- 协议继承链：BackdoorLLM（NeurIPS'25，`bboylyg/BackdoorLLM`，README 明文 "500 training / **200 test**"）→ CROW（ICML'25，`NayMyatMin/CROW`，`attack/DPA/` 整目录继承，eval 脚本全量单次 + `round(.,2)`）→ BV（ICLR'26，`JEKimLab/Backdoor-Vaccine`，仅带 negsentiment 的 backdoor200）。我们的 200 条 = 官方协议忠实复现，refusal 测试集为我们自建。
- **已确认的范围**：固定 200 条、所有输出均计入分母、单次二元判定的 ASR 粒度为 0.5pp。上游部分脚本会跳过空输出，使有效分母改变；因此不能仅根据小数位推出测试集规模或重复次数。
- **尚未确认**：论文各项数字具体用了多少样本、是否多次平均、如何处理空输出。目前没有作者运行记录或聚合脚本足以确定。此前关于“必然 50 次平均”“分母必须≥2000”“跨论文复制”“混合汇总”的断言撤回；数值重合和粒度兼容性均不能证明这些机制。
- ASR 判据逐字比对：negsentiment 关键词与 CROW **完全一致**（BackdoorLLM 原版多一个 "Negative"，CROW 已删）；生成配置逐位一致（fp32/贪心/top_p=0.75 陪衬/128tok/裸指令）；唯一小分歧：上游空输出跳过分母、我们计 0。

## 8. 协议约定

- **评测**：一律用 IP_P3（论文协议），禁止在报告表格中使用 arm-A 数字
- **表格**：保留两位小数
- **GPU**：先查 nvidia-smi，只用空卡，最多 2 张，不碰活跃卡
- **新实验**：新目录，不覆盖既有结果
- **主表**：只用我们实测数字，† 标注引用值

## 9. Utility 评测专节（接手者必读）

### 协议要点（与论文逐项对齐）

| 要素 | 论文做法 | 我们做法 | 一致？ |
|---|---|---|---|
| 工具 | lm-eval harness | lm-eval 0.4.13 | ✅（版本差 <1pt） |
| shot | zero-shot（代码 `num_shot=0`） | zero-shot | ✅ |
| 数据 | 全量测试集 | 全量 | ✅ |
| MCQ 口径 | acc_norm | acc_norm | ✅（校准确认） |
| 二分类口径 | acc | acc | ✅ |
| GSM8k 口径 | strict-match | strict-match | ✅ |
| Clean 校准 | Avg 59.97 | Avg 60.18 | ✅ 偏差+0.2pt |
| Attacked 校准 | Avg 55.62 | Avg 55.86 | ✅ 偏差+0.2pt |

### ⚠️ 三个踩过的坑

1. **GSM8k 的 48-token 截断**：v1 净化锚定生成 `max_new_tokens=48` → 所有训练目标 ≤48 tok → 模型学会提前发 EOS → GSM8k=0%。v2fix 改 256 后恢复。**永远不要用短 token 限制生成锚定目标**。

2. **acc vs acc_norm**：HellaSwag 用 `acc` 得 58%，用 `acc_norm` 得 76%（差 18pt）。OBQA/ARC 同理。论文 Table 3 一定用的是 `acc_norm`（Clean 校准逐基准确认）。**MCQ 任务必须用 `acc_norm`**。

3. **datasets 版本兼容**：用户手动下载的缓存由 datasets 4.x 生成，MMLU 的 `List` 特征类型在 3.x 报 `Feature type 'List' not found`。解决方案：升级到 datasets 4.8.5（`pip install datasets==4.*`），或 patch arrow 元数据。

### BV utility 全零问题（✅已修复，2026-11-20）

重跑 `lm_eval ... peft=outputs/baselines/bv_full` 即恢复正常（十件套 Avg 59.59）。此前全零为一次性评测故障，非模型问题。BV-full（抑制+微调@5e-5）的 ASR=36.5%（`outputs/baselines/logs/bv_asr.log`）——注意此前文档写的"BV ASR=3.5%"是仅抑制阶段的旧数，已过时。

### Utility 评测脚本

```bash
# 单模型十件套（~2h GPU）
bash ip_baselines/run_utility.sh <GPU> <adapter_path> <label>

# 完整模型（无 adapter，如 Wanda 剪枝产物）
lm_eval --model hf --model_args pretrained=<model_dir>,dtype=float16 \
  --tasks boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,piqa,gsm8k,mmlu \
  --batch_size auto --seed 0 --output_path <out>
```

### 下一步 utility 工作

1. **修复 BV utility 全零**（排查 adapter 加载或 forward 问题）
2. **v2fix utility 已完成**：Avg=55.63，GSM8k=8.34（从 v1 的 0 恢复）
3. **改进方向**：锚定集改用 ground truth 标签（`data/finetune_clean.json`，200 条带标准答案），预期 Avg 从 55.63 拉回 ~58-59

## 9.5 论文写作状态（2026-11-20）

- 引言草稿 v1 + 评审：`ip_paper/IP_PAPER_INTRO_DRAFT_v1_REVIEW_20261120.md`（草稿原文在其 §一，评审在 §二）
- 定位：从"防御方法论文"转为"有效触发结构分析范式"（名义结构 / 表征交互 / 有效行为条件三层区分）。数据依据：`ip_e21_structural/IP_E21_20261120_RESULTS.md`、`IP_E14R2_20261113_ASR_RESULTS.md`。
- 写作前必须先定：防御主线（净化 ASR、签名扫荡）是降为应用小节还是留给第二篇；评审建议保留"结构识别的防御回报"一段。
- 实验承诺清单（引言已立的债）：子集级行为激活阶梯、跨协议配对、跨位置配对、表征-行为一致性比较、中性词对照、refusal 任务、第二模型、多 seed。
- 注意：refusal 上的行为判定须用上游 9 词判据（见 §3 警示 3），否则与论文口径不可比。

## 10. 详细历史文档

```
IP_E14R2_20261113_ASR_RESULTS.md    # §1-§23 全部实验结果与结论
ip_p1_purify/VERSIONS.md             # 净化版本管理
ip_archive_superseded/               # 已归档的旧版代码与产物
```

---

**交接终点：下一个最有价值的操作是 P0（用 lr=2e-4 重跑 FT 基线），一行改动 + 1 小时 GPU。**
