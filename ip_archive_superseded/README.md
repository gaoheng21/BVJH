# 已归档的旧版本实验代码（IP_E2 ~ E15P1）

归档日期：2026-11-16。这些是迭代过程中的历史版本，被最终四阶段流水线取代，仅作复现凭证保留。

**保留在仓库根目录的现行代码**（勿动）：

```text
IP_E8R1_20261111_e4_primitives.py     # 冻结 E8 原语（ip_e14_lib 的 HELPER 依赖）
IP_E11R1_20261111_contexts.json       # 16 参考任务（utils/E17P1/E19P1 依赖）
IP_E12R1_20261112_contexts.json       # E12 语境文件（variable_coalitions/E16P1 依赖）
ip_e14_lib/                            # 冻结 E12R1/E12R3/E13R1V2 机制库
ip_e14r2_asr_audit/                    # E14R2 ASR 审计（A/A0/B/C/D 臂）
ip_e16p1_layered/                      # E16P1 双臂验证 + E16P2 位置忠实 + E16P3 最小签名扫荡
ip_e17p1_score/                        # E17P1 预注册分数变体（E20P1 的依赖）
ip_e19p1_incremental/                  # E19P1 增量流水线 + 签名缓存池
ip_e20p1_cleancenter/                  # E20P1 clean-only 中心消融（A1 修复）
IP_E14R2_20261113_ASR_RESULTS.md       # 总结果文档（§1-§14）
IP_E14R2_20261113_attack_specs_and_asr_audit_plan.md  # 攻击规范文档
```

**归档内容与去留理由**：

| 归档项 | 被谁取代 |
|---|---|
| IP_E2~E9（交互诊断、读出消融、E8 基线 detector） | 机制已并入冻结 lib；E8 结果文件在 outputs/ 不受影响 |
| IP_E10（反事实/协议/token 审计） | 结论并入 E14R2 审计与协议设计 |
| IP_E11（自动玩家/二阶交互/全量） | 并入 E12R1 搜索（lib 内副本为准） |
| IP_E12R1/R2V2/R3 根目录副本 | ip_e14_lib/ 内冻结副本为唯一现行版 |
| IP_E13R1/R1V2 根目录副本 | 同上（E13R1V2_core/experiment 在 lib 内） |
| IP_E14R1 根目录副本（跨攻击驱动） | E20P1 为现行全量检测形态；E14 冻结输出在 outputs/ |
| ip_e15p1_aligned/（协议对齐重测） | 已证伪的假设实验（负结果，数据在 outputs/e15p1_aligned/） |

各包的 checksums/tests/zip 随包归档。outputs/ 下全部历史结果**未做任何移动**。
