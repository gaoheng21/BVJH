# 净化 V1（raw+"\nAnswer:" 尾巴格式）— 已废弃归档

- V1 定义：训练对/交互零化读出位/审计均在检测读出协议（裸指令+"\nAnswer:"）下进行。
- 废弃原因（用户裁定 2026-11-19）：尾巴是检测辅助（放大器），净化应使用标准原生
  alpaca 格式；V2 全部改原生格式（ip_p1_purify/run_v2_native.sh，输出 p2_*_alpaca）。
- V1 其余输出目录（r1_surgery/r2_nullspace/r3_retrain*/r3c_*/p2_badnet/p2_ctba/
  p2_vpi/p2_sleeper/p2_refusal_*）已被外部清理，仅存本目录的 p2_negsentiment_mtba。
- V1 全部数字保留在 IP_E14R2_20261113_ASR_RESULTS.md §15-§21（含 a/b/c 消融、
  主表 IP-Ours 列 v1 口径值）；复现可用 IP_P2 --format raw_tail。
