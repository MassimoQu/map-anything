# OPV2V 文档整理（基于最新 Image-Only 评测）

> 目的：统一“哪些结论可信、哪些仅供历史参考”，避免继续引用带有外参/深度泄漏的旧结果。

## 1) 可信基线（Image-Only 批量评测 2025-11-26）
- 基准文件：`opv2v_batch_eval_analysis_image_only.md`、`opv2v_eval_report_image_only.md`。评估前剥离外参与 `depth_z`，脚本 `scripts/batch_eval.py` 固定 10 帧子集（`summary_test.json`），GPU 1/4/5 并行运行。
- 结论概要：
  - **Stage1 单车**：Pose Abs≈0.058 m / 0.62°，`scale_err≈3.3`，可作为部署级单车基线。
  - **Stage1 协同**：Pose Abs≈10.85 m，协同输入会破坏主车几何，不适合作为多车方案。
  - **Stage2 单车**：Pose Abs≈0.16 m，但 `scale_err≈16.6`，缺少协同视角时尺度发散。
  - **Stage2 协同**：Pose Abs≈2.09 m / 1.50°，但 `scale_err≈19.0`；在 BEV IoUᶠ 上领先 (0.065) 但仍不满足协同需求。
- 点云/BEV：过滤 `z∈[-1,4.2] m`、`r<120 m` 后，Stage2 协同 BEV IoUᶠ 最优但 Chamferᵣ 受大尺度偏移拖累。预训练模型在 image-only 下整体退化（单车 Pose≈6 m）。

## 2) 当前仍应使用的文档
- `opv2v_batch_eval_analysis_image_only.md`: Image-only 评测细节、命令、结果解读（最新权威）。
- `opv2v_eval_report_image_only.md`: 面向项目进展的简版汇报，基于同一批评测。
- `pose_evaluation_issue.md`: 解释外参/深度泄漏的来源与修复方式（`strip_external_calibration_inputs`），评测必须遵守。
- `opv2v_training_upgrade_plan.md`: 三项可落地改进（mask-aware depth、scale 蒸馏、多车一致性）及推荐顺序。
- `opv2v_vehicle_id_adaptation.md`: Vehicle-ID embedding 方案，适配多车身份感知。
- `opv2v_coop_training_summary.md`: Stage1/2/3 Hydra 命令与依赖梳理，可直接复现训练。
- `opv2v_unified_notes.md`: 数据路径、软链、环境与可视化/训练主线的“一页纸”。
- `opv2v_visualization_core_summary.md` / `opv2v_pointcloud_visualization.md`: 可视化脚本说明、无头查看方法。
- `mapanything_ft/docs/multi_car_training_plan.md`: 圆柱全景 4n→n 聚合方案与实现细节。
- `mapanything_ft/docs/opv2v_cyl_coop_debug_plan.md`: 20251201 圆柱训练失败的排查 checklist（确保使用 `mapanything_ft` 环境 + 固定步数）。

## 3) 已归档/不要再据此下结论的文档
- `opv2v_eval_report.md`: 早期 color_compare 流程仍注入 YAML 外参与深度，导致“厘米级协同”高估，仅保留历史脉络。
- `opv2v_single_agent_eval_report.md`: 旧版 `batch_eval_single_fixed.py` 抽样 6 帧，仍喂入外参/深度；相对位姿指标与现行绝对坐标系不一致。
- `opv2v_session_summary.md` 及其中 CPU/GPU 报告数字：使用泄漏输入与不同样本集，不能代表部署表现。
- 任何引用“Stage2 协同 20 cm / Pretrain 4 cm”级别数字的笔记都来自上述旧流程，应一律视作过期。

## 4) 近期工作优先级（结合最新基线）
1. **扩展 Image-Only 评测样本**：沿用 `scripts/batch_eval.py` + `summary_test.json`，扩到 ≥20 帧并记录方差，形成长期回归集。
2. **修正 Stage2 尺度漂移**：在训练中加入 `scale_err` 约束或蒸馏预训练尺度（参考 `opv2v_training_upgrade_plan.md`），确保协同模式不再 19 级偏移。
3. **针对协同失败帧做逐相机剖析**：复用代表性点云/metrics CSV，对 `stage2/coop` 的 worst case 输出做对齐图，定位是哪台相机引入漂移。
4. **推进圆柱 4n→n 方案**：按 `multi_car_training_plan.md` + `opv2v_cyl_coop_debug_plan.md` 复现数据/训练，先确保 loader/mask/steps 一致，再验证收敛。
5. **可视化与部署对齐**：继续使用 Plotly HTML/Matplotlib 离线查看；任何推理脚本务必调用 `strip_external_calibration_inputs`，保持与部署模态一致。

> 若未来有新评测或训练改动，请在对应文档开头注明日期与适用范围，并在本汇总同步更新“可信基线”和“已归档”列表。
