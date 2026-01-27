# Map-Anything 文档导航（分层版）

> 先看“必读三件套”，再按子目录查详情；过期内容集中在 `archive/`。

## 顶层
- `opv2v_docs_consolidated.md`：权威入口，写明最新结论/优先级/归档。

## 必读三件套（5 分钟上手）
- `opv2v_docs_consolidated.md`（本文件夹根）
- `eval/opv2v_batch_eval_analysis_image_only.md`：标准评测流程 + 数字（剥离外参/深度）。
- `training/opv2v_coop_training_summary.md`：Stage1/2/3 训练命令与依赖。

## 目录速览
- `eval/`：评测与假设  
  - `opv2v_eval_report_image_only.md`（汇报版数值）  
  - `pose_evaluation_issue.md`（必须 strip 外参/深度的原因）
- `training/`：训练改进 / 方案  
  - `opv2v_training_upgrade_plan.md`（mask-aware 深度/尺度蒸馏/多车一致性）  
  - `opv2v_vehicle_id_adaptation.md`（多车身份嵌入）  
  - `opv2v_unified_notes.md`（数据/环境/流程一页纸）
- `viz/`：可视化  
  - `opv2v_visualization_core_summary.md`（脚本与依赖拆解）  
  - `opv2v_pointcloud_visualization.md`（无头查看/HTML 发布指南）
- `assets/`：配套图示
- `archive/`：早期含外参/深度泄漏或流程过期的文档（见 `archive/README.md`）

## 相关外部文档
- 圆柱 4n→n 多车方案与调试：`mapanything_ft/docs/`（活跃，见其 README）。

新增评测/方案时：更新本导航，必要时把旧文档移入 `archive/` 并注明日期/范围。***
