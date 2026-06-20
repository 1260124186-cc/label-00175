# Fab 模型标定报告

**生成时间**：2026-06-20 22:36:45  
**反演耗时**：0.21 s  
**反演方法**：`lmfit`  
**收敛状态**：✅ 成功  
**消息**：`ftol` termination condition is satisfied.  

## 1. CD-SEM 数据集概览

| 项目 | 值 |
|------|------|
| FAB 厂 |  |
| 工艺节点 |  |
| 掩模组 ID |  |
| 晶圆 ID |  |
| 批次 ID |  |
| 放大倍率 | 4.0× |
| 数据点数量 | 592 |
| Focus 范围 | (-150.0, 150.0) nm |
| Dose 范围 | (0.85, 1.15) |
| 目标 CD 范围 | (45.0, 100.0) nm |
| 实测 CD 范围 | (0.117, 175.327) nm |
| 图形类型 | contact_hole, isolated_line, line_space |

## 2. 标定参数结果

| 参数名 | 标定值 | ±1σ 不确定度 | 单位 | 物理含义 |
|--------|--------|---------------|------|----------|
| `resist_threshold` | 0.286457 | ±0.002885 | norm.intensity | 光刻胶显影阈值（归一化光强） |
| `diffusion_length` | 12.712065 | ±0.250420 | nm | PEB 后酸扩散长度的 RMS 值 |
| `na_effective` | 1.335492 | ±0.001717 | - | 有效数值孔径（偏振/填充因子修正） |
| `dose_to_clear` | 0.718379 | ±6.796290 | relative | 大开阔区域刚好完全曝光的剂量（相对标称剂量） |
| `resist_contrast` | 10.000000 | ±562.554804 | - | 光刻胶对比度 γ (H-D 曲线斜率) |
| `sigma_effective` | 0.781389 | ±0.006493 | - | 有效部分相干因子 |
| `wavelength_effective` 🔒(固定) | 193.000000 | ±0.000000 | nm | 有效光源波长（考虑吸收偏移等） |

### 参数相关系数矩阵

| | `resist_threshold` | `diffusion_length` | `na_effective` | `dose_to_clear` | `resist_contrast` | `sigma_effective` |
|---|---|---|---|---|---|---|
| `resist_threshold` | +1.000 | +0.874 | +0.118 | +0.229 | +0.230 | +0.749 |
| `diffusion_length` | +0.874 | +1.000 | +0.070 | -0.010 | -0.010 | +0.740 |
| `na_effective` | +0.118 | +0.070 | +1.000 | +0.031 | +0.031 | -0.195 |
| `dose_to_clear` | +0.229 | -0.010 | +0.031 | +1.000 | +1.000 | +0.170 |
| `resist_contrast` | +0.230 | -0.010 | +0.031 | +1.000 | +1.000 | +0.170 |
| `sigma_effective` | +0.749 | +0.740 | -0.195 | +0.170 | +0.170 | +1.000 |

⚠️ **高相关参数对（|r|≥0.7，可能影响可辨识性）：**

- `resist_threshold` ↔ `diffusion_length`: r = +0.874
- `resist_threshold` ↔ `sigma_effective`: r = +0.749
- `diffusion_length` ↔ `sigma_effective`: r = +0.740
- `dose_to_clear` ↔ `resist_contrast`: r = +1.000

## 3. 反演统计

- 数据点数：592
- 自由参数数：6
- 自由度 (dof)：586
- 迭代次数：29
- 最终代价函数：5.347186e+02
- 卡方 χ²：742.6648
- 约化卡方 χ²/dof：1.2673

## 4. 模型拟合质量

| 指标 | 训练集 | 测试集 |
|------|--------|--------|
| MAE_nm | 1.0855 | 1.1318 |
| MAPE_pct | 12.0919 | 11.8258 |
| R2 | 0.9992 | 0.9991 |
| RMSE_nm | 1.3441 | 1.3920 |
| max_error_nm | 3.7016 | 3.9894 |

## 6. 可视化结果

### Measured Vs Predicted

![measured_vs_predicted.png](measured_vs_predicted.png)

### Bossung Curves

![bossung_curves.png](bossung_curves.png)

### Parameter Convergence

![parameter_convergence.png](parameter_convergence.png)

### Residual Analysis

![residual_analysis.png](residual_analysis.png)

## 7. 结论与建议

✅ 反演收敛。
✅ 训练集 RMSE < 2 nm，模型拟合良好。
✅ R² ≥ 0.95，解释方差足够。
