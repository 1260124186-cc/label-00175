import type { OpticalSystem, Optimization } from './config'

export type WorkflowType = 'opc' | 'smo' | 'ilt' | 'process_window' | 'batch' | 'simulation'

export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'starting'

export interface OPCConfigParams {
  epe_threshold: number
  epe_convergence_threshold: number
  max_iterations: number
  min_hotspot_area: number
  hotspot_dilation: number
  edge_offset_step: number
  max_edge_offset: number
  corner_bias_size: number
  line_end_extension: number
  line_end_width: number
  sraf_enable: boolean
  sraf_min_distance: number
  sraf_max_distance: number
  sraf_width: number
  sraf_length: number
  sraf_spacing: number
  sraf_min_feature_size: number
  sraf_max_aspect_ratio: number
  optimizer_enable: boolean
  optimizer_max_iter: number
  optimizer_learning_rate: number
  optimizer_epe_weight: number
  wafer_threshold: number
  verbose: boolean
}

export interface SourceConstraintsParams {
  energy_conservation: boolean
  energy_target: number
  sigma_target: number | null
  sigma_tolerance: number
  smoothness_weight: number
  smoothness_type: 'tv' | 'gaussian'
  gaussian_sigma: number
  non_negative: boolean
  support_radius: number | null
  support_radius_inner: number | null
}

export interface SMOConfigParams {
  strategy: 'alternating' | 'joint_gradient' | 'source_first'
  max_outer_iterations: number
  source_max_iter: number
  mask_max_iter: number
  joint_max_iter: number
  source_learning_rate: number
  mask_learning_rate: number
  joint_learning_rate_source: number
  joint_learning_rate_mask: number
  tol: number
  convergence_patience: number
  source_init_type: string
  source_constraints: SourceConstraintsParams
  wafer_threshold: number
  use_wafer_image_loss: boolean
  pvb_weight: number
  verbose: boolean
}

export interface ILTComplexityParams {
  perimeter_weight: number
  vertex_weight: number
  sub_feature_weight: number
  sub_feature_min_area: number
  sub_feature_max_area: number
}

export interface ILTConfigParams {
  max_iter: number
  learning_rate: number
  optimizer_type: 'gradient_projection' | 'adam_projection' | 'sgd_projection'
  convergence_tol: number
  convergence_patience: number
  transmission_level: 'binary' | 'ternary' | 'continuous'
  quantization_start_iter: number
  quantization_schedule: 'step' | 'linear' | 'cosine'
  quantization_strength: number
  resist_steepness: number
  wafer_threshold: number
  l2_wafer_weight: number
  complexity: ILTComplexityParams
  binary_penalty_weight: number
  tv_smooth_weight: number
  verbose: boolean
}

export interface ProcessWindowConfig {
  focus_range: [number, number, number]
  dose_range: [number, number, number]
  cd_tolerance: number
  epe_tolerance: number | null
  threshold: number
  save_visualizations: boolean
}

export interface BatchOptimizationConfig {
  source: string
  layer: number | null
  optical_system?: OpticalSystem
  optimization?: Optimization
  max_workers: number | null
  max_retries: number
  save_optimized_masks: boolean
  output_dir: string | null
  stop_on_first_failure: boolean
}

export interface WorkflowTask {
  task_id: string
  task_type: WorkflowType
  status: TaskStatus
  progress: number
  message?: string
  error?: string
  created_at?: number
  started_at?: number
  finished_at?: number
  result_summary?: Record<string, any>
  stage?: string
  current_loss?: number
  iteration?: number
}

export interface TaskSubmitResponse {
  success: boolean
  message: string
  task_id: string
  task_type: WorkflowType
  status: TaskStatus
}

export interface TaskListResponse {
  count: number
  tasks: WorkflowTask[]
}

export interface GdsLayerInfo {
  layer: number
  datatype: number
  name?: string
  cell_count?: number
}

export interface GdsFileInfo {
  filename: string
  path: string
  size: number
  uploaded_at: number
  layers: GdsLayerInfo[]
  cells: string[]
}

export interface BossungDataPoint {
  focus: number
  dose: number
  cd: number
  epe?: number
  valid: boolean
}

export interface ProcessWindowMetrics {
  max_exposure_latitude?: number
  depth_of_focus?: number
  process_window_area?: number
  nominal_cd?: number
  cd_uniformity?: number
  focus_points?: number
  dose_points?: number
}

export interface BatchTaskSummary {
  total: number
  succeeded: number
  failed: number
  skipped: number
  avg_mse?: number
  avg_ssim?: number
  elapsed_seconds?: number
}

export interface ExperimentRecord {
  id: string
  name: string
  description?: string
  workflow_type: WorkflowType
  created_at: number
  config_snapshot: Record<string, any>
  result_metrics: Record<string, any>
  tags?: string[]
}
