export interface SourceParams {
  sigma_inner: number
  sigma_outer: number
  angle?: number | null
  opening_angle?: number | null
}

export interface OpticalSystem {
  wavelength: number
  na: number
  sigma: number
  pixel_size: number
  defocus: number
  magnification: number
  illumination_type: 'conventional' | 'annular' | 'dipole' | 'quasar' | 'custom'
  source_params: SourceParams
  tcc_mode: 'full_tcc' | 'socs' | 'kernel_2d'
  socs_num_terms: number
  use_socs: boolean
  technology_node: 'duv_arf' | 'euv'
  flare: number
  shadowing_model: 'none' | 'approximate' | 'rigorous'
  reflective_mask_attenuation: number
  zernike_coefficients: Record<string, number>
}

export interface SpatialWeight {
  enable: boolean
  edge_weight: number
  corner_weight: number
  line_end_weight: number
  base_weight: number
  edge_sigma: number
  corner_threshold: number
  line_end_threshold: number
  weight_erosion: boolean
  smooth_sigma: number
  normalize: boolean
}

export interface LossWeights {
  mse: number
  ssim: number
  pvb: number
  mask_complexity: number
  weighted_mse: number
  weighted_mae: number
}

export interface Regularization {
  type: 'null' | 'l1' | 'l2' | 'tv' | null
  strength: number
}

export interface Optimization {
  optimizer_type: 'gradient_descent' | 'bfgs' | 'newton' | 'genetic' | 'pso' | 'rl'
  max_iter: number
  learning_rate: number
  tol: number
  early_stop_patience: number
  lr_scheduler: 'step' | 'exponential' | 'cosine' | null
  lr_decay: number
  lr_step_size: number
  metric: 'mse' | 'mae' | 'ssim'
  use_composite_loss: boolean
  loss_weights: LossWeights
  spatial_weight: SpatialWeight
  regularization: Regularization
  bounds: [number, number]
  verbose: boolean
  random_seed: number | null
  population_size: number
  crossover_rate: number
  mutation_rate: number
  n_jobs: number
  rl_gamma: number
  rl_epsilon: number
  rl_epsilon_decay: number
}

export interface OutputConfig {
  save_dir: string
  save_images: boolean
  save_history: boolean
  image_format: 'png' | 'tiff'
  log_level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'
}

export interface ImagingConfig {
  resist_threshold: number
  apply_resist: boolean
}

export interface SimulationConfig {
  optical_system: OpticalSystem
  optimization: Optimization
  output: OutputConfig
  imaging: ImagingConfig
}
