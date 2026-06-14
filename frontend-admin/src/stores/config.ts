import { defineStore } from 'pinia'
import type { SimulationConfig } from '@/types/config'
import { configApi } from '@/api'

function createDefaultConfig(): SimulationConfig {
  return {
    optical_system: {
      wavelength: 193.0,
      na: 1.35,
      sigma: 0.75,
      pixel_size: 1.0,
      defocus: 0.0,
      magnification: 4.0,
      illumination_type: 'conventional',
      source_params: {
        sigma_inner: 0.0,
        sigma_outer: 0.75,
        angle: null,
        opening_angle: null
      },
      tcc_mode: 'socs',
      socs_num_terms: 5,
      use_socs: true,
      zernike_coefficients: {}
    },
    optimization: {
      optimizer_type: 'gradient_descent',
      max_iter: 100,
      learning_rate: 0.01,
      tol: 1e-6,
      early_stop_patience: 10,
      lr_scheduler: null,
      lr_decay: 0.95,
      lr_step_size: 20,
      metric: 'mse',
      use_composite_loss: false,
      loss_weights: {
        mse: 1.0,
        ssim: 0.0,
        pvb: 0.0,
        mask_complexity: 0.0,
        weighted_mse: 0.0,
        weighted_mae: 0.0
      },
      spatial_weight: {
        enable: false,
        edge_weight: 2.0,
        corner_weight: 5.0,
        line_end_weight: 4.0,
        base_weight: 1.0,
        edge_sigma: 1.0,
        corner_threshold: 0.3,
        line_end_threshold: 0.5,
        weight_erosion: true,
        smooth_sigma: 0.5,
        normalize: true
      },
      regularization: {
        type: null,
        strength: 0.0
      },
      bounds: [0.0, 1.0],
      verbose: true,
      random_seed: 42,
      population_size: 50,
      crossover_rate: 0.8,
      mutation_rate: 0.1,
      n_jobs: 1,
      rl_gamma: 0.99,
      rl_epsilon: 0.1,
      rl_epsilon_decay: 0.995
    },
    output: {
      save_dir: './results',
      save_images: true,
      save_history: true,
      image_format: 'png',
      log_level: 'INFO'
    },
    imaging: {
      resist_threshold: 0.3,
      apply_resist: true
    }
  }
}

export const useConfigStore = defineStore('config', {
  state: () => ({
    config: createDefaultConfig() as SimulationConfig,
    loading: false,
    savedFiles: [] as any[]
  }),

  actions: {
    async _fetchDefaultConfig() {
      try {
        const res: any = await configApi.getDefault()
        if (res.success && res.config) {
          this.config = res.config as SimulationConfig
        }
      } catch (e) {
        console.error('加载默认配置失败:', e)
        this.config = createDefaultConfig()
      }
    },

    async loadDefault() {
      this.loading = true
      try {
        await this._fetchDefaultConfig()
      } finally {
        this.loading = false
      }
    },

    async loadSaved(filename: string) {
      this.loading = true
      try {
        const res: any = await configApi.getSaved(filename)
        if (res.success && res.config) {
          this.config = res.config as SimulationConfig
        }
      } catch (e) {
        console.error('加载保存配置失败:', e)
        throw e
      } finally {
        this.loading = false
      }
    },

    async fetchSavedList() {
      try {
        const res: any = await configApi.listSaved()
        this.savedFiles = res.files || []
      } catch (e) {
        console.error('加载保存列表失败:', e)
        this.savedFiles = []
      }
    },

    async loadInitialData() {
      this.loading = true
      try {
        await Promise.all([this._fetchDefaultConfig(), this.fetchSavedList()])
      } finally {
        this.loading = false
      }
    },

    updateConfig(newConfig: SimulationConfig) {
      this.config = JSON.parse(JSON.stringify(newConfig))
    },

    resetToDefault() {
      this.config = createDefaultConfig()
    }
  }
})
