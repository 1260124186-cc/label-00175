import axios, { AxiosInstance, AxiosRequestConfig } from 'axios'
import { ElMessage } from 'element-plus'
import type { SimulationConfig } from '@/types/config'
import type {
  OPCConfigParams,
  SMOConfigParams,
  ILTConfigParams,
  ProcessWindowConfig,
  BatchOptimizationConfig,
  WorkflowTask,
  TaskSubmitResponse,
  TaskListResponse,
  GdsFileInfo,
  GdsLayersResponse,
  WorkflowType,
  TaskStatus,
  TaskResultResponse,
} from '@/types/workflow'

const service: AxiosInstance = axios.create({
  baseURL: '/',
  timeout: 30000,
})

service.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.error('Request error:', error)
    ElMessage.error(error?.response?.data?.detail || error.message || '请求失败')
    return Promise.reject(error)
  }
)

export interface ApiResponse<T = any> {
  success: boolean
  message?: string
  config?: T
  data?: T
  [key: string]: any
}

export const configApi = {
  getDefault: () =>
    service.get<any, ApiResponse<SimulationConfig>>('/api/config/default'),

  listSaved: () =>
    service.get<any, any>('/api/config/saved'),

  getSaved: (filename: string) =>
    service.get<any, ApiResponse<SimulationConfig>>(`/api/config/saved/${encodeURIComponent(filename)}`),

  save: (config: SimulationConfig, filename?: string) =>
    service.post<any, ApiResponse>('/api/config/save', { config, filename }),

  delete: (filename: string) =>
    service.delete<any, ApiResponse>(`/api/config/saved/${encodeURIComponent(filename)}`),

  validate: (config: SimulationConfig) =>
    service.post<any, ApiResponse>('/api/config/validate', config),
}

export const simulationApi = {
  run: (config: SimulationConfig, patternType: string = 'rectangle', patternParams: Record<string, any> = {}) =>
    service.post<any, ApiResponse>('/api/simulation/run', { config, pattern_type: patternType, pattern_params: patternParams }),

  listTasks: () =>
    service.get<any, any>('/api/simulation/tasks'),

  getTaskStatus: (taskId: string) =>
    service.get<any, any>(`/api/simulation/tasks/${taskId}`),
}

export const workflowApi = {
  runOpc: (
    opticalSystem: any,
    opcConfig: OPCConfigParams,
    patternType: string,
    patternParams: Record<string, any>,
    gdsFileId?: string,
    gdsLayer?: number | null,
    gdsDatatype?: number | null,
    gdsPixelSize?: number | null,
    gdsTargetSize?: [number, number] | null
  ): Promise<TaskSubmitResponse> =>
    service.post<any, TaskSubmitResponse>('/api/workflows/opc', {
      optical_system: opticalSystem,
      opc_config: opcConfig,
      pattern_type: patternType,
      pattern_params: patternParams,
      gds_file_id: gdsFileId ?? null,
      gds_layer: gdsLayer ?? null,
      gds_datatype: gdsDatatype ?? null,
      gds_pixel_size: gdsPixelSize ?? null,
      gds_target_size: gdsTargetSize ?? null,
    }),

  runSmo: (
    opticalSystem: any,
    smoConfig: SMOConfigParams,
    patternType: string,
    patternParams: Record<string, any>,
    gdsFileId?: string,
    gdsLayer?: number | null,
    gdsDatatype?: number | null,
    gdsPixelSize?: number | null,
    gdsTargetSize?: [number, number] | null
  ): Promise<TaskSubmitResponse> =>
    service.post<any, TaskSubmitResponse>('/api/workflows/smo', {
      optical_system: opticalSystem,
      smo_config: smoConfig,
      pattern_type: patternType,
      pattern_params: patternParams,
      gds_file_id: gdsFileId ?? null,
      gds_layer: gdsLayer ?? null,
      gds_datatype: gdsDatatype ?? null,
      gds_pixel_size: gdsPixelSize ?? null,
      gds_target_size: gdsTargetSize ?? null,
    }),

  runIlt: (
    opticalSystem: any,
    iltConfig: ILTConfigParams,
    patternType: string,
    patternParams: Record<string, any>,
    gdsFileId?: string,
    gdsLayer?: number | null,
    gdsDatatype?: number | null,
    gdsPixelSize?: number | null,
    gdsTargetSize?: [number, number] | null
  ): Promise<TaskSubmitResponse> =>
    service.post<any, TaskSubmitResponse>('/api/workflows/ilt', {
      optical_system: opticalSystem,
      ilt_config: iltConfig,
      pattern_type: patternType,
      pattern_params: patternParams,
      gds_file_id: gdsFileId ?? null,
      gds_layer: gdsLayer ?? null,
      gds_datatype: gdsDatatype ?? null,
      gds_pixel_size: gdsPixelSize ?? null,
      gds_target_size: gdsTargetSize ?? null,
    }),

  runProcessWindow: (
    opticalSystem: any,
    pwConfig: ProcessWindowConfig,
    patternType: string,
    patternParams: Record<string, any>,
    gdsFileId?: string,
    gdsLayer?: number | null,
    gdsDatatype?: number | null,
    gdsPixelSize?: number | null,
    gdsTargetSize?: [number, number] | null
  ): Promise<TaskSubmitResponse> =>
    service.post<any, TaskSubmitResponse>('/api/workflows/process-window', {
      optical_system: opticalSystem,
      pattern_type: patternType,
      pattern_params: patternParams,
      gds_file_id: gdsFileId ?? null,
      gds_layer: gdsLayer ?? null,
      gds_datatype: gdsDatatype ?? null,
      gds_pixel_size: gdsPixelSize ?? null,
      gds_target_size: gdsTargetSize ?? null,
      focus_range: pwConfig.focus_range,
      dose_range: pwConfig.dose_range,
      cd_tolerance: pwConfig.cd_tolerance,
      epe_tolerance: pwConfig.epe_tolerance,
      threshold: pwConfig.threshold,
      save_visualizations: pwConfig.save_visualizations,
    }),

  runBatch: (
    source: string,
    layer: number | null,
    opticalSystem: any,
    optimization: any,
    maxWorkers: number | null,
    maxRetries: number,
    saveOptimizedMasks: boolean,
    outputDir: string | null,
    stopOnFirstFailure: boolean
  ): Promise<TaskSubmitResponse> =>
    service.post<any, TaskSubmitResponse>('/api/workflows/batch', {
      source,
      layer,
      optical_system: opticalSystem,
      optimization,
      max_workers: maxWorkers,
      max_retries: maxRetries,
      save_optimized_masks: saveOptimizedMasks,
      output_dir: outputDir,
      stop_on_first_failure: stopOnFirstFailure,
    }),
}

export const taskApi = {
  list: (taskType?: WorkflowType, status?: TaskStatus): Promise<TaskListResponse> => {
    const params: Record<string, string> = {}
    if (taskType) params.task_type = taskType
    if (status) params.status = status
    return service.get<any, TaskListResponse>('/api/tasks', { params })
  },

  getStatus: (taskId: string): Promise<WorkflowTask> =>
    service.get<any, WorkflowTask>(`/api/tasks/${taskId}`),

  getResult: (taskId: string): Promise<TaskResultResponse> =>
    service.get<any, TaskResultResponse>(`/api/tasks/${taskId}/result`),

  download: (taskId: string): Promise<Blob> =>
    service.get<any, Blob>(`/api/tasks/${taskId}/download`, { responseType: 'blob' }),
}

export const gdsApi = {
  upload: (file: File): Promise<GdsFileInfo> => {
    const formData = new FormData()
    formData.append('file', file)
    return service.post<any, { success: boolean; message: string; file: GdsFileInfo }>('/api/gds/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 60000,
    }).then(res => res.file)
  },

  list: (): Promise<{ count: number; files: GdsFileInfo[] }> =>
    service.get<any, { count: number; files: GdsFileInfo[] }>('/api/gds/list'),

  getLayers: (fileId: string): Promise<GdsLayersResponse> =>
    service.get<any, GdsLayersResponse>(`/api/gds/${encodeURIComponent(fileId)}/layers`),

  delete: (fileId: string): Promise<any> =>
    service.delete<any, any>(`/api/gds/${encodeURIComponent(fileId)}`),
}

export default service
