import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, InternalAxiosRequestConfig } from 'axios'
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

const TOKEN_KEY = 'litho_auth_token'
const USER_KEY = 'litho_auth_user'

const service: AxiosInstance = axios.create({
  baseURL: '/',
  timeout: 30000,
})

function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

function setStoredToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token)
}

function clearStoredAuth() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}

function getStoredUser(): UserInfo | null {
  const raw = localStorage.getItem(USER_KEY)
  if (!raw) return null
  try {
    return JSON.parse(raw)
  } catch {
    return null
  }
}

function setStoredUser(user: UserInfo) {
  localStorage.setItem(USER_KEY, JSON.stringify(user))
}

export interface UserInfo {
  user_id: string
  username: string
  display_name: string
}

export interface AuthState {
  token: string | null
  user: UserInfo | null
  isAuthenticated: boolean
}

export function getAuthState(): AuthState {
  const token = getStoredToken()
  const user = getStoredUser()
  return {
    token,
    user,
    isAuthenticated: !!token && !!user,
  }
}

service.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = getStoredToken()
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

service.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (error?.response?.status === 401) {
      clearStoredAuth()
      const currentPath = window.location.pathname
      if (!currentPath.includes('/login')) {
        window.location.href = '/login'
      }
    }
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

export const authApi = {
  register: (username: string, password: string, displayName?: string) =>
    service.post<any, any>('/api/auth/register', {
      username,
      password,
      display_name: displayName || null,
    }),

  login: async (username: string, password: string): Promise<AuthState> => {
    const res = await service.post<any, any>('/api/auth/login', {
      username,
      password,
    })
    setStoredToken(res.access_token)
    setStoredUser(res.user)
    return {
      token: res.access_token,
      user: res.user,
      isAuthenticated: true,
    }
  },

  logout: () => {
    clearStoredAuth()
  },

  me: () =>
    service.get<any, any>('/api/auth/me'),
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

  runWithCustomMask: (config: SimulationConfig, maskData: number[][], pupilData?: number[][] | null) =>
    service.post<any, any>('/api/simulation/run', {
      config,
      pattern_type: 'custom_mask',
      pattern_params: {
        size: [maskData.length, maskData[0]?.length || 0],
        custom_mask: maskData,
        custom_pupil: pupilData || null
      }
    }),

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
