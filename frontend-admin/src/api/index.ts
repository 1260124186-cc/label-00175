import axios, { AxiosInstance, AxiosRequestConfig } from 'axios'
import { ElMessage } from 'element-plus'
import type {
  SimulationConfig,
} from '@/types/config'

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

export default service
