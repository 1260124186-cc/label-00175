<template>
  <div class="batch-queue-monitor">
    <el-row :gutter="20">
      <el-col :xs="24" :lg="8">
        <el-card class="card" shadow="hover">
          <div class="card-header">
            <div class="title">
              <el-icon size="20" color="#e6a23c"><FolderOpened /></el-icon>
              <span>批处理配置</span>
            </div>
            <el-tag type="warning" effect="plain" round size="small">Batch</el-tag>
          </div>

          <el-alert
            title="批量处理 GDS 版图中的多个 Cell，自动应用优化算法"
            type="warning"
            :closable="false"
            show-icon
            style="margin-bottom: 16px"
          />

          <el-tabs v-model="activeTab" type="border-card">
            <el-tab-pane label="输入源" name="source">
              <GdsUploader
                v-model="selectedGds"
                :selected-layer-value="selectedLayer"
                @select="onGdsSelect"
              />
            </el-tab-pane>

            <el-tab-pane label="批处理参数" name="params">
              <el-form label-width="140px" label-position="right">
                <el-divider content-position="left"><span class="divider-label">资源配置</span></el-divider>

                <el-form-item label="最大并发数">
                  <el-input-number v-model="batchConfig.max_workers_value" :min="1" :max="32" class="w-full" />
                </el-form-item>

                <el-form-item label="失败重试次数">
                  <el-input-number v-model="batchConfig.max_retries" :min="0" :max="10" class="w-full" />
                </el-form-item>

                <el-divider content-position="left"><span class="divider-label">输出设置</span></el-divider>

                <el-form-item label="保存优化掩模">
                  <el-switch v-model="batchConfig.save_optimized_masks" />
                </el-form-item>

                <el-form-item label="遇错即停">
                  <el-switch v-model="batchConfig.stop_on_first_failure" />
                </el-form-item>

                <el-form-item label="输出目录">
                  <el-input v-model="batchConfig.output_dir_value" placeholder="留空使用默认目录" />
                </el-form-item>

                <el-divider content-position="left"><span class="divider-label">优化器配置</span></el-divider>

                <el-form-item label="使用配置页参数">
                  <el-switch v-model="useConfigParams" />
                </el-form-item>

                <el-form-item label="最大迭代次数" v-if="!useConfigParams">
                  <el-input-number v-model="customMaxIter" :min="10" :max="500" class="w-full" />
                </el-form-item>

                <el-form-item label="学习率" v-if="!useConfigParams">
                  <el-input-number v-model="customLearningRate" :min="0.001" :step="0.005" :precision="4" class="w-full" />
                </el-form-item>
              </el-form>
            </el-tab-pane>
          </el-tabs>

          <div class="form-actions">
            <el-button type="warning" size="large" :icon="VideoPlay" :loading="isRunning" @click="handleRun" class="run-btn">
              开始批处理
            </el-button>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="16">
        <el-card class="card" shadow="hover">
          <div class="card-header">
            <div class="title">
              <el-icon size="20" color="#67c23a"><TrendCharts /></el-icon>
              <span>队列监控</span>
            </div>
            <div class="actions">
              <el-tag v-if="currentBatchId" type="success" effect="light" size="small">
                Batch: {{ currentBatchId.slice(0, 8) }}
              </el-tag>
              <el-button type="primary" link size="small" @click="refreshQueue" :loading="loading">
                <el-icon><Refresh /></el-icon>
                刷新
              </el-button>
            </div>
          </div>

          <el-row :gutter="16" class="stats-row">
            <el-col :span="6">
              <div class="stat-card stat-pending">
                <div class="stat-icon"><el-icon><Timer /></el-icon></div>
                <div class="stat-info">
                  <span class="stat-value">{{ stats.pending }}</span>
                  <span class="stat-label">等待中</span>
                </div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-card stat-running">
                <div class="stat-icon"><el-icon><Cpu /></el-icon></div>
                <div class="stat-info">
                  <span class="stat-value">{{ stats.running }}</span>
                  <span class="stat-label">运行中</span>
                </div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-card stat-completed">
                <div class="stat-icon"><el-icon><CircleCheck /></el-icon></div>
                <div class="stat-info">
                  <span class="stat-value">{{ stats.completed }}</span>
                  <span class="stat-label">已完成</span>
                </div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-card stat-failed">
                <div class="stat-icon"><el-icon><CircleClose /></el-icon></div>
                <div class="stat-info">
                  <span class="stat-value">{{ stats.failed }}</span>
                  <span class="stat-label">失败</span>
                </div>
              </div>
            </el-col>
          </el-row>

          <div class="progress-section" v-if="totalTasks > 0">
            <div class="progress-header">
              <span>总进度</span>
              <span>{{ completedCount }} / {{ totalTasks }}</span>
            </div>
            <el-progress
              :percentage="Math.round(overallProgress)"
              :stroke-width="12"
              :status="overallStatus"
            />
          </div>

          <el-table
            :data="queueTasks"
            v-loading="loading"
            style="width: 100%"
            size="small"
            height="340"
            stripe
          >
            <el-table-column type="index" label="#" width="50" align="center" />
            <el-table-column prop="name" label="Cell 名称" min-width="160">
              <template #default="{ row }">
                <div class="cell-name">
                  <el-icon size="14" color="#409eff"><Grid /></el-icon>
                  <span>{{ row.name || 'Unknown' }}</span>
                </div>
              </template>
            </el-table-column>
            <el-table-column prop="status" label="状态" width="90" align="center">
              <template #default="{ row }">
                <el-tag :type="statusType(row.status)" size="small" effect="dark">
                  {{ statusLabel(row.status) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="进度" min-width="140">
              <template #default="{ row }">
                <el-progress
                  :percentage="Math.round(row.progress || 0)"
                  :stroke-width="6"
                  :status="progressStatus(row.status)"
                  :text-inside="false"
                />
              </template>
            </el-table-column>
            <el-table-column prop="mse" label="MSE" width="100" align="center">
              <template #default="{ row }">
                <span v-if="row.mse !== undefined" :class="metricClass(row.mse)">
                  {{ formatNumber(row.mse) }}
                </span>
                <span v-else>—</span>
              </template>
            </el-table-column>
            <el-table-column prop="ssim" label="SSIM" width="100" align="center">
              <template #default="{ row }">
                <span v-if="row.ssim !== undefined" :class="ssimClass(row.ssim)">
                  {{ formatNumber(row.ssim) }}
                </span>
                <span v-else>—</span>
              </template>
            </el-table-column>
            <el-table-column prop="duration" label="耗时" width="90" align="center">
              <template #default="{ row }">
                {{ row.duration ? formatDuration(row.duration) : '—' }}
              </template>
            </el-table-column>
            <el-table-column label="操作" width="80" align="center" fixed="right">
              <template #default="{ row }">
                <el-button type="primary" link size="small" @click="viewTask(row)">
                  详情
                </el-button>
              </template>
            </el-table-column>
          </el-table>

          <div class="queue-empty" v-if="queueTasks.length === 0 && !loading">
            <el-icon :size="48" color="#dcdfe6"><Files /></el-icon>
            <p>暂无批处理任务</p>
            <p class="hint">选择 GDS 文件并配置参数后开始批处理</p>
          </div>
        </el-card>

        <el-card v-if="selectedTask" class="card detail-card" shadow="hover" style="margin-top: 16px">
          <div class="card-header">
            <div class="title">
              <el-icon size="18" color="#409eff"><DataBoard /></el-icon>
              <span>任务详情</span>
              <el-tag size="small">{{ selectedTask.name }}</el-tag>
            </div>
            <el-button type="primary" link size="small" @click="selectedTask = null">
              关闭
            </el-button>
          </div>

          <el-descriptions :column="2" border size="small">
            <el-descriptions-item label="状态">
              <el-tag :type="statusType(selectedTask.status)" size="small" effect="dark">
                {{ statusLabel(selectedTask.status) }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="进度">
              {{ Math.round(selectedTask.progress || 0) }}%
            </el-descriptions-item>
            <el-descriptions-item label="MSE" v-if="selectedTask.mse !== undefined">
              <span :class="metricClass(selectedTask.mse)">{{ formatNumber(selectedTask.mse) }}</span>
            </el-descriptions-item>
            <el-descriptions-item label="SSIM" v-if="selectedTask.ssim !== undefined">
              <span :class="ssimClass(selectedTask.ssim)">{{ formatNumber(selectedTask.ssim) }}</span>
            </el-descriptions-item>
          </el-descriptions>

          <el-alert
            v-if="selectedTask.error"
            :title="selectedTask.error"
            type="error"
            :closable="false"
            show-icon
            style="margin-top: 12px"
          />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import {
  FolderOpened, VideoPlay, TrendCharts, Refresh, Timer, Cpu,
  CircleCheck, CircleClose, Grid, Files, DataBoard
} from '@element-plus/icons-vue'
import { useConfigStore } from '@/stores/config'
import { workflowApi, taskApi } from '@/api'
import taskWs from '@/api/websocket'
import GdsUploader from './GdsUploader.vue'
import type { BatchOptimizationConfig, WorkflowTask } from '@/types/workflow'

const configStore = useConfigStore()

const activeTab = ref('source')
const selectedGds = ref('')
const selectedLayer = ref<number | null>(null)
const isRunning = ref(false)
const loading = ref(false)
const currentBatchId = ref('')
const selectedTask = ref<any>(null)
const useConfigParams = ref(true)
const customMaxIter = ref(100)
const customLearningRate = ref(0.01)

const batchConfig = reactive({
  max_workers_value: 4,
  max_retries: 2,
  save_optimized_masks: true,
  stop_on_first_failure: false,
  output_dir_value: '',
})

const queueTasks = ref<any[]>([])

const stats = computed(() => {
  const result = { pending: 0, running: 0, completed: 0, failed: 0 }
  for (const t of queueTasks.value) {
    const s = t.status
    if (s === 'pending' || s === 'starting') result.pending++
    else if (s === 'running') result.running++
    else if (s === 'completed') result.completed++
    else if (s === 'failed') result.failed++
  }
  return result
})

const totalTasks = computed(() => queueTasks.value.length)
const completedCount = computed(() => stats.value.completed + stats.value.failed)

const overallProgress = computed(() => {
  if (totalTasks.value === 0) return 0
  const totalProgress = queueTasks.value.reduce((sum, t) => sum + (t.progress || 0), 0)
  return totalProgress / totalTasks.value
})

const overallStatus = computed((): '' | 'success' | 'warning' | 'danger' | 'exception' => {
  if (stats.value.failed > 0) return 'exception'
  if (stats.value.completed === totalTasks.value && totalTasks.value > 0) return 'success'
  if (stats.value.running > 0) return ''
  return ''
})

let wsUnsubscribe: (() => void) | null = null
let pollTimer: ReturnType<typeof setInterval> | null = null

function onGdsSelect(file: any, layer: number, datatype: number) {
  selectedLayer.value = layer
  generateMockQueue(file.filename)
}

function generateMockQueue(filename: string) {
  const mockCells = [
    'AND2_X1', 'NAND2_X1', 'NOR2_X1', 'INV_X1',
    'DFF_X1', 'MUX2_X1', 'XOR2_X1', 'AOI21_X1',
    'OAI21_X1', 'ADD_HALF', 'ADD_FULL', 'COMP_X1',
  ]
  queueTasks.value = mockCells.map((name, i) => ({
    id: `${filename.replace('.gds', '')}_${name}`,
    name,
    status: i < 3 ? 'running' : i < 6 ? 'completed' : 'pending',
    progress: i < 3 ? 20 + i * 25 : i < 6 ? 100 : 0,
    mse: i < 6 ? 0.001 + Math.random() * 0.01 : undefined,
    ssim: i < 6 ? 0.9 + Math.random() * 0.09 : undefined,
    duration: i < 6 ? 30 + Math.random() * 60 : undefined,
    error: i === 4 ? '收敛失败' : undefined,
  }))
  queueTasks.value[4].status = 'failed'
}

async function handleRun() {
  if (!selectedGds.value) {
    ElMessage.warning('请选择 GDS 文件')
    return
  }
  const config = (configStore as any).config
  if (!config?.optical_system) {
    ElMessage.warning('请先到「参数配置」页加载光学系统配置')
    return
  }

  isRunning.value = true
  try {
    const optConfig = useConfigParams.value
      ? config.optimization
      : {
          ...config.optimization,
          max_iter: customMaxIter.value,
          learning_rate: customLearningRate.value,
        }

    const res = await workflowApi.runBatch(
      selectedGds.value,
      selectedLayer.value,
      config.optical_system,
      optConfig,
      batchConfig.max_workers_value,
      batchConfig.max_retries,
      batchConfig.save_optimized_masks,
      batchConfig.output_dir_value || null,
      batchConfig.stop_on_first_failure
    )

    if (res.success) {
      currentBatchId.value = res.task_id
      ElMessage.success(`批处理任务已提交：${res.task_id}`)
      startPolling()
    } else {
      ElMessage.error(res.message || '提交失败')
    }
  } catch (e: any) {
    ElMessage.error(e?.message || '提交失败')
  } finally {
    isRunning.value = false
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer)
  pollTimer = setInterval(() => {
    refreshQueue()
  }, 3000)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

async function refreshQueue() {
  if (!currentBatchId.value) return
  loading.value = true
  try {
    const res: any = await taskApi.getStatus(currentBatchId.value)
    if (res.status === 'completed' || res.status === 'failed') {
      stopPolling()
    }
  } catch (e) {
  } finally {
    loading.value = false
  }
}

function viewTask(task: any) {
  selectedTask.value = { ...task }
}

function statusType(s: string) {
  switch (s) {
    case 'completed': return 'success'
    case 'running':   return 'primary'
    case 'failed':    return 'danger'
    case 'starting':  return 'warning'
    case 'pending':   return 'info'
    default:          return 'info'
  }
}

function statusLabel(s: string) {
  const m: Record<string, string> = {
    pending: '等待',
    starting: '启动',
    running: '运行中',
    completed: '完成',
    failed: '失败',
  }
  return m[s] || s
}

function progressStatus(s: string): '' | 'success' | 'warning' | 'danger' | 'exception' {
  switch (s) {
    case 'completed': return 'success'
    case 'failed':    return 'exception'
    case 'starting':  return 'warning'
    default:          return ''
  }
}

function metricClass(v: number): string {
  if (typeof v !== 'number') return ''
  if (v < 0.01) return 'metric-good'
  if (v < 0.1) return 'metric-mid'
  return 'metric-bad'
}

function ssimClass(v: number): string {
  if (typeof v !== 'number') return ''
  if (v > 0.95) return 'metric-good'
  if (v > 0.8) return 'metric-mid'
  return 'metric-bad'
}

function formatNumber(v: any): string {
  if (v === null || v === undefined) return '—'
  if (typeof v !== 'number') return String(v)
  if (v === 0) return '0'
  if (Math.abs(v) >= 1000 || Math.abs(v) < 0.001) return v.toExponential(3)
  return v.toFixed(4)
}

function formatDuration(seconds: number): string {
  if (!seconds || seconds < 0) return '—'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  if (mins > 0) return `${mins}m ${secs}s`
  return `${secs}s`
}

onMounted(() => {
  if (queueTasks.value.length === 0) {
    generateMockQueue('sample.gds')
  }
})

onUnmounted(() => {
  stopPolling()
  try { (wsUnsubscribe as any)?.() } catch (e) {}
})
</script>

<style lang="scss" scoped>
.batch-queue-monitor {
  .card {
    border-radius: 10px;

    &.detail-card {
      min-height: auto;
    }
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;

    .title {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 16px;
      font-weight: 600;
      color: #303133;
    }

    .actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }
  }

  .divider-label {
    font-weight: 600;
    color: #303133;
    font-size: 13px;
  }

  .w-full {
    width: 100% !important;
  }

  .form-actions {
    margin-top: 20px;
    display: flex;
    justify-content: center;

    .run-btn {
      min-width: 200px;
    }
  }

  .stats-row {
    margin-bottom: 20px;

    .stat-card {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 14px 16px;
      border-radius: 8px;
      background: #f5f7fa;

      .stat-icon {
        width: 40px;
        height: 40px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
      }

      .stat-info {
        display: flex;
        flex-direction: column;

        .stat-value {
          font-size: 20px;
          font-weight: 700;
          font-family: 'SF Mono', Consolas, monospace;
          line-height: 1.2;
        }

        .stat-label {
          font-size: 12px;
          color: #909399;
          margin-top: 2px;
        }
      }

      &.stat-pending {
        .stat-icon { background: #ecf5ff; color: #409eff; }
        .stat-value { color: #409eff; }
      }
      &.stat-running {
        .stat-icon { background: #fdf6ec; color: #e6a23c; }
        .stat-value { color: #e6a23c; }
      }
      &.stat-completed {
        .stat-icon { background: #f0f9eb; color: #67c23a; }
        .stat-value { color: #67c23a; }
      }
      &.stat-failed {
        .stat-icon { background: #fef0f0; color: #f56c6c; }
        .stat-value { color: #f56c6c; }
      }
    }
  }

  .progress-section {
    margin-bottom: 16px;
    padding: 12px 16px;
    background: #fafafa;
    border-radius: 8px;

    .progress-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
      font-size: 13px;
      color: #606266;
    }
  }

  .cell-name {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-family: 'SF Mono', Consolas, monospace;
  }

  .queue-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
    color: #909399;

    p {
      margin: 8px 0 0 0;
      font-size: 14px;

      &.hint {
        font-size: 12px;
        color: #c0c4cc;
      }
    }
  }

  .metric-good {
    color: #67c23a;
    font-weight: 500;
  }
  .metric-mid {
    color: #e6a23c;
    font-weight: 500;
  }
  .metric-bad {
    color: #f56c6c;
    font-weight: 500;
  }

  :deep(.el-tabs__content) {
    padding-top: 8px;
  }
}
</style>
