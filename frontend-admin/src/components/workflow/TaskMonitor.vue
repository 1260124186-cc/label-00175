<template>
  <div class="task-monitor">
    <el-card class="card" shadow="hover">
      <div class="card-header">
        <div class="title">
          <el-icon size="18" color="#67c23a"><List /></el-icon>
          <span>任务列表</span>
          <el-tag size="small" type="info" effect="plain">{{ workflowLabel }}</el-tag>
        </div>
        <div class="actions">
          <el-badge :value="runningCount" :hidden="runningCount === 0" class="badge">
            <el-tag size="small" type="primary" effect="light" round>
              共 {{ taskList.length }} 个
            </el-tag>
          </el-badge>
          <el-button type="primary" link size="small" @click="fetchTasks" :loading="loading">
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </div>

      <el-table
        :data="taskList"
        v-loading="loading"
        style="width: 100%"
        size="small"
        empty-text="暂无任务"
        stripe
        height="280"
      >
        <el-table-column prop="task_id" label="任务 ID" width="110" align="center">
          <template #default="{ row }">
            <code class="task-id">{{ row.task_id.slice(0, 8) }}</code>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="70" align="center">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)" size="small" effect="dark">
              {{ statusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="进度" min-width="100">
          <template #default="{ row }">
            <el-progress
              :percentage="Math.round(row.progress)"
              :stroke-width="6"
              :status="progressStatus(row.status)"
              :text-inside="row.progress === 100"
            />
          </template>
        </el-table-column>
        <el-table-column label="操作" width="60" align="center" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link size="small" @click="viewTask(row)">
              详情
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-card v-if="currentTask" class="card card-detail" shadow="hover">
      <div class="card-header">
        <div class="title">
          <el-icon size="18" color="#e6a23c"><DataBoard /></el-icon>
          <span>任务详情</span>
          <code class="task-id-small">{{ currentTask.task_id }}</code>
        </div>
        <div class="ws-status" v-if="['starting', 'running', 'pending'].includes(currentTask.status)">
          <el-tag :type="wsConnected ? 'success' : 'info'" size="small" effect="light">
            <el-icon size="12"><Connection /></el-icon>
            {{ wsConnected ? '实时连接' : '连接中...' }}
          </el-tag>
        </div>
      </div>

      <el-descriptions :column="1" border size="small" class="desc-base">
        <el-descriptions-item label="状态">
          <el-tag :type="statusType(currentTask.status)" size="small" effect="dark">
            {{ statusLabel(currentTask.status) }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="进度">
          <el-progress
            :percentage="Math.round(currentTask.progress)"
            :stroke-width="8"
            :status="progressStatus(currentTask.status)"
          />
        </el-descriptions-item>
        <el-descriptions-item label="当前阶段" v-if="currentStage || currentTask.stage">
          <span class="stage-label">{{ currentStage || currentTask.stage }}</span>
        </el-descriptions-item>
        <el-descriptions-item label="迭代次数" v-if="currentIteration !== null || currentTask.iteration !== undefined">
          <span class="iteration-label">第 {{ currentIteration ?? currentTask.iteration }} 次</span>
        </el-descriptions-item>
        <el-descriptions-item label="当前损失" v-if="currentLoss !== null || currentTask.current_loss !== undefined">
          <span class="loss-value">{{ formatNumber(currentLoss ?? currentTask.current_loss) }}</span>
        </el-descriptions-item>
      </el-descriptions>

      <div v-if="maskThumbnail" class="mask-thumbnail-section">
        <div class="section-title">
          <el-icon size="14"><Picture /></el-icon>
          <span>当前掩模预览</span>
        </div>
        <div class="mask-thumbnail-wrapper">
          <img :src="'data:image/png;base64,' + maskThumbnail" alt="Mask" class="mask-thumbnail" />
        </div>
      </div>

      <el-alert
        v-if="currentTask.error"
        :title="currentTask.error"
        type="error"
        :closable="false"
        show-icon
        style="margin-top: 12px"
      />

      <template v-if="currentTask.result_summary">
        <el-descriptions :column="1" border size="small" style="margin-top: 12px" class="desc-result">
          <el-descriptions-item
            v-for="(val, key) in currentTask.result_summary"
            :key="key"
            :label="metricLabel(String(key))"
          >
            <span :class="metricClass(String(key), val)">{{ formatNumber(val) }}</span>
          </el-descriptions-item>
        </el-descriptions>
      </template>

      <div class="detail-actions" v-if="currentTask.status === 'completed'">
        <el-button size="small" type="primary" plain @click="downloadResult">
          <el-icon><Download /></el-icon>
          下载结果
        </el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import {
  List, Refresh, DataBoard, Connection, Picture, Download
} from '@element-plus/icons-vue'
import { taskApi } from '@/api'
import taskWs from '@/api/websocket'
import type { WorkflowTask, WorkflowType } from '@/types/workflow'

interface Props {
  taskType: WorkflowType
}

const props = defineProps<Props>()

const emit = defineEmits<{
  (e: 'task-complete', taskId: string): void
  (e: 'task-failed', taskId: string): void
}>()

const loading = ref(false)
const taskList = ref<WorkflowTask[]>([])
const currentTask = ref<WorkflowTask | null>(null)
const wsConnected = ref(false)
const currentLoss = ref<number | null>(null)
const currentIteration = ref<number | null>(null)
const currentStage = ref('')
const maskThumbnail = ref('')

let wsUnsubscribe: (() => void) | null = null

const workflowLabel = computed(() => {
  const labels: Record<string, string> = {
    opc: 'OPC',
    smo: 'SMO',
    ilt: 'ILT',
    process_window: '工艺窗口',
    batch: '批处理',
    simulation: '仿真',
  }
  return labels[props.taskType] || props.taskType
})

const runningCount = computed(() =>
  taskList.value.filter(t => ['starting', 'running'].includes(t.status)).length
)

onMounted(() => {
  fetchTasks()
})

onUnmounted(() => {
  disconnectWebSocket()
})

async function fetchTasks() {
  loading.value = true
  try {
    const res: any = await taskApi.list(props.taskType)
    taskList.value = res.tasks || []
  } catch (e) {
    taskList.value = []
  } finally {
    loading.value = false
  }
}

function viewTask(task: WorkflowTask) {
  currentTask.value = JSON.parse(JSON.stringify(task))
  currentStage.value = task.stage || ''
  currentLoss.value = task.current_loss ?? null
  currentIteration.value = task.iteration ?? null
  maskThumbnail.value = ''

  if (['starting', 'running', 'pending'].includes(task.status)) {
    connectWebSocket(task.task_id)
  } else {
    disconnectWebSocket()
  }
}

function connectWebSocket(taskId: string) {
  disconnectWebSocket()

  try {
    taskWs.connect(taskId)
    wsUnsubscribe = taskWs.onMessage(handleWebSocketMessage)
    wsConnected.value = taskWs.isConnected.value
  } catch (e) {
    console.error('WebSocket 连接失败:', e)
  }
}

function disconnectWebSocket() {
  if (wsUnsubscribe) {
    try {
      wsUnsubscribe()
    } catch (e) {}
    wsUnsubscribe = null
  }
  wsConnected.value = false
  currentLoss.value = null
  currentIteration.value = null
  currentStage.value = ''
  maskThumbnail.value = ''
}

function handleWebSocketMessage(msg: any) {
  const taskId = msg.task_id

  if (msg.type === 'progress') {
    const updateData: any = {}
    if (msg.progress !== undefined) updateData.progress = msg.progress
    if (msg.message !== undefined) updateData.message = msg.message
    if (msg.stage !== undefined) updateData.stage = msg.stage
    if (msg.loss !== undefined) updateData.current_loss = msg.loss
    if (msg.iteration !== undefined) updateData.iteration = msg.iteration

    const idx = taskList.value.findIndex(t => t.task_id === taskId)
    if (idx >= 0) {
      taskList.value[idx] = { ...taskList.value[idx], ...updateData }
    }

    if (currentTask.value?.task_id === taskId) {
      currentTask.value = { ...currentTask.value, ...updateData }
      if (msg.loss !== undefined) currentLoss.value = msg.loss
      if (msg.iteration !== undefined) currentIteration.value = msg.iteration
      if (msg.stage !== undefined) currentStage.value = msg.stage
      if (msg.mask_thumbnail !== undefined) maskThumbnail.value = msg.mask_thumbnail
    }
  } else if (msg.type === 'task_complete') {
    const idx = taskList.value.findIndex(t => t.task_id === taskId)
    if (idx >= 0) {
      taskList.value[idx].status = 'completed'
      taskList.value[idx].progress = 100
      taskList.value[idx].result_summary = msg.result
    }
    if (currentTask.value && currentTask.value.task_id === taskId) {
      currentTask.value.status = 'completed'
      currentTask.value.progress = 100
      currentTask.value.result_summary = msg.result
    }
    ElMessage.success(`任务 ${taskId.slice(0, 8)} 完成 ✅`)
    disconnectWebSocket()
    fetchTasks()
    emit('task-complete', taskId)
  } else if (msg.type === 'task_failed') {
    const idx = taskList.value.findIndex(t => t.task_id === taskId)
    if (idx >= 0) {
      taskList.value[idx].status = 'failed'
      taskList.value[idx].error = msg.error
    }
    if (currentTask.value && currentTask.value.task_id === taskId) {
      currentTask.value.status = 'failed'
      currentTask.value.error = msg.error
    }
    ElMessage.error(`任务 ${taskId.slice(0, 8)} 失败`)
    disconnectWebSocket()
    fetchTasks()
    emit('task-failed', taskId)
  } else if (msg.type === 'stage_change') {
    const idx = taskList.value.findIndex(t => t.task_id === taskId)
    if (idx >= 0 && msg.stage !== undefined) {
      taskList.value[idx].stage = msg.stage
    }
    if (currentTask.value && currentTask.value.task_id === taskId && msg.stage !== undefined) {
      currentTask.value.stage = msg.stage
      currentStage.value = msg.stage
    }
  } else if (msg.type === 'connected') {
    wsConnected.value = true
  }
}

async function downloadResult() {
  if (!currentTask.value) return
  try {
    const res: any = await taskApi.download(currentTask.value.task_id)
    const blob = new Blob([res], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `task_${currentTask.value.task_id}.json`
    a.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    ElMessage.error('下载失败')
  }
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

function progressStatus(s: string): '' | 'success' | 'warning' | 'danger' | 'exception' {
  switch (s) {
    case 'completed': return 'success'
    case 'failed':    return 'exception'
    case 'starting':  return 'warning'
    default:          return ''
  }
}

function statusLabel(s: string) {
  const m: Record<string, string> = {
    pending: '等待中',
    starting: '启动中',
    running: '运行中',
    completed: '已完成',
    failed: '失败'
  }
  return m[s] || s
}

function metricLabel(k: string) {
  const m: Record<string, string> = {
    mse: 'MSE 均方误差',
    ssim: 'SSIM 结构相似度',
    mae: 'MAE 平均绝对误差',
    psnr: 'PSNR 峰值信噪比',
    iterations: '迭代次数',
    final_loss: '最终损失',
    hotspots_detected: '检测热点数',
    hotspots_remaining: '剩余热点数',
    mask_shape: '掩模尺寸',
    source_shape: '光源尺寸',
    transmission_levels: '透射率等级',
    total: '总任务数',
    succeeded: '成功数',
    failed: '失败数',
    skipped: '跳过数',
    avg_mse: '平均 MSE',
    avg_ssim: '平均 SSIM',
    elapsed_seconds: '耗时(s)',
    max_exposure_latitude: '最大曝光宽容度',
    depth_of_focus: '焦深',
    process_window_area: '工艺窗口面积',
    nominal_cd: '标称 CD',
    cd_uniformity: 'CD 均匀性',
    focus_points: '焦点数',
    dose_points: '剂量点',
  }
  return m[k] || k
}

function metricClass(k: string, v: any) {
  if (typeof v !== 'number') return ''
  if (k === 'ssim' || k === 'avg_ssim') {
    if (v > 0.95) return 'metric-good'
    if (v > 0.8) return 'metric-mid'
    return 'metric-bad'
  }
  if (k === 'mse' || k === 'mae' || k === 'avg_mse' || k === 'final_loss') {
    if (v < 0.01) return 'metric-good'
    if (v < 0.1) return 'metric-mid'
    return 'metric-bad'
  }
  return ''
}

function formatNumber(v: any): string {
  if (v === null || v === undefined) return '—'
  if (typeof v !== 'number') return String(v)
  if (v === 0) return '0'
  if (Math.abs(v) >= 1000 || Math.abs(v) < 0.001) return v.toExponential(4)
  return v.toFixed(6)
}

watch(() => props.taskType, () => {
  fetchTasks()
})

defineExpose({
  fetchTasks,
  viewTask,
})
</script>

<style lang="scss" scoped>
.task-monitor {
  display: flex;
  flex-direction: column;
  gap: 16px;

  .card {
    border-radius: 10px;

    &.card-detail {
      min-height: auto;
    }
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;

    .title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 600;
      font-size: 14px;
      color: #303133;
    }

    .actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .badge {
      margin-right: 4px;
    }
  }

  .task-id {
    font-size: 11px;
    font-family: 'SF Mono', Consolas, monospace;
    padding: 2px 4px;
    background: #f4f4f5;
    border-radius: 3px;
    color: #606266;
  }

  .task-id-small {
    font-size: 11px;
    font-family: 'SF Mono', Consolas, monospace;
    padding: 2px 6px;
    background: #fdf6ec;
    color: #b88230;
    border-radius: 3px;
    font-weight: 400;
  }

  .ws-status {
    :deep(.el-tag) {
      display: flex;
      align-items: center;
      gap: 4px;
    }
  }

  .desc-base { margin-top: 4px; }

  .stage-label {
    font-weight: 500;
    color: #409eff;
  }

  .iteration-label {
    font-family: 'SF Mono', Consolas, monospace;
    font-weight: 600;
    color: #606266;
  }

  .loss-value {
    font-family: 'SF Mono', Consolas, monospace;
    font-size: 14px;
    font-weight: 600;
    color: #e6a23c;
  }

  .mask-thumbnail-section {
    margin-top: 12px;

    .section-title {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 600;
      color: #606266;
      margin-bottom: 8px;
    }

    .mask-thumbnail-wrapper {
      display: flex;
      justify-content: center;
      padding: 10px;
      background: #f5f7fa;
      border-radius: 6px;
      border: 1px solid #e4e7ed;
    }

    .mask-thumbnail {
      max-width: 96px;
      max-height: 96px;
      image-rendering: pixelated;
      border: 1px solid #dcdfe6;
      border-radius: 4px;
      background: #fff;
    }
  }

  .detail-actions {
    margin-top: 12px;
    display: flex;
    justify-content: flex-end;
  }

  .metric-good {
    color: #67c23a;
    font-weight: 600;
  }
  .metric-mid {
    color: #e6a23c;
    font-weight: 600;
  }
  .metric-bad {
    color: #f56c6c;
    font-weight: 600;
  }
}
</style>
