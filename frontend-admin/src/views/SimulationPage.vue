<template>
  <div class="simulation-page">
    <el-row :gutter="20">
      <!-- ============ 左侧：参数表单 ============ -->
      <el-col :xs="24" :lg="12">
        <el-card class="card card-left" shadow="hover">
          <div class="card-header">
            <div class="title">
              <el-icon size="20" color="#409eff"><Cpu /></el-icon>
              <span>运行光刻仿真</span>
            </div>
            <el-tag type="primary" effect="plain" round size="small">实时计算</el-tag>
          </div>

          <!-- 主表单：label-width 100px -->
          <el-form :model="runForm" label-width="100px" label-position="right" size="default">

            <!-- 图案类型 -->
            <el-form-item label="图案类型" required>
              <el-select v-model="runForm.pattern_type" style="width: 100%">
                <el-option label="矩形 (Rectangle)" value="rectangle" />
                <el-option label="十字 (Cross)" value="cross" />
                <el-option label="L 型 (L-shape)" value="l_shape" />
                <el-option label="阵列 (Array)" value="array" />
                <el-option label="随机测试 (Random)" value="random" />
              </el-select>
            </el-form-item>

            <el-divider content-position="left">
              <span class="divider-label">图案尺寸参数</span>
            </el-divider>

            <!-- 图像尺寸：label-width 100px -->
            <el-row :gutter="24">
              <el-col :span="12">
                <el-form-item label="图像高度" required>
                  <el-input-number
                    v-model="runForm.params.size[0]"
                    :min="16"
                    :max="1024"
                    :step="8"
                    controls-position="right"
                    class="w-full"
                  />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="图像宽度" required>
                  <el-input-number
                    v-model="runForm.params.size[1]"
                    :min="16"
                    :max="1024"
                    :step="8"
                    controls-position="right"
                    class="w-full"
                  />
                </el-form-item>
              </el-col>
            </el-row>

            <!-- 矩形坐标参数：独立 form，更短的 label-width；2 行 × 2 列 -->
            <template v-if="runForm.pattern_type === 'rectangle'">
              <el-divider content-position="left">
                <span class="divider-label">矩形坐标（像素）</span>
              </el-divider>

              <el-form
                :model="runForm.params"
                label-width="72px"
                label-position="right"
                class="rect-form"
                size="default"
                inline-message
              >
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="X 起始" prop="x_start"
                      :rules="[{ required: true, message: '必填', trigger: 'blur' }]">
                      <el-input-number
                        v-model="runForm.params.x_start"
                        :min="0"
                        :max="runForm.params.size[1]"
                        :step="1"
                        :controls="false"
                        class="w-full input-block"
                      />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="X 结束" prop="x_end"
                      :rules="[{ required: true, message: '必填', trigger: 'blur' },
                               { type: 'number', min: 0, max: runForm.params.size[1],
                                 message: `必须 ≤ ${runForm.params.size[1]}` }]">
                      <el-input-number
                        v-model="runForm.params.x_end"
                        :min="0"
                        :max="runForm.params.size[1]"
                        :step="1"
                        :controls="false"
                        class="w-full input-block"
                      />
                    </el-form-item>
                  </el-col>
                </el-row>

                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="Y 起始" prop="y_start"
                      :rules="[{ required: true, message: '必填', trigger: 'blur' }]">
                      <el-input-number
                        v-model="runForm.params.y_start"
                        :min="0"
                        :max="runForm.params.size[0]"
                        :step="1"
                        :controls="false"
                        class="w-full input-block"
                      />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="Y 结束" prop="y_end"
                      :rules="[{ required: true, message: '必填', trigger: 'blur' },
                               { type: 'number', min: 0, max: runForm.params.size[0],
                                 message: `必须 ≤ ${runForm.params.size[0]}` }]">
                      <el-input-number
                        v-model="runForm.params.y_end"
                        :min="0"
                        :max="runForm.params.size[0]"
                        :step="1"
                        :controls="false"
                        class="w-full input-block"
                      />
                    </el-form-item>
                  </el-col>
                </el-row>

                <div class="coord-hint">
                  <el-icon size="13"><InfoFilled /></el-icon>
                  <span>
                    坐标范围将随图像尺寸自动调整：
                    X ∈ [0, {{ runForm.params.size[1] }}] &nbsp;·&nbsp;
                    Y ∈ [0, {{ runForm.params.size[0] }}]
                  </span>
                </div>
              </el-form>
            </template>

            <el-alert
              title="仿真使用「参数配置」页面中设置的光学系统、损失权重及正则化参数"
              type="info"
              :closable="false"
              show-icon
              style="margin: 12px 0 20px"
            />

            <!-- 操作按钮 -->
            <div class="form-actions">
              <el-button
                type="primary"
                :icon="VideoPlay"
                :loading="isRunning"
                size="default"
                @click="handleRun"
                class="run-btn"
              >
                开始仿真
              </el-button>
              <el-button
                :icon="RefreshRight"
                @click="fetchTasks"
                size="default"
                plain
              >
                刷新任务
              </el-button>
              <span class="run-hint" v-if="!configStore.config?.optical_system">
                <el-icon size="13"><Warning /></el-icon>
                请先到「参数配置」页加载配置
              </span>
            </div>
          </el-form>
        </el-card>
      </el-col>

      <!-- ============ 右侧：任务列表 + 详情 ============ -->
      <el-col :xs="24" :lg="12">
        <el-card class="card" shadow="hover">
          <div class="card-header">
            <div class="title">
              <el-icon size="20" color="#67c23a"><List /></el-icon>
              <span>仿真任务列表</span>
            </div>
            <el-badge :value="runningCount" :hidden="runningCount === 0" class="badge">
              <el-tag type="primary" effect="light" size="small" round>
                共 {{ taskList.length }} 个
              </el-tag>
            </el-badge>
          </div>

          <el-table
            :data="taskList"
            v-loading="taskLoading"
            style="width: 100%"
            empty-text="暂无任务，点击左侧「开始仿真」提交"
            stripe
            size="small"
          >
            <el-table-column prop="task_id" label="任务 ID" width="140" align="center">
              <template #default="{ row }">
                <code class="task-id">{{ row.task_id }}</code>
              </template>
            </el-table-column>
            <el-table-column prop="status" label="状态" width="100" align="center">
              <template #default="{ row }">
                <el-tag :type="statusType(row.status)" size="small" effect="dark">
                  {{ statusLabel(row.status) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="progress" label="进度" min-width="160">
              <template #default="{ row }">
                <el-progress
                  :percentage="row.progress"
                  :stroke-width="8"
                  :status="progressStatus(row.status)"
                  :text-inside="row.progress === 100"
                />
              </template>
            </el-table-column>
            <el-table-column label="操作" width="80" align="center" fixed="right">
              <template #default="{ row }">
                <el-button type="primary" link size="small" @click="viewTask(row)">
                  查看
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>

        <!-- 任务详情 -->
        <el-card class="card card-detail" v-if="currentTask" shadow="hover">
          <div class="card-header">
            <div class="title">
              <el-icon size="20" color="#e6a23c"><DataBoard /></el-icon>
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

          <el-descriptions :column="2" border size="small" class="desc-base">
            <el-descriptions-item label="任务 ID" :span="1">
              <code>{{ currentTask.task_id }}</code>
            </el-descriptions-item>
            <el-descriptions-item label="状态" :span="1">
              <el-tag :type="statusType(currentTask.status)" size="small" effect="dark">
                {{ statusLabel(currentTask.status) }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="进度" :span="2">
              <el-progress
                :percentage="currentTask.progress"
                :stroke-width="10"
                :status="progressStatus(currentTask.status)"
              />
            </el-descriptions-item>
            <el-descriptions-item label="当前阶段" v-if="currentStage || currentTask.stage">
              <span class="stage-label">{{ currentStage || currentTask.stage }}</span>
            </el-descriptions-item>
            <el-descriptions-item label="迭代次数" v-if="currentIteration !== null || currentTask.iteration !== undefined">
              <span class="iteration-label">
                第 {{ currentIteration ?? currentTask.iteration }} 次
              </span>
            </el-descriptions-item>
            <el-descriptions-item label="当前损失" v-if="currentLoss !== null || currentTask.current_loss !== undefined" :span="2">
              <span class="loss-value">
                {{ formatNumber(currentLoss ?? currentTask.current_loss) }}
              </span>
            </el-descriptions-item>
          </el-descriptions>

          <!-- 掩模缩略图 -->
          <div v-if="maskThumbnail" class="mask-thumbnail-section">
            <div class="section-title">
              <el-icon size="14"><Picture /></el-icon>
              <span>当前掩模预览</span>
            </div>
            <div class="mask-thumbnail-wrapper">
              <img
                :src="'data:image/png;base64,' + maskThumbnail"
                alt="Mask Thumbnail"
                class="mask-thumbnail"
              />
            </div>
          </div>

          <!-- 错误 -->
          <el-alert
            v-if="currentTask.error"
            :title="currentTask.error"
            type="error"
            :closable="false"
            show-icon
            style="margin-top: 12px"
          />

          <!-- 成功结果 -->
          <template v-if="currentTask.result">
            <el-descriptions
              :column="2"
              border
              size="small"
              style="margin-top: 12px"
              class="desc-result"
            >
              <el-descriptions-item
                v-for="(val, key) in currentTask.result.initial_metrics"
                :key="key"
                :label="`初始指标 · ${metricLabel(String(key))}`"
              >
                <span :class="metricClass(String(key), val)">{{ formatNumber(val) }}</span>
              </el-descriptions-item>
              <el-descriptions-item label="图案类型" v-if="currentTask.result.pattern_type">
                {{ patternTypeLabel(currentTask.result.pattern_type) }}
              </el-descriptions-item>
              <el-descriptions-item label="图案尺寸" v-if="currentTask.result.pattern_size">
                {{ currentTask.result.pattern_size.join(' × ') }} px
              </el-descriptions-item>
              <el-descriptions-item label="晶圆图像尺寸" v-if="currentTask.result.wafer_image_shape">
                {{ currentTask.result.wafer_image_shape.join(' × ') }} px
              </el-descriptions-item>
            </el-descriptions>
          </template>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import {
  VideoPlay, RefreshRight, List, DataBoard,
  InfoFilled, Warning, Connection, Picture
} from '@element-plus/icons-vue'
import { useConfigStore } from '@/stores/config'
import { simulationApi } from '@/api'
import taskWs, { type TaskProgressMessage } from '@/api/websocket'

const configStore = useConfigStore()

const isRunning = ref(false)
const taskLoading = ref(false)
const taskList = ref<any[]>([])
const currentTask = ref<any>(null)
const wsConnected = ref(false)
const currentLoss = ref<number | null>(null)
const currentIteration = ref<number | null>(null)
const currentStage = ref('')
const maskThumbnail = ref('')

let wsUnsubscribe: (() => void) | null = null

const runningCount = computed(() =>
  taskList.value.filter(t => ['starting', 'running'].includes(t.status)).length
)

const runForm = reactive({
  pattern_type: 'rectangle',
  params: {
    size: [64, 64] as [number, number],
    x_start: 20,
    x_end: 44,
    y_start: 20,
    y_end: 44
  }
})

onMounted(() => {
  fetchTasks()
  if (configStore.loading === false && !configStore.config?.optical_system) {
    configStore.loadDefault()
  }
})

async function handleRun() {
  if (!configStore.config?.optical_system) {
    ElMessage.warning('配置未加载，正在从服务器加载默认配置...')
    await configStore.loadDefault()
  }
  isRunning.value = true
  try {
    const patternParams = {
      size: runForm.params.size,
      x_start: runForm.params.x_start,
      x_end: runForm.params.x_end,
      y_start: runForm.params.y_start,
      y_end: runForm.params.y_end
    }
    const res: any = await simulationApi.run(
      configStore.config,
      runForm.pattern_type,
      patternParams
    )
    if (res.success) {
      ElMessage.success(`仿真任务已提交：${res.task_id}`)
      await fetchTasks()
      // 自动打开详情
      const t = taskList.value.find(x => x.task_id === res.task_id)
      if (t) viewTask(t)
    } else {
      ElMessage.error(res.message || '提交失败')
    }
  } catch (e: any) {
    ElMessage.error(e?.message || '提交失败，请检查后端服务是否启动')
  } finally {
    isRunning.value = false
  }
}

async function fetchTasks() {
  taskLoading.value = true
  try {
    const res: any = await simulationApi.listTasks()
    taskList.value = res.tasks || []
  } catch (e) {
    taskList.value = []
  } finally {
    taskLoading.value = false
  }
}

function handleWebSocketMessage(msg: TaskProgressMessage) {
  const taskId = msg.task_id

  // 更新任务列表中的任务
  const idx = taskList.value.findIndex((t) => t.task_id === taskId)

  if (msg.type === 'progress') {
    const updateData: any = {}
    if (msg.progress !== undefined) updateData.progress = msg.progress
    if (msg.message !== undefined) updateData.message = msg.message
    if (msg.stage !== undefined) updateData.stage = msg.stage
    if (msg.loss !== undefined) updateData.current_loss = msg.loss
    if (msg.iteration !== undefined) updateData.iteration = msg.iteration

    if (idx >= 0) {
      taskList.value[idx] = { ...taskList.value[idx], ...updateData }
    }

    // 更新当前查看的任务
    if (currentTask.value?.task_id === taskId) {
      currentTask.value = { ...currentTask.value, ...updateData }
      if (msg.loss !== undefined) currentLoss.value = msg.loss
      if (msg.iteration !== undefined) currentIteration.value = msg.iteration
      if (msg.stage !== undefined) currentStage.value = msg.stage
      if (msg.mask_thumbnail !== undefined) maskThumbnail.value = msg.mask_thumbnail
    }
  } else if (msg.type === 'task_complete') {
    if (idx >= 0) {
      taskList.value[idx].status = 'completed'
      taskList.value[idx].progress = 100
      taskList.value[idx].result_summary = msg.result
    }
    if (currentTask.value?.task_id === taskId) {
      currentTask.value.status = 'completed'
      currentTask.value.progress = 100
      currentTask.value.result = msg.result
      currentTask.value.result_summary = msg.result
    }
    ElMessage.success(`任务 ${taskId} 完成 ✅`)
    disconnectWebSocket()
    fetchTasks()
  } else if (msg.type === 'task_failed') {
    if (idx >= 0) {
      taskList.value[idx].status = 'failed'
      taskList.value[idx].error = msg.error
    }
    if (currentTask.value?.task_id === taskId) {
      currentTask.value.status = 'failed'
      currentTask.value.error = msg.error
    }
    ElMessage.error(`任务 ${taskId} 失败：${msg.error || '未知错误'}`)
    disconnectWebSocket()
    fetchTasks()
  } else if (msg.type === 'stage_change') {
    if (idx >= 0 && msg.stage !== undefined) {
      taskList.value[idx].stage = msg.stage
      if (msg.message !== undefined) {
        taskList.value[idx].message = msg.message
      }
    }
    if (currentTask.value?.task_id === taskId) {
      if (msg.stage !== undefined) {
        currentTask.value.stage = msg.stage
        currentStage.value = msg.stage
      }
      if (msg.message !== undefined) {
        currentTask.value.message = msg.message
      }
    }
  } else if (msg.type === 'connected') {
    wsConnected.value = true
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

function viewTask(task: any) {
  currentTask.value = JSON.parse(JSON.stringify(task))

  // 更新详情页的显示数据
  currentStage.value = task.stage || ''
  currentLoss.value = task.current_loss ?? null
  currentIteration.value = task.iteration ?? null
  maskThumbnail.value = ''

  // 如果任务正在运行，建立 WebSocket 连接
  if (['starting', 'running', 'pending'].includes(task.status)) {
    connectWebSocket(task.task_id)
  } else {
    disconnectWebSocket()
  }
}

onUnmounted(() => {
  disconnectWebSocket()
})

function statusType(s: string) {
  switch (s) {
    case 'completed': return 'success'
    case 'running':   return 'primary'
    case 'failed':    return 'danger'
    case 'starting':  return 'warning'
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
    psnr: 'PSNR 峰值信噪比'
  }
  return m[k] || k
}

function metricClass(k: string, v: any) {
  if (typeof v !== 'number') return ''
  if (k === 'ssim') {
    if (v > 0.95) return 'metric-good'
    if (v > 0.8) return 'metric-mid'
    return 'metric-bad'
  }
  if (v < 0.01) return 'metric-good'
  if (v < 0.1) return 'metric-mid'
  return 'metric-bad'
}

function patternTypeLabel(t: string) {
  const m: Record<string, string> = {
    rectangle: '矩形',
    cross: '十字',
    l_shape: 'L 型',
    array: '阵列',
    random: '随机'
  }
  return m[t] || t
}

function formatNumber(v: any) {
  if (v === null || v === undefined) return '—'
  if (typeof v !== 'number') return String(v)
  if (v === 0) return '0'
  if (Math.abs(v) >= 1000 || Math.abs(v) < 0.001) return v.toExponential(4)
  return v.toFixed(6)
}
</script>

<style lang="scss" scoped>
.simulation-page {
  .card {
    border-radius: 10px;
    min-height: 420px;

    &.card-left {
      min-height: 560px;
    }
    &.card-detail {
      margin-top: 20px;
      min-height: auto;
    }
  }

  .divider-label {
    font-weight: 600;
    color: #303133;
    font-size: 13px;
  }

  // 输入框统一宽度
  .w-full {
    width: 100% !important;
  }

  // 矩形坐标的数字输入：彻底移除按钮 + 100% 宽度，防止控件挤压变形
  .rect-form {
    :deep(.el-input-number) {
      width: 100% !important;

      // 移除右侧上下箭头按钮，确保输入框在窄空间内不挤压
      .el-input-number__increase,
      .el-input-number__decrease {
        display: none;
      }
      .el-input__wrapper {
        padding-right: 11px !important;  // 因为去掉了按钮，padding 对称
      }
    }
  }

  .coord-hint {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    margin-bottom: 12px;
    background: #f4f8ff;
    border: 1px dashed #b3d8ff;
    border-radius: 6px;
    color: #606266;
    font-size: 12px;
    line-height: 1.5;
  }

  .form-actions {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    padding-top: 8px;

    .run-btn {
      min-width: 140px;
    }
    .run-hint {
      display: flex;
      align-items: center;
      gap: 4px;
      font-size: 12px;
      color: #e6a23c;
    }
  }

  .badge {
    margin-right: 8px;
  }

  .task-id {
    font-size: 12px;
    font-family: 'SF Mono', Consolas, monospace;
    padding: 2px 6px;
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
    margin-left: 10px;
    font-weight: 400;
  }

  .desc-base { margin-top: 4px; }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .ws-status {
    :deep(.el-tag) {
      display: flex;
      align-items: center;
      gap: 4px;
    }
  }

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
    margin-top: 16px;

    .section-title {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      font-weight: 600;
      color: #606266;
      margin-bottom: 10px;
    }

    .mask-thumbnail-wrapper {
      display: flex;
      justify-content: center;
      padding: 12px;
      background: #f5f7fa;
      border-radius: 8px;
      border: 1px solid #e4e7ed;
    }

    .mask-thumbnail {
      max-width: 128px;
      max-height: 128px;
      image-rendering: pixelated;
      border: 1px solid #dcdfe6;
      border-radius: 4px;
      background: #fff;
    }
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

// 响应式：小屏时取消 label 右对齐，改为顶部
@media (max-width: 992px) {
  .simulation-page :deep(.el-form) {
    label-width: 88px !important;
  }
  .simulation-page :deep(.rect-form .el-form) {
    label-width: 64px !important;
  }
}
</style>
