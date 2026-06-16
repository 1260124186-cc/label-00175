<template>
  <div class="bossung-chart-page">
    <el-row :gutter="20">
      <el-col :xs="24" :lg="8">
        <el-card class="card" shadow="hover">
          <div class="card-header">
            <div class="title">
              <el-icon size="20" color="#909399"><Odometer /></el-icon>
              <span>工艺窗口分析</span>
            </div>
            <el-tag type="info" effect="plain" round size="small">Process Window</el-tag>
          </div>

          <el-alert
            title="通过扫描离焦量和曝光剂量，分析工艺窗口和 Bossung 曲线"
            type="info"
            :closable="false"
            show-icon
            style="margin-bottom: 16px"
          />

          <el-form label-width="120px" label-position="right">
            <el-form-item label="图案类型">
              <el-select v-model="patternType" style="width: 100%">
                <el-option label="矩形 (Rectangle)" value="rectangle" />
                <el-option label="十字 (Cross)" value="cross" />
                <el-option label="L 型 (L-shape)" value="l_shape" />
                <el-option label="阵列 (Array)" value="array" />
              </el-select>
            </el-form-item>

            <el-divider content-position="left"><span class="divider-label">扫描范围</span></el-divider>

            <el-form-item label="离焦范围 (nm)">
              <div class="range-inputs">
                <el-input-number v-model="pwConfig.focus_range[0]" :min="-500" :max="0" :step="10" :precision="0" style="width: 100%" />
                <span class="range-sep">~</span>
                <el-input-number v-model="pwConfig.focus_range[1]" :min="0" :max="500" :step="10" :precision="0" style="width: 100%" />
              </div>
            </el-form-item>
            <el-form-item label="焦点数">
              <el-input-number v-model="pwConfig.focus_range[2]" :min="3" :max="51" :step="2" style="width: 100%" />
            </el-form-item>

            <el-form-item label="剂量范围">
              <div class="range-inputs">
                <el-input-number v-model="pwConfig.dose_range[0]" :min="0.5" :max="1" :step="0.05" :precision="2" style="width: 100%" />
                <span class="range-sep">~</span>
                <el-input-number v-model="pwConfig.dose_range[1]" :min="1" :max="2" :step="0.05" :precision="2" style="width: 100%" />
              </div>
            </el-form-item>
            <el-form-item label="剂量点数">
              <el-input-number v-model="pwConfig.dose_range[2]" :min="3" :max="51" :step="2" style="width: 100%" />
            </el-form-item>

            <el-divider content-position="left"><span class="divider-label">容差设置</span></el-divider>

            <el-form-item label="CD 相对容差">
              <el-slider v-model="pwConfig.cd_tolerance" :min="0.01" :max="0.3" :step="0.01" />
            </el-form-item>

            <el-form-item label="EPE 容差 (nm)">
              <el-input-number
                v-model="epeToleranceValue"
                :min="0"
                :step="0.5"
                :precision="1"
                style="width: 100%"
                placeholder="None (不检查)"
              />
            </el-form-item>

            <el-form-item label="光刻胶阈值">
              <el-slider v-model="pwConfig.threshold" :min="0.1" :max="0.9" :step="0.05" />
            </el-form-item>

            <el-form-item label="保存可视化">
              <el-switch v-model="pwConfig.save_visualizations" />
            </el-form-item>
          </el-form>

          <div class="form-actions">
            <el-button type="primary" size="large" :icon="VideoPlay" :loading="isRunning" @click="handleRun" class="run-btn">
              运行工艺窗口分析
            </el-button>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="16">
        <el-card class="card chart-card" shadow="hover">
          <div class="card-header">
            <div class="title">
              <el-icon size="20" color="#67c23a"><DataLine /></el-icon>
              <span>Bossung 图</span>
            </div>
            <div class="chart-actions">
              <el-tag v-if="taskId" type="success" effect="light" size="small">
                任务: {{ taskId.slice(0, 8) }}
              </el-tag>
              <el-button type="primary" link size="small" @click="refreshChart">
                <el-icon><Refresh /></el-icon>
                刷新
              </el-button>
            </div>
          </div>

          <div class="chart-container">
            <canvas ref="chartCanvas" class="bossung-canvas"></canvas>
            <div v-if="!hasData && !isRunning" class="chart-placeholder">
              <el-icon :size="64" color="#dcdfe6"><DataLine /></el-icon>
              <p>暂无数据，请运行工艺窗口分析</p>
            </div>
            <div v-if="isRunning" class="chart-loading">
              <el-icon class="is-loading" :size="48" color="#409eff"><Loading /></el-icon>
              <p>正在计算工艺窗口...</p>
            </div>
          </div>

          <div class="chart-legend">
            <div class="legend-item">
              <span class="legend-color process-window"></span>
              <span>工艺窗口</span>
            </div>
            <div class="legend-item">
              <span class="legend-color cd-contour"></span>
              <span>CD 等值线</span>
            </div>
            <div class="legend-item">
              <span class="legend-color nominal-point"></span>
              <span>标称工作点</span>
            </div>
          </div>
        </el-card>

        <el-card class="card metrics-card" shadow="hover">
          <div class="card-header">
            <div class="title">
              <el-icon size="18" color="#e6a23c"><Trophy /></el-icon>
              <span>工艺窗口指标</span>
            </div>
          </div>

          <el-row :gutter="16">
            <el-col :xs="12" :sm="8" v-for="(metric, key) in metrics" :key="key">
              <div class="metric-card">
                <div class="metric-label">{{ metric.label }}</div>
                <div class="metric-value" :class="metric.class">
                  {{ metric.value }}
                  <span class="metric-unit" v-if="metric.unit">{{ metric.unit }}</span>
                </div>
              </div>
            </el-col>
          </el-row>
        </el-card>

        <el-card v-if="taskId" class="card" shadow="hover" style="margin-top: 16px">
          <div class="card-header">
            <div class="title">
              <el-icon size="18" color="#409eff"><List /></el-icon>
              <span>任务状态</span>
            </div>
            <el-tag :type="statusType(currentTask?.status || '')" size="small" effect="dark">
              {{ statusLabel(currentTask?.status || '') }}
            </el-tag>
          </div>

          <el-progress
            v-if="currentTask"
            :percentage="Math.round(currentTask.progress)"
            :status="progressStatus(currentTask.status)"
            :stroke-width="8"
          />
          <p v-if="currentTask?.stage" class="task-stage">
            当前阶段: {{ currentTask.stage }}
          </p>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import {
  Odometer, VideoPlay, DataLine, Refresh, Trophy, List, Loading
} from '@element-plus/icons-vue'
import { useConfigStore } from '@/stores/config'
import { workflowApi, taskApi } from '@/api'
import taskWs from '@/api/websocket'
import type { ProcessWindowConfig, ProcessWindowMetrics, WorkflowTask, TaskResultResponse } from '@/types/workflow'

const configStore = useConfigStore()

const chartCanvas = ref<HTMLCanvasElement | null>(null)
const isRunning = ref(false)
const taskId = ref('')
const currentTask = ref<WorkflowTask | null>(null)
const bossungData = ref<any[][]>([])
const hasData = ref(false)

const patternType = ref('rectangle')

const defaultPwConfig: ProcessWindowConfig = {
  focus_range: [-150, 150, 11],
  dose_range: [0.85, 1.15, 11],
  cd_tolerance: 0.1,
  epe_tolerance: null,
  threshold: 0.3,
  save_visualizations: false,
}

const pwConfig = reactive<ProcessWindowConfig>({ ...defaultPwConfig })

const epeToleranceValue = computed({
  get: () => pwConfig.epe_tolerance ?? 0,
  set: (val: number) => {
    pwConfig.epe_tolerance = val <= 0 ? null : val
  },
})

const metrics = computed(() => {
  const summary = currentTask.value?.result_summary
  if (!summary) {
    return {
      el: { label: '曝光宽容度', value: '—', unit: '%', class: '' },
      dof: { label: '焦深 (DOF)', value: '—', unit: 'nm', class: '' },
      pwa: { label: '工艺窗口面积', value: '—', unit: '', class: '' },
      ncd: { label: '标称 CD', value: '—', unit: '', class: '' },
      cdu: { label: 'CD 均匀性', value: '—', unit: '%', class: '' },
      fp: { label: '扫描点数', value: '—', unit: '', class: '' },
    }
  }

  const fmt = (v: any, unit = '', decimals = 2) => {
    if (v === undefined || v === null) return '—'
    return typeof v === 'number' ? v.toFixed(decimals) : String(v)
  }

  return {
    el: { label: '曝光宽容度', value: fmt(summary.max_exposure_latitude, '%'), unit: '', class: 'metric-good' },
    dof: { label: '焦深 (DOF)', value: fmt(summary.depth_of_focus, 'nm'), unit: '', class: 'metric-good' },
    pwa: { label: '工艺窗口面积', value: fmt(summary.process_window_area, ''), unit: '', class: '' },
    ncd: { label: '标称 CD', value: fmt(summary.nominal_cd, ''), unit: '', class: '' },
    cdu: { label: 'CD 均匀性', value: fmt(summary.cd_uniformity, '%'), unit: '', class: 'metric-mid' },
    fp: { label: '扫描点数', value: `${summary.focus_points || 0} × ${summary.dose_points || 0}`, unit: '', class: '' },
  }
})

let wsUnsubscribe: (() => void) | null = null

async function handleRun() {
  const config = (configStore as any).config
  if (!config?.optical_system) {
    ElMessage.warning('请先到「参数配置」页加载光学系统配置')
    return
  }

  isRunning.value = true
  hasData.value = false
  try {
    const patternParams = {
      size: [64, 64],
      x_start: 20,
      x_end: 44,
      y_start: 20,
      y_end: 44,
    }

    const res = await workflowApi.runProcessWindow(
      config.optical_system,
      pwConfig,
      patternType.value,
      patternParams
    )

    if (res.success) {
      taskId.value = res.task_id
      ElMessage.success(`工艺窗口分析任务已提交：${res.task_id}`)
      connectWebSocket(res.task_id)
      pollTaskStatus()
    } else {
      ElMessage.error(res.message || '提交失败')
    }
  } catch (e: any) {
    ElMessage.error(e?.message || '提交失败')
  } finally {
    isRunning.value = false
  }
}

function connectWebSocket(tid: string) {
  if (wsUnsubscribe) {
    try { wsUnsubscribe() } catch (e) {}
    wsUnsubscribe = null
  }
  try {
    taskWs.connect(tid)
    wsUnsubscribe = taskWs.onMessage(handleWsMessage)
  } catch (e) {
    console.error('WS 连接失败:', e)
  }
}

function handleWsMessage(msg: any) {
  if (msg.type === 'progress' && currentTask.value) {
    if (msg.progress !== undefined) currentTask.value.progress = msg.progress
    if (msg.stage !== undefined) currentTask.value.stage = msg.stage
    if (msg.message !== undefined) currentTask.value.message = msg.message
  } else if (msg.type === 'task_complete') {
    if (currentTask.value) {
      currentTask.value.status = 'completed'
      currentTask.value.progress = 100
      currentTask.value.result_summary = msg.result
    }
    ElMessage.success('工艺窗口分析完成 ✅')
    fetchTaskResult()
    disconnectWebSocket()
  } else if (msg.type === 'task_failed') {
    if (currentTask.value) {
      currentTask.value.status = 'failed'
      currentTask.value.error = msg.error
    }
    ElMessage.error(`任务失败: ${msg.error || '未知错误'}`)
    disconnectWebSocket()
  }
}

function disconnectWebSocket() {
  if (wsUnsubscribe) {
    try { wsUnsubscribe() } catch (e) {}
    wsUnsubscribe = null
  }
}

async function pollTaskStatus() {
  if (!taskId.value) return
  try {
    const res: any = await taskApi.getStatus(taskId.value)
    currentTask.value = res
    if (res.status === 'completed') {
      fetchTaskResult()
    }
  } catch (e) {}
}

async function fetchTaskResult() {
  if (!taskId.value) return
  try {
    const res: TaskResultResponse = await taskApi.getResult(taskId.value)
    const detail = res.result_detail
    if (detail && detail.focus_values && detail.dose_values && detail.cd_matrix) {
      parseBossungData(detail)
      hasData.value = true
      nextTick(() => drawBossungChart())
    }
  } catch (e) {
    console.error('获取结果失败:', e)
  }
}

function parseBossungData(detail: Record<string, any>) {
  const focusValues: number[] = detail.focus_values || []
  const doseValues: number[] = detail.dose_values || []
  const cdMatrix: number[][] = detail.cd_matrix || []
  const cdTolerance = pwConfig.cd_tolerance
  const nominalCd = detail.nominal_cd || (cdMatrix.length > 0 && cdMatrix[0].length > 0 ? cdMatrix[0][Math.floor(cdMatrix[0].length / 2)] : 45)

  const data: any[][] = []
  for (let i = 0; i < focusValues.length; i++) {
    const row: any[] = []
    const focus = focusValues[i]
    for (let j = 0; j < doseValues.length; j++) {
      const dose = doseValues[j]
      const cd = cdMatrix[i]?.[j] ?? 0
      const cdDiff = nominalCd > 0 ? Math.abs(cd - nominalCd) / nominalCd : 0
      const valid = cdDiff <= cdTolerance
      row.push({
        focus,
        dose,
        cd,
        valid,
      })
    }
    data.push(row)
  }
  bossungData.value = data
}

function drawBossungChart() {
  const canvas = chartCanvas.value
  if (!canvas || !bossungData.value.length) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const rect = canvas.parentElement?.getBoundingClientRect()
  const width = rect?.width || 600
  const height = 400
  const dpr = window.devicePixelRatio || 1

  canvas.width = width * dpr
  canvas.height = height * dpr
  canvas.style.width = width + 'px'
  canvas.style.height = height + 'px'
  ctx.scale(dpr, dpr)

  const padding = { top: 40, right: 60, bottom: 50, left: 60 }
  const plotW = width - padding.left - padding.right
  const plotH = height - padding.top - padding.bottom

  const data = bossungData.value
  const nRows = data.length
  const nCols = data[0]?.length || 0
  if (nRows < 2 || nCols < 2) return

  const minFocus = data[0][0].focus
  const maxFocus = data[nRows - 1][0].focus
  const minDose = data[0][0].dose
  const maxDose = data[0][nCols - 1].dose

  const xScale = (dose: number) => padding.left + ((dose - minDose) / (maxDose - minDose)) * plotW
  const yScale = (focus: number) => padding.top + plotH - ((focus - minFocus) / (maxFocus - minFocus)) * plotH

  ctx.clearRect(0, 0, width, height)

  const cellW = plotW / (nCols - 1)
  const cellH = plotH / (nRows - 1)

  let cdMin = Infinity, cdMax = -Infinity
  for (let i = 0; i < nRows; i++) {
    for (let j = 0; j < nCols; j++) {
      const cd = data[i][j].cd
      if (cd < cdMin) cdMin = cd
      if (cd > cdMax) cdMax = cd
    }
  }

  for (let i = 0; i < nRows - 1; i++) {
    for (let j = 0; j < nCols - 1; j++) {
      const x = padding.left + j * cellW
      const y = padding.top + i * cellH

      const v1 = data[i][j].valid
      const v2 = data[i][j + 1].valid
      const v3 = data[i + 1][j].valid
      const v4 = data[i + 1][j + 1].valid
      const validCount = (v1 ? 1 : 0) + (v2 ? 1 : 0) + (v3 ? 1 : 0) + (v4 ? 1 : 0)

      if (validCount > 0) {
        const alpha = validCount / 4
        ctx.fillStyle = `rgba(103, 194, 58, ${alpha * 0.6})`
        ctx.fillRect(x, y, cellW, cellH)
      }
    }
  }

  const cdLevels = [cdMin + (cdMax - cdMin) * 0.25, cdMin + (cdMax - cdMin) * 0.5, cdMin + (cdMax - cdMin) * 0.75]
  const colors = ['#f56c6c', '#e6a23c', '#67c23a']

  for (let l = 0; l < cdLevels.length; l++) {
    ctx.strokeStyle = colors[l]
    ctx.lineWidth = 1.5
    ctx.setLineDash(l === 1 ? [] : [4, 4])
    ctx.beginPath()

    for (let i = 0; i < nRows - 1; i++) {
      for (let j = 0; j < nCols - 1; j++) {
        const p00 = data[i][j]
        const p10 = data[i][j + 1]
        const p01 = data[i + 1][j]
        const p11 = data[i + 1][j + 1]

        const points: [number, number][] = []

        const sides = [
          [p00.cd, p10.cd, padding.left + j * cellW, padding.top + i * cellH, padding.left + (j + 1) * cellW, padding.top + i * cellH],
          [p10.cd, p11.cd, padding.left + (j + 1) * cellW, padding.top + i * cellH, padding.left + (j + 1) * cellW, padding.top + (i + 1) * cellH],
          [p01.cd, p11.cd, padding.left + j * cellW, padding.top + (i + 1) * cellH, padding.left + (j + 1) * cellW, padding.top + (i + 1) * cellH],
          [p00.cd, p01.cd, padding.left + j * cellW, padding.top + i * cellH, padding.left + j * cellW, padding.top + (i + 1) * cellH],
        ]

        for (const s of sides) {
          const [v1, v2, x1, y1, x2, y2] = s
          if ((v1 - cdLevels[l]) * (v2 - cdLevels[l]) < 0) {
            const t = (cdLevels[l] - v1) / (v2 - v1)
            points.push([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t])
          }
        }

        if (points.length === 2) {
          ctx.moveTo(points[0][0], points[0][1])
          ctx.lineTo(points[1][0], points[1][1])
        }
      }
    }

    ctx.stroke()
  }

  ctx.setLineDash([])

  ctx.strokeStyle = '#909399'
  ctx.lineWidth = 1
  ctx.strokeRect(padding.left, padding.top, plotW, plotH)

  const doseTicks = 5
  for (let i = 0; i <= doseTicks; i++) {
    const dose = minDose + (maxDose - minDose) * i / doseTicks
    const x = xScale(dose)
    ctx.strokeStyle = '#e4e7ed'
    ctx.beginPath()
    ctx.moveTo(x, padding.top)
    ctx.lineTo(x, padding.top + plotH)
    ctx.stroke()

    ctx.fillStyle = '#606266'
    ctx.font = '11px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(dose.toFixed(2), x, padding.top + plotH + 20)
  }

  const focusTicks = 5
  for (let i = 0; i <= focusTicks; i++) {
    const focus = minFocus + (maxFocus - minFocus) * i / focusTicks
    const y = yScale(focus)
    ctx.strokeStyle = '#e4e7ed'
    ctx.beginPath()
    ctx.moveTo(padding.left, y)
    ctx.lineTo(padding.left + plotW, y)
    ctx.stroke()

    ctx.fillStyle = '#606266'
    ctx.font = '11px sans-serif'
    ctx.textAlign = 'right'
    ctx.fillText(focus.toFixed(0), padding.left - 8, y + 4)
  }

  ctx.fillStyle = '#303133'
  ctx.font = '12px sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText('曝光剂量 (Dose)', padding.left + plotW / 2, height - 12)

  ctx.save()
  ctx.translate(18, padding.top + plotH / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.textAlign = 'center'
  ctx.fillText('离焦量 (Focus) nm', 0, 0)
  ctx.restore()

  const nomDose = 1.0
  const nomFocus = 0
  const nx = xScale(nomDose)
  const ny = yScale(nomFocus)

  ctx.strokeStyle = '#409eff'
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.arc(nx, ny, 6, 0, Math.PI * 2)
  ctx.stroke()
  ctx.fillStyle = '#409eff'
  ctx.beginPath()
  ctx.arc(nx, ny, 3, 0, Math.PI * 2)
  ctx.fill()

  ctx.fillStyle = '#409eff'
  ctx.font = '11px sans-serif'
  ctx.textAlign = 'left'
  ctx.fillText('标称点', nx + 10, ny + 4)
}

function refreshChart() {
  if (taskId.value) {
    pollTaskStatus()
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

function progressStatus(s: string): '' | 'success' | 'warning' | 'danger' | 'exception' {
  switch (s) {
    case 'completed': return 'success'
    case 'failed':    return 'exception'
    case 'starting':  return 'warning'
    default:          return ''
  }
}

onMounted(() => {
  if (chartCanvas.value) {
    nextTick(() => drawBossungChart())
  }
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  disconnectWebSocket()
  window.removeEventListener('resize', handleResize)
})

function handleResize() {
  if (hasData.value && chartCanvas.value) {
    drawBossungChart()
  }
}
</script>

<style lang="scss" scoped>
.bossung-chart-page {
  .card {
    border-radius: 10px;
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

    .chart-actions {
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

  .range-inputs {
    display: flex;
    align-items: center;
    gap: 8px;

    .range-sep {
      color: #909399;
      font-size: 12px;
      flex-shrink: 0;
    }
  }

  .form-actions {
    margin-top: 20px;
    display: flex;
    justify-content: center;

    .run-btn {
      min-width: 200px;
    }
  }

  .chart-card {
    .chart-container {
      position: relative;
      width: 100%;
      height: 400px;
      background: #fafafa;
      border-radius: 8px;
      overflow: hidden;
    }

    .bossung-canvas {
      display: block;
      width: 100%;
      height: 100%;
    }

    .chart-placeholder,
    .chart-loading {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: #909399;
      gap: 12px;

      p {
        margin: 0;
        font-size: 14px;
      }

      .is-loading {
        animation: spin 1s linear infinite;
      }
    }

    .chart-loading {
      color: #409eff;
    }
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .chart-legend {
    margin-top: 16px;
    display: flex;
    justify-content: center;
    gap: 24px;

    .legend-item {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: #606266;

      .legend-color {
        width: 20px;
        height: 14px;
        border-radius: 2px;

        &.process-window {
          background: rgba(103, 194, 58, 0.6);
        }
        &.cd-contour {
          background: linear-gradient(to right, #f56c6c, #e6a23c, #67c23a);
        }
        &.nominal-point {
          background: #409eff;
          border-radius: 50%;
          width: 14px;
        }
      }
    }
  }

  .metrics-card {
    margin-top: 16px;

    .metric-card {
      background: #f5f7fa;
      border-radius: 8px;
      padding: 16px 12px;
      text-align: center;
      margin-bottom: 12px;

      .metric-label {
        font-size: 12px;
        color: #909399;
        margin-bottom: 8px;
      }

      .metric-value {
        font-size: 20px;
        font-weight: 600;
        font-family: 'SF Mono', Consolas, monospace;
        color: #303133;

        &.metric-good {
          color: #67c23a;
        }
        &.metric-mid {
          color: #e6a23c;
        }
        &.metric-bad {
          color: #f56c6c;
        }

        .metric-unit {
          font-size: 12px;
          font-weight: 400;
          color: #909399;
          margin-left: 2px;
        }
      }
    }
  }

  .task-stage {
    margin: 8px 0 0 0;
    font-size: 12px;
    color: #606266;
  }

  :deep(.el-slider) {
    margin: 0;
  }
}
</style>
