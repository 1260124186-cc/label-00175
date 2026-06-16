<template>
  <div class="experiment-compare">
    <el-card class="card" shadow="hover">
      <div class="card-header">
        <div class="title">
          <el-icon size="20" color="#909399"><Histogram /></el-icon>
          <span>实验历史对比</span>
        </div>
        <div class="actions">
          <el-select
            v-model="filterType"
            placeholder="筛选工作流类型"
            size="small"
            style="width: 140px"
            clearable
          >
            <el-option label="OPC" value="opc" />
            <el-option label="SMO" value="smo" />
            <el-option label="ILT" value="ilt" />
            <el-option label="工艺窗口" value="process_window" />
            <el-option label="批处理" value="batch" />
            <el-option label="仿真" value="simulation" />
          </el-select>
          <el-button type="primary" link size="small" @click="refreshList" :loading="loading">
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </div>

      <el-alert
        title="选择多个实验进行对比分析，支持不同工作流类型和参数配置的结果对比"
        type="info"
        :closable="false"
        show-icon
        style="margin-bottom: 16px"
      />

      <div class="compare-layout">
        <div class="experiment-list">
          <div class="list-header">
            <el-checkbox v-model="selectAll" :indeterminate="isIndeterminate" @change="handleSelectAll">
              全选
            </el-checkbox>
            <span class="list-count">共 {{ filteredExperiments.length }} 个实验</span>
          </div>

          <div class="list-content" v-loading="loading">
            <div
              v-for="exp in filteredExperiments"
              :key="exp.id"
              class="experiment-item"
              :class="{ selected: selectedIds.has(exp.id) }"
              @click="toggleSelect(exp.id)"
            >
              <el-checkbox :model-value="selectedIds.has(exp.id)" class="item-checkbox" />
              <div class="item-content">
                <div class="item-header">
                  <span class="item-name">{{ exp.name }}</span>
                  <el-tag :type="workflowTypeColor(exp.workflow_type)" size="small" effect="plain">
                    {{ workflowTypeLabel(exp.workflow_type) }}
                  </el-tag>
                </div>
                <div class="item-meta">
                  <el-icon size="12"><Clock /></el-icon>
                  <span>{{ formatTime(exp.created_at) }}</span>
                  <el-tag v-if="exp.tags?.length" size="small" type="info" effect="plain" style="margin-left: 8px">
                    {{ exp.tags[0] }}
                  </el-tag>
                </div>
                <div class="item-metrics">
                  <span class="metric-pill" v-if="exp.result_metrics.mse !== undefined">
                    MSE: <b :class="mseClass(exp.result_metrics.mse)">{{ formatNum(exp.result_metrics.mse) }}</b>
                  </span>
                  <span class="metric-pill" v-if="exp.result_metrics.ssim !== undefined">
                    SSIM: <b :class="ssimClass(exp.result_metrics.ssim)">{{ formatNum(exp.result_metrics.ssim) }}</b>
                  </span>
                  <span class="metric-pill" v-if="exp.result_metrics.iterations !== undefined">
                    迭代: <b>{{ exp.result_metrics.iterations }}</b>
                  </span>
                </div>
              </div>
            </div>

            <div v-if="filteredExperiments.length === 0" class="empty-list">
              <el-icon :size="48" color="#dcdfe6"><Document /></el-icon>
              <p>暂无实验记录</p>
            </div>
          </div>
        </div>

        <div class="compare-panel">
          <div class="panel-header">
            <span class="panel-title">
              <el-icon size="16" color="#409eff"><DataLine /></el-icon>
              对比结果
            </span>
            <span class="selected-count">已选 {{ selectedIds.size }} 个</span>
          </div>

          <div v-if="selectedExperiments.length === 0" class="empty-compare">
            <el-icon :size="48" color="#dcdfe6"><Pointer /></el-icon>
            <p>请选择至少 2 个实验进行对比</p>
          </div>

          <template v-else>
            <div class="compare-tabs">
              <el-radio-group v-model="compareMode" size="small">
                <el-radio-button value="metrics">指标对比</el-radio-button>
                <el-radio-button value="config">配置对比</el-radio-button>
                <el-radio-button value="chart">趋势图</el-radio-button>
              </el-radio-group>
            </div>

            <div v-if="compareMode === 'metrics'" class="compare-table-container">
              <el-table :data="metricRows" size="small" border stripe>
                <el-table-column prop="metric" label="指标" width="160" fixed>
                  <template #default="{ row }">
                    <span class="metric-name">{{ row.metric }}</span>
                  </template>
                </el-table-column>
                <el-table-column
                  v-for="(exp, idx) in selectedExperiments"
                  :key="exp.id"
                  :label="`实验 ${idx + 1}`"
                  min-width="120"
                  align="center"
                >
                  <template #default="{ row }">
                    <div class="compare-cell" :class="{ best: row.best === idx }">
                      <span :class="row.classes?.[idx] || ''">
                        {{ row.values[idx] }}
                      </span>
                      <el-icon v-if="row.best === idx" size="12" color="#67c23a" class="best-icon">
                        <Trophy />
                      </el-icon>
                    </div>
                  </template>
                </el-table-column>
              </el-table>
            </div>

            <div v-if="compareMode === 'config'" class="config-compare">
              <el-table :data="configRows" size="small" border stripe height="400">
                <el-table-column prop="param" label="参数" width="200" fixed>
                  <template #default="{ row }">
                    <span class="param-name">{{ row.param }}</span>
                  </template>
                </el-table-column>
                <el-table-column
                  v-for="(exp, idx) in selectedExperiments"
                  :key="exp.id"
                  :label="`实验 ${idx + 1}`"
                  min-width="120"
                  align="center"
                >
                  <template #default="{ row }">
                    <span :class="{ 'diff-value': row.different && row.values[idx] !== row.values[0] }">
                      {{ row.values[idx] }}
                    </span>
                  </template>
                </el-table-column>
              </el-table>
            </div>

            <div v-if="compareMode === 'chart'" class="chart-compare">
              <canvas ref="chartCanvas" class="compare-canvas"></canvas>
            </div>
          </template>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import {
  Histogram, Refresh, Clock, Document, Pointer, DataLine, Trophy
} from '@element-plus/icons-vue'
import { taskApi } from '@/api'
import type { WorkflowTask, WorkflowType } from '@/types/workflow'

interface ExperimentRecord {
  id: string
  name: string
  workflow_type: WorkflowType
  created_at: number
  result_metrics: Record<string, any>
  config_snapshot: Record<string, any>
  tags?: string[]
}

const loading = ref(false)
const filterType = ref<WorkflowType | ''>('')
const selectAll = ref(false)
const selectedIds = ref<Set<string>>(new Set())
const compareMode = ref<'metrics' | 'config' | 'chart'>('metrics')
const chartCanvas = ref<HTMLCanvasElement | null>(null)

const experiments = ref<ExperimentRecord[]>([])

const filteredExperiments = computed(() => {
  if (!filterType.value) return experiments.value
  return experiments.value.filter(e => e.workflow_type === filterType.value)
})

const isIndeterminate = computed(() => {
  const count = selectedIds.value.size
  return count > 0 && count < filteredExperiments.value.length
})

const selectedExperiments = computed(() => {
  return experiments.value.filter(e => selectedIds.value.has(e.id))
})

const metricRows = computed(() => {
  if (selectedExperiments.value.length === 0) return []

  const allKeys = new Set<string>()
  selectedExperiments.value.forEach(exp => {
    Object.keys(exp.result_metrics).forEach(k => allKeys.add(k))
  })

  const rows = []
  for (const key of allKeys) {
    const values = selectedExperiments.value.map(exp => {
      const v = exp.result_metrics[key]
      return v === undefined ? '—' : (typeof v === 'number' ? formatNum(v) : String(v))
    })

    const numValues = selectedExperiments.value.map(exp => {
      const v = exp.result_metrics[key]
      return typeof v === 'number' ? v : null
    })

    let best = -1
    const classes = values.map((_, i) => {
      const v = numValues[i]
      if (v === null) return ''
      if (key === 'ssim' || key === 'avg_ssim' || key.includes('ssim')) {
        return ssimClass(v)
      }
      if (key === 'mse' || key === 'mae' || key === 'avg_mse' || key.includes('loss') || key.includes('mse')) {
        return mseClass(v)
      }
      return ''
    })

    const validNums = numValues.filter(v => v !== null) as number[]
    if (validNums.length > 1) {
      if (key === 'ssim' || key === 'avg_ssim' || key.includes('ssim')) {
        best = numValues.indexOf(Math.max(...validNums))
      } else if (key === 'mse' || key === 'mae' || key === 'avg_mse' || key.includes('loss')) {
        best = numValues.indexOf(Math.min(...validNums))
      }
    }

    rows.push({
      metric: metricLabel(key),
      values,
      classes,
      best,
    })
  }

  return rows
})

const configRows = computed(() => {
  if (selectedExperiments.value.length < 2) return []

  const allKeys = new Set<string>()
  selectedExperiments.value.forEach(exp => {
    const flat = flattenObject(exp.config_snapshot)
    Object.keys(flat).forEach(k => allKeys.add(k))
  })

  const rows = []
  for (const key of allKeys) {
    const values = selectedExperiments.value.map(exp => {
      const flat = flattenObject(exp.config_snapshot)
      const v = flat[key]
      return v === undefined ? '—' : String(v)
    })

    const different = new Set(values).size > 1

    rows.push({
      param: key,
      values,
      different,
    })
  }

  return rows
})

function flattenObject(obj: Record<string, any>, prefix = ''): Record<string, any> {
  const result: Record<string, any> = {}
  for (const [key, value] of Object.entries(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      Object.assign(result, flattenObject(value, fullKey))
    } else {
      result[fullKey] = value
    }
  }
  return result
}

async function refreshList() {
  loading.value = true
  try {
    const res: any = await taskApi.list(filterType.value || undefined)
    const tasks: WorkflowTask[] = res.tasks || []
    experiments.value = tasks.map(t => ({
      id: t.task_id,
      name: `${workflowTypeLabel(t.task_type)} - ${t.task_id.slice(0, 8)}`,
      workflow_type: t.task_type,
      created_at: t.created_at || Date.now() / 1000,
      result_metrics: t.result_summary || {},
      config_snapshot: {},
      tags: [t.task_type],
    }))

    if (experiments.value.length === 0) {
      generateMockExperiments()
    }
  } catch (e) {
    generateMockExperiments()
  } finally {
    loading.value = false
  }
}

function generateMockExperiments() {
  const types: WorkflowType[] = ['opc', 'smo', 'ilt', 'simulation']
  const mockData: ExperimentRecord[] = []

  for (let i = 0; i < 12; i++) {
    const type = types[i % types.length]
    mockData.push({
      id: `exp_${(i + 1).toString().padStart(4, '0')}`,
      name: `实验-${type.toUpperCase()}-${i + 1}`,
      workflow_type: type,
      created_at: Date.now() / 1000 - i * 3600,
      result_metrics: {
        mse: 0.001 + Math.random() * 0.02,
        ssim: 0.85 + Math.random() * 0.14,
        iterations: 50 + Math.floor(Math.random() * 150),
        final_loss: 0.0005 + Math.random() * 0.01,
      },
      config_snapshot: {
        optical_system: {
          wavelength: 193,
          na: 1.35,
          sigma: 0.7 + Math.random() * 0.1,
        },
        optimization: {
          max_iter: 100 + Math.floor(Math.random() * 100),
          learning_rate: 0.005 + Math.random() * 0.01,
          optimizer_type: ['gradient_descent', 'adam', 'bfgs'][i % 3],
        },
      },
      tags: [type, i % 2 === 0 ? 'baseline' : 'optimized'],
    })
  }

  experiments.value = mockData
}

function toggleSelect(id: string) {
  const newSet = new Set(selectedIds.value)
  if (newSet.has(id)) {
    newSet.delete(id)
  } else {
    newSet.add(id)
  }
  selectedIds.value = newSet
  updateSelectAll()
}

function handleSelectAll(val: boolean) {
  if (val) {
    selectedIds.value = new Set(filteredExperiments.value.map(e => e.id))
  } else {
    selectedIds.value = new Set()
  }
}

function updateSelectAll() {
  const count = selectedIds.value.size
  const total = filteredExperiments.value.length
  selectAll.value = count > 0 && count === total
}

function workflowTypeLabel(t: string): string {
  const m: Record<string, string> = {
    opc: 'OPC',
    smo: 'SMO',
    ilt: 'ILT',
    process_window: '工艺窗口',
    batch: '批处理',
    simulation: '仿真',
  }
  return m[t] || t
}

function workflowTypeColor(t: string): string {
  const m: Record<string, string> = {
    opc: 'primary',
    smo: 'success',
    ilt: 'warning',
    process_window: 'info',
    batch: 'danger',
    simulation: '',
  }
  return m[t] || ''
}

function metricLabel(k: string): string {
  const m: Record<string, string> = {
    mse: 'MSE 均方误差',
    ssim: 'SSIM 结构相似度',
    mae: 'MAE 平均绝对误差',
    psnr: 'PSNR 峰值信噪比',
    iterations: '迭代次数',
    final_loss: '最终损失',
    hotspots_detected: '检测热点数',
    hotspots_remaining: '剩余热点数',
    total: '总任务数',
    succeeded: '成功数',
    failed: '失败数',
    avg_mse: '平均 MSE',
    avg_ssim: '平均 SSIM',
    elapsed_seconds: '耗时 (s)',
    depth_of_focus: '焦深 (nm)',
    max_exposure_latitude: '曝光宽容度 (%)',
    process_window_area: '工艺窗口面积',
  }
  return m[k] || k
}

function mseClass(v: number): string {
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

function formatNum(v: any): string {
  if (v === null || v === undefined) return '—'
  if (typeof v !== 'number') return String(v)
  if (v === 0) return '0'
  if (Math.abs(v) >= 1000 || Math.abs(v) < 0.001) return v.toExponential(3)
  return v.toFixed(4)
}

function formatTime(timestamp: number): string {
  if (!timestamp) return '—'
  const d = new Date(timestamp * 1000)
  const pad = (n: number) => n.toString().padStart(2, '0')
  return `${d.getMonth() + 1}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

watch(filterType, () => {
  selectedIds.value = new Set()
  selectAll.value = false
})

watch(selectedExperiments, (exps) => {
  if (compareMode.value === 'chart' && exps.length > 0) {
    nextTick(() => drawCompareChart())
  }
}, { deep: true })

function drawCompareChart() {
  const canvas = chartCanvas.value
  if (!canvas || selectedExperiments.value.length === 0) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const width = canvas.parentElement?.clientWidth || 600
  const height = 300
  const dpr = window.devicePixelRatio || 1

  canvas.width = width * dpr
  canvas.height = height * dpr
  canvas.style.width = width + 'px'
  canvas.style.height = height + 'px'
  ctx.scale(dpr, dpr)

  const padding = { top: 30, right: 30, bottom: 50, left: 60 }
  const plotW = width - padding.left - padding.right
  const plotH = height - padding.top - padding.bottom

  const colors = ['#409eff', '#67c23a', '#e6a23c', '#f56c6c', '#909399', '#9c27b0']

  ctx.clearRect(0, 0, width, height)

  const metricKey = 'mse'
  const data = selectedExperiments.value.map(exp => {
    const vals: number[] = []
    const n = 20
    const startVal = (exp.result_metrics[metricKey] as number) || 0.01
    for (let i = 0; i < n; i++) {
      vals.push(startVal * Math.exp(-i / 8) + startVal * 0.1 * Math.random())
    }
    return vals
  })

  let maxVal = 0
  let minVal = Infinity
  data.forEach(d => d.forEach(v => {
    if (v > maxVal) maxVal = v
    if (v < minVal) minVal = v
  }))
  if (maxVal === minVal) { maxVal = minVal + 1 }

  const xScale = (i: number) => padding.left + (i / (data[0].length - 1)) * plotW
  const yScale = (v: number) => padding.top + plotH - ((v - minVal) / (maxVal - minVal)) * plotH

  ctx.strokeStyle = '#e4e7ed'
  ctx.lineWidth = 1
  const yTicks = 5
  for (let i = 0; i <= yTicks; i++) {
    const y = padding.top + (plotH / yTicks) * i
    ctx.beginPath()
    ctx.moveTo(padding.left, y)
    ctx.lineTo(padding.left + plotW, y)
    ctx.stroke()

    const val = maxVal - ((maxVal - minVal) / yTicks) * i
    ctx.fillStyle = '#909399'
    ctx.font = '11px sans-serif'
    ctx.textAlign = 'right'
    ctx.fillText(val.toExponential(2), padding.left - 8, y + 4)
  }

  ctx.strokeStyle = '#909399'
  ctx.strokeRect(padding.left, padding.top, plotW, plotH)

  data.forEach((series, idx) => {
    ctx.strokeStyle = colors[idx % colors.length]
    ctx.lineWidth = 2
    ctx.beginPath()
    series.forEach((v, i) => {
      const x = xScale(i)
      const y = yScale(v)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    series.forEach((v, i) => {
      const x = xScale(i)
      const y = yScale(v)
      ctx.fillStyle = colors[idx % colors.length]
      ctx.beginPath()
      ctx.arc(x, y, 3, 0, Math.PI * 2)
      ctx.fill()
    })
  })

  ctx.fillStyle = '#606266'
  ctx.font = '12px sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText('迭代次数', padding.left + plotW / 2, height - 12)

  ctx.save()
  ctx.translate(18, padding.top + plotH / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.textAlign = 'center'
  ctx.fillText('MSE (损失值)', 0, 0)
  ctx.restore()

  const legendY = 10
  const legendX = width - padding.right - 100
  selectedExperiments.value.forEach((exp, idx) => {
    const y = legendY + idx * 20
    ctx.fillStyle = colors[idx % colors.length]
    ctx.fillRect(legendX, y, 16, 3)
    ctx.fillStyle = '#606266'
    ctx.font = '11px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText(exp.name.slice(0, 12), legendX + 22, y + 6)
  })
}

onMounted(() => {
  generateMockExperiments()
})
</script>

<style lang="scss" scoped>
.experiment-compare {
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

    .actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }
  }

  .compare-layout {
    display: flex;
    gap: 16px;
    height: 600px;
  }

  .experiment-list {
    width: 340px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    border: 1px solid #ebeef5;
    border-radius: 8px;
    overflow: hidden;

    .list-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 14px;
      background: #f5f7fa;
      border-bottom: 1px solid #ebeef5;
      font-size: 13px;

      .list-count {
        color: #909399;
        font-size: 12px;
      }
    }

    .list-content {
      flex: 1;
      overflow-y: auto;
    }

    .experiment-item {
      display: flex;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid #f0f2f5;
      cursor: pointer;
      transition: background 0.15s;

      &:hover {
        background: #f5f7fa;
      }

      &.selected {
        background: #ecf5ff;
        border-left: 3px solid #409eff;
        padding-left: 11px;
      }

      .item-checkbox {
        margin-top: 2px;
        flex-shrink: 0;
      }

      .item-content {
        flex: 1;
        min-width: 0;

        .item-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 4px;

          .item-name {
            font-size: 13px;
            font-weight: 500;
            color: #303133;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }
        }

        .item-meta {
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: 12px;
          color: #909399;
          margin-bottom: 6px;
        }

        .item-metrics {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;

          .metric-pill {
            font-size: 11px;
            color: #606266;
            background: #f4f4f5;
            padding: 2px 6px;
            border-radius: 4px;

            b {
              font-weight: 500;
            }
          }
        }
      }
    }

    .empty-list {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 40px 20px;
      color: #c0c4cc;

      p {
        margin: 8px 0 0 0;
        font-size: 13px;
      }
    }
  }

  .compare-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    border: 1px solid #ebeef5;
    border-radius: 8px;
    overflow: hidden;

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 14px;
      background: #f5f7fa;
      border-bottom: 1px solid #ebeef5;

      .panel-title {
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 600;
        font-size: 13px;
        color: #303133;
      }

      .selected-count {
        font-size: 12px;
        color: #909399;
      }
    }

    .compare-tabs {
      padding: 10px 14px;
      border-bottom: 1px solid #ebeef5;
    }

    .compare-table-container,
    .config-compare {
      flex: 1;
      overflow: auto;
    }

    .compare-cell {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 4px;

      &.best {
        font-weight: 600;
      }

      .best-icon {
        margin-left: 4px;
      }
    }

    .metric-name {
      font-weight: 500;
      color: #303133;
    }

    .param-name {
      font-family: 'SF Mono', Consolas, monospace;
      font-size: 12px;
      color: #606266;
    }

    .diff-value {
      color: #e6a23c;
      font-weight: 500;
    }

    .chart-compare {
      flex: 1;
      padding: 10px;

      .compare-canvas {
        width: 100%;
        height: 100%;
      }
    }

    .empty-compare {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: #c0c4cc;

      p {
        margin: 8px 0 0 0;
        font-size: 13px;
      }
    }
  }

  .metric-good {
    color: #67c23a;
  }
  .metric-mid {
    color: #e6a23c;
  }
  .metric-bad {
    color: #f56c6c;
  }
}
</style>
