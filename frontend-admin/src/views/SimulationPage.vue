<template>
  <div class="simulation-page">
    <el-row :gutter="20">
      <el-col :span="12">
        <el-card class="card">
          <div class="card-header">
            <div class="title">
              <el-icon size="20" color="#409eff"><Cpu /></el-icon>
              <span>运行光刻仿真</span>
            </div>
          </div>

          <el-form :model="runForm" label-width="120px" label-position="right">
            <el-form-item label="图案类型">
              <el-select v-model="runForm.pattern_type" style="width: 100%">
                <el-option label="矩形 (Rectangle)" value="rectangle" />
                <el-option label="十字 (Cross)" value="cross" />
                <el-option label="L型 (L-shape)" value="l_shape" />
                <el-option label="阵列 (Array)" value="array" />
                <el-option label="随机测试 (Random)" value="random" />
              </el-select>
            </el-form-item>

            <el-divider content-position="left">图案尺寸参数</el-divider>

            <el-row :gutter="16">
              <el-col :span="12">
                <el-form-item label="图像高度">
                  <el-input-number
                    v-model="runForm.params.size[0]"
                    :min="16"
                    :max="1024"
                    :step="8"
                    controls-position="right"
                    style="width: 100%"
                  />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="图像宽度">
                  <el-input-number
                    v-model="runForm.params.size[1]"
                    :min="16"
                    :max="1024"
                    :step="8"
                    controls-position="right"
                    style="width: 100%"
                  />
                </el-form-item>
              </el-col>
            </el-row>

            <el-row :gutter="16" v-if="runForm.pattern_type === 'rectangle'">
              <el-col :span="6">
                <el-form-item label="X起始">
                  <el-input-number v-model="runForm.params.x_start" :min="0" :step="1" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="6">
                <el-form-item label="X结束">
                  <el-input-number v-model="runForm.params.x_end" :min="0" :step="1" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="6">
                <el-form-item label="Y起始">
                  <el-input-number v-model="runForm.params.y_start" :min="0" :step="1" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
              <el-col :span="6">
                <el-form-item label="Y结束">
                  <el-input-number v-model="runForm.params.y_end" :min="0" :step="1" controls-position="right" style="width: 100%" />
                </el-form-item>
              </el-col>
            </el-row>

            <el-alert
              title="提示：仿真使用当前「参数配置」页面中的光学系统和优化器参数"
              type="info"
              :closable="false"
              show-icon
              style="margin-bottom: 16px"
            />

            <el-form-item>
              <el-button
                type="primary"
                :icon="VideoPlay"
                :loading="isRunning"
                @click="handleRun"
                style="width: 160px"
              >
                开始仿真
              </el-button>
              <el-button :icon="List" @click="fetchTasks">刷新任务</el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card class="card">
          <div class="card-header">
            <div class="title">
              <el-icon size="20" color="#67c23a"><List /></el-icon>
              <span>仿真任务列表</span>
            </div>
          </div>

          <el-table :data="taskList" v-loading="taskLoading" style="width: 100%" empty-text="暂无任务">
            <el-table-column prop="task_id" label="任务ID" width="140" />
            <el-table-column prop="status" label="状态" width="100">
              <template #default="{ row }">
                <el-tag :type="statusType(row.status)" size="small">
                  {{ statusLabel(row.status) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="progress" label="进度" width="160">
              <template #default="{ row }">
                <el-progress :percentage="row.progress" :stroke-width="10" />
              </template>
            </el-table-column>
            <el-table-column label="操作" width="80">
              <template #default="{ row }">
                <el-button type="primary" link size="small" @click="viewTask(row)">
                  查看
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>

        <el-card class="card" v-if="currentTask" style="margin-top: 20px">
          <div class="card-header">
            <div class="title">
              <el-icon size="20" color="#e6a23c"><DataBoard /></el-icon>
              <span>任务详情 - {{ currentTask.task_id }}</span>
            </div>
          </div>
          <el-descriptions :column="2" border size="small">
            <el-descriptions-item label="任务ID">{{ currentTask.task_id }}</el-descriptions-item>
            <el-descriptions-item label="状态">
              <el-tag :type="statusType(currentTask.status)" size="small">
                {{ statusLabel(currentTask.status) }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="进度">
              <el-progress :percentage="currentTask.progress" :stroke-width="8" />
            </el-descriptions-item>
            <el-descriptions-item label="错误信息" v-if="currentTask.error">
              <span style="color: #f56c6c">{{ currentTask.error }}</span>
            </el-descriptions-item>
          </el-descriptions>

          <el-descriptions :column="2" border size="small" style="margin-top: 12px" v-if="currentTask.result">
            <template v-for="(val, key) in currentTask.result.initial_metrics" :key="key">
              <el-descriptions-item :label="`初始指标 - ${metricLabel(key)}`">
                {{ formatNumber(val) }}
              </el-descriptions-item>
            </template>
          </el-descriptions>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { VideoPlay, List, DataBoard } from '@element-plus/icons-vue'
import { useConfigStore } from '@/stores/config'
import { simulationApi } from '@/api'

const configStore = useConfigStore()

const isRunning = ref(false)
const taskLoading = ref(false)
const taskList = ref<any[]>([])
const currentTask = ref<any>(null)

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
  if (configStore.loading === false && !configStore.config) {
    configStore.loadDefault()
  }
})

async function handleRun() {
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
      ElMessage.success('仿真任务已提交')
      await fetchTasks()
      pollTask(res.task_id)
    } else {
      ElMessage.error(res.message || '提交失败')
    }
  } catch (e: any) {
    ElMessage.error('提交失败')
  } finally {
    isRunning.value = false
  }
}

async function fetchTasks() {
  taskLoading.value = true
  try {
    const res: any = await simulationApi.listTasks()
    taskList.value = res.tasks || []
  } finally {
    taskLoading.value = false
  }
}

function pollTask(taskId: string) {
  const interval = setInterval(async () => {
    try {
      const res: any = await simulationApi.getTaskStatus(taskId)
      const idx = taskList.value.findIndex((t) => t.task_id === taskId)
      if (idx >= 0) {
        taskList.value[idx] = res
      } else {
        taskList.value.unshift(res)
      }
      if (currentTask.value?.task_id === taskId) {
        currentTask.value = res
      }
      if (['completed', 'failed'].includes(res.status)) {
        clearInterval(interval)
        if (res.status === 'completed') {
          ElMessage.success(`任务 ${taskId} 完成`)
        } else {
          ElMessage.error(`任务 ${taskId} 失败: ${res.error}`)
        }
      }
    } catch (e) {
      clearInterval(interval)
    }
  }, 2000)
}

function viewTask(task: any) {
  currentTask.value = task
  if (['starting', 'running'].includes(task.status)) {
    pollTask(task.task_id)
  }
}

function statusType(s: string) {
  switch (s) {
    case 'completed': return 'success'
    case 'running': return 'primary'
    case 'failed': return 'danger'
    case 'starting': return 'warning'
    default: return 'info'
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
  const m: Record<string, string> = { mse: 'MSE', ssim: 'SSIM', mae: 'MAE' }
  return m[k] || k
}
function formatNumber(v: any) {
  if (typeof v !== 'number') return v
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(4)
  return v.toFixed(6)
}
</script>

<style lang="scss" scoped>
.simulation-page {
  .card {
    border-radius: 8px;
    min-height: 400px;
  }
}
</style>
