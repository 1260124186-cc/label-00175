<template>
  <div class="opc-console">
    <el-row :gutter="20">
      <el-col :xs="24" :lg="14">
        <el-card class="card" shadow="hover">
          <div class="card-header">
            <div class="title">
              <el-icon size="20" color="#409eff"><MagicStick /></el-icon>
              <span>OPC 光学邻近校正</span>
            </div>
            <el-tag type="primary" effect="plain" round size="small">Optical Proximity Correction</el-tag>
          </div>

          <el-alert
            title="OPC 通过边缘偏移、SRAF 插入和精细优化，修正光刻工艺中的邻近效应"
            type="info"
            :closable="false"
            show-icon
            style="margin-bottom: 16px"
          />

          <el-tabs v-model="activeSection" type="border-card">
            <el-tab-pane label="图案输入" name="pattern">
              <el-form label-width="100px" label-position="right">
                <el-form-item label="输入源">
                  <el-radio-group v-model="inputSource">
                    <el-radio value="synthetic">测试图案</el-radio>
                    <el-radio value="gds">GDS 版图</el-radio>
                  </el-radio-group>
                </el-form-item>

                <template v-if="inputSource === 'synthetic'">
                  <el-form-item label="图案类型">
                    <el-select v-model="patternType" style="width: 100%">
                      <el-option label="矩形 (Rectangle)" value="rectangle" />
                      <el-option label="十字 (Cross)" value="cross" />
                      <el-option label="L 型 (L-shape)" value="l_shape" />
                      <el-option label="阵列 (Array)" value="array" />
                    </el-select>
                  </el-form-item>

                  <el-row :gutter="24">
                    <el-col :span="12">
                      <el-form-item label="图像高度">
                        <el-input-number
                          v-model="patternParams.size[0]"
                          :min="16" :max="512" :step="8"
                          controls-position="right"
                          class="w-full"
                        />
                      </el-form-item>
                    </el-col>
                    <el-col :span="12">
                      <el-form-item label="图像宽度">
                        <el-input-number
                          v-model="patternParams.size[1]"
                          :min="16" :max="512" :step="8"
                          controls-position="right"
                          class="w-full"
                        />
                      </el-form-item>
                    </el-col>
                  </el-row>

                  <el-row :gutter="24" v-if="patternType === 'rectangle'">
                    <el-col :span="12">
                      <el-form-item label="X 起始">
                        <el-input-number v-model="patternParams.x_start" :min="0" :controls="false" class="w-full" />
                      </el-form-item>
                    </el-col>
                    <el-col :span="12">
                      <el-form-item label="X 结束">
                        <el-input-number v-model="patternParams.x_end" :min="0" :controls="false" class="w-full" />
                      </el-form-item>
                    </el-col>
                    <el-col :span="12">
                      <el-form-item label="Y 起始">
                        <el-input-number v-model="patternParams.y_start" :min="0" :controls="false" class="w-full" />
                      </el-form-item>
                    </el-col>
                    <el-col :span="12">
                      <el-form-item label="Y 结束">
                        <el-input-number v-model="patternParams.y_end" :min="0" :controls="false" class="w-full" />
                      </el-form-item>
                    </el-col>
                  </el-row>
                </template>

                <template v-else>
                  <GdsUploader
                    v-model="selectedGdsFile"
                    :selected-layer-value="selectedGdsLayer"
                    @select="onGdsSelect"
                  />
                </template>
              </el-form>
            </el-tab-pane>

            <el-tab-pane label="OPC 参数" name="params">
              <el-form label-width="140px" label-position="right" size="default">
                <el-divider content-position="left"><span class="divider-label">EPE 热点检测</span></el-divider>
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="EPE 阈值 (nm)">
                      <el-input-number v-model="opcConfig.epe_threshold" :min="0.1" :step="0.5" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="收敛阈值 (nm)">
                      <el-input-number v-model="opcConfig.epe_convergence_threshold" :min="0.1" :step="0.1" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="最大迭代次数">
                      <el-input-number v-model="opcConfig.max_iterations" :min="1" :max="100" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="最小热点面积">
                      <el-input-number v-model="opcConfig.min_hotspot_area" :min="1" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>

                <el-divider content-position="left"><span class="divider-label">边缘偏移</span></el-divider>
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="偏移步长 (px)">
                      <el-input-number v-model="opcConfig.edge_offset_step" :min="0.1" :step="0.1" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="最大偏移量 (px)">
                      <el-input-number v-model="opcConfig.max_edge_offset" :min="0.5" :step="0.5" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>
                <el-form-item label="热点膨胀 (px)">
                  <el-input-number v-model="opcConfig.hotspot_dilation" :min="0" :max="10" class="w-full" />
                </el-form-item>

                <el-divider content-position="left"><span class="divider-label">几何修正</span></el-divider>
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="拐角 Serif (px)">
                      <el-input-number v-model="opcConfig.corner_bias_size" :min="0.1" :step="0.5" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="线端延伸 (px)">
                      <el-input-number v-model="opcConfig.line_end_extension" :min="0.1" :step="0.5" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="线端宽度 (px)">
                      <el-input-number v-model="opcConfig.line_end_width" :min="0.1" :step="0.5" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>

                <el-divider content-position="left"><span class="divider-label">SRAF 辅助特征</span></el-divider>
                <el-form-item label="启用 SRAF">
                  <el-switch v-model="opcConfig.sraf_enable" />
                </el-form-item>
                <template v-if="opcConfig.sraf_enable">
                  <el-row :gutter="24">
                    <el-col :span="12">
                      <el-form-item label="最小距离 (px)">
                        <el-input-number v-model="opcConfig.sraf_min_distance" :min="0.5" :step="0.5" class="w-full" />
                      </el-form-item>
                    </el-col>
                    <el-col :span="12">
                      <el-form-item label="最大距离 (px)">
                        <el-input-number v-model="opcConfig.sraf_max_distance" :min="1" :step="0.5" class="w-full" />
                      </el-form-item>
                    </el-col>
                  </el-row>
                  <el-row :gutter="24">
                    <el-col :span="12">
                      <el-form-item label="SRAF 宽度 (px)">
                        <el-input-number v-model="opcConfig.sraf_width" :min="0.5" :step="0.1" class="w-full" />
                      </el-form-item>
                    </el-col>
                    <el-col :span="12">
                      <el-form-item label="SRAF 长度 (px)">
                        <el-input-number v-model="opcConfig.sraf_length" :min="1" :step="0.5" class="w-full" />
                      </el-form-item>
                    </el-col>
                  </el-row>
                  <el-row :gutter="24">
                    <el-col :span="12">
                      <el-form-item label="SRAF 间距 (px)">
                        <el-input-number v-model="opcConfig.sraf_spacing" :min="0.5" :step="0.5" class="w-full" />
                      </el-form-item>
                    </el-col>
                    <el-col :span="12">
                      <el-form-item label="最小尺寸 (px)">
                        <el-input-number v-model="opcConfig.sraf_min_feature_size" :min="0.5" :step="0.1" class="w-full" />
                      </el-form-item>
                    </el-col>
                  </el-row>
                </template>

                <el-divider content-position="left"><span class="divider-label">精细优化</span></el-divider>
                <el-form-item label="启用优化器">
                  <el-switch v-model="opcConfig.optimizer_enable" />
                </el-form-item>
                <template v-if="opcConfig.optimizer_enable">
                  <el-row :gutter="24">
                    <el-col :span="12">
                      <el-form-item label="最大迭代次数">
                        <el-input-number v-model="opcConfig.optimizer_max_iter" :min="1" :max="200" class="w-full" />
                      </el-form-item>
                    </el-col>
                    <el-col :span="12">
                      <el-form-item label="学习率">
                        <el-input-number v-model="opcConfig.optimizer_learning_rate" :min="0.001" :step="0.01" class="w-full" />
                      </el-form-item>
                    </el-col>
                  </el-row>
                  <el-form-item label="EPE 损失权重">
                    <el-input-number v-model="opcConfig.optimizer_epe_weight" :min="0" :step="0.1" class="w-full" />
                  </el-form-item>
                </template>

                <el-divider content-position="left"><span class="divider-label">晶圆成像</span></el-divider>
                <el-form-item label="光刻胶阈值">
                  <el-input-number v-model="opcConfig.wafer_threshold" :min="0" :max="1" :step="0.05" class="w-full" />
                </el-form-item>
                <el-form-item label="详细日志">
                  <el-switch v-model="opcConfig.verbose" />
                </el-form-item>
              </el-form>
            </el-tab-pane>
          </el-tabs>

          <div class="form-actions">
            <el-button type="primary" size="large" :icon="VideoPlay" :loading="isRunning" @click="handleRun" class="run-btn">
              运行 OPC 工作流
            </el-button>
            <el-button :icon="RefreshRight" @click="resetConfig">重置参数</el-button>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="10">
        <TaskMonitor task-type="opc" @task-complete="onTaskComplete" />
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import { MagicStick, VideoPlay, RefreshRight } from '@element-plus/icons-vue'
import { useConfigStore } from '@/stores/config'
import { workflowApi } from '@/api'
import GdsUploader from './GdsUploader.vue'
import TaskMonitor from './TaskMonitor.vue'
import type { OPCConfigParams } from '@/types/workflow'

const configStore = useConfigStore()

const activeSection = ref('pattern')
const inputSource = ref<'synthetic' | 'gds'>('synthetic')
const patternType = ref('rectangle')
const patternParams = reactive({
  size: [64, 64] as [number, number],
  x_start: 20,
  x_end: 44,
  y_start: 20,
  y_end: 44,
})
const selectedGdsFile = ref('')
const selectedGdsLayer = ref<number | null>(null)

const isRunning = ref(false)

const defaultOpcConfig: OPCConfigParams = {
  epe_threshold: 3.0,
  epe_convergence_threshold: 1.0,
  max_iterations: 10,
  min_hotspot_area: 4,
  hotspot_dilation: 2,
  edge_offset_step: 0.5,
  max_edge_offset: 3.0,
  corner_bias_size: 1.0,
  line_end_extension: 2.0,
  line_end_width: 2.0,
  sraf_enable: true,
  sraf_min_distance: 2.0,
  sraf_max_distance: 5.0,
  sraf_width: 1.0,
  sraf_length: 4.0,
  sraf_spacing: 2.0,
  sraf_min_feature_size: 1.0,
  sraf_max_aspect_ratio: 10.0,
  optimizer_enable: true,
  optimizer_max_iter: 20,
  optimizer_learning_rate: 0.05,
  optimizer_epe_weight: 1.0,
  wafer_threshold: 0.3,
  verbose: true,
}

const opcConfig = reactive<OPCConfigParams>({ ...defaultOpcConfig })

function onGdsSelect(file: any, layer: number, datatype: number) {
  selectedGdsLayer.value = layer
}

async function handleRun() {
  const config = (configStore as any).config
  if (!config?.optical_system) {
    ElMessage.warning('请先到「参数配置」页加载光学系统配置')
    return
  }

  if (inputSource.value === 'gds' && !selectedGdsFile.value) {
    ElMessage.warning('请选择 GDS 文件')
    return
  }

  isRunning.value = true
  try {
    const patternParamsObj: Record<string, any> = {
      size: patternParams.size,
      x_start: patternParams.x_start,
      x_end: patternParams.x_end,
      y_start: patternParams.y_start,
      y_end: patternParams.y_end,
    }

    const res = await workflowApi.runOpc(
      config.optical_system,
      opcConfig,
      patternType.value,
      patternParamsObj
    )

    if (res.success) {
      ElMessage.success(`OPC 任务已提交：${res.task_id}`)
      emitTaskSubmit(res.task_id)
    } else {
      ElMessage.error(res.message || '提交失败')
    }
  } catch (e: any) {
    ElMessage.error(e?.message || '提交失败')
  } finally {
    isRunning.value = false
  }
}

function resetConfig() {
  Object.assign(opcConfig, defaultOpcConfig)
}

const emit = defineEmits<{
  (e: 'task-submit', taskId: string): void
  (e: 'task-complete', taskId: string): void
}>()

function emitTaskSubmit(taskId: string) {
  emit('task-submit', taskId)
}

function onTaskComplete(taskId: string) {
  emit('task-complete', taskId)
}
</script>

<style lang="scss" scoped>
.opc-console {
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
    gap: 12px;
    justify-content: flex-end;

    .run-btn {
      min-width: 180px;
    }
  }

  :deep(.el-tabs__content) {
    padding-top: 8px;
  }
}
</style>
