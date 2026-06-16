<template>
  <div class="ilt-console">
    <el-row :gutter="20">
      <el-col :xs="24" :lg="14">
        <el-card class="card" shadow="hover">
          <div class="card-header">
            <div class="title">
              <el-icon size="20" color="#e6a23c"><Aim /></el-icon>
              <span>ILT 反演光刻技术</span>
            </div>
            <el-tag type="warning" effect="plain" round size="small">Inverse Lithography Technology</el-tag>
          </div>

          <el-alert
            title="ILT 通过反演优化直接生成复杂掩模，实现最佳的成像分辨率和工艺窗口"
            type="warning"
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
                        <el-input-number v-model="patternParams.size[0]" :min="16" :max="512" :step="8" controls-position="right" class="w-full" />
                      </el-form-item>
                    </el-col>
                    <el-col :span="12">
                      <el-form-item label="图像宽度">
                        <el-input-number v-model="patternParams.size[1]" :min="16" :max="512" :step="8" controls-position="right" class="w-full" />
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
                  <GdsUploader v-model="selectedGdsFile" :selected-layer-value="selectedGdsLayer" @select="onGdsSelect" />
                </template>
              </el-form>
            </el-tab-pane>

            <el-tab-pane label="ILT 参数" name="params">
              <el-form label-width="160px" label-position="right" size="default">
                <el-divider content-position="left"><span class="divider-label">优化器配置</span></el-divider>
                <el-form-item label="优化器类型">
                  <el-select v-model="iltConfig.optimizer_type" style="width: 100%">
                    <el-option label="梯度投影 (Gradient Projection)" value="gradient_projection" />
                    <el-option label="Adam 投影 (Adam Projection)" value="adam_projection" />
                    <el-option label="SGD 投影 (SGD Projection)" value="sgd_projection" />
                  </el-select>
                </el-form-item>
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="最大迭代次数">
                      <el-input-number v-model="iltConfig.max_iter" :min="10" :max="2000" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="学习率">
                      <el-input-number v-model="iltConfig.learning_rate" :min="0.001" :step="0.005" :precision="4" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="收敛容差">
                      <el-input-number v-model="iltConfig.convergence_tol" :min="1e-10" :step="1e-6" :precision="8" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="收敛耐心值">
                      <el-input-number v-model="iltConfig.convergence_patience" :min="0" :max="200" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>

                <el-divider content-position="left"><span class="divider-label">透射率与量化</span></el-divider>
                <el-form-item label="透射率等级">
                  <el-radio-group v-model="iltConfig.transmission_level">
                    <el-radio value="binary">二值 (Binary)</el-radio>
                    <el-radio value="ternary">三值 (Ternary)</el-radio>
                    <el-radio value="continuous">连续 (Continuous)</el-radio>
                  </el-radio-group>
                </el-form-item>

                <template v-if="iltConfig.transmission_level !== 'continuous'">
                  <el-row :gutter="24">
                    <el-col :span="12">
                      <el-form-item label="量化起始迭代">
                        <el-input-number v-model="iltConfig.quantization_start_iter" :min="0" :max="1000" class="w-full" />
                      </el-form-item>
                    </el-col>
                    <el-col :span="12">
                      <el-form-item label="量化调度">
                        <el-select v-model="iltConfig.quantization_schedule" style="width: 100%">
                          <el-option label="阶跃式 (Step)" value="step" />
                          <el-option label="线性 (Linear)" value="linear" />
                          <el-option label="余弦 (Cosine)" value="cosine" />
                        </el-select>
                      </el-form-item>
                    </el-col>
                  </el-row>
                  <el-form-item label="量化强度">
                    <el-slider v-model="iltConfig.quantization_strength" :min="0" :max="1" :step="0.05" />
                  </el-form-item>
                </template>

                <el-divider content-position="left"><span class="divider-label">光刻胶模型</span></el-divider>
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="光刻胶陡度 k">
                      <el-input-number v-model="iltConfig.resist_steepness" :min="1" :max="200" :step="5" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="光刻胶阈值">
                      <el-input-number v-model="iltConfig.wafer_threshold" :min="0" :max="1" :step="0.05" :precision="2" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>

                <el-divider content-position="left"><span class="divider-label">损失函数权重</span></el-divider>
                <el-form-item label="晶圆 L2 损失权重">
                  <el-slider v-model="iltConfig.l2_wafer_weight" :min="0" :max="5" :step="0.1" />
                </el-form-item>
                <el-form-item label="二值化惩罚权重">
                  <el-slider v-model="iltConfig.binary_penalty_weight" :min="0" :max="2" :step="0.05" />
                </el-form-item>
                <el-form-item label="TV 平滑权重">
                  <el-slider v-model="iltConfig.tv_smooth_weight" :min="0" :max="1" :step="0.01" />
                </el-form-item>

                <el-divider content-position="left"><span class="divider-label">复杂度惩罚</span></el-divider>
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="周长惩罚权重">
                      <el-input-number v-model="iltConfig.complexity.perimeter_weight" :min="0" :step="0.001" :precision="4" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="顶点数惩罚权重">
                      <el-input-number v-model="iltConfig.complexity.vertex_weight" :min="0" :step="0.001" :precision="4" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="辅助特征权重">
                      <el-input-number v-model="iltConfig.complexity.sub_feature_weight" :min="0" :step="0.001" :precision="4" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="最小特征面积">
                      <el-input-number v-model="iltConfig.complexity.sub_feature_min_area" :min="1" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>
                <el-form-item label="最大特征面积">
                  <el-input-number v-model="iltConfig.complexity.sub_feature_max_area" :min="10" class="w-full" />
                </el-form-item>

                <el-form-item label="详细日志">
                  <el-switch v-model="iltConfig.verbose" />
                </el-form-item>
              </el-form>
            </el-tab-pane>
          </el-tabs>

          <div class="form-actions">
            <el-button type="warning" size="large" :icon="VideoPlay" :loading="isRunning" @click="handleRun" class="run-btn">
              运行 ILT 工作流
            </el-button>
            <el-button :icon="RefreshRight" @click="resetConfig">重置参数</el-button>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="10">
        <TaskMonitor task-type="ilt" @task-complete="onTaskComplete" />
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import { Aim, VideoPlay, RefreshRight } from '@element-plus/icons-vue'
import { useConfigStore } from '@/stores/config'
import { workflowApi } from '@/api'
import GdsUploader from './GdsUploader.vue'
import TaskMonitor from './TaskMonitor.vue'
import type { ILTConfigParams } from '@/types/workflow'

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

const defaultIltConfig: ILTConfigParams = {
  max_iter: 200,
  learning_rate: 0.01,
  optimizer_type: 'adam_projection',
  convergence_tol: 1e-6,
  convergence_patience: 20,
  transmission_level: 'continuous',
  quantization_start_iter: 100,
  quantization_schedule: 'linear',
  quantization_strength: 1.0,
  resist_steepness: 50.0,
  wafer_threshold: 0.3,
  l2_wafer_weight: 1.0,
  complexity: {
    perimeter_weight: 0.0,
    vertex_weight: 0.0,
    sub_feature_weight: 0.0,
    sub_feature_min_area: 2,
    sub_feature_max_area: 100,
  },
  binary_penalty_weight: 0.0,
  tv_smooth_weight: 0.0,
  verbose: true,
}

const iltConfig = reactive<ILTConfigParams>(JSON.parse(JSON.stringify(defaultIltConfig)))

function onGdsSelect(file: any, layer: number, datatype: number) {
  selectedGdsLayer.value = layer
}

async function handleRun() {
  if (!configStore.config?.optical_system) {
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

    const res: any = await workflowApi.runIlt(
      configStore.config.optical_system,
      iltConfig,
      patternType.value,
      patternParamsObj
    )

    if (res.success) {
      ElMessage.success(`ILT 任务已提交：${res.task_id}`)
      emit('task-submit', res.task_id)
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
  Object.assign(iltConfig, JSON.parse(JSON.stringify(defaultIltConfig)))
}

const emit = defineEmits<{
  (e: 'task-submit', taskId: string): void
  (e: 'task-complete', taskId: string): void
}>()

function onTaskComplete(taskId: string) {
  emit('task-complete', taskId)
}
</script>

<style lang="scss" scoped>
.ilt-console {
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

  :deep(.el-slider) {
    margin: 0;
  }
}
</style>
