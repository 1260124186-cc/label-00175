<template>
  <div class="smo-console">
    <el-row :gutter="20">
      <el-col :xs="24" :lg="14">
        <el-card class="card" shadow="hover">
          <div class="card-header">
            <div class="title">
              <el-icon size="20" color="#67c23a"><Sunny /></el-icon>
              <span>SMO 光源掩模协同优化</span>
            </div>
            <el-tag type="success" effect="plain" round size="small">Source Mask Optimization</el-tag>
          </div>

          <el-alert
            title="SMO 同时优化光源形状和掩模图形，最大化光刻工艺窗口和成像质量"
            type="success"
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

            <el-tab-pane label="SMO 参数" name="params">
              <el-form label-width="160px" label-position="right" size="default">
                <el-divider content-position="left"><span class="divider-label">优化策略</span></el-divider>
                <el-form-item label="优化策略">
                  <el-select v-model="smoConfig.strategy" style="width: 100%">
                    <el-option label="交替优化 (Alternating)" value="alternating" />
                    <el-option label="联合梯度 (Joint Gradient)" value="joint_gradient" />
                    <el-option label="光源优先 (Source First)" value="source_first" />
                  </el-select>
                </el-form-item>
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="外层迭代次数">
                      <el-input-number v-model="smoConfig.max_outer_iterations" :min="1" :max="100" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="收敛耐心值">
                      <el-input-number v-model="smoConfig.convergence_patience" :min="0" :max="50" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>
                <el-form-item label="收敛容差">
                  <el-input-number v-model="smoConfig.tol" :min="1e-8" :step="1e-6" :precision="6" class="w-full" />
                </el-form-item>

                <el-divider content-position="left"><span class="divider-label">交替优化参数</span></el-divider>
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="光源最大迭代">
                      <el-input-number v-model="smoConfig.source_max_iter" :min="10" :max="500" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="掩模最大迭代">
                      <el-input-number v-model="smoConfig.mask_max_iter" :min="10" :max="500" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="光源学习率">
                      <el-input-number v-model="smoConfig.source_learning_rate" :min="0.001" :step="0.005" :precision="4" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="掩模学习率">
                      <el-input-number v-model="smoConfig.mask_learning_rate" :min="0.001" :step="0.005" :precision="4" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>

                <el-divider content-position="left"><span class="divider-label">联合优化参数</span></el-divider>
                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="联合最大迭代">
                      <el-input-number v-model="smoConfig.joint_max_iter" :min="50" :max="1000" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="联合光源 LR">
                      <el-input-number v-model="smoConfig.joint_learning_rate_source" :min="0.001" :step="0.001" :precision="4" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>
                <el-form-item label="联合掩模 LR">
                  <el-input-number v-model="smoConfig.joint_learning_rate_mask" :min="0.001" :step="0.001" :precision="4" class="w-full" />
                </el-form-item>

                <el-divider content-position="left"><span class="divider-label">光源初始化与约束</span></el-divider>
                <el-form-item label="光源初始化类型">
                  <el-select v-model="smoConfig.source_init_type" style="width: 100%">
                    <el-option label="传统照明 (Conventional)" value="conventional" />
                    <el-option label="环形照明 (Annular)" value="annular" />
                    <el-option label="偶极照明 (Dipole)" value="dipole" />
                    <el-option label="四极照明 (Quasar)" value="quasar" />
                    <el-option label="均匀圆盘 (Uniform Disk)" value="uniform_disk" />
                    <el-option label="随机初始化 (Random)" value="random" />
                  </el-select>
                </el-form-item>

                <el-form-item label="能量守恒约束">
                  <el-switch v-model="smoConfig.source_constraints.energy_conservation" />
                </el-form-item>
                <template v-if="smoConfig.source_constraints.energy_conservation">
                  <el-form-item label="目标总能量">
                    <el-input-number v-model="smoConfig.source_constraints.energy_target" :min="0.1" :step="0.1" class="w-full" />
                  </el-form-item>
                </template>

                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="目标 Sigma">
                      <el-input-number v-model="sigmaTargetValue" :min="0" :max="1" :step="0.05" :precision="3" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="Sigma 容差">
                      <el-input-number v-model="smoConfig.source_constraints.sigma_tolerance" :min="0" :step="0.01" :precision="3" class="w-full" />
                    </el-form-item>
                  </el-col>
                </el-row>

                <el-row :gutter="24">
                  <el-col :span="12">
                    <el-form-item label="平滑权重">
                      <el-input-number v-model="smoConfig.source_constraints.smoothness_weight" :min="0" :step="0.005" :precision="4" class="w-full" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="平滑类型">
                      <el-select v-model="smoConfig.source_constraints.smoothness_type" style="width: 100%">
                        <el-option label="TV 全变差" value="tv" />
                        <el-option label="高斯平滑" value="gaussian" />
                      </el-select>
                    </el-form-item>
                  </el-col>
                </el-row>

                <el-form-item label="非负约束">
                  <el-switch v-model="smoConfig.source_constraints.non_negative" />
                </el-form-item>

                <el-divider content-position="left"><span class="divider-label">损失函数</span></el-divider>
                <el-form-item label="使用晶圆图像损失">
                  <el-switch v-model="smoConfig.use_wafer_image_loss" />
                </el-form-item>
                <el-form-item label="PVB 损失权重">
                  <el-input-number v-model="smoConfig.pvb_weight" :min="0" :step="0.1" :precision="2" class="w-full" />
                </el-form-item>
                <el-form-item label="光刻胶阈值">
                  <el-input-number v-model="smoConfig.wafer_threshold" :min="0" :max="1" :step="0.05" class="w-full" />
                </el-form-item>

                <el-form-item label="详细日志">
                  <el-switch v-model="smoConfig.verbose" />
                </el-form-item>
              </el-form>
            </el-tab-pane>
          </el-tabs>

          <div class="form-actions">
            <el-button type="success" size="large" :icon="VideoPlay" :loading="isRunning" @click="handleRun" class="run-btn">
              运行 SMO 工作流
            </el-button>
            <el-button :icon="RefreshRight" @click="resetConfig">重置参数</el-button>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="10">
        <TaskMonitor task-type="smo" @task-complete="onTaskComplete" />
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { Sunny, VideoPlay, RefreshRight } from '@element-plus/icons-vue'
import { useConfigStore } from '@/stores/config'
import { workflowApi } from '@/api'
import GdsUploader from './GdsUploader.vue'
import TaskMonitor from './TaskMonitor.vue'
import type { SMOConfigParams } from '@/types/workflow'

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

const defaultSmoConfig: SMOConfigParams = {
  strategy: 'alternating',
  max_outer_iterations: 20,
  source_max_iter: 50,
  mask_max_iter: 100,
  joint_max_iter: 200,
  source_learning_rate: 0.005,
  mask_learning_rate: 0.01,
  joint_learning_rate_source: 0.003,
  joint_learning_rate_mask: 0.008,
  tol: 1e-5,
  convergence_patience: 5,
  source_init_type: 'conventional',
  source_constraints: {
    energy_conservation: true,
    energy_target: 1.0,
    sigma_target: null,
    sigma_tolerance: 0.02,
    smoothness_weight: 0.01,
    smoothness_type: 'tv',
    gaussian_sigma: 1.5,
    non_negative: true,
    support_radius: null,
    support_radius_inner: null,
  },
  wafer_threshold: 0.3,
  use_wafer_image_loss: true,
  pvb_weight: 0.0,
  verbose: true,
}

const smoConfig = reactive<SMOConfigParams>(JSON.parse(JSON.stringify(defaultSmoConfig)))

const sigmaTargetValue = computed({
  get: () => smoConfig.source_constraints.sigma_target ?? 0.75,
  set: (val: number) => {
    smoConfig.source_constraints.sigma_target = val <= 0 ? null : val
  },
})

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

    const res = await workflowApi.runSmo(
      config.optical_system,
      smoConfig,
      patternType.value,
      patternParamsObj
    )

    if (res.success) {
      ElMessage.success(`SMO 任务已提交：${res.task_id}`)
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
  Object.assign(smoConfig, JSON.parse(JSON.stringify(defaultSmoConfig)))
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
.smo-console {
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
