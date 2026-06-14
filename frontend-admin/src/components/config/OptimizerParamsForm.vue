<template>
  <div class="form-group-wrap">
    <h3 class="form-section-title">优化器参数</h3>
    <el-form :model="formData" label-width="140px" label-position="right">
      <el-row :gutter="24">
        <el-col :span="12">
          <el-form-item label="优化器类型">
            <el-select v-model="formData.optimizer_type" style="width: 100%">
              <el-option label="梯度下降 (Gradient Descent)" value="gradient_descent" />
              <el-option label="BFGS (拟牛顿法)" value="bfgs" />
              <el-option label="Newton (牛顿法)" value="newton" />
              <el-option label="遗传算法 (Genetic)" value="genetic" />
              <el-option label="粒子群 (PSO)" value="pso" />
              <el-option label="强化学习 (RL)" value="rl" />
            </el-select>
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="最大迭代次数">
            <el-input-number
              v-model="formData.max_iter"
              :min="1"
              :max="100000"
              :step="10"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24" v-if="isGradientBased">
        <el-col :span="12">
          <el-form-item label="学习率">
            <el-input-number
              v-model="formData.learning_rate"
              :min="1e-8"
              :max="10"
              :step="0.001"
              :precision="6"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="收敛容差">
            <el-input-number
              v-model="formData.tol"
              :min="1e-12"
              :max="1"
              :step="1e-6"
              :precision="10"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24">
        <el-col :span="12">
          <el-form-item label="早停耐心值">
            <el-input-number
              v-model="formData.early_stop_patience"
              :min="0"
              :max="500"
              :step="1"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="优化目标指标">
            <el-select v-model="formData.metric" style="width: 100%" :disabled="formData.use_composite_loss">
              <el-option label="均方误差 (MSE)" value="mse" />
              <el-option label="平均绝对误差 (MAE)" value="mae" />
              <el-option label="结构相似度 (SSIM)" value="ssim" />
            </el-select>
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24" v-if="isGradientBased">
        <el-col :span="12">
          <el-form-item label="学习率调度器">
            <el-select v-model="formData.lr_scheduler" style="width: 100%" clearable placeholder="不使用调度器">
              <el-option label="阶梯衰减 (Step)" value="step" />
              <el-option label="指数衰减 (Exponential)" value="exponential" />
              <el-option label="余弦退火 (Cosine)" value="cosine" />
            </el-select>
          </el-form-item>
        </el-col>
        <el-col :span="12" v-if="formData.lr_scheduler">
          <el-form-item label="学习率衰减率">
            <el-input-number
              v-model="formData.lr_decay"
              :min="0.1"
              :max="0.999"
              :step="0.01"
              :precision="4"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24" v-if="formData.lr_scheduler === 'step'">
        <el-col :span="12">
          <el-form-item label="调度步长">
            <el-input-number
              v-model="formData.lr_step_size"
              :min="1"
              :max="1000"
              :step="1"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24">
        <el-col :span="12">
          <el-form-item label="掩模值下限">
            <el-input-number
              v-model="formData.bounds[0]"
              :min="0"
              :max="1"
              :step="0.01"
              :precision="3"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="掩模值上限">
            <el-input-number
              v-model="formData.bounds[1]"
              :min="0"
              :max="1"
              :step="0.01"
              :precision="3"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24">
        <el-col :span="12">
          <el-form-item label="启用复合损失">
            <el-switch v-model="formData.use_composite_loss" />
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="输出详细日志">
            <el-switch v-model="formData.verbose" />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24">
        <el-col :span="12">
          <el-form-item label="随机种子">
            <el-input-number
              v-model="formData.random_seed"
              :min="0"
              :step="1"
              :controls="true"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="并行进程数">
            <el-input-number
              v-model="formData.n_jobs"
              :min="-1"
              :step="1"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-divider content-position="left" v-if="isHeuristic">启发式算法参数</el-divider>
      <el-row :gutter="24" v-if="isHeuristic">
        <el-col :span="8">
          <el-form-item label="种群大小">
            <el-input-number
              v-model="formData.population_size"
              :min="2"
              :max="5000"
              :step="10"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item label="交叉概率">
            <el-input-number
              v-model="formData.crossover_rate"
              :min="0"
              :max="1"
              :step="0.05"
              :precision="3"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item label="变异概率">
            <el-input-number
              v-model="formData.mutation_rate"
              :min="0"
              :max="1"
              :step="0.01"
              :precision="3"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-divider content-position="left" v-if="isRL">强化学习参数</el-divider>
      <el-row :gutter="24" v-if="isRL">
        <el-col :span="8">
          <el-form-item label="折扣因子 γ">
            <el-input-number
              v-model="formData.rl_gamma"
              :min="0"
              :max="1"
              :step="0.01"
              :precision="3"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item label="初始探索率 ε">
            <el-input-number
              v-model="formData.rl_epsilon"
              :min="0"
              :max="1"
              :step="0.01"
              :precision="3"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item label="探索率衰减">
            <el-input-number
              v-model="formData.rl_epsilon_decay"
              :min="0.5"
              :max="1"
              :step="0.001"
              :precision="4"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { reactive, computed, watch } from 'vue'
import type { Optimization } from '@/types/config'

const props = defineProps<{
  modelValue: Optimization
}>()

const emit = defineEmits<{
  'update:modelValue': [value: Optimization]
}>()

const formData = reactive<Optimization>({
  ...props.modelValue,
  loss_weights: { ...props.modelValue.loss_weights },
  spatial_weight: { ...props.modelValue.spatial_weight },
  regularization: { ...props.modelValue.regularization },
  bounds: [...props.modelValue.bounds] as [number, number]
})

const isGradientBased = computed(
  () => ['gradient_descent', 'bfgs', 'newton'].includes(formData.optimizer_type)
)
const isHeuristic = computed(() => ['genetic', 'pso'].includes(formData.optimizer_type))
const isRL = computed(() => formData.optimizer_type === 'rl')

watch(
  () => props.modelValue,
  (val) => {
    Object.assign(formData, val)
    formData.loss_weights = { ...val.loss_weights }
    formData.spatial_weight = { ...val.spatial_weight }
    formData.regularization = { ...val.regularization }
    formData.bounds = [...val.bounds] as [number, number]
  },
  { deep: true }
)

watch(
  formData,
  (val) => {
    emit('update:modelValue', JSON.parse(JSON.stringify(val)))
  },
  { deep: true }
)
</script>
