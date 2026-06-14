<template>
  <div class="form-group-wrap">
    <h3 class="form-section-title">损失权重配置</h3>
    <el-alert
      title="复合损失公式：Loss = w_mse·MSE + w_ssim·(1-SSIM) + w_pvb·PVB + w_mask·TV(mask) + w_wmse·WMSE + w_wmae·WMAE"
      type="info"
      :closable="false"
      style="margin-bottom: 16px"
      show-icon
    />
    <el-form :model="formData" label-width="180px" label-position="right">
      <el-row :gutter="24">
        <el-col :span="8">
          <el-form-item label="MSE 权重 (w_mse)">
            <el-slider
              v-model="formData.mse"
              :min="0"
              :max="10"
              :step="0.05"
              :show-input="true"
              input-size="small"
            />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item label="SSIM 权重 (w_ssim)">
            <el-slider
              v-model="formData.ssim"
              :min="0"
              :max="10"
              :step="0.05"
              :show-input="true"
              input-size="small"
            />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item label="PVB 权重 (w_pvb)">
            <el-slider
              v-model="formData.pvb"
              :min="0"
              :max="10"
              :step="0.05"
              :show-input="true"
              input-size="small"
            />
          </el-form-item>
        </el-col>
      </el-row>
      <el-row :gutter="24">
        <el-col :span="8">
          <el-form-item label="掩模复杂度 (TV)">
            <el-slider
              v-model="formData.mask_complexity"
              :min="0"
              :max="10"
              :step="0.05"
              :show-input="true"
              input-size="small"
            />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item label="空间加权 MSE">
            <el-slider
              v-model="formData.weighted_mse"
              :min="0"
              :max="10"
              :step="0.05"
              :show-input="true"
              input-size="small"
            />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item label="空间加权 MAE">
            <el-slider
              v-model="formData.weighted_mae"
              :min="0"
              :max="10"
              :step="0.05"
              :show-input="true"
              input-size="small"
            />
          </el-form-item>
        </el-col>
      </el-row>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { reactive, watch } from 'vue'
import type { LossWeights } from '@/types/config'

const props = defineProps<{
  modelValue: LossWeights
}>()

const emit = defineEmits<{
  'update:modelValue': [value: LossWeights]
}>()

const formData = reactive<LossWeights>({ ...props.modelValue })

watch(
  () => props.modelValue,
  (val) => Object.assign(formData, val),
  { deep: true }
)

watch(
  formData,
  (val) => emit('update:modelValue', { ...val }),
  { deep: true }
)
</script>
