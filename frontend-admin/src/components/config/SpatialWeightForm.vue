<template>
  <div class="form-group-wrap">
    <div class="form-header">
      <h3 class="form-section-title" style="margin: 0">空间加权误差配置</h3>
      <el-switch v-model="formData.enable" />
    </div>
    <el-alert
      title="对边缘、拐角、线端等关键区域设置更高的权重，提升优化效果"
      type="warning"
      :closable="false"
      style="margin-bottom: 16px"
      show-icon
    />
    <el-form :model="formData" label-width="180px" label-position="right" :disabled="!formData.enable">
      <el-row :gutter="24">
        <el-col :span="8">
          <el-form-item label="边缘区域权重">
            <el-input-number
              v-model="formData.edge_weight"
              :min="1"
              :max="20"
              :step="0.5"
              :precision="2"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item label="拐角区域权重">
            <el-input-number
              v-model="formData.corner_weight"
              :min="1"
              :max="50"
              :step="0.5"
              :precision="2"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item label="线端区域权重">
            <el-input-number
              v-model="formData.line_end_weight"
              :min="1"
              :max="50"
              :step="0.5"
              :precision="2"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24">
        <el-col :span="8">
          <el-form-item label="基础区域权重">
            <el-input-number
              v-model="formData.base_weight"
              :min="0.1"
              :max="5"
              :step="0.1"
              :precision="2"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item label="边缘检测 sigma">
            <el-input-number
              v-model="formData.edge_sigma"
              :min="0.1"
              :max="10"
              :step="0.1"
              :precision="2"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item label="拐角检测阈值">
            <el-slider
              v-model="formData.corner_threshold"
              :min="0"
              :max="1"
              :step="0.05"
              :show-input="true"
              input-size="small"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24">
        <el-col :span="8">
          <el-form-item label="线端检测阈值">
            <el-slider
              v-model="formData.line_end_threshold"
              :min="0"
              :max="1"
              :step="0.05"
              :show-input="true"
              input-size="small"
            />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item label="权重平滑 sigma">
            <el-input-number
              v-model="formData.smooth_sigma"
              :min="0"
              :max="5"
              :step="0.1"
              :precision="2"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item label="归一化权重到均值1">
            <el-switch v-model="formData.normalize" />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24">
        <el-col :span="8">
          <el-form-item label="形态学腐蚀权重">
            <el-switch v-model="formData.weight_erosion" />
          </el-form-item>
        </el-col>
      </el-row>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { reactive, watch } from 'vue'
import type { SpatialWeight } from '@/types/config'

const props = defineProps<{
  modelValue: SpatialWeight
}>()

const emit = defineEmits<{
  'update:modelValue': [value: SpatialWeight]
}>()

const formData = reactive<SpatialWeight>({ ...props.modelValue })

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

<style lang="scss" scoped>
.form-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 12px;
  border-bottom: 1px solid #ebeef5;
  margin-bottom: 16px;
}
</style>
