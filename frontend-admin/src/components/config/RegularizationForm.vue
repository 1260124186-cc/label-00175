<template>
  <div class="form-group-wrap">
    <h3 class="form-section-title">正则化配置</h3>
    <el-form :model="formData" label-width="180px" label-position="right">
      <el-row :gutter="24">
        <el-col :span="12">
          <el-form-item label="正则化类型">
            <el-select v-model="formData.type" style="width: 100%" clearable placeholder="不使用正则化">
              <el-option label="L1 正则化 (Lasso)" value="l1" />
              <el-option label="L2 正则化 (Ridge)" value="l2" />
              <el-option label="全变差正则化 (TV)" value="tv" />
            </el-select>
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="正则化强度">
            <el-input-number
              v-model="formData.strength"
              :min="0"
              :max="100"
              :step="0.001"
              :precision="5"
              controls-position="right"
              :disabled="!formData.type"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { reactive, watch } from 'vue'
import type { Regularization } from '@/types/config'

const props = defineProps<{
  modelValue: Regularization
}>()

const emit = defineEmits<{
  'update:modelValue': [value: Regularization]
}>()

const formData = reactive<Regularization>({ ...props.modelValue })

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
