<template>
  <div class="form-group-wrap">
    <h3 class="form-section-title">输出与成像配置</h3>
    <el-row :gutter="24">
      <el-col :span="12">
        <h4 class="sub-title">输出配置</h4>
        <el-form :model="outputData" label-width="140px" label-position="right">
          <el-form-item label="结果保存目录">
            <el-input v-model="outputData.save_dir" placeholder="./results" />
          </el-form-item>
          <el-form-item label="图像格式">
            <el-select v-model="outputData.image_format" style="width: 100%">
              <el-option label="PNG" value="png" />
              <el-option label="TIFF" value="tiff" />
            </el-select>
          </el-form-item>
          <el-form-item label="日志级别">
            <el-select v-model="outputData.log_level" style="width: 100%">
              <el-option label="DEBUG" value="DEBUG" />
              <el-option label="INFO" value="INFO" />
              <el-option label="WARNING" value="WARNING" />
              <el-option label="ERROR" value="ERROR" />
            </el-select>
          </el-form-item>
          <el-form-item label="保存图像">
            <el-switch v-model="outputData.save_images" />
          </el-form-item>
          <el-form-item label="保存历史数据">
            <el-switch v-model="outputData.save_history" />
          </el-form-item>
        </el-form>
      </el-col>
      <el-col :span="12">
        <h4 class="sub-title">成像模拟配置</h4>
        <el-form :model="imagingData" label-width="140px" label-position="right">
          <el-form-item label="光刻胶阈值">
            <el-slider
              v-model="imagingData.resist_threshold"
              :min="0"
              :max="1"
              :step="0.01"
              :show-input="true"
              input-size="small"
            />
          </el-form-item>
          <el-form-item label="应用光刻胶响应">
            <el-switch v-model="imagingData.apply_resist" />
          </el-form-item>
        </el-form>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { reactive, watch } from 'vue'
import type { OutputConfig, ImagingConfig } from '@/types/config'

const props = defineProps<{
  outputValue: OutputConfig
  imagingValue: ImagingConfig
}>()

const emit = defineEmits<{
  'update:outputValue': [value: OutputConfig]
  'update:imagingValue': [value: ImagingConfig]
}>()

const outputData = reactive<OutputConfig>({ ...props.outputValue })
const imagingData = reactive<ImagingConfig>({ ...props.imagingValue })

watch(
  () => props.outputValue,
  (val) => Object.assign(outputData, val),
  { deep: true }
)
watch(
  () => props.imagingValue,
  (val) => Object.assign(imagingData, val),
  { deep: true }
)

watch(outputData, (val) => emit('update:outputValue', { ...val }), { deep: true })
watch(imagingData, (val) => emit('update:imagingValue', { ...val }), { deep: true })
</script>

<style lang="scss" scoped>
.sub-title {
  font-size: 14px;
  font-weight: 600;
  color: #606266;
  margin: 0 0 16px 0;
  padding-left: 8px;
  border-left: 2px solid #a0cfff;
}
</style>
