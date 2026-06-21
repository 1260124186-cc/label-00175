<template>
  <div class="top-controls">
    <div class="controls-left">
      <div class="size-controls">
        <span class="control-label">画布尺寸</span>
        <el-input-number
          v-model="localWidth"
          :min="16"
          :max="2048"
          :step="8"
          size="small"
          controls-position="right"
          @change="handleSizeChange"
          style="width: 100px"
        />
        <span class="size-sep">×</span>
        <el-input-number
          v-model="localHeight"
          :min="16"
          :max="2048"
          :step="8"
          size="small"
          controls-position="right"
          @change="handleSizeChange"
          style="width: 100px"
        />
        <span class="size-unit">px</span>
      </div>

      <el-divider direction="vertical" />

      <div class="mode-toggle">
        <el-radio-group v-model="editMode" size="small" @change="handleModeChange">
          <el-radio-button label="mask">
            <el-icon><Grid /></el-icon>
            <span>掩模模式</span>
          </el-radio-button>
          <el-radio-button label="pupil">
            <el-icon><Sunny /></el-icon>
            <span>Pupil 模式</span>
          </el-radio-button>
        </el-radio-group>
      </div>
    </div>

    <div class="controls-center">
      <el-button
        type="primary"
        :icon="VideoPlay"
        :loading="submitting"
        size="default"
        @click="handleSubmitSimulation"
        class="submit-btn"
      >
        提交仿真
      </el-button>
      <el-button
        :icon="Download"
        size="default"
        @click="handleExportMenu"
        ref="exportBtnRef"
      >
        导出
      </el-button>
      <el-button
        :icon="Upload"
        size="default"
        @click="handleImport"
      >
        导入
      </el-button>
      <input
        ref="fileInputRef"
        type="file"
        accept=".png,.json,.jpg,.jpeg"
        style="display: none"
        @change="handleFileSelect"
      />
    </div>

    <div class="controls-right">
      <el-button size="small" :icon="RefreshLeft" @click="$emit('clear-active-layer')">
        清空当前层
      </el-button>
      <el-button size="small" type="danger" :icon="Delete" @click="$emit('clear-all')">
        清空全部
      </el-button>
    </div>

    <el-dropdown
      ref="exportDropdownRef"
      @command="handleExportCommand"
      trigger="click"
    >
      <span style="display: none"></span>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item command="png_mask">
            <el-icon><Picture /></el-icon>
            导出掩模为 PNG
          </el-dropdown-item>
          <el-dropdown-item command="png_pupil">
            <el-icon><Picture /></el-icon>
            导出 Pupil 为 PNG
          </el-dropdown-item>
          <el-dropdown-item command="json_mask">
            <el-icon><Document /></el-icon>
            导出掩模数据 (JSON)
          </el-dropdown-item>
          <el-dropdown-item command="json_all">
            <el-icon><Document /></el-icon>
            导出全部数据 (JSON)
          </el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import {
  VideoPlay, Download, Upload, RefreshLeft, Delete,
  Grid, Sunny, Picture, Document
} from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'

interface Props {
  width: number
  height: number
  editMode: 'mask' | 'pupil'
  submitting?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  submitting: false
})

const emit = defineEmits<{
  (e: 'size-change', width: number, height: number): void
  (e: 'mode-change', mode: 'mask' | 'pupil'): void
  (e: 'submit-simulation'): void
  (e: 'export', format: string): void
  (e: 'import', file: File): void
  (e: 'clear-active-layer'): void
  (e: 'clear-all'): void
}>()

const localWidth = ref(props.width)
const localHeight = ref(props.height)
const exportBtnRef = ref()
const fileInputRef = ref<HTMLInputElement | null>(null)
const exportDropdownRef = ref()

watch(() => props.width, (val) => {
  localWidth.value = val
})

watch(() => props.height, (val) => {
  localHeight.value = val
})

function handleSizeChange() {
  emit('size-change', localWidth.value, localHeight.value)
}

function handleModeChange(val: string) {
  emit('mode-change', val as 'mask' | 'pupil')
}

async function handleSubmitSimulation() {
  try {
    await ElMessageBox.confirm(
      '确认提交当前掩模和光源配置到后端进行仿真？',
      '提交仿真',
      {
        confirmButtonText: '确认提交',
        cancelButtonText: '取消',
        type: 'info'
      }
    )
    emit('submit-simulation')
  } catch (e) {
    // User cancelled
  }
}

function handleExportMenu() {
  if (exportDropdownRef.value) {
    exportDropdownRef.value.handleClick()
  }
}

function handleExportCommand(command: string) {
  emit('export', command)
  ElMessage.success('导出成功')
}

function handleImport() {
  fileInputRef.value?.click()
}

function handleFileSelect(e: Event) {
  const target = e.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    emit('import', file)
  }
  target.value = ''
}
</script>

<style lang="scss" scoped>
.top-controls {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  background: #fff;
  border-bottom: 1px solid #e4e7ed;
  gap: 16px;
  flex-wrap: wrap;

  .controls-left,
  .controls-center,
  .controls-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .control-label {
    font-size: 13px;
    color: #606266;
    margin-right: 4px;
  }

  .size-controls {
    display: flex;
    align-items: center;
    gap: 6px;

    .size-sep {
      color: #909399;
      font-weight: 500;
    }

    .size-unit {
      color: #909399;
      font-size: 12px;
    }
  }

  .mode-toggle {
    :deep(.el-radio-button__inner) {
      display: flex;
      align-items: center;
      gap: 4px;
    }
  }

  .submit-btn {
    min-width: 100px;
    font-weight: 500;
  }
}

@media (max-width: 1024px) {
  .top-controls {
    .controls-right {
      display: none;
    }
  }
}
</style>
