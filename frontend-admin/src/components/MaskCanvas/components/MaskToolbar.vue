<template>
  <div class="mask-toolbar">
    <div class="toolbar-section">
      <div class="section-title">绘制工具</div>
      <div class="tool-group">
        <el-tooltip content="画笔 (B)" placement="right">
          <button
            class="tool-btn"
            :class="{ active: currentTool === 'brush' }"
            @click="$emit('tool-change', 'brush')"
          >
            <el-icon><Brush /></el-icon>
          </button>
        </el-tooltip>
        <el-tooltip content="橡皮擦 (E)" placement="right">
          <button
            class="tool-btn"
            :class="{ active: currentTool === 'eraser' }"
            @click="$emit('tool-change', 'eraser')"
          >
            <el-icon><Delete /></el-icon>
          </button>
        </el-tooltip>
        <el-tooltip content="多边形 (P)" placement="right">
          <button
            class="tool-btn"
            :class="{ active: currentTool === 'polygon' }"
            @click="$emit('tool-change', 'polygon')"
          >
            <el-icon><Connection /></el-icon>
          </button>
        </el-tooltip>
        <el-tooltip content="矩形 (R)" placement="right">
          <button
            class="tool-btn"
            :class="{ active: currentTool === 'rectangle' }"
            @click="$emit('tool-change', 'rectangle')"
          >
            <el-icon><Grid /></el-icon>
          </button>
        </el-tooltip>
        <el-tooltip content="圆形 (C)" placement="right">
          <button
            class="tool-btn"
            :class="{ active: currentTool === 'circle' }"
            @click="$emit('tool-change', 'circle')"
          >
            <el-icon><CircleCheck /></el-icon>
          </button>
        </el-tooltip>
      </div>
    </div>

    <el-divider />

    <div class="toolbar-section">
      <div class="section-title">光源编辑</div>
      <div class="tool-group">
        <el-tooltip content="Pupil 画笔" placement="right">
          <button
            class="tool-btn pupil"
            :class="{ active: currentTool === 'pupil_brush' }"
            @click="$emit('tool-change', 'pupil_brush')"
          >
            <el-icon><Sunny /></el-icon>
          </button>
        </el-tooltip>
        <el-tooltip content="Pupil 橡皮擦" placement="right">
          <button
            class="tool-btn pupil"
            :class="{ active: currentTool === 'pupil_eraser' }"
            @click="$emit('tool-change', 'pupil_eraser')"
          >
            <el-icon><Moon /></el-icon>
          </button>
        </el-tooltip>
      </div>
    </div>

    <el-divider />

    <div class="toolbar-section">
      <div class="section-title">视图</div>
      <div class="tool-group">
        <el-tooltip content="移动 (H)" placement="right">
          <button
            class="tool-btn"
            :class="{ active: currentTool === 'move' }"
            @click="$emit('tool-change', 'move')"
          >
            <el-icon><Rank /></el-icon>
          </button>
        </el-tooltip>
        <el-tooltip content="缩放 (Z)" placement="right">
          <button
            class="tool-btn"
            :class="{ active: currentTool === 'zoom' }"
            @click="$emit('tool-change', 'zoom')"
          >
            <el-icon><ZoomIn /></el-icon>
          </button>
        </el-tooltip>
        <el-tooltip content="适应视图" placement="right">
          <button class="tool-btn" @click="$emit('fit-view')">
            <el-icon><FullScreen /></el-icon>
          </button>
        </el-tooltip>
        <el-tooltip content="重置视图" placement="right">
          <button class="tool-btn" @click="$emit('reset-view')">
            <el-icon><RefreshRight /></el-icon>
          </button>
        </el-tooltip>
      </div>
    </div>

    <el-divider />

    <div class="toolbar-section">
      <div class="section-title">撤销/重做</div>
      <div class="tool-group">
        <el-tooltip content="撤销 (Ctrl+Z)" placement="right">
          <button class="tool-btn" :disabled="!canUndo" @click="$emit('undo')">
            <el-icon><DArrowLeft /></el-icon>
          </button>
        </el-tooltip>
        <el-tooltip content="重做 (Ctrl+Y)" placement="right">
          <button class="tool-btn" :disabled="!canRedo" @click="$emit('redo')">
            <el-icon><DArrowRight /></el-icon>
          </button>
        </el-tooltip>
      </div>
    </div>

    <el-divider />

    <div class="toolbar-section">
      <div class="section-title">画笔设置</div>
      <div class="brush-settings">
        <div class="setting-item">
          <span class="setting-label">大小</span>
          <el-slider
            :model-value="brushSize"
            :min="1"
            :max="100"
            size="small"
            @update:model-value="$emit('brush-size-change', $event)"
          />
          <span class="setting-value">{{ brushSize }}px</span>
        </div>
        <div class="setting-item">
          <span class="setting-label">硬度</span>
          <el-slider
            :model-value="brushHardness"
            :min="0"
            :max="1"
            :step="0.1"
            size="small"
            @update:model-value="$emit('brush-hardness-change', $event)"
          />
          <span class="setting-value">{{ Math.round(brushHardness * 100) }}%</span>
        </div>
        <div class="setting-item">
          <span class="setting-label">不透明度</span>
          <el-slider
            :model-value="brushOpacity"
            :min="0"
            :max="1"
            :step="0.05"
            size="small"
            @update:model-value="$emit('brush-opacity-change', $event)"
          />
          <span class="setting-value">{{ Math.round(brushOpacity * 100) }}%</span>
        </div>
        <div class="setting-item">
          <span class="setting-label">颜色</span>
          <input
            type="color"
            :value="brushColor"
            class="color-picker"
            @input="$emit('brush-color-change', ($event.target as HTMLInputElement).value)"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  Brush, Delete, Connection, Grid, CircleCheck,
  Sunny, Moon, Rank, ZoomIn, FullScreen, RefreshRight,
  DArrowLeft, DArrowRight
} from '@element-plus/icons-vue'
import type { ToolType } from '../types'

interface Props {
  currentTool: ToolType
  brushSize: number
  brushOpacity: number
  brushHardness: number
  brushColor: string
  canUndo: boolean
  canRedo: boolean
}

defineProps<Props>()

defineEmits<{
  (e: 'tool-change', tool: ToolType): void
  (e: 'brush-size-change', size: number): void
  (e: 'brush-opacity-change', opacity: number): void
  (e: 'brush-hardness-change', hardness: number): void
  (e: 'brush-color-change', color: string): void
  (e: 'undo'): void
  (e: 'redo'): void
  (e: 'fit-view'): void
  (e: 'reset-view'): void
}>()
</script>

<style lang="scss" scoped>
.mask-toolbar {
  width: 220px;
  padding: 16px 12px;
  background: #fff;
  border-right: 1px solid #e4e7ed;
  display: flex;
  flex-direction: column;
  gap: 4px;
  overflow-y: auto;

  .toolbar-section {
    .section-title {
      font-size: 12px;
      font-weight: 600;
      color: #606266;
      margin-bottom: 8px;
      padding-left: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
  }

  .tool-group {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .tool-btn {
    width: 36px;
    height: 36px;
    border: 1px solid #dcdfe6;
    border-radius: 6px;
    background: #fff;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #606266;
    transition: all 0.2s;

    &:hover {
      border-color: #409eff;
      color: #409eff;
      background: #ecf5ff;
    }

    &.active {
      background: #409eff;
      border-color: #409eff;
      color: #fff;

      &:hover {
        background: #66b1ff;
        border-color: #66b1ff;
      }
    }

    &.pupil.active {
      background: #e6a23c;
      border-color: #e6a23c;
      color: #fff;

      &:hover {
        background: #ebb563;
        border-color: #ebb563;
      }
    }

    &:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }
  }

  .brush-settings {
    display: flex;
    flex-direction: column;
    gap: 12px;

    .setting-item {
      display: flex;
      flex-direction: column;
      gap: 4px;

      .setting-label {
        font-size: 12px;
        color: #606266;
      }

      .setting-value {
        font-size: 11px;
        color: #909399;
        text-align: right;
        font-family: 'SF Mono', Consolas, monospace;
      }

      :deep(.el-slider) {
        margin: 0;
      }

      .color-picker {
        width: 100%;
        height: 32px;
        border: 1px solid #dcdfe6;
        border-radius: 4px;
        cursor: pointer;
        padding: 2px;
      }
    }
  }
}
</style>
