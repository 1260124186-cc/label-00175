<template>
  <div class="layer-panel">
    <div class="panel-header">
      <div class="title">
        <el-icon size="16"><Files /></el-icon>
        <span>图层</span>
      </div>
      <div class="actions">
        <el-tooltip content="新建图层" placement="top">
          <button class="action-btn" @click="handleAddLayer">
            <el-icon><Plus /></el-icon>
          </button>
        </el-tooltip>
        <el-tooltip content="复制图层" placement="top">
          <button class="action-btn" :disabled="!activeLayerId" @click="$emit('duplicate-layer', activeLayerId!)">
            <el-icon><DocumentCopy /></el-icon>
          </button>
        </el-tooltip>
        <el-tooltip content="删除图层" placement="top">
          <button class="action-btn danger" :disabled="!activeLayerId || layers.length <= 1" @click="$emit('remove-layer', activeLayerId!)">
            <el-icon><Delete /></el-icon>
          </button>
        </el-tooltip>
      </div>
    </div>

    <div class="layer-list">
      <div
        v-for="layer in reversedLayers"
        :key="layer.id"
        class="layer-item"
        :class="{ active: layer.id === activeLayerId, 'layer-mask': layer.type === 'mask', 'layer-pupil': layer.type === 'pupil', 'layer-reference': layer.type === 'reference' }"
        @click="$emit('set-active-layer', layer.id)"
      >
        <div class="layer-left">
          <button class="visibility-btn" @click.stop="$emit('toggle-visibility', layer.id)">
            <el-icon v-if="layer.visible"><View /></el-icon>
            <el-icon v-else><Hide /></el-icon>
          </button>
          <div class="layer-thumb">
            <canvas
              :ref="el => setLayerThumb(layer.id, el as HTMLCanvasElement)"
              width="32"
              height="32"
              class="thumb-canvas"
            />
          </div>
        </div>

        <div class="layer-info">
          <div class="layer-name">{{ layer.name }}</div>
          <div class="layer-type-tag">
            <el-tag :type="layerTypeTag(layer.type).type" size="small" effect="plain">
              {{ layerTypeTag(layer.type).label }}
            </el-tag>
          </div>
        </div>

        <div class="layer-right">
          <div class="layer-actions">
            <button class="move-btn" @click.stop="$emit('move-layer', layer.id, 'up')" :disabled="isFirstLayer(layer.id)">
              <el-icon><ArrowUp /></el-icon>
            </button>
            <button class="move-btn" @click.stop="$emit('move-layer', layer.id, 'down')" :disabled="isLastLayer(layer.id)">
              <el-icon><ArrowDown /></el-icon>
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="opacity-section" v-if="activeLayer">
      <div class="section-label">不透明度</div>
      <el-slider
        :model-value="activeLayer.opacity"
        :min="0"
        :max="1"
        :step="0.05"
        size="small"
        @update:model-value="handleOpacityChange"
      />
      <div class="opacity-value">{{ Math.round((activeLayer.opacity || 1) * 100) }}%</div>
    </div>

    <div class="blend-section" v-if="activeLayer">
      <div class="section-label">混合模式</div>
      <el-select
        :model-value="activeLayer.blendMode"
        size="small"
        style="width: 100%"
        @update:model-value="handleBlendModeChange"
      >
        <el-option label="正常" value="source-over" />
        <el-option label="正片叠底" value="multiply" />
        <el-option label="滤色" value="screen" />
        <el-option label="叠加" value="overlay" />
        <el-option label="变暗" value="darken" />
        <el-option label="变亮" value="lighten" />
      </el-select>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch, nextTick } from 'vue'
import {
  Files, Plus, Delete, DocumentCopy,
  View, Hide, ArrowUp, ArrowDown
} from '@element-plus/icons-vue'
import type { Layer, LayerType } from '../types'

interface Props {
  layers: Layer[]
  activeLayerId: string | null
}

const props = defineProps<Props>()

const emit = defineEmits<{
  (e: 'set-active-layer', id: string): void
  (e: 'toggle-visibility', id: string): void
  (e: 'remove-layer', id: string): void
  (e: 'duplicate-layer', id: string): void
  (e: 'move-layer', id: string, direction: 'up' | 'down'): void
  (e: 'add-layer', type: LayerType): void
  (e: 'change-opacity', id: string, opacity: number): void
  (e: 'change-blend-mode', id: string, mode: GlobalCompositeOperation): void
}>()

const thumbCanvases = ref<Map<string, HTMLCanvasElement>>(new Map())

const reversedLayers = computed(() => [...props.layers].reverse())

const activeLayer = computed(() =>
  props.layers.find(l => l.id === props.activeLayerId) || null
)

function layerTypeTag(type: LayerType): { type: string; label: string } {
  switch (type) {
    case 'mask':
      return { type: 'primary', label: '掩模' }
    case 'pupil':
      return { type: 'warning', label: 'Pupil' }
    case 'reference':
      return { type: 'info', label: '参考' }
    default:
      return { type: 'info', label: type }
  }
}

function isFirstLayer(layerId: string): boolean {
  return props.layers[0]?.id === layerId
}

function isLastLayer(layerId: string): boolean {
  return props.layers[props.layers.length - 1]?.id === layerId
}

function setLayerThumb(layerId: string, canvas: HTMLCanvasElement | null) {
  if (canvas) {
    thumbCanvases.value.set(layerId, canvas)
    renderLayerThumb(layerId)
  }
}

function renderLayerThumb(layerId: string) {
  const canvas = thumbCanvases.value.get(layerId)
  const layer = props.layers.find(l => l.id === layerId)
  if (!canvas || !layer || !layer.data) return

  const ctx = canvas.getContext('2d')!
  ctx.clearRect(0, 0, 32, 32)

  const tempCanvas = document.createElement('canvas')
  tempCanvas.width = layer.width
  tempCanvas.height = layer.height
  const tempCtx = tempCanvas.getContext('2d')!
  tempCtx.putImageData(layer.data, 0, 0)

  ctx.drawImage(tempCanvas, 0, 0, 32, 32)
}

function handleAddLayer() {
  emit('add-layer', 'mask')
}

function handleOpacityChange(val: number) {
  if (props.activeLayerId) {
    emit('change-opacity', props.activeLayerId, val)
  }
}

function handleBlendModeChange(val: GlobalCompositeOperation) {
  if (props.activeLayerId) {
    emit('change-blend-mode', props.activeLayerId, val)
  }
}

watch(
  () => props.layers,
  () => {
    nextTick(() => {
      props.layers.forEach(layer => {
        renderLayerThumb(layer.id)
      })
    })
  },
  { deep: true }
)
</script>

<style lang="scss" scoped>
.layer-panel {
  width: 240px;
  background: #fff;
  border-left: 1px solid #e4e7ed;
  display: flex;
  flex-direction: column;
  overflow: hidden;

  .panel-header {
    padding: 12px 16px;
    border-bottom: 1px solid #ebeef5;
    display: flex;
    justify-content: space-between;
    align-items: center;

    .title {
      display: flex;
      align-items: center;
      gap: 6px;
      font-weight: 600;
      font-size: 14px;
      color: #303133;
    }

    .actions {
      display: flex;
      gap: 4px;
    }

    .action-btn {
      width: 28px;
      height: 28px;
      border: none;
      background: transparent;
      cursor: pointer;
      border-radius: 4px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #606266;
      transition: all 0.2s;

      &:hover {
        background: #ecf5ff;
        color: #409eff;
      }

      &.danger:hover {
        background: #fef0f0;
        color: #f56c6c;
      }

      &:disabled {
        opacity: 0.4;
        cursor: not-allowed;
      }
    }
  }

  .layer-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }

  .layer-item {
    display: flex;
    align-items: center;
    padding: 8px;
    border: 1px solid transparent;
    border-radius: 6px;
    cursor: pointer;
    margin-bottom: 4px;
    transition: all 0.2s;

    &:hover {
      background: #f5f7fa;
      border-color: #e4e7ed;
    }

    &.active {
      background: #ecf5ff;
      border-color: #409eff;
    }

    &.layer-pupil.active {
      background: #fdf6ec;
      border-color: #e6a23c;
    }

    .layer-left {
      display: flex;
      align-items: center;
      gap: 8px;

      .visibility-btn {
        width: 24px;
        height: 24px;
        border: none;
        background: transparent;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #606266;
        border-radius: 4px;

        &:hover {
          background: #fff;
          color: #409eff;
        }
      }

      .layer-thumb {
        width: 32px;
        height: 32px;
        background: #f0f0f0;
        border: 1px solid #dcdfe6;
        border-radius: 4px;
        overflow: hidden;

        .thumb-canvas {
          width: 100%;
          height: 100%;
          image-rendering: pixelated;
        }
      }
    }

    .layer-info {
      flex: 1;
      min-width: 0;
      padding: 0 8px;

      .layer-name {
        font-size: 13px;
        color: #303133;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .layer-type-tag {
        margin-top: 2px;
      }
    }

    .layer-right {
      .layer-actions {
        display: flex;
        flex-direction: column;
        gap: 2px;

        .move-btn {
          width: 20px;
          height: 18px;
          border: none;
          background: transparent;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #909399;
          border-radius: 3px;
          font-size: 12px;

          &:hover {
            background: #fff;
            color: #409eff;
          }

          &:disabled {
            opacity: 0.3;
            cursor: not-allowed;
          }
        }
      }
    }
  }

  .opacity-section,
  .blend-section {
    padding: 12px 16px;
    border-top: 1px solid #ebeef5;

    .section-label {
      font-size: 12px;
      color: #606266;
      margin-bottom: 8px;
    }

    .opacity-value {
      text-align: right;
      font-size: 11px;
      color: #909399;
      font-family: 'SF Mono', Consolas, monospace;
    }
  }
}
</style>
