<template>
  <div class="mask-canvas-editor">
    <TopControls
      :width="canvasWidth"
      :height="canvasHeight"
      :edit-mode="editMode"
      :submitting="submitting"
      @size-change="handleSizeChange"
      @mode-change="handleModeChange"
      @submit-simulation="handleSubmitSimulation"
      @export="handleExport"
      @import="handleImport"
      @clear-active-layer="handleClearActiveLayer"
      @clear-all="handleClearAll"
    />

    <div class="editor-body">
      <MaskToolbar
        :current-tool="currentTool"
        :brush-size="brushSize"
        :brush-opacity="brushOpacity"
        :brush-hardness="brushHardness"
        :brush-color="brushColor"
        :can-undo="canUndo"
        :can-redo="canRedo"
        @tool-change="handleToolChange"
        @brush-size-change="brushSize = $event"
        @brush-opacity-change="brushOpacity = $event"
        @brush-hardness-change="brushHardness = $event"
        @brush-color-change="brushColor = $event"
        @undo="handleUndo"
        @redo="handleRedo"
        @fit-view="handleFitView"
        @reset-view="handleResetView"
      />

      <div class="canvas-area">
        <MaskCanvasCore
          ref="canvasCoreRef"
          :width="canvasWidth"
          :height="canvasHeight"
          :layers="layers"
          :active-layer-id="activeLayerId"
          :current-tool="currentTool"
          :brush-size="brushSize"
          :brush-opacity="brushOpacity"
          :brush-hardness="brushHardness"
          :brush-color="brushColor"
          @layer-update="handleLayerUpdate"
          @draw-start="handleDrawStart"
          @draw-end="handleDrawEnd"
          @polygon-complete="handlePolygonComplete"
        />
      </div>

      <LayerPanel
        :layers="layers"
        :active-layer-id="activeLayerId"
        @set-active-layer="setActiveLayer"
        @toggle-visibility="toggleLayerVisibility"
        @remove-layer="handleRemoveLayer"
        @duplicate-layer="handleDuplicateLayer"
        @move-layer="handleMoveLayer"
        @add-layer="handleAddLayer"
        @change-opacity="handleLayerOpacityChange"
        @change-blend-mode="handleLayerBlendModeChange"
      />
    </div>

    <div v-if="showTaskDialog" class="task-dialog-overlay" @click="showTaskDialog = false">
      <div class="task-dialog" @click.stop>
        <div class="dialog-header">
          <h3>仿真任务已提交</h3>
          <el-icon class="close-btn" @click="showTaskDialog = false"><Close /></el-icon>
        </div>
        <div class="dialog-body">
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item label="任务 ID">
              <code>{{ currentTaskId }}</code>
            </el-descriptions-item>
            <el-descriptions-item label="状态">
              <el-tag type="primary">运行中</el-tag>
            </el-descriptions-item>
          </el-descriptions>
          <p class="dialog-tip">
            可在「仿真运行」或「RET 工作流」页面查看任务进度和结果。
          </p>
        </div>
        <div class="dialog-footer">
          <el-button @click="showTaskDialog = false">关闭</el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { Close } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { ToolType, LayerType, Layer, Point } from './types'
import { useLayers } from './composables/useLayers'
import { useUndoRedo } from './composables/useUndoRedo'
import { useMaskExport } from './composables/useMaskExport'
import MaskCanvasCore from './components/MaskCanvasCore.vue'
import MaskToolbar from './components/MaskToolbar.vue'
import LayerPanel from './components/LayerPanel.vue'
import TopControls from './components/TopControls.vue'
import { useConfigStore } from '@/stores/config'

interface Props {
  width?: number
  height?: number
  initialMaskData?: number[][]
  initialPupilData?: number[][]
  pixelSize?: number
}

const props = withDefaults(defineProps<Props>(), {
  width: 256,
  height: 256,
  pixelSize: 1.0
})

const emit = defineEmits<{
  (e: 'mask-change', data: number[][]): void
  (e: 'pupil-change', data: number[][] | null): void
  (e: 'simulation-submitted', taskId: string): void
  (e: 'change', data: { mask: number[][]; pupil: number[][] | null }): void
}>()

const configStore = useConfigStore()

const canvasWidth = ref(props.width)
const canvasHeight = ref(props.height)
const editMode = ref<'mask' | 'pupil'>('mask')
const currentTool = ref<ToolType>('brush')
const brushSize = ref(8)
const brushOpacity = ref(1)
const brushHardness = ref(0.8)
const brushColor = ref('#ffffff')
const submitting = ref(false)
const currentTaskId = ref('')
const showTaskDialog = ref(false)
const canvasCoreRef = ref<InstanceType<typeof MaskCanvasCore> | null>(null)

const {
  layers,
  activeLayerId,
  activeLayer,
  createLayer,
  addLayer,
  removeLayer,
  setActiveLayer,
  toggleLayerVisibility,
  setLayerOpacity,
  getLayerById,
  getLayersByType,
  resizeAllLayers,
  clearLayer,
  duplicateLayer: doDuplicateLayer,
  moveLayer: doMoveLayer,
  createEmptyImageData
} = useLayers(props.width, props.height)

const { canUndo, canRedo, saveState, undo, redo, clearHistory, cloneLayers } = useUndoRedo()

const {
  layerToGrayscaleArray,
  grayscaleArrayToImageData,
  exportMaskData,
  submitSimulation,
  downloadMaskAsImage,
  downloadMaskAsJSON,
  loadMaskFromImage,
  loadMaskFromJSON
} = useMaskExport()

const maskLayers = computed(() => getLayersByType('mask'))
const pupilLayers = computed(() => getLayersByType('pupil'))

onMounted(() => {
  initDefaultLayers()
  if (props.initialMaskData) {
    loadMaskData(props.initialMaskData, 'mask')
  }
  if (props.initialPupilData) {
    loadMaskData(props.initialPupilData, 'pupil')
  }
  nextTick(() => {
    saveState(layers.value, activeLayerId.value)
  })
  window.addEventListener('keydown', handleKeyDown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyDown)
})

function initDefaultLayers() {
  const maskLayer = createLayer('掩模 1', 'mask')
  addLayer(maskLayer)

  const pupilLayer = createLayer('Pupil 1', 'pupil')
  pupilLayer.visible = false
  addLayer(pupilLayer)

  setActiveLayer(maskLayer.id)
}

function loadMaskData(data: number[][], type: LayerType) {
  const h = data.length
  const w = data[0]?.length || 0
  if (h === 0 || w === 0) return

  const imageData = grayscaleArrayToImageData(data, w, h)
  const layerList = getLayersByType(type)
  if (layerList.length > 0) {
    const layer = layerList[0]
    layer.data = imageData
    layer.width = w
    layer.height = h
  }
}

function handleToolChange(tool: ToolType) {
  currentTool.value = tool
}

function handleSizeChange(width: number, height: number) {
  canvasWidth.value = width
  canvasHeight.value = height
  resizeAllLayers(width, height)
  saveState(layers.value, activeLayerId.value)
  emitChange()
}

function handleModeChange(mode: 'mask' | 'pupil') {
  editMode.value = mode
  const typeLayers = getLayersByType(mode)
  if (typeLayers.length > 0) {
    setActiveLayer(typeLayers[0].id)
    typeLayers[0].visible = true
  }
}

function handleDrawStart() {
}

function handleDrawEnd() {
  saveState(layers.value, activeLayerId.value)
  emitChange()
}

function handleLayerUpdate(layerId: string, data: ImageData) {
  const layer = getLayerById(layerId)
  if (layer) {
    layer.data = data
  }
}

function handlePolygonComplete(points: Point[]) {
  if (!activeLayer.value || !activeLayer.value.data) return

  const canvas = document.createElement('canvas')
  canvas.width = activeLayer.value.width
  canvas.height = activeLayer.value.height
  const ctx = canvas.getContext('2d')!
  ctx.putImageData(activeLayer.value.data, 0, 0)

  ctx.beginPath()
  if (points.length > 0) {
    ctx.moveTo(points[0].x, points[0].y)
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y)
    }
  }
  ctx.closePath()
  ctx.fillStyle = brushColor.value
  ctx.globalAlpha = brushOpacity.value
  ctx.fill()
  ctx.globalAlpha = 1

  const newData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  activeLayer.value.data = newData

  saveState(layers.value, activeLayerId.value)
  emitChange()
}

function handleUndo() {
  const state = undo()
  if (state) {
    layers.value = state.layers
    if (state.activeLayerId) {
      activeLayerId.value = state.activeLayerId
    }
    emitChange()
  }
}

function handleRedo() {
  const state = redo()
  if (state) {
    layers.value = state.layers
    if (state.activeLayerId) {
      activeLayerId.value = state.activeLayerId
    }
    emitChange()
  }
}

function handleFitView() {
  canvasCoreRef.value?.fitToView()
}

function handleResetView() {
  canvasCoreRef.value?.resetView()
}

function handleRemoveLayer(layerId: string) {
  const layer = getLayerById(layerId)
  if (!layer) return

  if (layers.value.length <= 1) {
    ElMessage.warning('至少保留一个图层')
    return
  }

  ElMessageBox.confirm(`确认删除图层 "${layer.name}"？`, '删除图层', {
    type: 'warning',
    confirmButtonText: '删除',
    cancelButtonText: '取消'
  }).then(() => {
    removeLayer(layerId)
    saveState(layers.value, activeLayerId.value)
    emitChange()
  }).catch(() => {})
}

function handleDuplicateLayer(layerId: string) {
  const newLayer = doDuplicateLayer(layerId)
  if (newLayer) {
    setActiveLayer(newLayer.id)
    saveState(layers.value, activeLayerId.value)
    emitChange()
  }
}

function handleMoveLayer(layerId: string, direction: 'up' | 'down') {
  doMoveLayer(layerId, direction)
  saveState(layers.value, activeLayerId.value)
}

function handleAddLayer(type: LayerType) {
  const count = getLayersByType(type).length + 1
  const name = type === 'mask' ? `掩模 ${count}` : type === 'pupil' ? `Pupil ${count}` : `参考 ${count}`
  const layer = createLayer(name, type)
  addLayer(layer)
  setActiveLayer(layer.id)
  saveState(layers.value, activeLayerId.value)
  emitChange()
}

function handleLayerOpacityChange(layerId: string, opacity: number) {
  setLayerOpacity(layerId, opacity)
}

function handleLayerBlendModeChange(layerId: string, mode: GlobalCompositeOperation) {
  const layer = getLayerById(layerId)
  if (layer) {
    layer.blendMode = mode
  }
}

function handleClearActiveLayer() {
  if (!activeLayerId.value) return
  ElMessageBox.confirm('确认清空当前图层？', '清空图层', {
    type: 'warning',
    confirmButtonText: '清空',
    cancelButtonText: '取消'
  }).then(() => {
    clearLayer(activeLayerId.value!)
    saveState(layers.value, activeLayerId.value)
    emitChange()
  }).catch(() => {})
}

function handleClearAll() {
  ElMessageBox.confirm('确认清空所有图层？此操作不可撤销。', '清空全部', {
    type: 'warning',
    confirmButtonText: '清空全部',
    cancelButtonText: '取消',
    confirmButtonClass: 'el-button--danger'
  }).then(() => {
    layers.value.forEach(layer => {
      if (layer.data) {
        layer.data = createEmptyImageData(layer.width, layer.height)
      }
    })
    clearHistory()
    saveState(layers.value, activeLayerId.value)
    emitChange()
    ElMessage.success('已清空全部图层')
  }).catch(() => {})
}

function handleExport(format: string) {
  const maskLayer = maskLayers.value.find(l => l.visible) || maskLayers.value[0]
  const pupilLayer = pupilLayers.value.find(l => l.visible) || pupilLayers.value[0]

  switch (format) {
    case 'png_mask':
      if (maskLayer) {
        downloadMaskAsImage(maskLayer, 'mask.png')
      }
      break
    case 'png_pupil':
      if (pupilLayer) {
        downloadMaskAsImage(pupilLayer, 'pupil.png')
      } else {
        ElMessage.warning('没有 Pupil 图层')
      }
      break
    case 'json_mask':
      if (maskLayer) {
        const data = exportMaskData(maskLayer, null, props.pixelSize)
        downloadMaskAsJSON(data, 'mask_data.json')
      }
      break
    case 'json_all':
      if (maskLayer) {
        const data = exportMaskData(maskLayer, pupilLayer || null, props.pixelSize)
        downloadMaskAsJSON(data, 'mask_pupil_data.json')
      }
      break
  }
}

async function handleImport(file: File) {
  try {
    if (file.name.endsWith('.json')) {
      const data = await loadMaskFromJSON(file)
      canvasWidth.value = data.width
      canvasHeight.value = data.height
      resizeAllLayers(data.width, data.height)

      if (data.maskData && maskLayers.value.length > 0) {
        const imageData = grayscaleArrayToImageData(data.maskData, data.width, data.height)
        maskLayers.value[0].data = imageData
      }
      if (data.pupilData && pupilLayers.value.length > 0) {
        const imageData = grayscaleArrayToImageData(data.pupilData, data.width, data.height)
        pupilLayers.value[0].data = imageData
      }
    } else {
      const imageData = await loadMaskFromImage(file, canvasWidth.value, canvasHeight.value)
      if (activeLayer.value) {
        activeLayer.value.data = imageData
      }
    }

    saveState(layers.value, activeLayerId.value)
    emitChange()
    ElMessage.success('导入成功')
  } catch (e) {
    ElMessage.error('导入失败：' + (e as Error).message)
  }
}

async function handleSubmitSimulation() {
  const maskLayer = maskLayers.value.find(l => l.visible) || maskLayers.value[0]
  if (!maskLayer) {
    ElMessage.error('没有可用的掩模图层')
    return
  }

  const pupilLayer = pupilLayers.value.find(l => l.visible) || null

  submitting.value = true
  try {
    if (!configStore.config?.optical_system) {
      await configStore.loadDefault()
    }

    const maskData = layerToGrayscaleArray(maskLayer)
    const pupilData = pupilLayer ? layerToGrayscaleArray(pupilLayer) : null

    const result: any = await submitSimulation(
      {
        maskData,
        pupilData,
        config: configStore.config
      },
      configStore.config
    )

    if (result.success || result.task_id) {
      const taskId = result.task_id
      currentTaskId.value = taskId
      showTaskDialog.value = true
      emit('simulation-submitted', taskId)
      ElMessage.success(`仿真任务已提交：${taskId}`)
    } else {
      ElMessage.error(result.message || '提交失败')
    }
  } catch (e: any) {
    ElMessage.error(e?.message || '提交失败，请检查后端服务')
  } finally {
    submitting.value = false
  }
}

function emitChange() {
  const maskLayer = maskLayers.value.find(l => l.visible) || maskLayers.value[0]
  const pupilLayer = pupilLayers.value.find(l => l.visible) || null

  const maskData = maskLayer ? layerToGrayscaleArray(maskLayer) : []
  const pupilData = pupilLayer ? layerToGrayscaleArray(pupilLayer) : null

  emit('mask-change', maskData)
  emit('pupil-change', pupilData)
  emit('change', { mask: maskData, pupil: pupilData })
}

function handleKeyDown(e: KeyboardEvent) {
  if (e.ctrlKey || e.metaKey) {
    if (e.key === 'z') {
      e.preventDefault()
      if (e.shiftKey) {
        handleRedo()
      } else {
        handleUndo()
      }
    } else if (e.key === 'y') {
      e.preventDefault()
      handleRedo()
    }
    return
  }

  switch (e.key.toLowerCase()) {
    case 'b':
      currentTool.value = editMode.value === 'mask' ? 'brush' : 'pupil_brush'
      break
    case 'e':
      currentTool.value = editMode.value === 'mask' ? 'eraser' : 'pupil_eraser'
      break
    case 'p':
      currentTool.value = 'polygon'
      break
    case 'r':
      currentTool.value = 'rectangle'
      break
    case 'c':
      currentTool.value = 'circle'
      break
    case 'h':
      currentTool.value = 'move'
      break
    case 'z':
      currentTool.value = 'zoom'
      break
  }
}

watch(editMode, (mode) => {
  if (mode === 'mask') {
    if (['pupil_brush', 'pupil_eraser'].includes(currentTool.value)) {
      currentTool.value = currentTool.value === 'pupil_brush' ? 'brush' : 'eraser'
    }
  } else {
    if (['brush', 'eraser'].includes(currentTool.value)) {
      currentTool.value = currentTool.value === 'brush' ? 'pupil_brush' : 'pupil_eraser'
    }
  }
})

defineExpose({
  layers,
  activeLayerId,
  currentTool,
  canvasWidth,
  canvasHeight,
  editMode,
  exportMaskData,
  submitSimulation,
  fitToView: handleFitView,
  resetView: handleResetView
})
</script>

<style lang="scss" scoped>
.mask-canvas-editor {
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  min-height: 500px;
  background: #f5f7fa;
  border-radius: 8px;
  overflow: hidden;

  .editor-body {
    flex: 1;
    display: flex;
    min-height: 0;
  }

  .canvas-area {
    flex: 1;
    position: relative;
    min-width: 0;
  }
}

.task-dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;

  .task-dialog {
    width: 400px;
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);

    .dialog-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 20px;
      border-bottom: 1px solid #ebeef5;

      h3 {
        margin: 0;
        font-size: 16px;
        color: #303133;
      }

      .close-btn {
        cursor: pointer;
        color: #909399;
        font-size: 20px;

        &:hover {
          color: #409eff;
        }
      }
    }

    .dialog-body {
      padding: 20px;

      .dialog-tip {
        margin: 16px 0 0 0;
        font-size: 13px;
        color: #909399;
        line-height: 1.6;
      }

      code {
        background: #f4f4f5;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 12px;
      }
    }

    .dialog-footer {
      padding: 12px 20px;
      border-top: 1px solid #ebeef5;
      text-align: right;
    }
  }
}
</style>
