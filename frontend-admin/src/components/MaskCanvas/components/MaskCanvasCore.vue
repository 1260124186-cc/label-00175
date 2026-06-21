<template>
  <div class="mask-canvas-container" ref="containerRef">
    <div class="canvas-wrapper" ref="canvasWrapperRef">
      <canvas
        ref="displayCanvasRef"
        class="display-canvas"
        :style="{
          transform: `translate(${panX}px, ${panY}px) scale(${zoom})`,
          transformOrigin: '0 0',
          cursor: canvasCursor
        }"
        @mousedown="handleMouseDown"
        @mousemove="handleMouseMove"
        @mouseup="handleMouseUp"
        @mouseleave="handleMouseUp"
        @wheel="handleWheel"
        @contextmenu.prevent
      />
      <canvas
        v-if="isPolygonDrawing || isDrawing"
        ref="overlayCanvasRef"
        class="overlay-canvas"
        :style="{
          transform: `translate(${panX}px, ${panY}px) scale(${zoom})`,
          transformOrigin: '0 0',
          pointerEvents: 'none'
        }"
      />
    </div>

    <div class="canvas-info">
      <span class="info-item">缩放: {{ (zoom * 100).toFixed(0) }}%</span>
      <span class="info-item">尺寸: {{ canvasWidth }} × {{ canvasHeight }} px</span>
      <span class="info-item" v-if="mousePos">
        坐标: ({{ mousePos.x }}, {{ mousePos.y }})
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import type { ToolType, Layer, Point } from '../types'
import { useCanvasDrawing } from '../composables/useCanvasDrawing'

interface Props {
  width?: number
  height?: number
  layers: Layer[]
  activeLayerId: string | null
  currentTool: ToolType
  brushSize?: number
  brushOpacity?: number
  brushHardness?: number
  brushColor?: string
}

const props = withDefaults(defineProps<Props>(), {
  width: 256,
  height: 256,
  brushSize: 8,
  brushOpacity: 1,
  brushHardness: 0.8,
  brushColor: '#ffffff'
})

const emit = defineEmits<{
  (e: 'layer-update', layerId: string, data: ImageData): void
  (e: 'draw-start'): void
  (e: 'draw-end'): void
  (e: 'zoom-change', zoom: number): void
  (e: 'pan-change', x: number, y: number): void
  (e: 'polygon-complete', points: Point[]): void
}>()

const containerRef = ref<HTMLDivElement | null>(null)
const canvasWrapperRef = ref<HTMLDivElement | null>(null)
const displayCanvasRef = ref<HTMLCanvasElement | null>(null)
const overlayCanvasRef = ref<HTMLCanvasElement | null>(null)

const {
  zoom,
  panX,
  panY,
  isDrawing,
  lastPoint,
  polygonPoints,
  isPolygonDrawing,
  brushSettings,
  initTempCanvas,
  tempCtx,
  getCanvasPoint,
  drawBrush,
  drawLine,
  drawPolygon,
  renderAllLayers,
  clearTempCanvas,
  handleZoom,
  handlePan,
  addPolygonPoint,
  cancelPolygon
} = useCanvasDrawing()

const canvasWidth = computed(() => props.width)
const canvasHeight = computed(() => props.height)
const mousePos = ref<Point | null>(null)

const canvasCursor = computed(() => {
  switch (props.currentTool) {
    case 'move':
      return isDrawing.value ? 'grabbing' : 'grab'
    case 'zoom':
      return 'zoom-in'
    case 'picker':
      return 'crosshair'
    case 'polygon':
      return 'crosshair'
    default:
      return 'crosshair'
  }
})

const activeLayer = computed(() =>
  props.layers.find(l => l.id === props.activeLayerId) || null
)

let displayCtx: CanvasRenderingContext2D | null = null
let overlayCtx: CanvasRenderingContext2D | null = null
let startPanPoint: Point | null = null

onMounted(() => {
  initCanvases()
  initTempCanvas(props.width, props.height)
  render()
})

function initCanvases() {
  if (displayCanvasRef.value) {
    displayCanvasRef.value.width = props.width
    displayCanvasRef.value.height = props.height
    displayCtx = displayCanvasRef.value.getContext('2d')
  }
  if (overlayCanvasRef.value) {
    overlayCanvasRef.value.width = props.width
    overlayCanvasRef.value.height = props.height
    overlayCtx = overlayCanvasRef.value.getContext('2d')
  }
}

function render() {
  if (!displayCtx) return
  renderAllLayers(displayCtx, props.layers, true)
}

function renderOverlay() {
  if (!overlayCtx) return
  overlayCtx.clearRect(0, 0, props.width, props.height)

  if (props.currentTool === 'polygon' && polygonPoints.value.length > 0) {
    overlayCtx.beginPath()
    overlayCtx.moveTo(polygonPoints.value[0].x, polygonPoints.value[0].y)
    for (let i = 1; i < polygonPoints.value.length; i++) {
      overlayCtx.lineTo(polygonPoints.value[i].x, polygonPoints.value[i].y)
    }
    overlayCtx.strokeStyle = props.brushColor
    overlayCtx.lineWidth = 2
    overlayCtx.setLineDash([5, 5])
    overlayCtx.stroke()
    overlayCtx.setLineDash([])

    polygonPoints.value.forEach(p => {
      overlayCtx!.beginPath()
      overlayCtx!.arc(p.x, p.y, 4, 0, Math.PI * 2)
      overlayCtx!.fillStyle = props.brushColor
      overlayCtx!.fill()
      overlayCtx!.strokeStyle = '#fff'
      overlayCtx!.lineWidth = 2
      overlayCtx!.stroke()
    })
  }
}

function handleMouseDown(e: MouseEvent) {
  if (!displayCanvasRef.value || !activeLayer.value || !tempCtx.value) return

  const rect = displayCanvasRef.value.getBoundingClientRect()
  const point = getCanvasPoint(e.clientX, e.clientY, rect)

  if (e.button === 1 || (e.button === 0 && props.currentTool === 'move')) {
    isDrawing.value = true
    startPanPoint = { x: e.clientX, y: e.clientY }
    return
  }

  if (e.button === 2) {
    if (props.currentTool === 'polygon' && isPolygonDrawing.value) {
      if (polygonPoints.value.length >= 3) {
        emit('polygon-complete', [...polygonPoints.value])
      }
      cancelPolygon()
      renderOverlay()
    }
    return
  }

  switch (props.currentTool) {
    case 'brush':
    case 'eraser':
    case 'pupil_brush':
    case 'pupil_eraser':
      isDrawing.value = true
      lastPoint.value = point
      emit('draw-start')
      clearTempCanvas()
      const isEraser = props.currentTool === 'eraser' || props.currentTool === 'pupil_eraser'
      drawBrush(tempCtx.value, point.x, point.y, isEraser)
      break

    case 'polygon':
      if (!isPolygonDrawing.value) {
        isPolygonDrawing.value = true
        polygonPoints.value = []
      }
      addPolygonPoint(point)
      renderOverlay()
      break

    case 'rectangle':
    case 'circle':
      isDrawing.value = true
      lastPoint.value = point
      emit('draw-start')
      clearTempCanvas()
      break

    case 'zoom':
      if (e.button === 0) {
        handleZoom(1, e.clientX, e.clientY)
      } else {
        handleZoom(-1, e.clientX, e.clientY)
      }
      emit('zoom-change', zoom.value)
      break
  }
}

function handleMouseMove(e: MouseEvent) {
  if (!displayCanvasRef.value || !tempCtx.value) return

  const rect = displayCanvasRef.value.getBoundingClientRect()
  const point = getCanvasPoint(e.clientX, e.clientY, rect)
  mousePos.value = point

  if (props.currentTool === 'move' && isDrawing.value && startPanPoint) {
    const dx = e.clientX - startPanPoint.x
    const dy = e.clientY - startPanPoint.y
    handlePan(dx, dy)
    startPanPoint = { x: e.clientX, y: e.clientY }
    emit('pan-change', panX.value, panY.value)
    return
  }

  if (!isDrawing.value || !activeLayer.value) return

  switch (props.currentTool) {
    case 'brush':
    case 'eraser':
    case 'pupil_brush':
    case 'pupil_eraser':
      if (lastPoint.value) {
        const isEraser = props.currentTool === 'eraser' || props.currentTool === 'pupil_eraser'
        drawLine(tempCtx.value, lastPoint.value, point, isEraser)
        lastPoint.value = point
      }
      break

    case 'rectangle':
      if (lastPoint.value) {
        overlayCtx?.clearRect(0, 0, props.width, props.height)
        const x = Math.min(lastPoint.value.x, point.x)
        const y = Math.min(lastPoint.value.y, point.y)
        const w = Math.abs(point.x - lastPoint.value.x)
        const h = Math.abs(point.y - lastPoint.value.y)
        overlayCtx!.fillStyle = props.brushColor
        overlayCtx!.globalAlpha = props.brushOpacity
        overlayCtx!.fillRect(x, y, w, h)
        overlayCtx!.globalAlpha = 1
      }
      break

    case 'circle':
      if (lastPoint.value) {
        overlayCtx?.clearRect(0, 0, props.width, props.height)
        const radius = Math.sqrt(
          (point.x - lastPoint.value.x) ** 2 + (point.y - lastPoint.value.y) ** 2
        )
        overlayCtx!.beginPath()
        overlayCtx!.arc(lastPoint.value.x, lastPoint.value.y, radius, 0, Math.PI * 2)
        overlayCtx!.fillStyle = props.brushColor
        overlayCtx!.globalAlpha = props.brushOpacity
        overlayCtx!.fill()
        overlayCtx!.globalAlpha = 1
      }
      break

    case 'polygon':
      if (isPolygonDrawing.value && polygonPoints.value.length > 0) {
        renderOverlay()
        const lastP = polygonPoints.value[polygonPoints.value.length - 1]
        overlayCtx!.beginPath()
        overlayCtx!.moveTo(lastP.x, lastP.y)
        overlayCtx!.lineTo(point.x, point.y)
        overlayCtx!.strokeStyle = props.brushColor
        overlayCtx!.lineWidth = 1
        overlayCtx!.setLineDash([3, 3])
        overlayCtx!.stroke()
        overlayCtx!.setLineDash([])
      }
      break
  }
}

function handleMouseUp(e: MouseEvent) {
  if (!activeLayer.value || !tempCtx.value) {
    isDrawing.value = false
    lastPoint.value = null
    startPanPoint = null
    return
  }

  if (props.currentTool === 'move') {
    isDrawing.value = false
    startPanPoint = null
    return
  }

  if (isDrawing.value) {
    switch (props.currentTool) {
      case 'brush':
      case 'eraser':
      case 'pupil_brush':
      case 'pupil_eraser':
        if (activeLayer.value.data) {
          const canvas = document.createElement('canvas')
          canvas.width = activeLayer.value.width
          canvas.height = activeLayer.value.height
          const ctx = canvas.getContext('2d')!
          ctx.putImageData(activeLayer.value.data, 0, 0)
          ctx.drawImage(tempCtx.value.canvas, 0, 0)
          const newData = ctx.getImageData(0, 0, activeLayer.value.width, activeLayer.value.height)
          emit('layer-update', activeLayer.value.id, newData)
        }
        break

      case 'rectangle':
      case 'circle':
        if (activeLayer.value.data && overlayCtx) {
          const canvas = document.createElement('canvas')
          canvas.width = activeLayer.value.width
          canvas.height = activeLayer.value.height
          const ctx = canvas.getContext('2d')!
          ctx.putImageData(activeLayer.value.data, 0, 0)
          ctx.drawImage(overlayCtx.canvas, 0, 0)
          const newData = ctx.getImageData(0, 0, activeLayer.value.width, activeLayer.value.height)
          emit('layer-update', activeLayer.value.id, newData)
        }
        overlayCtx?.clearRect(0, 0, props.width, props.height)
        break
    }

    clearTempCanvas()
    emit('draw-end')
  }

  isDrawing.value = false
  lastPoint.value = null
}

function handleWheel(e: WheelEvent) {
  e.preventDefault()
  if (!displayCanvasRef.value) return

  const rect = displayCanvasRef.value.getBoundingClientRect()
  handleZoom(-e.deltaY, e.clientX - rect.left, e.clientY - rect.top)
  emit('zoom-change', zoom.value)
}

watch(() => props.layers, () => {
  nextTick(() => render())
}, { deep: true })

watch([() => props.width, () => props.height], () => {
  initCanvases()
  initTempCanvas(props.width, props.height)
  render()
})

watch(() => props.brushSize, (val) => {
  brushSettings.size = val
})

watch(() => props.brushOpacity, (val) => {
  brushSettings.opacity = val
})

watch(() => props.brushHardness, (val) => {
  brushSettings.hardness = val
})

watch(() => props.brushColor, (val) => {
  brushSettings.color = val
})

function fitToView() {
  if (!containerRef.value) return
  const containerRect = containerRef.value.getBoundingClientRect()
  const scaleX = containerRect.width / props.width
  const scaleY = containerRect.height / props.height
  const newZoom = Math.min(scaleX, scaleY) * 0.9

  zoom.value = Math.max(0.1, newZoom)
  panX.value = (containerRect.width - props.width * zoom.value) / 2
  panY.value = (containerRect.height - props.height * zoom.value) / 2

  emit('zoom-change', zoom.value)
  emit('pan-change', panX.value, panY.value)
}

function resetView() {
  zoom.value = 1
  panX.value = 0
  panY.value = 0
  emit('zoom-change', zoom.value)
  emit('pan-change', panX.value, panY.value)
}

defineExpose({
  fitToView,
  resetView,
  zoom,
  panX,
  panY
})
</script>

<style lang="scss" scoped>
.mask-canvas-container {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: #1a1a2e;
  background-image:
    linear-gradient(45deg, #2a2a4e 25%, transparent 25%),
    linear-gradient(-45deg, #2a2a4e 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #2a2a4e 75%),
    linear-gradient(-45deg, transparent 75%, #2a2a4e 75%);
  background-size: 20px 20px;
  background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
}

.canvas-wrapper {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.display-canvas,
.overlay-canvas {
  position: absolute;
  top: 0;
  left: 0;
  image-rendering: pixelated;
  image-rendering: crisp-edges;
}

.overlay-canvas {
  z-index: 1;
}

.canvas-info {
  position: absolute;
  bottom: 8px;
  left: 8px;
  display: flex;
  gap: 16px;
  padding: 6px 12px;
  background: rgba(0, 0, 0, 0.6);
  border-radius: 4px;
  color: #fff;
  font-size: 12px;
  font-family: 'SF Mono', Consolas, monospace;
  pointer-events: none;
  z-index: 10;

  .info-item {
    opacity: 0.9;
  }
}
</style>
