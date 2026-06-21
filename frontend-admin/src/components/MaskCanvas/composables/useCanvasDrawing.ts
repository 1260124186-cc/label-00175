import { ref, computed, reactive } from 'vue'
import type {
  ToolType,
  Point,
  BrushSettings,
  Layer,
  PolygonShape
} from '../types'

export function useCanvasDrawing() {
  const currentTool = ref<ToolType>('brush')
  const isDrawing = ref(false)
  const lastPoint = ref<Point | null>(null)
  const polygonPoints = ref<Point[]>([])
  const isPolygonDrawing = ref(false)

  const brushSettings = reactive<BrushSettings>({
    size: 8,
    hardness: 0.8,
    opacity: 1,
    flow: 1,
    color: '#ffffff'
  })

  const zoom = ref(1)
  const panX = ref(0)
  const panY = ref(0)

  const tempCanvas = ref<HTMLCanvasElement | null>(null)
  const tempCtx = ref<CanvasRenderingContext2D | null>(null)

  function initTempCanvas(width: number, height: number) {
    tempCanvas.value = document.createElement('canvas')
    tempCanvas.value.width = width
    tempCanvas.value.height = height
    tempCtx.value = tempCanvas.value.getContext('2d')
  }

  function getCanvasPoint(
    clientX: number,
    clientY: number,
    canvasRect: DOMRect
  ): Point {
    const x = (clientX - canvasRect.left - panX.value) / zoom.value
    const y = (clientY - canvasRect.top - panY.value) / zoom.value
    return { x: Math.round(x), y: Math.round(y) }
  }

  function drawBrush(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    isEraser: boolean = false
  ) {
    const radius = brushSettings.size / 2
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius)

    if (isEraser) {
      ctx.globalCompositeOperation = 'destination-out'
    } else {
      ctx.globalCompositeOperation = 'source-over'
    }

    const alpha = brushSettings.opacity * brushSettings.flow
    const innerAlpha = alpha
    const outerAlpha = alpha * (1 - brushSettings.hardness)

    gradient.addColorStop(0, `rgba(255, 255, 255, ${innerAlpha})`)
    gradient.addColorStop(brushSettings.hardness, `rgba(255, 255, 255, ${innerAlpha})`)
    gradient.addColorStop(1, `rgba(255, 255, 255, ${outerAlpha})`)

    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.arc(x, y, radius, 0, Math.PI * 2)
    ctx.fill()

    ctx.globalCompositeOperation = 'source-over'
  }

  function drawLine(
    ctx: CanvasRenderingContext2D,
    from: Point,
    to: Point,
    isEraser: boolean = false
  ) {
    const dist = Math.sqrt((to.x - from.x) ** 2 + (to.y - from.y) ** 2)
    const steps = Math.max(1, Math.ceil(dist / (brushSettings.size * 0.1)))

    for (let i = 0; i <= steps; i++) {
      const t = i / steps
      const x = from.x + (to.x - from.x) * t
      const y = from.y + (to.y - from.y) * t
      drawBrush(ctx, x, y, isEraser)
    }
  }

  function drawPolygon(
    ctx: CanvasRenderingContext2D,
    points: Point[],
    fill: boolean = true
  ) {
    if (points.length < 2) return

    ctx.beginPath()
    ctx.moveTo(points[0].x, points[0].y)
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y)
    }
    ctx.closePath()

    if (fill) {
      ctx.fillStyle = brushSettings.color
      ctx.globalAlpha = brushSettings.opacity
      ctx.fill()
      ctx.globalAlpha = 1
    } else {
      ctx.strokeStyle = brushSettings.color
      ctx.lineWidth = brushSettings.size
      ctx.globalAlpha = brushSettings.opacity
      ctx.stroke()
      ctx.globalAlpha = 1
    }
  }

  function drawRectangle(
    ctx: CanvasRenderingContext2D,
    from: Point,
    to: Point,
    fill: boolean = true
  ) {
    const x = Math.min(from.x, to.x)
    const y = Math.min(from.y, to.y)
    const w = Math.abs(to.x - from.x)
    const h = Math.abs(to.y - from.y)

    if (fill) {
      ctx.fillStyle = brushSettings.color
      ctx.globalAlpha = brushSettings.opacity
      ctx.fillRect(x, y, w, h)
      ctx.globalAlpha = 1
    } else {
      ctx.strokeStyle = brushSettings.color
      ctx.lineWidth = brushSettings.size
      ctx.globalAlpha = brushSettings.opacity
      ctx.strokeRect(x, y, w, h)
      ctx.globalAlpha = 1
    }
  }

  function drawCircle(
    ctx: CanvasRenderingContext2D,
    center: Point,
    radius: number,
    fill: boolean = true
  ) {
    ctx.beginPath()
    ctx.arc(center.x, center.y, radius, 0, Math.PI * 2)

    if (fill) {
      ctx.fillStyle = brushSettings.color
      ctx.globalAlpha = brushSettings.opacity
      ctx.fill()
      ctx.globalAlpha = 1
    } else {
      ctx.strokeStyle = brushSettings.color
      ctx.lineWidth = brushSettings.size
      ctx.globalAlpha = brushSettings.opacity
      ctx.stroke()
      ctx.globalAlpha = 1
    }
  }

  function applyToLayer(layer: Layer, sourceCtx: CanvasRenderingContext2D) {
    if (!layer.data) return

    const canvas = document.createElement('canvas')
    canvas.width = layer.width
    canvas.height = layer.height
    const ctx = canvas.getContext('2d')!

    ctx.putImageData(layer.data, 0, 0)
    ctx.globalCompositeOperation = 'source-over'
    ctx.globalAlpha = 1
    ctx.drawImage(sourceCtx.canvas, 0, 0)

    layer.data = ctx.getImageData(0, 0, layer.width, layer.height)
  }

  function renderLayerToCanvas(
    ctx: CanvasRenderingContext2D,
    layer: Layer,
    clear: boolean = false
  ) {
    if (!layer.visible || !layer.data) return

    if (clear) {
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    }

    ctx.save()
    ctx.globalAlpha = layer.opacity
    ctx.globalCompositeOperation = layer.blendMode

    const canvas = document.createElement('canvas')
    canvas.width = layer.width
    canvas.height = layer.height
    const layerCtx = canvas.getContext('2d')!
    layerCtx.putImageData(layer.data, 0, 0)

    ctx.drawImage(canvas, 0, 0)
    ctx.restore()
  }

  function renderAllLayers(
    ctx: CanvasRenderingContext2D,
    layers: Layer[],
    clear: boolean = true
  ) {
    if (clear) {
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    }

    layers.forEach(layer => {
      renderLayerToCanvas(ctx, layer, false)
    })
  }

  function clearTempCanvas() {
    if (tempCtx.value && tempCanvas.value) {
      tempCtx.value.clearRect(0, 0, tempCanvas.value.width, tempCanvas.value.height)
    }
  }

  function setTool(tool: ToolType) {
    currentTool.value = tool
    isPolygonDrawing.value = false
    polygonPoints.value = []
  }

  function setBrushSize(size: number) {
    brushSettings.size = Math.max(1, size)
  }

  function setBrushOpacity(opacity: number) {
    brushSettings.opacity = Math.max(0, Math.min(1, opacity))
  }

  function setBrushHardness(hardness: number) {
    brushSettings.hardness = Math.max(0, Math.min(1, hardness))
  }

  function setBrushColor(color: string) {
    brushSettings.color = color
  }

  function handleZoom(delta: number, centerX?: number, centerY?: number) {
    const oldZoom = zoom.value
    const newZoom = Math.max(0.1, Math.min(10, zoom.value * (delta > 0 ? 1.1 : 0.9)))

    if (centerX !== undefined && centerY !== undefined) {
      const scale = newZoom / oldZoom
      panX.value = centerX - (centerX - panX.value) * scale
      panY.value = centerY - (centerY - panY.value) * scale
    }

    zoom.value = newZoom
  }

  function handlePan(dx: number, dy: number) {
    panX.value += dx
    panY.value += dy
  }

  function resetView() {
    zoom.value = 1
    panX.value = 0
    panY.value = 0
  }

  function fitToView(canvasWidth: number, canvasHeight: number, contentWidth: number, contentHeight: number) {
    const scaleX = canvasWidth / contentWidth
    const scaleY = canvasHeight / contentHeight
    zoom.value = Math.min(scaleX, scaleY) * 0.9

    panX.value = (canvasWidth - contentWidth * zoom.value) / 2
    panY.value = (canvasHeight - contentHeight * zoom.value) / 2
  }

  function addPolygonPoint(point: Point) {
    polygonPoints.value.push({ ...point })
  }

  function closePolygon() {
    isPolygonDrawing.value = false
  }

  function cancelPolygon() {
    polygonPoints.value = []
    isPolygonDrawing.value = false
  }

  return {
    currentTool,
    isDrawing,
    lastPoint,
    polygonPoints,
    isPolygonDrawing,
    brushSettings,
    zoom,
    panX,
    panY,
    tempCanvas,
    tempCtx,
    initTempCanvas,
    getCanvasPoint,
    drawBrush,
    drawLine,
    drawPolygon,
    drawRectangle,
    drawCircle,
    applyToLayer,
    renderLayerToCanvas,
    renderAllLayers,
    clearTempCanvas,
    setTool,
    setBrushSize,
    setBrushOpacity,
    setBrushHardness,
    setBrushColor,
    handleZoom,
    handlePan,
    resetView,
    fitToView,
    addPolygonPoint,
    closePolygon,
    cancelPolygon
  }
}
