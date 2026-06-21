import { ref, computed } from 'vue'
import type { Layer, LayerType } from '../types'

let layerIdCounter = 0

function generateLayerId(): string {
  return `layer_${++layerIdCounter}_${Date.now()}`
}

export function useLayers(initialWidth: number, initialHeight: number) {
  const layers = ref<Layer[]>([])
  const activeLayerId = ref<string | null>(null)

  const activeLayer = computed(() =>
    layers.value.find(l => l.id === activeLayerId.value) || null
  )

  const visibleLayers = computed(() =>
    layers.value.filter(l => l.visible)
  )

  function createEmptyImageData(width: number, height: number): ImageData {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')!
    return ctx.createImageData(width, height)
  }

  function createLayer(
    name: string,
    type: LayerType,
    options?: Partial<Omit<Layer, 'id' | 'name' | 'type'>>
  ): Layer {
    const width = options?.width || initialWidth
    const height = options?.height || initialHeight
    const layer: Layer = {
      id: generateLayerId(),
      name,
      type,
      visible: true,
      opacity: 1,
      blendMode: 'source-over',
      data: createEmptyImageData(width, height),
      width,
      height,
      ...options
    }
    return layer
  }

  function addLayer(layer: Layer, index?: number) {
    if (index !== undefined) {
      layers.value.splice(index, 0, layer)
    } else {
      layers.value.push(layer)
    }
    if (!activeLayerId.value) {
      activeLayerId.value = layer.id
    }
  }

  function removeLayer(layerId: string) {
    const index = layers.value.findIndex(l => l.id === layerId)
    if (index >= 0) {
      layers.value.splice(index, 1)
      if (activeLayerId.value === layerId) {
        activeLayerId.value = layers.value[layers.value.length - 1]?.id || null
      }
    }
  }

  function setActiveLayer(layerId: string) {
    activeLayerId.value = layerId
  }

  function toggleLayerVisibility(layerId: string) {
    const layer = layers.value.find(l => l.id === layerId)
    if (layer) {
      layer.visible = !layer.visible
    }
  }

  function setLayerOpacity(layerId: string, opacity: number) {
    const layer = layers.value.find(l => l.id === layerId)
    if (layer) {
      layer.opacity = Math.max(0, Math.min(1, opacity))
    }
  }

  function moveLayer(layerId: string, direction: 'up' | 'down') {
    const index = layers.value.findIndex(l => l.id === layerId)
    if (index < 0) return

    const newIndex = direction === 'up' ? index - 1 : index + 1
    if (newIndex < 0 || newIndex >= layers.value.length) return

    const [layer] = layers.value.splice(index, 1)
    layers.value.splice(newIndex, 0, layer)
  }

  function duplicateLayer(layerId: string): Layer | null {
    const layer = layers.value.find(l => l.id === layerId)
    if (!layer) return null

    const newLayer: Layer = {
      ...layer,
      id: generateLayerId(),
      name: `${layer.name} 副本`,
      data: layer.data
        ? new ImageData(
            new Uint8ClampedArray(layer.data.data),
            layer.data.width,
            layer.data.height
          )
        : null
    }

    const index = layers.value.findIndex(l => l.id === layerId)
    layers.value.splice(index + 1, 0, newLayer)
    return newLayer
  }

  function clearLayer(layerId: string) {
    const layer = layers.value.find(l => l.id === layerId)
    if (layer && layer.data) {
      const ctx = document.createElement('canvas').getContext('2d')!
      layer.data = ctx.createImageData(layer.width, layer.height)
    }
  }

  function getLayerById(layerId: string): Layer | undefined {
    return layers.value.find(l => l.id === layerId)
  }

  function getLayersByType(type: LayerType): Layer[] {
    return layers.value.filter(l => l.type === type)
  }

  function resizeAllLayers(newWidth: number, newHeight: number) {
    layers.value.forEach(layer => {
      if (layer.data) {
        const tempCanvas = document.createElement('canvas')
        tempCanvas.width = layer.width
        tempCanvas.height = layer.height
        const tempCtx = tempCanvas.getContext('2d')!
        tempCtx.putImageData(layer.data, 0, 0)

        const newCanvas = document.createElement('canvas')
        newCanvas.width = newWidth
        newCanvas.height = newHeight
        const newCtx = newCanvas.getContext('2d')!
        newCtx.drawImage(tempCanvas, 0, 0, newWidth, newHeight)

        layer.data = newCtx.getImageData(0, 0, newWidth, newHeight)
        layer.width = newWidth
        layer.height = newHeight
      }
    })
  }

  function resetLayers() {
    layers.value = []
    activeLayerId.value = null
  }

  return {
    layers,
    activeLayerId,
    activeLayer,
    visibleLayers,
    createLayer,
    addLayer,
    removeLayer,
    setActiveLayer,
    toggleLayerVisibility,
    setLayerOpacity,
    moveLayer,
    duplicateLayer,
    clearLayer,
    getLayerById,
    getLayersByType,
    resizeAllLayers,
    createEmptyImageData,
    resetLayers
  }
}
