import { ref, computed } from 'vue'
import type { HistoryState, Layer } from '../types'

const MAX_HISTORY = 50

export function useUndoRedo() {
  const historyStack = ref<HistoryState[]>([])
  const historyIndex = ref(-1)

  const canUndo = computed(() => historyIndex.value > 0)
  const canRedo = computed(() => historyIndex.value < historyStack.value.length - 1)

  function cloneLayers(layers: Layer[]): Layer[] {
    return layers.map(layer => ({
      ...layer,
      data: layer.data ? new ImageData(
        new Uint8ClampedArray(layer.data.data),
        layer.data.width,
        layer.data.height
      ) : null
    }))
  }

  function saveState(layers: Layer[], activeLayerId: string | null) {
    const state: HistoryState = {
      layers: cloneLayers(layers),
      activeLayerId
    }

    if (historyIndex.value < historyStack.value.length - 1) {
      historyStack.value = historyStack.value.slice(0, historyIndex.value + 1)
    }

    historyStack.value.push(state)

    if (historyStack.value.length > MAX_HISTORY) {
      historyStack.value.shift()
    } else {
      historyIndex.value++
    }
  }

  function undo(): HistoryState | null {
    if (!canUndo.value) return null
    historyIndex.value--
    return historyStack.value[historyIndex.value]
      ? {
          layers: cloneLayers(historyStack.value[historyIndex.value].layers),
          activeLayerId: historyStack.value[historyIndex.value].activeLayerId
        }
      : null
  }

  function redo(): HistoryState | null {
    if (!canRedo.value) return null
    historyIndex.value++
    return historyStack.value[historyIndex.value]
      ? {
          layers: cloneLayers(historyStack.value[historyIndex.value].layers),
          activeLayerId: historyStack.value[historyIndex.value].activeLayerId
        }
      : null
  }

  function clearHistory() {
    historyStack.value = []
    historyIndex.value = -1
  }

  return {
    historyStack,
    historyIndex,
    canUndo,
    canRedo,
    saveState,
    undo,
    redo,
    clearHistory,
    cloneLayers
  }
}
