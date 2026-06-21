import type { Layer, MaskExportData, SimulationSubmitParams } from '../types'
import { simulationApi } from '@/api'

export function useMaskExport() {
  function layerToGrayscaleArray(layer: Layer): number[][] {
    if (!layer.data) return []

    const { data, width, height } = layer.data
    const result: number[][] = []

    for (let y = 0; y < height; y++) {
      const row: number[] = []
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4
        const alpha = data[idx + 3] / 255
        const value = data[idx] / 255 * alpha
        row.push(value)
      }
      result.push(row)
    }

    return result
  }

  function grayscaleArrayToImageData(
    array: number[][],
    width: number,
    height: number
  ): ImageData {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')!
    const imageData = ctx.createImageData(width, height)

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4
        const value = array[y]?.[x] ?? 0
        const byteVal = Math.round(Math.max(0, Math.min(1, value)) * 255)
        imageData.data[idx] = byteVal
        imageData.data[idx + 1] = byteVal
        imageData.data[idx + 2] = byteVal
        imageData.data[idx + 3] = 255
      }
    }

    return imageData
  }

  function exportMaskData(
    maskLayer: Layer,
    pupilLayer?: Layer | null,
    pixelSize: number = 1.0
  ): MaskExportData {
    return {
      width: maskLayer.width,
      height: maskLayer.height,
      maskData: layerToGrayscaleArray(maskLayer),
      pupilData: pupilLayer ? layerToGrayscaleArray(pupilLayer) : null,
      pixelSize
    }
  }

  async function submitSimulation(
    params: SimulationSubmitParams,
    config: any
  ): Promise<any> {
    try {
      const patternParams = {
        size: [params.maskData.length, params.maskData[0]?.length || 0],
        custom_mask: params.maskData,
        custom_pupil: params.pupilData || null
      }

      const result = await simulationApi.run(
        config,
        'custom_mask',
        patternParams
      )
      return result
    } catch (error) {
      console.error('提交仿真失败:', error)
      throw error
    }
  }

  function downloadMaskAsImage(layer: Layer, filename: string = 'mask.png') {
    if (!layer.data) return

    const canvas = document.createElement('canvas')
    canvas.width = layer.width
    canvas.height = layer.height
    const ctx = canvas.getContext('2d')!
    ctx.putImageData(layer.data, 0, 0)

    const link = document.createElement('a')
    link.download = filename
    link.href = canvas.toDataURL('image/png')
    link.click()
  }

  function downloadMaskAsJSON(
    exportData: MaskExportData,
    filename: string = 'mask_data.json'
  ) {
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    })
    const link = document.createElement('a')
    link.download = filename
    link.href = URL.createObjectURL(blob)
    link.click()
    URL.revokeObjectURL(link.href)
  }

  function loadMaskFromImage(
    file: File,
    width: number,
    height: number
  ): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = () => {
        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height
        const ctx = canvas.getContext('2d')!
        ctx.drawImage(img, 0, 0, width, height)
        resolve(ctx.getImageData(0, 0, width, height))
      }
      img.onerror = reject
      img.src = URL.createObjectURL(file)
    })
  }

  function loadMaskFromJSON(file: File): Promise<MaskExportData> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        try {
          const data = JSON.parse(reader.result as string)
          resolve(data as MaskExportData)
        } catch (e) {
          reject(e)
        }
      }
      reader.onerror = reject
      reader.readAsText(file)
    })
  }

  return {
    layerToGrayscaleArray,
    grayscaleArrayToImageData,
    exportMaskData,
    submitSimulation,
    downloadMaskAsImage,
    downloadMaskAsJSON,
    loadMaskFromImage,
    loadMaskFromJSON
  }
}
