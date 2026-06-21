export type ToolType =
  | 'brush'
  | 'eraser'
  | 'polygon'
  | 'rectangle'
  | 'circle'
  | 'pupil_brush'
  | 'pupil_eraser'
  | 'move'
  | 'zoom'
  | 'picker'

export type LayerType = 'mask' | 'pupil' | 'reference'

export interface Point {
  x: number
  y: number
}

export interface Layer {
  id: string
  name: string
  type: LayerType
  visible: boolean
  opacity: number
  blendMode: GlobalCompositeOperation
  data: ImageData | null
  width: number
  height: number
}

export interface PolygonShape {
  id: string
  points: Point[]
  closed: boolean
  fill: boolean
  strokeWidth: number
  color: string
}

export interface BrushSettings {
  size: number
  hardness: number
  opacity: number
  flow: number
  color: string
}

export interface CanvasState {
  width: number
  height: number
  zoom: number
  panX: number
  panY: number
  currentTool: ToolType
  brushSettings: BrushSettings
  activeLayerId: string | null
  isDrawing: boolean
  lastPoint: Point | null
  polygonPoints: Point[]
  isPolygonDrawing: boolean
}

export interface HistoryState {
  layers: Layer[]
  activeLayerId: string | null
}

export interface MaskExportData {
  width: number
  height: number
  maskData: number[][]
  pupilData: number[][] | null
  pixelSize: number
}

export interface SimulationSubmitParams {
  maskData: number[][]
  pupilData?: number[][] | null
  config?: any
  patternType?: string
  patternParams?: Record<string, any>
}
