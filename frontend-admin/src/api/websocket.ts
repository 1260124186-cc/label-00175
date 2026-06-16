import { ref, onUnmounted, type Ref } from 'vue'

export interface TaskProgressMessage {
  type: string
  task_id: string
  progress?: number
  message?: string
  stage?: string
  loss?: number
  iteration?: number
  mask_thumbnail?: string
  result?: any
  error?: string
  [key: string]: any
}

export type WebSocketMessageHandler = (message: TaskProgressMessage) => void

class TaskWebSocket {
  private ws: WebSocket | null = null
  private taskId: string = ''
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null
  private reconnectAttempts: number = 0
  private maxReconnectAttempts: number = 5
  private reconnectDelay: number = 2000
  private handlers: Set<WebSocketMessageHandler> = new Set()
  
  public isConnected: Ref<boolean> = ref(false)
  public lastMessage: Ref<TaskProgressMessage | null> = ref(null)

  constructor() {}

  connect(taskId: string): void {
    if (this.ws && this.taskId === taskId && this.isConnected.value) {
      return
    }

    if (this.ws) {
      this.disconnect()
    }

    this.taskId = taskId
    this.reconnectAttempts = 0
    this._connect()
  }

  private _connect(): void {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const wsUrl = `${protocol}//${host}/ws/tasks/${this.taskId}`

    try {
      this.ws = new WebSocket(wsUrl)

      this.ws.onopen = () => {
        console.log(`[WebSocket] 连接已建立: ${this.taskId}`)
        this.isConnected.value = true
        this.reconnectAttempts = 0
        this._startHeartbeat()
      }

      this.ws.onmessage = (event) => {
        try {
          const message: TaskProgressMessage = JSON.parse(event.data)
          this.lastMessage.value = message
          this._notifyHandlers(message)
        } catch (e) {
          console.error('[WebSocket] 消息解析失败:', e)
        }
      }

      this.ws.onerror = (error) => {
        console.error('[WebSocket] 连接错误:', error)
      }

      this.ws.onclose = (event) => {
        console.log(`[WebSocket] 连接已关闭: code=${event.code}, reason=${event.reason}`)
        this.isConnected.value = false
        this._stopHeartbeat()
        this._tryReconnect()
      }
    } catch (e) {
      console.error('[WebSocket] 创建连接失败:', e)
      this._tryReconnect()
    }
  }

  private _startHeartbeat(): void {
    this._stopHeartbeat()
    this.heartbeatTimer = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' })
      }
    }, 20000)
  }

  private _stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  private _tryReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.warn('[WebSocket] 达到最大重连次数，停止重连')
      return
    }

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
    }

    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1)
    
    console.log(`[WebSocket] ${delay / 1000}秒后尝试重连 (${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
    
    this.reconnectTimer = setTimeout(() => {
      this._connect()
    }, delay)
  }

  send(data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    }
  }

  onMessage(handler: WebSocketMessageHandler): () => void {
    this.handlers.add(handler)
    return () => {
      this.handlers.delete(handler)
    }
  }

  private _notifyHandlers(message: TaskProgressMessage): void {
    this.handlers.forEach((handler) => {
      try {
        handler(message)
      } catch (e) {
        console.error('[WebSocket] 消息处理错误:', e)
      }
    })
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    
    this._stopHeartbeat()
    
    if (this.ws) {
      try {
        this.ws.close(1000, '客户端主动断开')
      } catch (e) {
        console.error('[WebSocket] 关闭连接失败:', e)
      }
      this.ws = null
    }
    
    this.isConnected.value = false
    this.handlers.clear()
    this.taskId = ''
    this.reconnectAttempts = 0
  }

  getTaskId(): string {
    return this.taskId
  }
}

const taskWs = new TaskWebSocket()

export function useTaskWebSocket(taskId: string | Ref<string>) {
  const taskIdRef = ref(typeof taskId === 'string' ? taskId : taskId.value)
  const isConnected = ref(false)
  const progress = ref(0)
  const message = ref('')
  const stage = ref('')
  const loss = ref<number | null>(null)
  const iteration = ref<number | null>(null)
  const maskThumbnail = ref('')
  const isCompleted = ref(false)
  const isFailed = ref(false)
  const error = ref('')
  const result = ref<any>(null)

  function handleMessage(msg: TaskProgressMessage) {
    if (msg.type === 'connected') {
      isConnected.value = true
    } else if (msg.type === 'progress') {
      if (msg.progress !== undefined) progress.value = msg.progress
      if (msg.message !== undefined) message.value = msg.message
      if (msg.stage !== undefined) stage.value = msg.stage
      if (msg.loss !== undefined) loss.value = msg.loss
      if (msg.iteration !== undefined) iteration.value = msg.iteration
      if (msg.mask_thumbnail !== undefined) maskThumbnail.value = msg.mask_thumbnail
    } else if (msg.type === 'stage_change') {
      if (msg.stage !== undefined) stage.value = msg.stage
      if (msg.message !== undefined) message.value = msg.message
    } else if (msg.type === 'task_complete') {
      isCompleted.value = true
      progress.value = 100
      result.value = msg.result
    } else if (msg.type === 'task_failed') {
      isFailed.value = true
      error.value = msg.error || '任务失败'
    }
  }

  function connect() {
    taskWs.connect(taskIdRef.value)
    
    const unsubscribe = taskWs.onMessage(handleMessage)
    
    isConnected.value = taskWs.isConnected.value
    
    return unsubscribe
  }

  const unsubscribe = connect()

  onUnmounted(() => {
    unsubscribe()
  })

  return {
    isConnected,
    progress,
    message,
    stage,
    loss,
    iteration,
    maskThumbnail,
    isCompleted,
    isFailed,
    error,
    result,
    disconnect: () => {
      unsubscribe()
      taskWs.disconnect()
    },
    send: (data: any) => taskWs.send(data),
  }
}

export default taskWs
