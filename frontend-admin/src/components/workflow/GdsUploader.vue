<template>
  <div class="gds-uploader">
    <el-card shadow="never" class="upload-card">
      <div class="card-header">
        <div class="title">
          <el-icon size="18" color="#409eff"><Upload /></el-icon>
          <span>GDS 版图上传</span>
        </div>
        <el-tag size="small" type="info" effect="plain">支持 .gds / .gdsii</el-tag>
      </div>

      <el-upload
        class="upload-dragger"
        drag
        :auto-upload="false"
        :show-file-list="false"
        accept=".gds,.gdsii"
        :on-change="handleFileChange"
        :disabled="uploading"
      >
        <el-icon class="upload-icon" :size="48">
          <UploadFilled />
        </el-icon>
        <div class="upload-text">
          <p class="main-text">将 GDS 文件拖到此处，或 <em>点击上传</em></p>
          <p class="sub-text">支持单文件上传，文件大小建议不超过 50MB</p>
        </div>
      </el-upload>

      <div v-if="uploading" class="upload-progress">
        <el-progress :percentage="uploadPercent" :status="'success'" :stroke-width="6" />
        <span class="uploading-label">正在解析 GDS 文件...</span>
      </div>

      <div v-if="selectedFile" class="selected-file">
        <div class="file-info">
          <el-icon size="16" color="#e6a23c"><Document /></el-icon>
          <span class="file-name">{{ selectedFile.name }}</span>
          <span class="file-size">{{ formatFileSize(selectedFile.size) }}</span>
        </div>
        <el-button type="danger" link size="small" @click="clearFile" :disabled="uploading">
          <el-icon><Delete /></el-icon>
          移除
        </el-button>
      </div>
    </el-card>

    <el-card v-if="gdsFiles.length > 0" shadow="never" class="files-card">
      <div class="card-header">
        <div class="title">
          <el-icon size="18" color="#67c23a"><Folder /></el-icon>
          <span>已上传 GDS 文件</span>
          <el-badge :value="gdsFiles.length" class="count-badge" />
        </div>
        <el-button type="primary" link size="small" @click="refreshFiles" :loading="loading">
          <el-icon><Refresh /></el-icon>
          刷新
        </el-button>
      </div>

      <el-table :data="gdsFiles" size="small" style="width: 100%" v-loading="loading" empty-text="暂无上传的文件">
        <el-table-column prop="filename" label="文件名" min-width="160">
          <template #default="{ row }">
            <div class="filename-cell">
              <el-icon size="14" color="#409eff"><Document /></el-icon>
              <span>{{ row.filename }}</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="大小" width="100" align="center">
          <template #default="{ row }">
            {{ formatFileSize(row.size) }}
          </template>
        </el-table-column>
        <el-table-column label="层数" width="80" align="center">
          <template #default="{ row }">
            {{ row.layers?.length || 0 }}
          </template>
        </el-table-column>
        <el-table-column label="上传时间" width="160" align="center">
          <template #default="{ row }">
            {{ formatTime(row.uploaded_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="100" align="center" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link size="small" @click="selectGdsFile(row)">
              选择
            </el-button>
            <el-button type="danger" link size="small" @click="deleteGdsFile(row)">
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-card v-if="selectedGds && showLayerSelect" shadow="never" class="layer-card">
      <div class="card-header">
        <div class="title">
          <el-icon size="18" color="#e6a23c"><Grid /></el-icon>
          <span>选择层 (Layer)</span>
        </div>
        <el-tag size="small">{{ selectedGds.filename }}</el-tag>
      </div>

      <el-form label-width="100px">
        <el-form-item label="GDS 层号" required>
          <el-select v-model="selectedLayer" placeholder="请选择层号" style="width: 100%">
            <el-option
              v-for="layer in selectedGds.layers"
              :key="`${layer.layer}/${layer.datatype}`"
              :value="layer.layer"
              :label="`Layer ${layer.layer} / Datatype ${layer.datatype}${layer.name ? ' - ' + layer.name : ''}`"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="Datatype">
          <el-select v-model="selectedDatatype" placeholder="自动 (0)" style="width: 100%">
            <el-option :value="0" label="0 - 默认" />
            <el-option
              v-for="layer in (selectedGds.layers || []).filter(l => l.layer === selectedLayer)"
              :key="layer.datatype"
              :value="layer.datatype"
              :label="String(layer.datatype)"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="Cell 列表" v-if="selectedGds.cells?.length">
          <el-select
            v-model="selectedCell"
            placeholder="选择顶层 Cell（可选）"
            filterable
            style="width: 100%"
          >
            <el-option
              v-for="cell in selectedGds.cells"
              :key="cell"
              :value="cell"
              :label="cell"
            />
          </el-select>
        </el-form-item>
      </el-form>

      <div class="cell-preview" v-if="selectedGds.cells?.length">
        <div class="preview-header">
          <el-icon size="14"><Grid /></el-icon>
          <span>共 {{ selectedGds.cells.length }} 个 Cell</span>
        </div>
        <div class="cell-tags">
          <el-tag
            v-for="cell in selectedGds.cells.slice(0, 20)"
            :key="cell"
            size="small"
            :type="selectedCell === cell ? 'primary' : 'info'"
            effect="plain"
            class="cell-tag"
            @click="selectedCell = selectedCell === cell ? '' : cell"
          >
            {{ cell }}
          </el-tag>
          <el-tag v-if="selectedGds.cells.length > 20" size="small" type="info" effect="plain">
            +{{ selectedGds.cells.length - 20 }} 更多
          </el-tag>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  Upload, UploadFilled, Document, Folder, Refresh, Delete, Grid
} from '@element-plus/icons-vue'
import { gdsApi } from '@/api'
import type { GdsFileInfo } from '@/types/workflow'

interface Props {
  showLayerSelect?: boolean
  modelValue?: string
  selectedLayerValue?: number | null
}

const props = withDefaults(defineProps<Props>(), {
  showLayerSelect: true,
  modelValue: '',
  selectedLayerValue: null,
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
  (e: 'update:selectedLayerValue', value: number | null): void
  (e: 'select', file: GdsFileInfo, layer: number, datatype: number): void
}>()

const uploading = ref(false)
const uploadPercent = ref(0)
const loading = ref(false)
const loadingLayers = ref(false)
const selectedFile = ref<File | null>(null)
const gdsFiles = ref<GdsFileInfo[]>([])
const selectedGds = ref<GdsFileInfo | null>(null)
const selectedLayer = ref<number | null>(null)
const selectedDatatype = ref(0)
const selectedCell = ref('')

onMounted(() => {
  loadGdsFiles()
})

watch(() => props.modelValue, (val) => {
  if (val && selectedGds.value?.file_id !== val) {
    const found = gdsFiles.value.find(f => f.file_id === val)
    if (found) {
      selectedGds.value = found
      loadGdsLayers(found)
    }
  }
})

watch(() => props.selectedLayerValue, (val) => {
  if (val !== null && selectedLayer.value !== val) {
    selectedLayer.value = val
  }
})

async function loadGdsFiles() {
  loading.value = true
  try {
    const res: any = await gdsApi.list()
    gdsFiles.value = res.files || []
  } catch (e) {
    console.error('加载 GDS 文件列表失败:', e)
  } finally {
    loading.value = false
  }
}

function refreshFiles() {
  loadGdsFiles()
}

async function loadGdsLayers(file: GdsFileInfo) {
  if (file.layers && file.layers.length > 0) return
  loadingLayers.value = true
  try {
    const res = await gdsApi.getLayers(file.file_id)
    file.layers = res.layers
    file.cells = res.cells
  } catch (e) {
    console.error('加载 GDS 层信息失败:', e)
  } finally {
    loadingLayers.value = false
  }
}

function handleFileChange(file: any) {
  if (!file.raw) return
  selectedFile.value = file.raw
  doUpload(file.raw)
}

async function doUpload(file: File) {
  uploading.value = true
  uploadPercent.value = 10
  try {
    const uploadedFile = await gdsApi.upload(file)
    uploadPercent.value = 100
    ElMessage.success(`上传成功：${uploadedFile.filename || file.name}`)
    await loadGdsFiles()
    const found = gdsFiles.value.find(f => f.file_id === uploadedFile.file_id)
    if (found) {
      selectGdsFile(found)
    }
  } catch (e: any) {
    ElMessage.error(e?.message || '上传失败')
  } finally {
    uploading.value = false
    uploadPercent.value = 0
  }
}

function clearFile() {
  selectedFile.value = null
}

function selectGdsFile(file: GdsFileInfo) {
  selectedGds.value = file
  emit('update:modelValue', file.file_id)
  loadGdsLayers(file).then(() => {
    if (file.layers?.length && selectedLayer.value === null) {
      selectedLayer.value = file.layers[0].layer
    }
    emit('select', file, selectedLayer.value || 0, selectedDatatype.value)
  })
}

async function deleteGdsFile(file: GdsFileInfo) {
  try {
    await ElMessageBox.confirm(`确认删除 GDS 文件 "${file.filename}"？`, '确认删除', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
    await gdsApi.delete(file.file_id)
    ElMessage.success('删除成功')
    await loadGdsFiles()
    if (selectedGds.value?.file_id === file.file_id) {
      selectedGds.value = null
      emit('update:modelValue', '')
    }
  } catch (e: any) {
    if (e !== 'cancel') {
      ElMessage.error(e?.message || '删除失败')
    }
  }
}

function formatFileSize(bytes: number): string {
  if (!bytes || bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return (bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i]
}

function formatTime(timestamp: number): string {
  if (!timestamp) return '—'
  const d = new Date(timestamp * 1000)
  const pad = (n: number) => n.toString().padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

watch([selectedLayer, selectedDatatype], () => {
  if (selectedGds.value && selectedLayer.value !== null) {
    emit('select', selectedGds.value, selectedLayer.value, selectedDatatype.value)
    emit('update:selectedLayerValue', selectedLayer.value)
  }
})
</script>

<style lang="scss" scoped>
.gds-uploader {
  display: flex;
  flex-direction: column;
  gap: 16px;

  .upload-card,
  .files-card,
  .layer-card {
    border-radius: 8px;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ebeef5;

    .title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 600;
      font-size: 14px;
      color: #303133;
    }

    .count-badge {
      margin-left: 4px;
    }
  }

  .upload-dragger {
    :deep(.el-upload-dragger) {
      padding: 24px 20px;
      background: #fafafa;
      border-radius: 8px;
      transition: all 0.2s;

      &:hover {
        background: #f0f7ff;
        border-color: #409eff;
      }
    }

    .upload-icon {
      color: #409eff;
      margin-bottom: 8px;
    }

    .upload-text {
      .main-text {
        font-size: 14px;
        color: #606266;
        margin: 0 0 4px 0;

        em {
          color: #409eff;
          font-style: normal;
        }
      }

      .sub-text {
        font-size: 12px;
        color: #909399;
        margin: 0;
      }
    }
  }

  .upload-progress {
    margin-top: 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;

    .uploading-label {
      font-size: 12px;
      color: #606266;
    }
  }

  .selected-file {
    margin-top: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    background: #fdf6ec;
    border: 1px solid #faecd8;
    border-radius: 6px;

    .file-info {
      display: flex;
      align-items: center;
      gap: 8px;

      .file-name {
        font-weight: 500;
        color: #b88230;
      }

      .file-size {
        font-size: 12px;
        color: #e6a23c;
      }
    }
  }

  .filename-cell {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
  }

  .layer-card {
    .cell-preview {
      margin-top: 8px;
      padding: 10px 12px;
      background: #f5f7fa;
      border-radius: 6px;

      .preview-header {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #606266;
        margin-bottom: 8px;
      }

      .cell-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;

        .cell-tag {
          cursor: pointer;
          transition: all 0.2s;

          &:hover {
            transform: translateY(-1px);
          }
        }
      }
    }
  }
}
</style>
