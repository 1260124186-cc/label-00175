<template>
  <div class="config-page">
    <el-card v-loading="configStore.loading" class="main-card">
      <div class="card-header">
        <div class="title">
          <el-icon size="20" color="#409eff"><Setting /></el-icon>
          <span>光刻仿真参数配置</span>
        </div>
        <div class="actions">
          <el-button type="info" @click="handleLoadDefault">
            <el-icon><Refresh /></el-icon>
            加载默认
          </el-button>
          <el-button @click="handleReset">
            <el-icon><RefreshLeft /></el-icon>
            重置
          </el-button>
          <el-dropdown trigger="click" @command="handleLoadSaved">
            <el-button type="success" plain>
              <el-icon><FolderOpened /></el-icon>
              加载已保存
              <el-icon><ArrowDown /></el-icon>
            </el-button>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item
                  v-for="file in configStore.savedFiles"
                  :key="file.filename"
                  :command="file.filename"
                >
                  {{ file.filename }}
                  <span style="color: #909399; font-size: 12px; margin-left: 8px">
                    {{ formatTime(file.modified) }}
                  </span>
                </el-dropdown-item>
                <el-dropdown-item v-if="!configStore.savedFiles.length" disabled>
                  暂无保存的配置
                </el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
          <el-button type="primary" plain @click="handleValidate">
            <el-icon><CircleCheck /></el-icon>
            验证配置
          </el-button>
          <el-button type="success" @click="handleSaveConfig">
            <el-icon><Download /></el-icon>
            保存配置
          </el-button>
        </div>
      </div>

      <el-tabs v-model="activeTab" type="card" class="config-tabs">
        <el-tab-pane label="光学系统" name="optical">
          <OpticalParamsForm v-model="configStore.config.optical_system" />
        </el-tab-pane>
        <el-tab-pane label="优化器" name="optimizer">
          <OptimizerParamsForm v-model="configStore.config.optimization" />
        </el-tab-pane>
        <el-tab-pane label="损失权重" name="loss">
          <LossWeightsForm v-model="configStore.config.optimization.loss_weights" />
        </el-tab-pane>
        <el-tab-pane label="空间加权" name="spatial">
          <SpatialWeightForm v-model="configStore.config.optimization.spatial_weight" />
        </el-tab-pane>
        <el-tab-pane label="正则化" name="regularization">
          <RegularizationForm v-model="configStore.config.optimization.regularization" />
        </el-tab-pane>
        <el-tab-pane label="输出与成像" name="output">
          <OutputImagingForm
            v-model:outputValue="configStore.config.output"
            v-model:imagingValue="configStore.config.imaging"
          />
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <el-dialog v-model="saveDialogVisible" title="保存配置" width="480px">
      <el-form :model="saveForm" label-width="100px">
        <el-form-item label="文件名">
          <el-input v-model="saveForm.filename" placeholder="自定义名称 (可选)" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="saveDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="doSaveConfig">确认保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useConfigStore } from '@/stores/config'
import { configApi } from '@/api'
import OpticalParamsForm from '@/components/config/OpticalParamsForm.vue'
import OptimizerParamsForm from '@/components/config/OptimizerParamsForm.vue'
import LossWeightsForm from '@/components/config/LossWeightsForm.vue'
import SpatialWeightForm from '@/components/config/SpatialWeightForm.vue'
import RegularizationForm from '@/components/config/RegularizationForm.vue'
import OutputImagingForm from '@/components/config/OutputImagingForm.vue'

const configStore = useConfigStore()
const activeTab = ref('optical')

const saveDialogVisible = ref(false)
const saveForm = ref({ filename: '' })

onMounted(async () => {
  try {
    await configStore.loadInitialData()
  } catch (e) {
    console.error('加载初始数据失败:', e)
  }
})

function handleLoadDefault() {
  ElMessageBox.confirm('确认加载默认配置？当前未保存的修改将丢失。', '提示', {
    confirmButtonText: '确认',
    cancelButtonText: '取消',
    type: 'warning'
  })
    .then(async () => {
      await configStore.loadDefault()
      ElMessage.success('已加载默认配置')
    })
    .catch(() => {})
}

function handleReset() {
  ElMessageBox.confirm('确认重置为默认值？当前未保存的修改将丢失。', '提示', {
    confirmButtonText: '确认',
    cancelButtonText: '取消',
    type: 'warning'
  })
    .then(() => {
      configStore.resetToDefault()
      ElMessage.success('已重置')
    })
    .catch(() => {})
}

async function handleLoadSaved(filename: string) {
  try {
    await configStore.loadSaved(filename)
    ElMessage.success(`已加载: ${filename}`)
  } catch (e: any) {
    ElMessage.error('加载失败')
  }
}

async function handleValidate() {
  try {
    const res: any = await configApi.validate(configStore.config)
    if (res.valid) {
      ElMessage.success('配置验证通过')
    } else {
      ElMessage.error(res.message || '配置验证失败')
    }
  } catch (e: any) {
    ElMessage.error(e?.message || '验证异常')
  }
}

function handleSaveConfig() {
  saveForm.value.filename = ''
  saveDialogVisible.value = true
}

async function doSaveConfig() {
  try {
    const filename = saveForm.value.filename?.trim() || undefined
    const res: any = await configApi.save(configStore.config, filename)
    if (res.success) {
      ElMessage.success(res.message || '保存成功')
      saveDialogVisible.value = false
      await configStore.fetchSavedList()
    } else {
      ElMessage.error(res.message || '保存失败')
    }
  } catch (e: any) {
    ElMessage.error('保存失败')
  }
}

function formatTime(timestamp: number) {
  const d = new Date(timestamp * 1000)
  const pad = (n: number) => n.toString().padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}
</script>

<style lang="scss" scoped>
.config-page {
  .main-card {
    border-radius: 8px;
  }

  .actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .config-tabs {
    margin-top: 8px;

    :deep(.el-tabs__content) {
      padding-top: 16px;
    }
  }
}
</style>
