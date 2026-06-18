<template>
  <div class="form-group-wrap">
    <h3 class="form-section-title">光学系统参数</h3>
    <el-form :model="formData" label-width="160px" label-position="right">
      <el-row :gutter="24">
        <el-col :span="24">
          <el-form-item label="技术节点">
            <el-select v-model="formData.technology_node" style="width: 100%" @change="onTechNodeChange">
              <el-option label="DUV ArF (193nm)" value="duv_arf" />
              <el-option label="EUV (13.5nm)" value="euv" />
            </el-select>
          </el-form-item>
        </el-col>
      </el-row>

      <el-divider content-position="left">核心光学参数</el-divider>

      <el-row :gutter="24">
        <el-col :span="12">
          <el-form-item label="光源波长 (nm)">
            <el-input-number
              v-model="formData.wavelength"
              :min="10"
              :max="1000"
              :step="1"
              :precision="2"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="数值孔径 (NA)">
            <el-input-number
              v-model="formData.na"
              :min="0.1"
              :max="2"
              :step="0.01"
              :precision="3"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24">
        <el-col :span="12">
          <el-form-item label="部分相干因子 σ">
            <el-input-number
              v-model="formData.sigma"
              :min="0"
              :max="1"
              :step="0.05"
              :precision="3"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="像素尺寸 (nm)">
            <el-input-number
              v-model="formData.pixel_size"
              :min="0.1"
              :max="100"
              :step="0.1"
              :precision="2"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24">
        <el-col :span="12">
          <el-form-item label="离焦量 (nm)">
            <el-input-number
              v-model="formData.defocus"
              :step="1"
              :precision="1"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="放大倍率">
            <el-input-number
              v-model="formData.magnification"
              :min="1"
              :step="0.5"
              :precision="2"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24">
        <el-col :span="12">
          <el-form-item label="照明模式">
            <el-select v-model="formData.illumination_type" style="width: 100%">
              <el-option label="常规照明 (Conventional)" value="conventional" />
              <el-option label="环形照明 (Annular)" value="annular" />
              <el-option label="偶极照明 (Dipole)" value="dipole" />
              <el-option label="四极照明 (Quasar)" value="quasar" />
              <el-option label="自定义 (Custom)" value="custom" />
            </el-select>
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="TCC 计算模式">
            <el-select v-model="formData.tcc_mode" style="width: 100%">
              <el-option label="完整TCC (高精度)" value="full_tcc" />
              <el-option label="SOCS 低秩近似 (平衡)" value="socs" />
              <el-option label="2D核近似 (最快)" value="kernel_2d" />
            </el-select>
          </el-form-item>
        </el-col>
      </el-row>

      <el-divider content-position="left">光源形状参数</el-divider>

      <el-row :gutter="24">
        <el-col :span="12">
          <el-form-item label="内环 σ (inner)">
            <el-input-number
              v-model="formData.source_params.sigma_inner"
              :min="0"
              :max="1"
              :step="0.05"
              :precision="3"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="外环 σ (outer)">
            <el-input-number
              v-model="formData.source_params.sigma_outer"
              :min="0.01"
              :max="1"
              :step="0.05"
              :precision="3"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24" v-if="formData.illumination_type === 'dipole' || formData.illumination_type === 'quasar'">
        <el-col :span="12">
          <el-form-item label="极角 (度)">
            <el-input-number
              v-model="formData.source_params.angle"
              :min="0"
              :max="360"
              :step="5"
              :precision="1"
              :controls="true"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="开口角 (度)">
            <el-input-number
              v-model="formData.source_params.opening_angle"
              :min="0"
              :max="180"
              :step="5"
              :precision="1"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-divider content-position="left">EUV 特有参数</el-divider>

      <el-row :gutter="24" v-if="formData.technology_node === 'euv'">
        <el-col :span="12">
          <el-form-item label="Flare 系数">
            <el-input-number
              v-model="formData.flare"
              :min="0"
              :max="1"
              :step="0.01"
              :precision="3"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="反射式掩模衰减">
            <el-input-number
              v-model="formData.reflective_mask_attenuation"
              :min="0"
              :max="1"
              :step="0.05"
              :precision="3"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-row :gutter="24" v-if="formData.technology_node === 'euv'">
        <el-col :span="12">
          <el-form-item label="阴影效应模型">
            <el-select v-model="formData.shadowing_model" style="width: 100%">
              <el-option label="不考虑 (None)" value="none" />
              <el-option label="近似几何模型 (Approximate)" value="approximate" />
              <el-option label="严格电磁模型 (Rigorous)" value="rigorous" />
            </el-select>
          </el-form-item>
        </el-col>
      </el-row>

      <el-divider content-position="left">SOCS 与像差</el-divider>

      <el-row :gutter="24">
        <el-col :span="12">
          <el-form-item label="SOCS 分解项数" v-if="formData.tcc_mode === 'socs'">
            <el-input-number
              v-model="formData.socs_num_terms"
              :min="1"
              :max="50"
              :step="1"
              controls-position="right"
              style="width: 100%"
            />
          </el-form-item>
        </el-col>
      </el-row>

      <el-form-item label="Zernike 像差系数">
        <div class="zernike-inputs">
          <div v-for="(item, idx) in zernikeList" :key="idx" class="zernike-item">
            <el-select
              v-model="item.name"
              placeholder="选择像差类型"
              size="small"
              style="width: 180px"
              @change="onZernikeChange"
            >
              <el-option
                v-for="opt in zernikeOptions"
                :key="opt.value"
                :label="opt.label"
                :value="opt.value"
              />
            </el-select>
            <el-input-number
              v-model="item.value"
              :step="0.01"
              :precision="4"
              size="small"
              controls-position="right"
              style="width: 140px"
            />
            <el-button type="danger" size="small" link @click="removeZernike(idx)">
              <el-icon><Delete /></el-icon>
            </el-button>
          </div>
          <el-button type="primary" size="small" plain @click="addZernike">
            <el-icon><Plus /></el-icon>
            添加像差
          </el-button>
        </div>
      </el-form-item>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, reactive } from 'vue'
import type { OpticalSystem } from '@/types/config'

const props = defineProps<{
  modelValue: OpticalSystem
}>()

const emit = defineEmits<{
  'update:modelValue': [value: OpticalSystem]
}>()

const formData = reactive<OpticalSystem>({ ...props.modelValue, source_params: { ...props.modelValue.source_params } })

const zernikeOptions = [
  { label: 'Piston (平移)', value: 'piston' },
  { label: 'Tilt X (X倾斜)', value: 'tilt_x' },
  { label: 'Tilt Y (Y倾斜)', value: 'tilt_y' },
  { label: 'Defocus (离焦)', value: 'defocus' },
  { label: 'Astigmatism X (X像散)', value: 'astigmatism_x' },
  { label: 'Astigmatism Y (Y像散)', value: 'astigmatism_y' },
  { label: 'Coma X (X彗差)', value: 'coma_x' },
  { label: 'Coma Y (Y彗差)', value: 'coma_y' },
  { label: 'Trefoil X (X三瓣)', value: 'trefoil_x' },
  { label: 'Trefoil Y (Y三瓣)', value: 'trefoil_y' },
  { label: 'Spherical (球差)', value: 'spherical' },
  { label: 'Sec. Astigmatism X', value: 'secondary_astigmatism_x' },
  { label: 'Sec. Astigmatism Y', value: 'secondary_astigmatism_y' },
  { label: 'Sec. Coma X', value: 'secondary_coma_x' },
  { label: 'Sec. Coma Y', value: 'secondary_coma_y' },
  { label: 'Sec. Spherical', value: 'secondary_spherical' }
]

interface ZernikeItem { name: string; value: number }
const zernikeList = ref<ZernikeItem[]>([])

function syncZernikeFromDict() {
  zernikeList.value = Object.entries(props.modelValue.zernike_coefficients || {}).map(
    ([name, value]) => ({ name, value })
  )
}
syncZernikeFromDict()

function onTechNodeChange(techNode: string) {
  if (techNode === 'euv') {
    formData.wavelength = 13.5
    formData.na = 0.33
    formData.pixel_size = 0.5
    formData.flare = 0.05
    formData.shadowing_model = 'approximate'
    formData.reflective_mask_attenuation = 0.6
  } else {
    formData.wavelength = 193.0
    formData.na = 1.35
    formData.pixel_size = 1.0
    formData.flare = 0.0
    formData.shadowing_model = 'none'
    formData.reflective_mask_attenuation = 0.0
  }
}

function onZernikeChange() {
  syncToDict()
}

function addZernike() {
  zernikeList.value.push({ name: 'spherical', value: 0.0 })
  syncToDict()
}

function removeZernike(idx: number) {
  zernikeList.value.splice(idx, 1)
  syncToDict()
}

function syncToDict() {
  formData.zernike_coefficients = {}
  for (const item of zernikeList.value) {
    if (item.name) formData.zernike_coefficients[item.name] = item.value
  }
}

watch(
  () => props.modelValue,
  (val) => {
    Object.assign(formData, val)
    formData.source_params = { ...val.source_params }
    syncZernikeFromDict()
  },
  { deep: true }
)

watch(
  formData,
  (val) => {
    syncToDict()
    emit('update:modelValue', JSON.parse(JSON.stringify(val)))
  },
  { deep: true }
)
</script>

<style lang="scss" scoped>
.zernike-inputs {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.zernike-item {
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>
