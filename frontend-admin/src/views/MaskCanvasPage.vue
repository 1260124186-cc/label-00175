<template>
  <div class="mask-canvas-page">
    <el-card class="page-card" shadow="never">
      <div class="card-header">
        <div class="title">
          <el-icon size="20" color="#409eff"><Grid /></el-icon>
          <span>交互式掩模 / 光源编辑器</span>
        </div>
        <el-tag type="success" effect="plain" round size="small">所见即所优</el-tag>
      </div>

      <div class="editor-container">
        <MaskCanvas
          ref="maskCanvasRef"
          :width="256"
          :height="256"
          @simulation-submitted="handleSimulationSubmitted"
          @change="handleCanvasChange"
        />
      </div>

      <div class="page-tips">
        <el-alert type="info" :closable="false" show-icon>
          <template #title>
            使用说明
          </template>
          <ul class="tips-list">
            <li><strong>快捷键：</strong>B 画笔 / E 橡皮 / P 多边形 / R 矩形 / C 圆形 / H 移动 / Z 缩放 / Ctrl+Z 撤销 / Ctrl+Y 重做</li>
            <li><strong>滚轮：</strong>缩放画布 · <strong>空格拖拽：</strong>平移画布</li>
            <li><strong>多边形：</strong>左键添加顶点，右键或双击闭合</li>
            <li><strong>掩模模式：</strong>编辑光刻掩模图案 · <strong>Pupil 模式：</strong>编辑光源光瞳分布</li>
            <li><strong>提交仿真：</strong>将当前编辑的掩模和光源配置提交到后端进行光刻仿真</li>
          </ul>
        </el-alert>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Grid } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import MaskCanvas from '@/components/MaskCanvas/index.vue'

const maskCanvasRef = ref<InstanceType<typeof MaskCanvas> | null>(null)

function handleSimulationSubmitted(taskId: string) {
  console.log('仿真任务已提交:', taskId)
}

function handleCanvasChange(data: { mask: number[][]; pupil: number[][] | null }) {
  console.log('画布数据更新:', data.mask.length, 'x', data.mask[0]?.length)
}
</script>

<style lang="scss" scoped>
.mask-canvas-page {
  padding: 20px;

  .page-card {
    border-radius: 10px;

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 16px;
      border-bottom: 1px solid #ebeef5;
      margin-bottom: 20px;

      .title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 18px;
        font-weight: 600;
        color: #303133;
      }
    }

    .editor-container {
      height: 600px;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid #e4e7ed;
    }

    .page-tips {
      margin-top: 20px;

      .tips-list {
        margin: 8px 0 0 0;
        padding-left: 20px;
        color: #606266;
        font-size: 13px;
        line-height: 1.8;

        li {
          margin-bottom: 4px;

          strong {
            color: #303133;
          }
        }
      }
    }
  }
}
</style>
