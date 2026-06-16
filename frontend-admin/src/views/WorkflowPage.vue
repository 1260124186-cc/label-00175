<template>
  <div class="workflow-page">
    <el-card class="page-header-card" shadow="never">
      <div class="page-header">
        <div class="header-left">
          <el-icon size="24" color="#409eff"><SetUp /></el-icon>
          <div class="header-text">
            <h2 class="page-title">RET 工作流工作台</h2>
            <p class="page-desc">光刻分辨率增强技术工作流：OPC / SMO / ILT 全流程仿真与优化</p>
          </div>
        </div>
        <div class="header-right">
          <el-tag type="primary" effect="dark" size="large">
            <el-icon size="12"><Cpu /></el-icon>
            &nbsp;光刻仿真 v1.0
          </el-tag>
        </div>
      </div>

      <el-tabs v-model="activeTab" class="workflow-tabs" type="card">
        <el-tab-pane name="console">
          <template #label>
            <span class="tab-label">
              <el-icon><MagicStick /></el-icon>
              优化操作台
            </span>
          </template>
        </el-tab-pane>
        <el-tab-pane name="process-window">
          <template #label>
            <span class="tab-label">
              <el-icon><Odometer /></el-icon>
              工艺窗口
            </span>
          </template>
        </el-tab-pane>
        <el-tab-pane name="batch">
          <template #label>
            <span class="tab-label">
              <el-icon><Files /></el-icon>
              批处理队列
            </span>
          </template>
        </el-tab-pane>
        <el-tab-pane name="compare">
          <template #label>
            <span class="tab-label">
              <el-icon><Histogram /></el-icon>
              实验对比
            </span>
          </template>
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <div class="tab-content">
      <div v-show="activeTab === 'console'" class="console-section">
        <el-tabs v-model="consoleTab" class="console-tabs">
          <el-tab-pane name="opc">
            <template #label>
              <span class="console-tab-label">
                <span class="tab-dot dot-opc"></span>
                OPC 光学邻近校正
              </span>
            </template>
          </el-tab-pane>
          <el-tab-pane name="smo">
            <template #label>
              <span class="console-tab-label">
                <span class="tab-dot dot-smo"></span>
                SMO 光源掩模协同
              </span>
            </template>
          </el-tab-pane>
          <el-tab-pane name="ilt">
            <template #label>
              <span class="console-tab-label">
                <span class="tab-dot dot-ilt"></span>
                ILT 反演光刻
              </span>
            </template>
          </el-tab-pane>
        </el-tabs>

        <div class="console-content">
          <OpcConsole v-show="consoleTab === 'opc'" @task-complete="onTaskComplete" />
          <SmoConsole v-show="consoleTab === 'smo'" @task-complete="onTaskComplete" />
          <IltConsole v-show="consoleTab === 'ilt'" @task-complete="onTaskComplete" />
        </div>
      </div>

      <div v-show="activeTab === 'process-window'">
        <BossungChart />
      </div>

      <div v-show="activeTab === 'batch'">
        <BatchQueueMonitor />
      </div>

      <div v-show="activeTab === 'compare'">
        <ExperimentCompare />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import {
  SetUp, MagicStick, Odometer, Files, Histogram, Cpu
} from '@element-plus/icons-vue'
import OpcConsole from '@/components/workflow/OpcConsole.vue'
import SmoConsole from '@/components/workflow/SmoConsole.vue'
import IltConsole from '@/components/workflow/IltConsole.vue'
import BossungChart from '@/components/workflow/BossungChart.vue'
import BatchQueueMonitor from '@/components/workflow/BatchQueueMonitor.vue'
import ExperimentCompare from '@/components/workflow/ExperimentCompare.vue'

const activeTab = ref('console')
const consoleTab = ref('opc')

function onTaskComplete(taskId: string) {
  ElMessage.success(`任务完成: ${taskId.slice(0, 8)}`)
}
</script>

<style lang="scss" scoped>
.workflow-page {
  .page-header-card {
    border-radius: 10px;
    margin-bottom: 16px;

    .page-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 4px;

      .header-left {
        display: flex;
        align-items: center;
        gap: 14px;

        .header-text {
          .page-title {
            margin: 0;
            font-size: 20px;
            font-weight: 600;
            color: #303133;
          }

          .page-desc {
            margin: 4px 0 0 0;
            font-size: 13px;
            color: #909399;
          }
        }
      }

      .header-right {
        display: flex;
        align-items: center;
        gap: 10px;
      }
    }

    .workflow-tabs {
      margin-top: 16px;

      :deep(.el-tabs__nav-wrap::after) {
        display: none;
      }

      .tab-label {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 14px;
        padding: 4px 8px;
      }
    }
  }

  .tab-content {
    min-height: 500px;
  }

  .console-section {
    .console-tabs {
      margin-bottom: 16px;

      :deep(.el-tabs__item) {
        font-size: 15px;
        height: 48px;
        line-height: 48px;
      }

      .console-tab-label {
        display: flex;
        align-items: center;
        gap: 8px;

        .tab-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          display: inline-block;

          &.dot-opc {
            background: #409eff;
            box-shadow: 0 0 8px rgba(64, 158, 255, 0.5);
          }
          &.dot-smo {
            background: #67c23a;
            box-shadow: 0 0 8px rgba(103, 194, 58, 0.5);
          }
          &.dot-ilt {
            background: #e6a23c;
            box-shadow: 0 0 8px rgba(230, 162, 60, 0.5);
          }
        }
      }
    }

    .console-content {
      animation: fadeIn 0.3s ease;
    }
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(8px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
}
</style>
