<template>
  <el-container class="layout-container">
    <el-aside :width="isCollapse ? '64px' : '220px'" class="sidebar">
      <div class="logo" :class="{ 'logo-collapsed': isCollapse }">
        <el-icon size="24" color="#409eff"><Microscope /></el-icon>
        <span v-if="!isCollapse" class="logo-text">光刻仿真</span>
      </div>
      <el-menu
        :default-active="activeMenu"
        :collapse="isCollapse"
        :collapse-transition="false"
        background-color="#304156"
        text-color="#bfcbd9"
        active-text-color="#409eff"
        router
      >
        <el-menu-item index="/config">
          <el-icon><Setting /></el-icon>
          <template #title>参数配置</template>
        </el-menu-item>
        <el-menu-item index="/simulation">
          <el-icon><Cpu /></el-icon>
          <template #title>仿真运行</template>
        </el-menu-item>
        <el-menu-item index="/workflow">
          <el-icon><SetUp /></el-icon>
          <template #title>RET 工作流</template>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header class="header">
        <div class="header-left">
          <el-icon class="collapse-btn" size="20" @click="toggleCollapse">
            <Fold v-if="!isCollapse" />
            <Expand v-else />
          </el-icon>
          <el-breadcrumb separator="/">
            <el-breadcrumb-item v-for="item in breadcrumbs" :key="item.path">
              {{ item.title }}
            </el-breadcrumb-item>
          </el-breadcrumb>
        </div>
        <div class="header-right">
          <el-tag type="success" effect="dark" size="small">光刻仿真工作台 v1.0</el-tag>
        </div>
      </el-header>

      <el-main class="main-content">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const isCollapse = ref(false)

const activeMenu = computed(() => route.path)

const breadcrumbs = computed(() => {
  const matched = route.matched.filter((m) => m.meta && m.meta.title)
  return matched.map((m) => ({
    path: m.path,
    title: m.meta.title as string
  }))
})

function toggleCollapse() {
  isCollapse.value = !isCollapse.value
}
</script>

<style lang="scss" scoped>
.layout-container {
  height: 100%;
}

.sidebar {
  background-color: #304156;
  transition: width 0.3s;
  overflow: hidden;

  :deep(.el-menu) {
    border-right: none;
  }
}

.logo {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: #fff;
  background-color: #263445;
  border-bottom: 1px solid #1f2d3d;

  &.logo-collapsed {
    padding: 0;
  }

  .logo-text {
    font-size: 16px;
    font-weight: 600;
    white-space: nowrap;
  }
}

.header {
  height: 60px;
  background: #fff;
  box-shadow: 0 1px 4px rgba(0, 21, 41, 0.08);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;

  .header-left {
    display: flex;
    align-items: center;
    gap: 20px;
  }

  .collapse-btn {
    cursor: pointer;
    color: #606266;
    transition: color 0.3s;

    &:hover {
      color: #409eff;
    }
  }
}

.main-content {
  background: #f5f7fa;
  padding: 20px;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.25s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
