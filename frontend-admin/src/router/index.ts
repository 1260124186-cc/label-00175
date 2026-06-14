import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import Layout from '@/layout/index.vue'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: Layout,
    redirect: '/config',
    children: [
      {
        path: 'config',
        name: 'Config',
        component: () => import('@/views/ConfigPage.vue'),
        meta: { title: '参数配置', icon: 'Setting' }
      },
      {
        path: 'simulation',
        name: 'Simulation',
        component: () => import('@/views/SimulationPage.vue'),
        meta: { title: '仿真运行', icon: 'Cpu' }
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
