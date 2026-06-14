# 调试会话: web-workbench-stuck

**状态**: [CLOSED]
**创建时间**: 2026-06-14
**问题描述**: web工作台docker启动后页面无响应，一直loading，仿真工作台卡住无法正常渲染参数设置页面

## 假设列表 (可证伪)

| ID | 假设 | 验证方法 | 状态 |
|----|------|----------|------|
| H1 | 前端构建/编译错误 - 某个Vue组件有语法错误导致预编译失败 | 查看Docker构建日志，检查是否有TypeScript/Vue编译错误 | ❌ 排除 |
| H2 | JavaScript运行时错误 - API响应格式不匹配或代码逻辑错误导致页面崩溃 | 浏览器控制台查看完整错误堆栈 | ⚠️ 部分相关 |
| H3 | Loading状态死锁 - API请求失败/超时，loading状态未正确关闭 | 检查store中loading状态变化，查看网络请求状态 | ✅ 确认 |
| H4 | 组件渲染循环 - 响应式数据无限更新导致页面卡住 | 检查组件watch/computed逻辑，查看是否有循环依赖 | ❌ 排除 |
| H5 | Pinia store 状态损坏 - 数据结构异常导致响应式系统崩溃 | 检查store初始化和API响应数据结构 | ❌ 排除 |

## 根本原因分析

### 根因1: Service Worker 缓存旧应用 (主要)
- **现象**: 浏览器请求中出现 `@vite/client`（Vite开发服务器特有），但构建产物中不存在此文件
- **原因**: 之前部署的"KeyFlow Pro"应用注册了Service Worker，拦截了 localhost:8080 的所有请求
- **证据**: 浏览器网络请求显示 `@vite/client` 404，控制台无应用代码报错

### 根因2: Loading状态管理设计缺陷 (次要)
- **现象**: `loadDefault()` 和 `fetchSavedList()` 并发调用时，内部的 `loading=true/false` 可能与外部调用冲突
- **原因**: 缺少统一的 loading 状态管理入口，并发异步操作可能导致状态不一致

### 根因3: 缺少超时保护机制 (加剧)
- **现象**: loading 状态可能永远卡在 true，即使 API 请求已完成
- **原因**: 异步 Promise 链缺少超时保护，极端情况下可能导致页面永久不可用

## 修复方案

### 1. Nginx 层 - 彻底清除 Service Worker 缓存
- 添加 `Clear-Site-Data: "storage", "cookies", "executionContexts"` 响应头
- 添加 `Service-Worker-Allowed: none` 阻止新 SW 注册
- 对 HTML 页面禁用所有缓存

### 2. 前端代码 - 多层 SW 清除机制
- `index.html` 中添加内联脚本，在任何其他脚本执行前清除 SW
- `main.ts` 中再次执行 SW 注销，作为双重保险

### 3. Pinia Store - 重构 Loading 状态管理
- 抽离 `_fetchDefaultConfig()` 内部方法，避免嵌套调用导致的 loading 冲突
- 新增 `loadInitialData()` 统一管理初始化 loading 状态
- 采用顺序 `await` 替代 `Promise.all`，简化异步流程

### 4. 双重超时保护
- 8秒独立安全定时器：无论代码执行情况如何，超时后强制关闭 loading
- `finally` 块确保无论成功失败都关闭 loading

## 验证结果

- ✅ 页面正常加载，52个交互元素可访问
- ✅ 参数配置页面完整渲染，所有参数值正确显示
- ✅ API 请求正常返回 200 OK
- ✅ Loading 遮罩正确消失
- ✅ 无 Service Worker 缓存问题

## 结论

根因: Service Worker 缓存旧应用 + Loading状态管理缺陷
修复: 已完成，所有功能恢复正常
