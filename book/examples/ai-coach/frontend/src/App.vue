<template>
  <div class="min-h-screen flex">
    <!-- 侧边栏 -->
    <aside class="w-56 bg-white border-r border-gray-200 flex flex-col">
      <!-- Logo -->
      <div class="p-4 border-b border-gray-100">
        <h1 class="text-lg font-bold text-coach-700">🎓 AI Coach</h1>
        <p class="text-xs text-gray-400 mt-1">你的虚拟学习教练</p>
      </div>

      <!-- 导航 -->
      <nav class="flex-1 p-3 space-y-1">
        <router-link
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors"
          :class="$route.path === item.path
            ? 'bg-coach-50 text-coach-700 font-medium'
            : 'text-gray-600 hover:bg-gray-50'"
        >
          <span class="text-lg">{{ item.icon }}</span>
          <span>{{ item.label }}</span>
        </router-link>
      </nav>

      <!-- 底部用户信息 -->
      <div class="p-3 border-t border-gray-100">
        <div v-if="auth.isLoggedIn" class="flex items-center justify-between">
          <div class="flex items-center gap-2 text-xs text-gray-500">
            <span class="w-2 h-2 bg-green-400 rounded-full"></span>
            <span>{{ auth.username }}</span>
          </div>
          <button @click="auth.logout(); $router.push('/login')" class="text-xs text-gray-400 hover:text-red-500">退出</button>
        </div>
        <div v-else class="text-xs text-gray-400">
          <router-link to="/login" class="hover:text-coach-600">登录</router-link>
        </div>
      </div>
    </aside>

    <!-- 主内容区 -->
    <main class="flex-1 flex flex-col overflow-hidden">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { useAuthStore } from './stores/auth.js'
const auth = useAuthStore()

const navItems = [
  { path: '/', icon: '💬', label: '对话教练' },
  { path: '/knowledge', icon: '📚', label: '知识库' },
  { path: '/learning', icon: '📊', label: '学习计划' },
]
</script>
