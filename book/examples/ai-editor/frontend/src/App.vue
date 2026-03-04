<template>
  <div class="min-h-screen flex">
    <!-- 侧边栏 -->
    <aside class="w-56 bg-white border-r border-gray-200 flex flex-col">
      <div class="p-4 border-b border-gray-100">
        <h1 class="text-lg font-bold text-editor-700">✏️ AI Editor</h1>
        <p class="text-xs text-gray-400 mt-1">你的虚拟书稿编辑</p>
      </div>

      <nav class="flex-1 p-3 space-y-1">
        <router-link
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors"
          :class="isActive(item.path)
            ? 'bg-editor-50 text-editor-700 font-medium'
            : 'text-gray-600 hover:bg-gray-50'"
        >
          <span class="text-lg">{{ item.icon }}</span>
          <span>{{ item.label }}</span>
        </router-link>
      </nav>

      <div class="p-3 border-t border-gray-100">
        <div v-if="auth.isLoggedIn" class="flex items-center justify-between">
          <div class="flex items-center gap-2 text-xs text-gray-500">
            <span class="w-2 h-2 bg-green-400 rounded-full"></span>
            <span>{{ auth.username }}</span>
          </div>
          <button @click="auth.logout(); router.push('/login')" class="text-xs text-gray-400 hover:text-red-500">退出</button>
        </div>
        <div v-else class="text-xs text-gray-400">
          <router-link to="/login" class="hover:text-editor-600">登录</router-link>
        </div>
      </div>
    </aside>

    <main class="flex-1 flex flex-col overflow-hidden">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from './stores/auth.js'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

const navItems = [
  { path: '/', icon: '📖', label: '章节管理' },
  { path: '/write', icon: '✍️', label: 'AI 写作' },
  { path: '/chat', icon: '💬', label: '编辑对话' },
]

function isActive(path) {
  if (path === '/') return route.path === '/' || route.path.startsWith('/edit/')
  return route.path === path
}
</script>
