<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-50">
    <div class="w-full max-w-sm">
      <div class="text-center mb-8">
        <p class="text-4xl mb-2">✏️</p>
        <h1 class="text-2xl font-bold text-gray-800">AI Editor</h1>
        <p class="text-sm text-gray-400 mt-1">你的虚拟书稿编辑</p>
      </div>

      <div class="bg-white rounded-xl border border-gray-200 p-6">
        <div class="flex gap-2 mb-6">
          <button
            @click="isLogin = true"
            class="flex-1 py-2 text-sm font-medium rounded-lg transition-colors"
            :class="isLogin ? 'bg-editor-500 text-white' : 'text-gray-500 hover:bg-gray-50'"
          >
            登录
          </button>
          <button
            @click="isLogin = false"
            class="flex-1 py-2 text-sm font-medium rounded-lg transition-colors"
            :class="!isLogin ? 'bg-editor-500 text-white' : 'text-gray-500 hover:bg-gray-50'"
          >
            注册
          </button>
        </div>

        <form @submit.prevent="handleSubmit" class="space-y-4">
          <input
            v-model="form.username"
            type="text"
            placeholder="用户名"
            class="w-full px-4 py-2.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-editor-400"
            required
          />
          <input
            v-if="!isLogin"
            v-model="form.email"
            type="email"
            placeholder="邮箱（可选）"
            class="w-full px-4 py-2.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-editor-400"
          />
          <input
            v-model="form.password"
            type="password"
            placeholder="密码"
            class="w-full px-4 py-2.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-editor-400"
            required
          />
          <button
            type="submit"
            :disabled="loading"
            class="w-full py-2.5 bg-editor-500 text-white rounded-lg text-sm font-medium hover:bg-editor-600 disabled:opacity-50 transition-colors"
          >
            {{ loading ? '请稍候...' : (isLogin ? '登录' : '注册') }}
          </button>
        </form>

        <p v-if="error" class="mt-3 text-xs text-red-500 text-center">{{ error }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth.js'

const router = useRouter()
const auth = useAuthStore()

const isLogin = ref(true)
const loading = ref(false)
const error = ref('')
const form = ref({ username: '', password: '', email: '' })

async function handleSubmit() {
  error.value = ''
  loading.value = true
  try {
    if (isLogin.value) {
      await auth.login(form.value.username, form.value.password)
    } else {
      await auth.register(form.value.username, form.value.password, form.value.email)
    }
    router.push('/')
  } catch (err) {
    error.value = err.response?.data?.detail || err.message
  } finally {
    loading.value = false
  }
}
</script>
