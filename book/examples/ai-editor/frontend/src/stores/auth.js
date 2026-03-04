/**
 * 认证状态管理 — Pinia Store
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 10000 })

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem('token') || '')
  const refreshToken = ref(localStorage.getItem('refreshToken') || '')
  const user = ref(JSON.parse(localStorage.getItem('user') || 'null'))

  const isLoggedIn = computed(() => !!token.value)
  const username = computed(() => user.value?.username || '')
  const role = computed(() => user.value?.role || 'user')

  function setAuth(data) {
    token.value = data.access_token
    refreshToken.value = data.refresh_token
    user.value = data.user
    localStorage.setItem('token', data.access_token)
    localStorage.setItem('refreshToken', data.refresh_token)
    localStorage.setItem('user', JSON.stringify(data.user))
  }

  function clearAuth() {
    token.value = ''
    refreshToken.value = ''
    user.value = null
    localStorage.removeItem('token')
    localStorage.removeItem('refreshToken')
    localStorage.removeItem('user')
  }

  async function register(username, password, email = '') {
    const { data } = await api.post('/auth/register', { username, password, email })
    setAuth(data)
    return data
  }

  async function login(username, password) {
    const { data } = await api.post('/auth/login', { username, password })
    setAuth(data)
    return data
  }

  function logout() {
    clearAuth()
  }

  return {
    token, refreshToken, user,
    isLoggedIn, username, role,
    register, login, logout, setAuth, clearAuth,
  }
})
