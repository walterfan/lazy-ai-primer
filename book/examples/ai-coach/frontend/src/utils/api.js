/**
 * API 客户端 — 封装所有后端接口调用
 */
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000, // AI 响应可能较慢
  headers: { 'Content-Type': 'application/json' },
})

// Token 拦截器：自动附加 Authorization header
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 401 响应拦截：自动跳转登录
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('refreshToken')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// ── 知识库 API ──────────────────────────────

export const knowledgeApi = {
  /** 上传文档 */
  uploadDocument(data) {
    return api.post('/knowledge/documents', data)
  },

  /** 上传文件 */
  uploadFile(file, tags = '') {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('tags', tags)
    return api.post('/knowledge/documents/file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },

  /** 列出文档 */
  listDocuments() {
    return api.get('/knowledge/documents')
  },

  /** 删除文档 */
  deleteDocument(docId) {
    return api.delete(`/knowledge/documents/${docId}`)
  },

  /** 查询知识库 */
  query(question, topK = 5) {
    return api.post('/knowledge/query', { question, top_k: topK })
  },

  /** 知识库统计 */
  getStats() {
    return api.get('/knowledge/stats')
  },
}

// ── 对话 API ──────────────────────────────

export const chatApi = {
  /** 发送消息 */
  sendMessage(message, sessionId = '', mode = 'coach') {
    return api.post('/chat/message', {
      message,
      session_id: sessionId,
      mode,
    })
  },

  /** 列出会话 */
  listSessions(limit = 20) {
    return api.get('/chat/sessions', { params: { limit } })
  },

  /** 获取会话详情 */
  getSession(sessionId) {
    return api.get(`/chat/sessions/${sessionId}`)
  },

  /** 删除会话 */
  deleteSession(sessionId) {
    return api.delete(`/chat/sessions/${sessionId}`)
  },
}

// ── 学习计划 API ──────────────────────────────

export const learningApi = {
  /** 创建学习目标 */
  createGoal(data) {
    return api.post('/learning/goals', data)
  },

  /** 列出学习目标 */
  listGoals(status = 'active') {
    return api.get('/learning/goals', { params: { status } })
  },

  /** 记录学习会话 */
  logSession(data) {
    return api.post('/learning/sessions', data)
  },

  /** 获取学习进度 */
  getProgress(goalId) {
    return api.get(`/learning/progress/${goalId}`)
  },

  /** 列出学习记录 */
  listSessions(goalId, limit = 20) {
    return api.get(`/learning/sessions/${goalId}`, { params: { limit } })
  },
}

// ── SSE 流式对话 ──────────────────────────────

export function streamChat(message, sessionId = '', mode = 'coach', onToken, onDone, onError) {
  const token = localStorage.getItem('token')
  
  fetch('/api/chat/message/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ message, session_id: sessionId, mode }),
  })
    .then(async (response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let currentSessionId = sessionId

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.type === 'start') {
                currentSessionId = data.session_id
              } else if (data.type === 'token' && data.token) {
                onToken?.(data.token, currentSessionId)
              } else if (data.type === 'done') {
                onDone?.(currentSessionId, data.sources || [])
              } else if (data.error) {
                onError?.(data.error)
              }
            } catch (e) {
              // ignore parse errors
            }
          }
        }
      }
    })
    .catch((err) => {
      onError?.(err.message)
    })
}

export default api
