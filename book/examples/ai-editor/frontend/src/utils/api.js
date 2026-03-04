/**
 * API 客户端 — AI Editor 后端接口
 */
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
})

// Token 拦截器
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

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

// ── 章节 API ──────────────────────────────

export const chaptersApi = {
  create(data) {
    return api.post('/chapters/', data)
  },
  list(bookId = 'default') {
    return api.get('/chapters/', { params: { book_id: bookId } })
  },
  get(id) {
    return api.get(`/chapters/${id}`)
  },
  update(id, data) {
    return api.put(`/chapters/${id}`, data)
  },
  delete(id) {
    return api.delete(`/chapters/${id}`)
  },
}

// ── 编辑 API ──────────────────────────────

export const editorApi = {
  edit(data) {
    return api.post('/editor/edit', data)
  },
  apply(chapterId, content) {
    return api.post(`/editor/apply/${chapterId}`, { content })
  },
  history(chapterId, limit = 20) {
    return api.get(`/editor/history/${chapterId}`, { params: { limit } })
  },
}

// ── 写作 API ──────────────────────────────

export const writerApi = {
  generate(data) {
    return api.post('/writer/generate', data)
  },
}

// ── 对话 API ──────────────────────────────

export const chatApi = {
  sendMessage(message, sessionId = '', chapterId = '') {
    return api.post('/chat/message', {
      message,
      session_id: sessionId,
      chapter_id: chapterId,
    })
  },
  listSessions(limit = 20) {
    return api.get('/chat/sessions', { params: { limit } })
  },
}

// ── SSE 流式对话 ──────────────────────────────

export function streamChat(message, sessionId = '', chapterId = '', onToken, onDone, onError) {
  const token = localStorage.getItem('token')

  fetch('/api/chat/message/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ message, session_id: sessionId, chapter_id: chapterId }),
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
                onDone?.(currentSessionId)
              } else if (data.error) {
                onError?.(data.error)
              }
            } catch (e) { /* ignore */ }
          }
        }
      }
    })
    .catch((err) => onError?.(err.message))
}

export default api
