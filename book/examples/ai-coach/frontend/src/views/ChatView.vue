<template>
  <div class="flex flex-col h-screen">
    <!-- 顶栏 -->
    <header class="bg-white border-b px-6 py-3 flex items-center justify-between shrink-0">
      <div>
        <h2 class="font-semibold text-gray-800">💬 AI 教练对话</h2>
        <p class="text-xs text-gray-400">选择模式与你的 AI 教练交流</p>
      </div>
      <div class="flex gap-2">
        <button
          v-for="m in modes"
          :key="m.value"
          @click="currentMode = m.value"
          class="px-3 py-1.5 rounded-full text-xs font-medium transition-colors"
          :class="currentMode === m.value
            ? 'bg-coach-500 text-white'
            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'"
        >
          {{ m.icon }} {{ m.label }}
        </button>
      </div>
    </header>

    <!-- 消息列表 -->
    <div ref="messageList" class="flex-1 overflow-y-auto p-6 space-y-4">
      <div v-if="messages.length === 0" class="text-center text-gray-400 mt-20">
        <p class="text-4xl mb-4">🎓</p>
        <p class="text-lg font-medium">你好！我是你的 AI 学习教练</p>
        <p class="text-sm mt-2">选择一个模式开始对话吧</p>
        <div class="flex gap-3 justify-center mt-6">
          <button
            v-for="s in starters"
            :key="s"
            @click="sendMessage(s)"
            class="px-4 py-2 bg-white border border-gray-200 rounded-lg text-sm text-gray-600 hover:border-coach-300 hover:text-coach-600 transition-colors"
          >
            {{ s }}
          </button>
        </div>
      </div>

      <div
        v-for="(msg, i) in messages"
        :key="i"
        class="flex"
        :class="msg.role === 'user' ? 'justify-end' : 'justify-start'"
      >
        <div
          class="max-w-2xl px-4 py-3 rounded-2xl text-sm leading-relaxed"
          :class="msg.role === 'user'
            ? 'bg-coach-500 text-white rounded-br-md'
            : 'bg-white border border-gray-100 shadow-sm rounded-bl-md'"
        >
          <div
            v-if="msg.role !== 'user'"
            class="markdown-body"
            v-html="renderMd(msg.content)"
          ></div>
          <span v-else>{{ msg.content }}</span>
        </div>
      </div>

      <div v-if="loading" class="flex justify-start">
        <div class="bg-white border border-gray-100 shadow-sm px-4 py-3 rounded-2xl rounded-bl-md">
          <span class="inline-flex gap-1">
            <span class="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style="animation-delay: 0ms"></span>
            <span class="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style="animation-delay: 150ms"></span>
            <span class="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style="animation-delay: 300ms"></span>
          </span>
        </div>
      </div>
    </div>

    <!-- 输入框 -->
    <div class="bg-white border-t px-6 py-4 shrink-0">
      <form @submit.prevent="sendMessage()" class="flex gap-3">
        <input
          v-model="input"
          type="text"
          :placeholder="inputPlaceholder"
          class="flex-1 px-4 py-2.5 border border-gray-200 rounded-xl text-sm focus:outline-none focus:border-coach-400 focus:ring-1 focus:ring-coach-200"
          :disabled="loading"
        />
        <button
          type="submit"
          :disabled="!input.trim() || loading"
          class="px-5 py-2.5 bg-coach-500 text-white rounded-xl text-sm font-medium hover:bg-coach-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          发送
        </button>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, computed } from 'vue'
import { chatApi, streamChat } from '../utils/api.js'
import { renderMarkdown } from '../utils/markdown.js'

const messages = ref([])
const input = ref('')
const loading = ref(false)
const currentMode = ref('coach')
const sessionId = ref('')
const messageList = ref(null)

const modes = [
  { value: 'coach', label: '教练', icon: '🎯' },
  { value: 'tutor', label: '导师', icon: '📖' },
  { value: 'quiz', label: '测验', icon: '✍️' },
]

const starters = [
  '帮我制定一个学习计划',
  '解释一下什么是 RAG',
  '今天学什么好？',
]

const inputPlaceholder = computed(() => {
  const map = {
    coach: '跟教练聊聊你的学习情况...',
    tutor: '问导师一个技术问题...',
    quiz: '准备好了，开始测验吧...',
  }
  return map[currentMode.value] || '输入消息...'
})

function renderMd(text) {
  return renderMarkdown(text)
}

async function sendMessage(text) {
  const msg = text || input.value.trim()
  if (!msg) return

  messages.value.push({ role: 'user', content: msg })
  input.value = ''
  loading.value = true

  await nextTick()
  scrollToBottom()

  // 添加一个空的教练消息，用于流式填充
  const coachMsg = { role: 'coach', content: '' }
  messages.value.push(coachMsg)

  streamChat(
    msg,
    sessionId.value,
    currentMode.value,
    // onToken
    (token, sid) => {
      sessionId.value = sid
      coachMsg.content += token
      scrollToBottom()
    },
    // onDone
    (sid, sources) => {
      sessionId.value = sid
      loading.value = false
      scrollToBottom()
    },
    // onError
    (err) => {
      coachMsg.content += `\n\n⚠️ ${err}`
      loading.value = false
    }
  )
}

function scrollToBottom() {
  if (messageList.value) {
    messageList.value.scrollTop = messageList.value.scrollHeight
  }
}
</script>
