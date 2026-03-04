<template>
  <div class="flex flex-col h-screen">
    <header class="bg-white border-b px-6 py-3 shrink-0">
      <h2 class="font-semibold text-gray-800">💬 编辑对话</h2>
      <p class="text-xs text-gray-400">与 AI 编辑讨论书稿内容</p>
    </header>

    <div ref="messageList" class="flex-1 overflow-y-auto p-6 space-y-4">
      <div v-if="messages.length === 0" class="text-center text-gray-400 mt-20">
        <p class="text-4xl mb-4">✏️</p>
        <p class="text-lg font-medium">你好！我是你的 AI 编辑助手</p>
        <p class="text-sm mt-2">我可以帮你讨论书稿、提供写作建议、解决写作瓶颈</p>
        <div class="flex gap-3 justify-center mt-6">
          <button
            v-for="s in starters"
            :key="s"
            @click="sendMessage(s)"
            class="px-4 py-2 bg-white border border-gray-200 rounded-lg text-sm text-gray-600 hover:border-editor-300 hover:text-editor-600 transition-colors"
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
            ? 'bg-editor-500 text-white rounded-br-md'
            : 'bg-white border border-gray-100 shadow-sm rounded-bl-md'"
        >
          <div v-if="msg.role !== 'user'" class="markdown-body" v-html="renderMd(msg.content)"></div>
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

    <div class="bg-white border-t px-6 py-4 shrink-0">
      <form @submit.prevent="sendMessage()" class="flex gap-3">
        <input
          v-model="input"
          type="text"
          placeholder="跟编辑聊聊你的书稿..."
          class="flex-1 px-4 py-2.5 border border-gray-200 rounded-xl text-sm focus:outline-none focus:border-editor-400 focus:ring-1 focus:ring-editor-200"
          :disabled="loading"
        />
        <button
          type="submit"
          :disabled="!input.trim() || loading"
          class="px-5 py-2.5 bg-editor-500 text-white rounded-xl text-sm font-medium hover:bg-editor-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          发送
        </button>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { chatApi, streamChat } from '../utils/api.js'
import { renderMarkdown } from '../utils/markdown.js'

const messages = ref([])
const input = ref('')
const loading = ref(false)
const sessionId = ref('')
const messageList = ref(null)

const starters = [
  '帮我审查第一章的结构',
  '这段话怎么写更好？',
  '帮我想一个章节大纲',
]

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

  const editorMsg = { role: 'editor', content: '' }
  messages.value.push(editorMsg)

  streamChat(
    msg,
    sessionId.value,
    '',
    (token, sid) => {
      sessionId.value = sid
      editorMsg.content += token
      scrollToBottom()
    },
    (sid) => {
      sessionId.value = sid
      loading.value = false
      scrollToBottom()
    },
    (err) => {
      editorMsg.content += `\n\n⚠️ ${err}`
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
