<template>
  <div class="flex flex-col h-screen">
    <header class="bg-white border-b px-6 py-3 shrink-0">
      <h2 class="font-semibold text-gray-800">✍️ AI 写作助手</h2>
      <p class="text-xs text-gray-400">让 AI 帮你生成书稿内容</p>
    </header>

    <div class="flex-1 overflow-y-auto p-6">
      <div class="max-w-4xl mx-auto space-y-6">

        <!-- 写作配置 -->
        <div class="bg-white rounded-xl border border-gray-200 p-6">
          <h3 class="font-medium text-gray-700 mb-4">📋 写作配置</h3>
          <div class="space-y-3">
            <input
              v-model="form.topic"
              type="text"
              placeholder="写作主题（如：RAG 在企业中的最佳实践）"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-editor-400"
            />
            <textarea
              v-model="form.outline"
              placeholder="大纲（可选，每行一个要点）..."
              rows="4"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-editor-400 resize-none"
            ></textarea>
            <textarea
              v-model="form.context"
              placeholder="上下文（可选，前后章节的摘要，帮助 AI 保持连贯性）..."
              rows="3"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-editor-400 resize-none"
            ></textarea>
            <div class="flex gap-3 items-center">
              <select
                v-model="form.style"
                class="px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-editor-400"
              >
                <option value="technical">📘 技术书籍风格</option>
                <option value="casual">📝 轻松博客风格</option>
                <option value="academic">🎓 学术论文风格</option>
              </select>
              <input
                v-model.number="form.word_count"
                type="number"
                placeholder="目标字数"
                min="500"
                max="10000"
                class="w-32 px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-editor-400"
              />
              <span class="text-xs text-gray-400">字</span>
              <button
                @click="generate"
                :disabled="!form.topic || generating"
                class="ml-auto px-5 py-2 bg-editor-500 text-white rounded-lg text-sm font-medium hover:bg-editor-600 disabled:opacity-50 transition-colors"
              >
                {{ generating ? '⏳ 生成中...' : '✨ 开始生成' }}
              </button>
            </div>
          </div>
        </div>

        <!-- 生成结果 -->
        <div v-if="result" class="bg-white rounded-xl border border-gray-200 p-6">
          <div class="flex items-center justify-between mb-4">
            <h3 class="font-medium text-gray-700">📄 生成结果（{{ result.word_count }} 字）</h3>
            <div class="flex gap-2">
              <button
                @click="copyResult"
                class="px-3 py-1.5 bg-gray-100 text-gray-600 rounded-lg text-xs hover:bg-gray-200 transition-colors"
              >
                📋 复制
              </button>
              <button
                @click="saveAsChapter"
                class="px-3 py-1.5 bg-editor-500 text-white rounded-lg text-xs font-medium hover:bg-editor-600 transition-colors"
              >
                💾 保存为章节
              </button>
            </div>
          </div>
          <div class="markdown-body text-sm border-t pt-4" v-html="renderMd(result.content)"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { writerApi, chaptersApi } from '../utils/api.js'
import { renderMarkdown } from '../utils/markdown.js'

const router = useRouter()
const generating = ref(false)
const result = ref(null)

const form = ref({
  topic: '',
  outline: '',
  style: 'technical',
  word_count: 2000,
  context: '',
})

function renderMd(text) {
  return renderMarkdown(text)
}

async function generate() {
  generating.value = true
  result.value = null
  try {
    const { data } = await writerApi.generate(form.value)
    result.value = data
  } catch (err) {
    alert('生成失败: ' + (err.response?.data?.detail || err.message))
  } finally {
    generating.value = false
  }
}

function copyResult() {
  if (result.value) {
    navigator.clipboard.writeText(result.value.content)
    alert('已复制到剪贴板 ✅')
  }
}

async function saveAsChapter() {
  if (!result.value) return
  const title = prompt('请输入章节标题：', form.value.topic)
  if (!title) return
  const number = parseInt(prompt('请输入章节编号：', '1'))
  if (isNaN(number)) return

  try {
    const { data } = await chaptersApi.create({
      number,
      title,
      content: result.value.content,
    })
    alert('已保存为章节 ✅')
    router.push(`/edit/${data.id}`)
  } catch (err) {
    alert('保存失败: ' + (err.response?.data?.detail || err.message))
  }
}
</script>
