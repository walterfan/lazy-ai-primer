<template>
  <div class="flex flex-col h-screen">
    <!-- 顶栏 -->
    <header class="bg-white border-b px-6 py-3 flex items-center justify-between shrink-0">
      <div>
        <h2 class="font-semibold text-gray-800">📚 知识库管理</h2>
        <p class="text-xs text-gray-400">上传文档构建你的个人知识库</p>
      </div>
      <div class="flex gap-2">
        <span class="text-xs text-gray-400 bg-gray-50 px-3 py-1.5 rounded-full">
          {{ stats.total_chunks || 0 }} 个知识块
        </span>
      </div>
    </header>

    <div class="flex-1 overflow-y-auto p-6">
      <div class="max-w-4xl mx-auto space-y-6">

        <!-- 上传区域 -->
        <div class="bg-white rounded-xl border border-gray-200 p-6">
          <h3 class="font-medium text-gray-700 mb-4">📤 添加知识</h3>

          <!-- 文本输入 -->
          <div class="space-y-3">
            <input
              v-model="newDoc.title"
              type="text"
              placeholder="文档标题"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-coach-400"
            />
            <textarea
              v-model="newDoc.content"
              placeholder="粘贴文档内容（支持 Markdown）..."
              rows="5"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-coach-400 resize-none"
            ></textarea>
            <div class="flex gap-3 items-center">
              <input
                v-model="newDoc.tags"
                type="text"
                placeholder="标签（逗号分隔）"
                class="flex-1 px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-coach-400"
              />
              <button
                @click="uploadText"
                :disabled="!newDoc.title || !newDoc.content"
                class="px-4 py-2 bg-coach-500 text-white rounded-lg text-sm font-medium hover:bg-coach-600 disabled:opacity-50 transition-colors"
              >
                添加文档
              </button>
            </div>
          </div>

          <!-- 文件上传 -->
          <div class="mt-4 pt-4 border-t border-gray-100">
            <label class="flex items-center justify-center gap-2 px-4 py-3 border-2 border-dashed border-gray-200 rounded-lg cursor-pointer hover:border-coach-300 transition-colors">
              <span class="text-gray-400 text-sm">📎 点击上传文件（TXT、MD、PDF）</span>
              <input type="file" class="hidden" accept=".txt,.md,.pdf" @change="uploadFile" />
            </label>
          </div>
        </div>

        <!-- 知识查询 -->
        <div class="bg-white rounded-xl border border-gray-200 p-6">
          <h3 class="font-medium text-gray-700 mb-4">🔍 知识查询</h3>
          <form @submit.prevent="queryKnowledge" class="flex gap-3">
            <input
              v-model="queryText"
              type="text"
              placeholder="输入问题，从知识库中检索答案..."
              class="flex-1 px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-coach-400"
            />
            <button
              type="submit"
              :disabled="!queryText.trim() || querying"
              class="px-4 py-2 bg-coach-500 text-white rounded-lg text-sm font-medium hover:bg-coach-600 disabled:opacity-50 transition-colors"
            >
              {{ querying ? '检索中...' : '检索' }}
            </button>
          </form>

          <div v-if="queryResult" class="mt-4 p-4 bg-gray-50 rounded-lg">
            <div class="markdown-body text-sm" v-html="renderMd(queryResult.answer)"></div>
            <div v-if="queryResult.sources?.length" class="mt-3 pt-3 border-t border-gray-200">
              <p class="text-xs text-gray-400 mb-2">📎 来源：</p>
              <div v-for="(s, i) in queryResult.sources" :key="i" class="text-xs text-gray-500 mb-1">
                [{{ s.title }}] 相关度: {{ (s.score * 100).toFixed(1) }}%
              </div>
            </div>
          </div>
        </div>

        <!-- 文档列表 -->
        <div class="bg-white rounded-xl border border-gray-200 p-6">
          <h3 class="font-medium text-gray-700 mb-4">📄 已有文档 ({{ documents.length }})</h3>
          <div v-if="documents.length === 0" class="text-center text-gray-400 py-8 text-sm">
            还没有文档，上传一些知识吧 📖
          </div>
          <div v-else class="space-y-2">
            <div
              v-for="doc in documents"
              :key="doc.id"
              class="flex items-center justify-between px-4 py-3 bg-gray-50 rounded-lg"
            >
              <div>
                <p class="text-sm font-medium text-gray-700">{{ doc.title }}</p>
                <p class="text-xs text-gray-400 mt-0.5">
                  {{ doc.chunk_count }} 个块 · {{ doc.source }}
                  <span v-if="doc.tags?.length"> · {{ doc.tags.join(', ') }}</span>
                </p>
              </div>
              <button
                @click="deleteDoc(doc.id)"
                class="text-xs text-red-400 hover:text-red-600 transition-colors"
              >
                删除
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { knowledgeApi } from '../utils/api.js'
import { renderMarkdown } from '../utils/markdown.js'

const documents = ref([])
const stats = ref({})
const queryText = ref('')
const queryResult = ref(null)
const querying = ref(false)

const newDoc = ref({ title: '', content: '', tags: '' })

function renderMd(text) {
  return renderMarkdown(text)
}

async function loadDocuments() {
  try {
    const { data } = await knowledgeApi.listDocuments()
    documents.value = data
  } catch (err) {
    console.error('加载文档失败:', err)
  }
}

async function loadStats() {
  try {
    const { data } = await knowledgeApi.getStats()
    stats.value = data
  } catch (err) {
    console.error('加载统计失败:', err)
  }
}

async function uploadText() {
  try {
    const tags = newDoc.value.tags.split(',').map(t => t.trim()).filter(Boolean)
    await knowledgeApi.uploadDocument({
      title: newDoc.value.title,
      content: newDoc.value.content,
      tags,
    })
    newDoc.value = { title: '', content: '', tags: '' }
    await loadDocuments()
    await loadStats()
  } catch (err) {
    alert('上传失败: ' + (err.response?.data?.detail || err.message))
  }
}

async function uploadFile(event) {
  const file = event.target.files[0]
  if (!file) return
  try {
    await knowledgeApi.uploadFile(file)
    await loadDocuments()
    await loadStats()
  } catch (err) {
    alert('上传失败: ' + (err.response?.data?.detail || err.message))
  }
}

async function deleteDoc(docId) {
  if (!confirm('确定删除这个文档？')) return
  try {
    await knowledgeApi.deleteDocument(docId)
    await loadDocuments()
    await loadStats()
  } catch (err) {
    alert('删除失败: ' + (err.response?.data?.detail || err.message))
  }
}

async function queryKnowledge() {
  if (!queryText.value.trim()) return
  querying.value = true
  queryResult.value = null
  try {
    const { data } = await knowledgeApi.query(queryText.value)
    queryResult.value = data
  } catch (err) {
    alert('查询失败: ' + (err.response?.data?.detail || err.message))
  } finally {
    querying.value = false
  }
}

onMounted(() => {
  loadDocuments()
  loadStats()
})
</script>
