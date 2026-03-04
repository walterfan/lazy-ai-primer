<template>
  <div class="flex flex-col h-screen">
    <header class="bg-white border-b px-6 py-3 flex items-center justify-between shrink-0">
      <div>
        <h2 class="font-semibold text-gray-800">📖 章节管理</h2>
        <p class="text-xs text-gray-400">管理书稿的所有章节</p>
      </div>
      <button
        @click="showCreate = !showCreate"
        class="px-4 py-2 bg-editor-500 text-white rounded-lg text-sm font-medium hover:bg-editor-600 transition-colors"
      >
        + 新建章节
      </button>
    </header>

    <div class="flex-1 overflow-y-auto p-6">
      <div class="max-w-4xl mx-auto space-y-4">

        <!-- 新建章节表单 -->
        <div v-if="showCreate" class="bg-white rounded-xl border border-gray-200 p-6">
          <h3 class="font-medium text-gray-700 mb-4">📝 新建章节</h3>
          <div class="space-y-3">
            <div class="flex gap-3">
              <input
                v-model.number="newChapter.number"
                type="number"
                placeholder="章节号"
                min="1"
                class="w-24 px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-editor-400"
              />
              <input
                v-model="newChapter.title"
                type="text"
                placeholder="章节标题"
                class="flex-1 px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-editor-400"
              />
            </div>
            <textarea
              v-model="newChapter.content"
              placeholder="章节内容（Markdown 格式）..."
              rows="6"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg text-sm font-mono focus:outline-none focus:border-editor-400 resize-none"
            ></textarea>
            <div class="flex gap-2 justify-end">
              <button @click="showCreate = false" class="px-4 py-2 text-gray-500 text-sm hover:text-gray-700">取消</button>
              <button
                @click="createChapter"
                :disabled="!newChapter.title"
                class="px-4 py-2 bg-editor-500 text-white rounded-lg text-sm font-medium hover:bg-editor-600 disabled:opacity-50 transition-colors"
              >
                创建
              </button>
            </div>
          </div>
        </div>

        <!-- 章节列表 -->
        <div v-if="chapters.length === 0 && !showCreate" class="text-center text-gray-400 py-20">
          <p class="text-4xl mb-4">📖</p>
          <p class="text-lg font-medium">还没有章节</p>
          <p class="text-sm mt-2">点击"新建章节"开始写作</p>
        </div>

        <div
          v-for="ch in chapters"
          :key="ch.id"
          class="bg-white rounded-xl border border-gray-200 p-5 hover:shadow-sm transition-shadow cursor-pointer"
          @click="$router.push(`/edit/${ch.id}`)"
        >
          <div class="flex items-center justify-between">
            <div>
              <h3 class="font-medium text-gray-800">
                第 {{ ch.number }} 章：{{ ch.title }}
              </h3>
              <p class="text-xs text-gray-400 mt-1">
                {{ ch.word_count }} 字 · 版本 {{ ch.version }}
                <span v-if="ch.updated_at"> · {{ formatDate(ch.updated_at) }}</span>
              </p>
            </div>
            <div class="flex gap-2">
              <button
                @click.stop="$router.push(`/edit/${ch.id}`)"
                class="px-3 py-1.5 bg-editor-50 text-editor-600 rounded-lg text-xs font-medium hover:bg-editor-100 transition-colors"
              >
                编辑
              </button>
              <button
                @click.stop="deleteChapter(ch.id)"
                class="px-3 py-1.5 text-red-400 hover:text-red-600 text-xs transition-colors"
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
import { chaptersApi } from '../utils/api.js'

const chapters = ref([])
const showCreate = ref(false)
const newChapter = ref({ number: 1, title: '', content: '' })

function formatDate(iso) {
  return new Date(iso).toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

async function loadChapters() {
  try {
    const { data } = await chaptersApi.list()
    chapters.value = data
    if (data.length > 0) {
      newChapter.value.number = Math.max(...data.map(c => c.number)) + 1
    }
  } catch (err) {
    console.error('加载章节失败:', err)
  }
}

async function createChapter() {
  try {
    await chaptersApi.create(newChapter.value)
    newChapter.value = { number: newChapter.value.number + 1, title: '', content: '' }
    showCreate.value = false
    await loadChapters()
  } catch (err) {
    alert('创建失败: ' + (err.response?.data?.detail || err.message))
  }
}

async function deleteChapter(id) {
  if (!confirm('确定删除这个章节？')) return
  try {
    await chaptersApi.delete(id)
    await loadChapters()
  } catch (err) {
    alert('删除失败: ' + (err.response?.data?.detail || err.message))
  }
}

onMounted(loadChapters)
</script>
