<template>
  <div class="flex flex-col h-screen">
    <!-- 顶栏 -->
    <header class="bg-white border-b px-6 py-3 flex items-center justify-between shrink-0">
      <div>
        <button @click="$router.push('/')" class="text-gray-400 hover:text-gray-600 text-sm mr-3">← 返回</button>
        <span class="font-semibold text-gray-800" v-if="chapter">
          第 {{ chapter.number }} 章：{{ chapter.title }}
        </span>
        <span class="text-xs text-gray-400 ml-2" v-if="chapter">
          {{ chapter.word_count }} 字 · v{{ chapter.version }}
        </span>
      </div>
      <div class="flex gap-2">
        <button
          v-for="action in editActions"
          :key="action.value"
          @click="executeEdit(action.value)"
          :disabled="editing"
          class="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
          :class="editing ? 'bg-gray-100 text-gray-400' : 'bg-gray-100 text-gray-600 hover:bg-editor-50 hover:text-editor-600'"
          :title="action.desc"
        >
          {{ action.icon }} {{ action.label }}
        </button>
      </div>
    </header>

    <!-- 编辑区域 -->
    <div class="flex-1 flex overflow-hidden">
      <!-- 左侧：编辑器 -->
      <div class="flex-1 flex flex-col border-r">
        <div class="px-4 py-2 bg-gray-50 border-b text-xs text-gray-400 flex justify-between">
          <span>📝 编辑器（Markdown）</span>
          <button @click="saveContent" class="text-editor-600 hover:text-editor-700 font-medium">💾 保存</button>
        </div>
        <textarea
          v-model="content"
          class="flex-1 p-4 text-sm font-mono leading-relaxed resize-none focus:outline-none"
          placeholder="在这里编写章节内容..."
        ></textarea>
      </div>

      <!-- 右侧：预览 / Diff -->
      <div class="w-1/2 flex flex-col">
        <div class="px-4 py-2 bg-gray-50 border-b text-xs text-gray-400 flex gap-3">
          <button
            @click="rightPanel = 'preview'"
            :class="rightPanel === 'preview' ? 'text-editor-600 font-medium' : ''"
          >
            👁 预览
          </button>
          <button
            v-if="editResult"
            @click="rightPanel = 'diff'"
            :class="rightPanel === 'diff' ? 'text-editor-600 font-medium' : ''"
          >
            📊 Diff
          </button>
          <button
            v-if="editResult"
            @click="rightPanel = 'edited'"
            :class="rightPanel === 'edited' ? 'text-editor-600 font-medium' : ''"
          >
            ✨ 编辑结果
          </button>
          <button
            v-if="editResult && editResult.action !== 'review'"
            @click="applyEdit"
            class="ml-auto text-green-600 hover:text-green-700 font-medium"
          >
            ✅ 应用修改
          </button>
        </div>
        <div class="flex-1 overflow-y-auto p-4">
          <!-- 预览 -->
          <div v-if="rightPanel === 'preview'" class="markdown-body text-sm" v-html="previewHtml"></div>

          <!-- Diff -->
          <div v-if="rightPanel === 'diff' && editResult" class="diff-view text-sm" v-html="editResult.diff_html"></div>

          <!-- 编辑结果 -->
          <div v-if="rightPanel === 'edited' && editResult">
            <div v-if="editResult.stats" class="mb-3 p-3 bg-gray-50 rounded-lg text-xs text-gray-500">
              原文 {{ editResult.stats.original_chars }} 字 → 编辑后 {{ editResult.stats.edited_chars }} 字
              (变化 {{ editResult.stats.change_ratio }}%)
            </div>
            <div class="markdown-body text-sm" v-html="renderMd(editResult.edited)"></div>
            <div v-if="editResult.suggestions?.length" class="mt-4 p-3 bg-yellow-50 rounded-lg">
              <p class="text-xs font-medium text-yellow-700 mb-2">💡 修改建议：</p>
              <ul class="text-xs text-yellow-600 space-y-1">
                <li v-for="(s, i) in editResult.suggestions" :key="i">{{ s }}</li>
              </ul>
            </div>
          </div>

          <!-- 加载中 -->
          <div v-if="editing" class="text-center text-gray-400 mt-10">
            <p class="text-2xl mb-2">⏳</p>
            <p class="text-sm">AI 编辑中，请稍候...</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { chaptersApi, editorApi } from '../utils/api.js'
import { renderMarkdown } from '../utils/markdown.js'

const route = useRoute()
const chapter = ref(null)
const content = ref('')
const editing = ref(false)
const editResult = ref(null)
const rightPanel = ref('preview')

const editActions = [
  { value: 'proofread', label: '校对', icon: '🔍', desc: '修正错别字、语法、标点' },
  { value: 'polish', label: '润色', icon: '✨', desc: '改善表达、提升可读性' },
  { value: 'expand', label: '扩写', icon: '📝', desc: '补充内容、增加细节' },
  { value: 'condense', label: '缩写', icon: '📐', desc: '精简内容、去除冗余' },
  { value: 'restructure', label: '重构', icon: '🔄', desc: '调整结构、重新组织' },
  { value: 'review', label: '审查', icon: '📋', desc: '给出修改建议' },
]

const previewHtml = computed(() => renderMarkdown(content.value))

function renderMd(text) {
  return renderMarkdown(text)
}

async function loadChapter() {
  try {
    const { data } = await chaptersApi.get(route.params.id)
    chapter.value = data
    content.value = data.content
  } catch (err) {
    alert('加载章节失败: ' + (err.response?.data?.detail || err.message))
  }
}

async function saveContent() {
  try {
    await chaptersApi.update(route.params.id, { content: content.value })
    await loadChapter()
    alert('保存成功 ✅')
  } catch (err) {
    alert('保存失败: ' + (err.response?.data?.detail || err.message))
  }
}

async function executeEdit(action) {
  editing.value = true
  editResult.value = null
  rightPanel.value = 'edited'

  // 获取选中文本或全文
  const textarea = document.querySelector('textarea')
  const selection = textarea?.value.substring(textarea.selectionStart, textarea.selectionEnd) || ''

  try {
    const { data } = await editorApi.edit({
      chapter_id: route.params.id,
      action,
      selection,
    })
    editResult.value = data
    if (data.diff_html) rightPanel.value = 'diff'
  } catch (err) {
    alert('编辑失败: ' + (err.response?.data?.detail || err.message))
  } finally {
    editing.value = false
  }
}

async function applyEdit() {
  if (!editResult.value || !confirm('确定应用此修改？将覆盖当前内容。')) return
  try {
    await editorApi.apply(route.params.id, editResult.value.edited)
    await loadChapter()
    editResult.value = null
    rightPanel.value = 'preview'
    alert('修改已应用 ✅')
  } catch (err) {
    alert('应用失败: ' + (err.response?.data?.detail || err.message))
  }
}

onMounted(loadChapter)
</script>
