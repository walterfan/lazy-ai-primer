<template>
  <div class="flex flex-col h-screen">
    <!-- 顶栏 -->
    <header class="bg-white border-b px-6 py-3 shrink-0">
      <h2 class="font-semibold text-gray-800">📊 学习计划</h2>
      <p class="text-xs text-gray-400">设定目标、记录学习、追踪进度</p>
    </header>

    <div class="flex-1 overflow-y-auto p-6">
      <div class="max-w-4xl mx-auto space-y-6">

        <!-- 创建目标 -->
        <div class="bg-white rounded-xl border border-gray-200 p-6">
          <h3 class="font-medium text-gray-700 mb-4">🎯 创建学习目标</h3>
          <div class="grid grid-cols-2 gap-3">
            <input
              v-model="newGoal.topic"
              type="text"
              placeholder="学习主题（如：Python 异步编程）"
              class="px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-coach-400"
            />
            <input
              v-model="newGoal.target"
              type="text"
              placeholder="目标描述（如：能独立编写异步爬虫）"
              class="px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-coach-400"
            />
            <input
              v-model.number="newGoal.daily_minutes"
              type="number"
              placeholder="每日学习时长（分钟）"
              class="px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-coach-400"
            />
            <button
              @click="createGoal"
              :disabled="!newGoal.topic || !newGoal.target"
              class="px-4 py-2 bg-coach-500 text-white rounded-lg text-sm font-medium hover:bg-coach-600 disabled:opacity-50 transition-colors"
            >
              创建目标
            </button>
          </div>
        </div>

        <!-- 目标列表 -->
        <div v-for="goal in goals" :key="goal.id" class="bg-white rounded-xl border border-gray-200 p-6">
          <div class="flex items-start justify-between mb-4">
            <div>
              <h3 class="font-medium text-gray-800">🎯 {{ goal.topic }}</h3>
              <p class="text-sm text-gray-500 mt-1">{{ goal.target }}</p>
              <p class="text-xs text-gray-400 mt-1">每日目标: {{ goal.daily_minutes }} 分钟</p>
            </div>
            <button
              @click="loadProgress(goal.id)"
              class="px-3 py-1.5 bg-coach-50 text-coach-600 rounded-lg text-xs font-medium hover:bg-coach-100 transition-colors"
            >
              查看进度
            </button>
          </div>

          <!-- 进度报告 -->
          <div v-if="progress[goal.id]" class="mt-4 p-4 bg-gray-50 rounded-lg">
            <div class="grid grid-cols-4 gap-4 mb-4">
              <div class="text-center">
                <p class="text-2xl font-bold text-coach-600">{{ progress[goal.id].total_hours }}</p>
                <p class="text-xs text-gray-400">总学习时长(h)</p>
              </div>
              <div class="text-center">
                <p class="text-2xl font-bold text-green-600">{{ progress[goal.id].streak_days }}</p>
                <p class="text-xs text-gray-400">连续天数</p>
              </div>
              <div class="text-center">
                <p class="text-2xl font-bold text-orange-500">{{ progress[goal.id].completion_pct }}%</p>
                <p class="text-xs text-gray-400">完成度</p>
              </div>
              <div class="text-center">
                <p class="text-2xl font-bold text-purple-600">{{ progress[goal.id].session_count }}</p>
                <p class="text-xs text-gray-400">学习次数</p>
              </div>
            </div>
            <div v-if="progress[goal.id].coach_feedback" class="p-3 bg-white rounded-lg border border-gray-100">
              <p class="text-xs text-gray-400 mb-1">🎓 教练反馈：</p>
              <div class="text-sm text-gray-700 markdown-body" v-html="renderMd(progress[goal.id].coach_feedback)"></div>
            </div>
          </div>

          <!-- 记录学习 -->
          <div class="mt-4 pt-4 border-t border-gray-100">
            <p class="text-xs text-gray-400 mb-2">📝 记录今天的学习：</p>
            <div class="flex gap-2">
              <input
                v-model.number="sessionForm[goal.id + '_min']"
                type="number"
                placeholder="分钟"
                min="1"
                class="w-20 px-2 py-1.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-coach-400"
              />
              <input
                v-model="sessionForm[goal.id + '_notes']"
                type="text"
                placeholder="学习笔记（可选）"
                class="flex-1 px-2 py-1.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-coach-400"
              />
              <select
                v-model.number="sessionForm[goal.id + '_diff']"
                class="w-24 px-2 py-1.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-coach-400"
              >
                <option :value="1">😊 简单</option>
                <option :value="2">🙂 较易</option>
                <option :value="3">😐 适中</option>
                <option :value="4">😓 较难</option>
                <option :value="5">🤯 很难</option>
              </select>
              <button
                @click="logSession(goal.id)"
                :disabled="!sessionForm[goal.id + '_min']"
                class="px-3 py-1.5 bg-green-500 text-white rounded-lg text-xs font-medium hover:bg-green-600 disabled:opacity-50 transition-colors"
              >
                记录
              </button>
            </div>
          </div>
        </div>

        <!-- 空状态 -->
        <div v-if="goals.length === 0" class="text-center text-gray-400 py-16">
          <p class="text-4xl mb-4">🎯</p>
          <p class="text-lg font-medium">还没有学习目标</p>
          <p class="text-sm mt-2">创建一个目标开始你的学习之旅吧！</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { learningApi } from '../utils/api.js'
import { renderMarkdown } from '../utils/markdown.js'

const goals = ref([])
const progress = reactive({})
const sessionForm = reactive({})

const newGoal = ref({
  topic: '',
  target: '',
  daily_minutes: 60,
})

function renderMd(text) {
  return renderMarkdown(text)
}

async function loadGoals() {
  try {
    const { data } = await learningApi.listGoals()
    goals.value = data
  } catch (err) {
    console.error('加载目标失败:', err)
  }
}

async function createGoal() {
  try {
    await learningApi.createGoal(newGoal.value)
    newGoal.value = { topic: '', target: '', daily_minutes: 60 }
    await loadGoals()
  } catch (err) {
    alert('创建失败: ' + (err.response?.data?.detail || err.message))
  }
}

async function logSession(goalId) {
  const minutes = sessionForm[goalId + '_min']
  if (!minutes) return
  try {
    await learningApi.logSession({
      goal_id: goalId,
      duration_minutes: minutes,
      notes: sessionForm[goalId + '_notes'] || '',
      difficulty: sessionForm[goalId + '_diff'] || 3,
    })
    sessionForm[goalId + '_min'] = null
    sessionForm[goalId + '_notes'] = ''
    sessionForm[goalId + '_diff'] = 3
    alert('学习记录已保存 ✅')
  } catch (err) {
    alert('记录失败: ' + (err.response?.data?.detail || err.message))
  }
}

async function loadProgress(goalId) {
  try {
    const { data } = await learningApi.getProgress(goalId)
    progress[goalId] = data
  } catch (err) {
    alert('加载进度失败: ' + (err.response?.data?.detail || err.message))
  }
}

onMounted(() => {
  loadGoals()
})
</script>
