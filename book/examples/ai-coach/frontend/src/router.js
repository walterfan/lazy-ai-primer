import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('./views/LoginView.vue'),
    meta: { public: true },
  },
  {
    path: '/',
    name: 'Chat',
    component: () => import('./views/ChatView.vue'),
  },
  {
    path: '/knowledge',
    name: 'Knowledge',
    component: () => import('./views/KnowledgeView.vue'),
  },
  {
    path: '/learning',
    name: 'Learning',
    component: () => import('./views/LearningView.vue'),
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// 路由守卫：未登录跳转到登录页
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  if (!to.meta.public && !token) {
    next('/login')
  } else {
    next()
  }
})

export default router
