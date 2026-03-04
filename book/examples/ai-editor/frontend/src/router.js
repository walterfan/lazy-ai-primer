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
    name: 'Chapters',
    component: () => import('./views/ChaptersView.vue'),
  },
  {
    path: '/edit/:id',
    name: 'Editor',
    component: () => import('./views/EditorView.vue'),
  },
  {
    path: '/write',
    name: 'Writer',
    component: () => import('./views/WriterView.vue'),
  },
  {
    path: '/chat',
    name: 'Chat',
    component: () => import('./views/ChatView.vue'),
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  if (!to.meta.public && !token) {
    next('/login')
  } else {
    next()
  }
})

export default router
