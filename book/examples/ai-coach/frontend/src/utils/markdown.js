/**
 * Markdown 渲染工具
 */
import { marked } from 'marked'
import hljs from 'highlight.js'

// 配置 marked
marked.setOptions({
  highlight(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value
    }
    return hljs.highlightAuto(code).value
  },
  breaks: true,
  gfm: true,
})

/**
 * 将 Markdown 文本渲染为 HTML
 */
export function renderMarkdown(text) {
  if (!text) return ''
  return marked.parse(text)
}
