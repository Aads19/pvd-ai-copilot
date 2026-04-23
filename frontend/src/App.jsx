import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

function normalizeApiBase(value) {
  return (value || '').trim().replace(/\/+$/, '')
}

const CONFIGURED_API_BASE = normalizeApiBase(import.meta.env.VITE_API_URL)
const API_BASE_CANDIDATES = Array.from(new Set(
  CONFIGURED_API_BASE ? [CONFIGURED_API_BASE, ''] : ['']
))

async function postChat(query) {
  const requestOptions = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  }

  let lastError = new Error('Failed to fetch')

  for (const base of API_BASE_CANDIDATES) {
    try {
      const response = await fetch(`${base}/chat`, requestOptions)
      if (!response.ok) {
        lastError = new Error(`Server error: ${response.status}`)
        if (base) continue
        throw lastError
      }
      return await response.json()
    } catch (error) {
      lastError = error
      if (base) continue
      throw error
    }
  }

  throw lastError
}

const TAG_STYLES = {
  Background: { color: '#f87171', bg: 'rgba(239,68,68,0.1)', border: 'rgba(239,68,68,0.25)' },
  Synthesis: { color: '#a78bfa', bg: 'rgba(124,58,237,0.12)', border: 'rgba(124,58,237,0.25)' },
  Characterization: { color: '#fbbf24', bg: 'rgba(245,158,11,0.1)', border: 'rgba(245,158,11,0.25)' },
  Analysis: { color: '#34d399', bg: 'rgba(16,185,129,0.1)', border: 'rgba(16,185,129,0.25)' },
}

const SUGGESTIONS = [
  'What are optimal sputtering pressures for TiO₂ thin films?',
  'How does substrate temperature affect PZT film crystallinity?',
  'Explain the HiPIMS process and its advantages over DC magnetron sputtering.',
  'What characterization techniques are used for PVD thin film analysis?',
]

function Tag({ label }) {
  const style = TAG_STYLES[label] || { color: '#00e5ff', bg: 'rgba(0,229,255,0.08)', border: 'rgba(0,229,255,0.2)' }
  return (
    <span className="tag" style={{ color: style.color, background: style.bg, borderColor: style.border }}>
      {label}
    </span>
  )
}

function SourceCard({ chunk, index }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="source-card">
      <button className="source-header" onClick={() => setOpen(o => !o)}>
        <span className="source-num">[{index + 1}]</span>
        <span className="source-title">{chunk.title || 'Untitled'}</span>
        <span className="source-score">{(chunk.score * 100).toFixed(1)}%</span>
        <span className="source-chevron">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="source-body">
          <div className="source-doi">
            DOI: <a href={`https://doi.org/${chunk.doi}`} target="_blank" rel="noreferrer">{chunk.doi}</a>
          </div>
        </div>
      )}
    </div>
  )
}

function Message({ msg }) {
  if (msg.role === 'user') {
    return (
      <div className="msg msg-user">
        <div className="msg-bubble user-bubble">
          <span>{msg.content}</span>
        </div>
      </div>
    )
  }

  if (msg.role === 'thinking') {
    return (
      <div className="msg msg-bot">
        <div className="bot-icon">⬡</div>
        <div className="msg-bubble bot-bubble thinking">
          <div className="thinking-steps">
            {msg.steps.map((step, i) => (
              <div key={i} className={`thinking-step ${step.done ? 'done' : step.active ? 'active' : ''}`}>
                <span className="step-dot">{step.done ? '✓' : step.active ? '◉' : '○'}</span>
                <span>{step.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="msg msg-bot">
      <div className="bot-icon">⬡</div>
      <div className="msg-content">
        <div className="msg-bubble bot-bubble">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
        </div>

        {msg.tags?.length > 0 && (
          <div className="msg-meta">
            <span className="meta-label">TAGS</span>
            {msg.tags.map(t => <Tag key={t} label={t} />)}
          </div>
        )}

        {msg.chunks?.length > 0 && (
          <div className="msg-sources">
            <div className="sources-header">
              <span className="meta-label">SOURCES</span>
              <span className="sources-count">{msg.chunks.length} retrieved</span>
            </div>
            {msg.chunks.map((c, i) => <SourceCard key={i} chunk={c} index={i} />)}
          </div>
        )}

        {msg.time && (
          <div className="msg-time">{msg.time}ms</div>
        )}
      </div>
    </div>
  )
}

const PIPELINE_STEPS = [
  { label: 'Routing query…' },
  { label: 'Expanding semantic query…' },
  { label: 'Generating HyDE document…' },
  { label: 'Retrieving from vector DB…' },
  { label: 'Reranking with CrossEncoder…' },
  { label: 'Generating answer via Gemini…' },
  { label: 'Paraphrasing via Llama…' },
]

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'bot',
      content: "Hello. I'm your **PVD AI Copilot** — trained on 409 Physical Vapor Deposition research papers.\n\nAsk me anything about thin film deposition, characterization, synthesis parameters, or materials analysis.",
      tags: [],
      chunks: [],
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const stepTimerRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const advanceSteps = (thinkingId, stepIdx) => {
    if (stepIdx >= PIPELINE_STEPS.length) return
    setMessages(prev => prev.map(m =>
      m.id === thinkingId
        ? {
            ...m,
            steps: PIPELINE_STEPS.map((s, i) => ({
              ...s,
              done: i < stepIdx,
              active: i === stepIdx,
            }))
          }
        : m
    ))
    const delay = stepIdx < 2 ? 900 : stepIdx < 5 ? 1400 : 2000
    stepTimerRef.current = setTimeout(() => advanceSteps(thinkingId, stepIdx + 1), delay)
  }

  const sendMessage = async (queryText) => {
    const q = (queryText || input).trim()
    if (!q || loading) return

    setInput('')
    setLoading(true)

    const userMsg = { role: 'user', content: q, id: Date.now() }
    const thinkingId = Date.now() + 1
    const thinkingMsg = {
      role: 'thinking',
      id: thinkingId,
      steps: PIPELINE_STEPS.map((s, i) => ({ ...s, done: false, active: i === 0 })),
    }

    setMessages(prev => [...prev, userMsg, thinkingMsg])
    advanceSteps(thinkingId, 0)

    try {
      const data = await postChat(q)

      clearTimeout(stepTimerRef.current)

      setMessages(prev => prev.map(m =>
        m.id === thinkingId
          ? {
              role: 'bot',
              id: thinkingId,
              content: data.answer || 'No answer returned.',
              tags: data.target_tags || [],
              chunks: data.chunks || [],
              time: data.processing_time_ms,
            }
          : m
      ))
    } catch (err) {
      clearTimeout(stepTimerRef.current)
      setMessages(prev => prev.map(m =>
        m.id === thinkingId
          ? { role: 'bot', id: thinkingId, content: `**Error:** ${err.message}`, tags: [], chunks: [] }
          : m
      ))
    } finally {
      setLoading(false)
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo">⬡ PVD</div>
          <div className="header-title">
            <span className="title-main">AI Copilot</span>
            <span className="title-sub">Physical Vapor Deposition Research</span>
          </div>
        </div>
        <div className="header-right">
          <div className="stat"><span>409</span> papers</div>
          <div className="stat"><span>8,381</span> chunks</div>
          <div className="status-dot" />
        </div>
      </header>

      {/* Messages */}
      <div className="messages">
        {messages.map((msg, i) => <Message key={msg.id || i} msg={msg} />)}

        {/* Suggestions (only shown before first user message) */}
        {messages.length === 1 && (
          <div className="suggestions">
            <div className="suggestions-label">Try asking:</div>
            <div className="suggestions-grid">
              {SUGGESTIONS.map((s, i) => (
                <button key={i} className="suggestion-btn" onClick={() => sendMessage(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="input-area">
        <div className="input-row">
          <textarea
            ref={inputRef}
            className="input-box"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask about PVD deposition, thin films, characterization…"
            rows={1}
            disabled={loading}
          />
          <button
            className={`send-btn ${loading ? 'loading' : ''}`}
            onClick={() => sendMessage()}
            disabled={loading || !input.trim()}
          >
            {loading ? '◌' : '→'}
          </button>
        </div>
        <div className="input-hint">Enter to send · Shift+Enter for new line · 7-agent RAG pipeline</div>
      </div>
    </div>
  )
}
