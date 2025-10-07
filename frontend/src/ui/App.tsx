import React, { useMemo, useState } from 'react'

async function apiPredict(text: string) {
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data?.detail || 'Request failed')
  return data as { label: 'spam' | 'ham', label_id: number, probabilities: { spam: number, ham: number } }
}

export function App() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<null | { label: 'spam' | 'ham', probabilities: { spam: number, ham: number } }>(null)

  const spamPercent = useMemo(() => result ? (result.probabilities.spam * 100).toFixed(1) : '0.0', [result])
  const hamPercent = useMemo(() => result ? (result.probabilities.ham * 100).toFixed(1) : '0.0', [result])

  async function onSubmit() {
    setError(null)
    setResult(null)
    const trimmed = text.trim()
    if (!trimmed) {
      setError('Please enter some text.')
      return
    }
    setLoading(true)
    try {
      const prediction = await apiPredict(trimmed)
      setResult({ label: prediction.label, probabilities: prediction.probabilities })
    } catch (e: any) {
      setError(e?.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>Spam Email Detector</h1>
      <p className="lead">Enter an email body below to classify it as spam or ham.</p>

      <div className="card">
        <label htmlFor="emailText">Email text</label>
        <textarea id="emailText" value={text} onChange={e => setText(e.target.value)} placeholder="Paste or type your email here..." />
        <div className="actions">
          <button onClick={onSubmit} disabled={loading}>{loading ? 'Checking…' : 'Check'}</button>
          <button onClick={() => { setText(''); setResult(null); setError(null); }} disabled={loading}>Clear</button>
        </div>
        {error && <div className="error" role="alert" style={{ marginTop: 12 }}>{error}</div>}
        {result && (
          <div className="result">
            {result.label === 'spam' ? (
              <span className="badge spam">SPAM</span>
            ) : (
              <span className="badge ham">HAM</span>
            )}
            &nbsp; Spam: {spamPercent}% • Ham: {hamPercent}%
          </div>
        )}
      </div>

      <div className="footer">Backend: POST <code>/api/predict</code> with JSON <code>{`{ text: string }`}</code>.</div>
    </div>
  )
}
