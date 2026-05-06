import { useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type UploadResponse = {
  filename: string
  pages: number
  characters: number
  message: string
}

type ChatResponse = {
  answer: string
  sources?: Array<{
    source: string
    chunk: string
    score: number | null
  }>
}

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
).replace(/\/$/, '')

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadInfo, setUploadInfo] = useState<UploadResponse | null>(null)
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<Array<{ source: string; chunk: string; score: number | null }>>([])
  const [errorMessage, setErrorMessage] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const [isAsking, setIsAsking] = useState(false)

  const canUpload = useMemo(() => Boolean(selectedFile) && !isUploading, [selectedFile, isUploading])
  const canAsk = useMemo(
    () => question.trim().length > 0 && !isAsking && Boolean(uploadInfo),
    [question, isAsking, uploadInfo],
  )

  const parseErrorDetail = async (response: Response): Promise<string> => {
    try {
      const payload = (await response.json()) as { detail?: string }
      return payload.detail ?? `Request failed with status ${response.status}`
    } catch {
      return `Request failed with status ${response.status}`
    }
  }

  const uploadFile = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!selectedFile) return

    setIsUploading(true)
    setErrorMessage('')
    setAnswer('')
    setSources([])

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(await parseErrorDetail(response))
      }

      const payload = (await response.json()) as UploadResponse
      setUploadInfo(payload)
    } catch (error) {
      setUploadInfo(null)
      setErrorMessage(error instanceof Error ? error.message : 'Failed to upload file.')
    } finally {
      setIsUploading(false)
    }
  }

  const askQuestion = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!question.trim()) return

    setIsAsking(true)
    setErrorMessage('')
    setAnswer('')
    setSources([])

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question.trim() }),
      })

      if (!response.ok) {
        throw new Error(await parseErrorDetail(response))
      }

      const payload = (await response.json()) as ChatResponse
      setAnswer(payload.answer)
      setSources(payload.sources ?? [])
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Failed to get answer.')
    } finally {
      setIsAsking(false)
    }
  }

  return (
    <main className="app">
      <header>
        <h1>DocuMind</h1>
        <p>Upload a document, index it with Chroma, and chat with RAG-powered answers.</p>
      </header>

      <section className="panel">
        <h2>1) Upload file</h2>
        <form onSubmit={uploadFile} className="stack">
          <input
            type="file"
            accept=".pdf,.docx,.xlsx,.pptx,.csv,.txt,.png,.jpg,.jpeg,.webp"
            onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
          />
          <button type="submit" disabled={!canUpload}>
            {isUploading ? 'Uploading and indexing...' : 'Upload and index'}
          </button>
        </form>
        {uploadInfo && (
          <div className="success">
            <strong>Ready:</strong> {uploadInfo.filename} ({uploadInfo.pages} pages,{' '}
            {uploadInfo.characters.toLocaleString()} characters)
          </div>
        )}
      </section>

      <section className="panel">
        <h2>2) Ask questions</h2>
        <form onSubmit={askQuestion} className="stack">
          <textarea
            rows={4}
            value={question}
            placeholder="Example: List all companies, roles, and employment durations."
            onChange={(event) => setQuestion(event.target.value)}
          />
          <button type="submit" disabled={!canAsk}>
            {isAsking ? 'Thinking...' : 'Ask'}
          </button>
        </form>
      </section>

      {errorMessage && (
        <section className="panel error">
          <h2>Error</h2>
          <p>{errorMessage}</p>
        </section>
      )}

      {answer && (
        <section className="panel">
          <h2>Answer</h2>
          <div className="answer">{answer}</div>
        </section>
      )}

      {sources.length > 0 && (
        <section className="panel">
          <h2>Sources</h2>
          <div className="stack">
            {sources.map((source, index) => (
              <div key={`${source.source}-${index}`} className="sourceCard">
                <p>
                  <strong>{source.source}</strong>
                  {typeof source.score === 'number' ? ` (score: ${source.score.toFixed(3)})` : ''}
                </p>
                <p className="sourceChunk">{source.chunk}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      <footer>
        <code>{API_BASE_URL}</code>
      </footer>
    </main>
  )
}

export default App
