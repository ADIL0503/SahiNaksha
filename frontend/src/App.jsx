import { useMemo, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [jobId, setJobId] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const detections = result?.detections || []
  const averageConfidence = useMemo(() => {
    if (!detections.length) return 0
    return detections.reduce((sum, item) => sum + item.confidence, 0) / detections.length
  }, [detections])

  function chooseFile(event) {
    const selected = event.target.files?.[0]
    if (!selected) return
    setFile(selected)
    setPreview(URL.createObjectURL(selected))
    setResult(null)
    setJobId('')
    setError('')
  }

  async function runAnalysis() {
    if (!file) return
    setBusy(true)
    setError('')
    try {
      const form = new FormData()
      form.append('file', file)
      const upload = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: form })
      const uploaded = await upload.json()
      if (!upload.ok) throw new Error(uploaded.detail || 'Upload failed')
      setJobId(uploaded.job_id)

      const prediction = await fetch(`${API_BASE}/api/predict/${uploaded.job_id}`, { method: 'POST' })
      const body = await prediction.json()
      if (!prediction.ok) throw new Error(body.detail || 'Prediction failed')
      setResult(body)
    } catch (err) {
      setError(err.message || 'Unable to connect to the SahiNaksha API')
    } finally {
      setBusy(false)
    }
  }

  function downloadGeoJSON() {
    if (!result) return
    const blob = new Blob([JSON.stringify({ type: 'FeatureCollection', features: result.features || [] }, null, 2)], { type: 'application/geo+json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `sahinaksha-${jobId || 'result'}.geojson`
    link.click()
    URL.revokeObjectURL(url)
  }

  const width = result?.image?.width || 2000
  const height = result?.image?.height || 2000

  return (
    <main className="shell">
      <header className="hero">
        <div>
          <span className="eyebrow">SIH 2026 · PS-26012</span>
          <h1>SahiNaksha</h1>
          <p>AI-assisted urban building-footprint extraction from aerial imagery.</p>
        </div>
        <div className="status">● Prototype</div>
      </header>

      <section className="grid">
        <article className="card controls">
          <h2>1. Upload imagery</h2>
          <label className="dropzone">
            <input type="file" accept=".jpg,.jpeg,.png,.tif,.tiff" onChange={chooseFile} />
            <strong>{file ? file.name : 'Choose an aerial image'}</strong>
            <span>JPG, PNG or GeoTIFF</span>
          </label>
          <button disabled={!file || busy} onClick={runAnalysis}>{busy ? 'Running AI…' : 'Detect buildings'}</button>
          {result && <button className="secondary" onClick={downloadGeoJSON}>Export GeoJSON</button>}
          {error && <div className="error">{error}</div>}
          {jobId && <small>Job: {jobId}</small>}
        </article>

        <article className="card preview-card">
          <h2>2. Inspection view</h2>
          <div className="preview">
            {preview ? <img src={preview} alt="Uploaded aerial imagery" /> : <span>Upload an image to preview it</span>}
            {preview && detections.map((item, index) => (
              <svg key={index} className="overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
                <polygon points={item.polygon.map(([x, y]) => `${x},${y}`).join(' ')} />
              </svg>
            ))}
          </div>
        </article>

        <article className="card metrics">
          <h2>3. Results</h2>
          <div className="metric"><span>Buildings detected</span><strong>{result?.count ?? '—'}</strong></div>
          <div className="metric"><span>Mean confidence</span><strong>{result ? `${(averageConfidence * 100).toFixed(1)}%` : '—'}</strong></div>
          <div className="metric"><span>Geometry check</span><strong>{result ? (result.validation?.valid ? 'Valid' : 'Review') : '—'}</strong></div>
          <p className="note">Exported polygons are in image-pixel coordinates unless a separate georeferencing step is applied. Building footprints are not legal cadastral parcel boundaries.</p>
        </article>
      </section>
    </main>
  )
}
