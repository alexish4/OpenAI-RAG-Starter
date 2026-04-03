import React, { useMemo, useState } from "react";
import axios from "axios";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [ingesting, setIngesting] = useState(false);
  const [docStatus, setDocStatus] = useState(null); // {doc_id, chunks_added}
  const [error, setError] = useState("");

  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [messages, setMessages] = useState([]); // {role: "user"|"assistant", text}
  const [sources, setSources] = useState([]);   // from API

  const canAsk = useMemo(() => !!docStatus?.doc_id, [docStatus]);

  async function handleIngest(e) {
    e.preventDefault();
    setError("");

    if (!file) {
      setError("Please choose a PDF first.");
      return;
    }

    try {
      setIngesting(true);
      setDocStatus(null);
      setSources([]);
      setMessages([]);

      const form = new FormData();
      form.append("file", file);

      const res = await axios.post("/api/ingest/pdf", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setDocStatus(res.data);
      setMessages([
        { role: "assistant", text: `✅ Ingested: ${file.name} (${res.data.chunks_added} chunks)` },
      ]);
      } catch (err) {
        if (err?.response?.status === 413) {
          setError("📄 Document too large. Please upload a smaller PDF.");
        } else {
          setError(err?.response?.data?.detail || err.message || "Failed to ingest PDF.");
        }
      }
  }

  async function handleAsk(e) {
    e.preventDefault();
    setError("");

    const q = question.trim();
    if (!q) return;

    if (!docStatus?.doc_id) {
      setError("Please ingest a PDF first.");
      return;
    }

    try {
      setAsking(true);
      setSources([]);

      setMessages((m) => [...m, { role: "user", text: q }]);
      setQuestion("");

      const res = await axios.post("/api/ask", {
        question: q,
        doc_id: docStatus.doc_id,
        top_k: 5,
      });

      setMessages((m) => [...m, { role: "assistant", text: res.data.answer }]);
      setSources(res.data.sources || []);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "Failed to ask question.");
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1 className="brandTitle">OpenAI RAG Starter</h1>
          <p className="sub">
            Upload a PDF → ask questions → get grounded answers with sources.
          </p>
          <a
            href="https://github.com/alexish4/OpenAI-RAG-Starter"
            target="_blank"
            rel="noopener noreferrer"
            className="githubLink"
          >
            🔗 View on GitHub
          </a>
        </div>
        <div className="pill">API: {API_BASE}</div>
      </header>

      <div className="grid">
        {/* LEFT: Upload + Chat */}
        <section className="card">
          <h2>1) Upload a PDF</h2>
          <form className="row" onSubmit={handleIngest}>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            <button className="btn btnPrimary" type="submit" disabled={!file || ingesting}>
              {ingesting ? "Ingesting..." : "Ingest"}
            </button>
          </form>

          {docStatus && (
            <div className="status">
              <div><strong>doc_id:</strong> {docStatus.doc_id}</div>
              <div><strong>chunks_added:</strong> {docStatus.chunks_added}</div>
            </div>
          )}

          <hr className="divider" />

          <h2>2) Ask</h2>
          <form className="row" onSubmit={handleAsk}>
            <input
              className="textInput"
              placeholder="Ask something about the uploaded PDF..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={asking}
            />
            <button className="btn btnGreen" type="submit" disabled={asking || !question.trim() || !canAsk}>
              {asking ? "Asking..." : "Ask"}
            </button>
          </form>

          {error && <div className="error">{error}</div>}

          <div className="chat">
            {messages.length === 0 ? (
              <div className="empty">
                Upload a PDF, then ask a question like: <br />
                <code>What is the main topic of this document?</code>
              </div>
            ) : (
              messages.map((m, i) => (
                <div key={i} className={`msg ${m.role}`} style={{ "--i": i }}>
                  <div className="role">{m.role}</div>
                  <div className="bubble">{m.text}</div>
                </div>
              ))
            )}
          </div>
        </section>

        {/* RIGHT: Sources */}
        <section className="card">
          <h2>Sources</h2>
          <p className="sub">Top retrieved chunks used as context.</p>

          {sources.length === 0 ? (
            <div className="empty">Ask a question to see source chunks here.</div>
          ) : (
            <div className="sources">
              {sources.map((s, i) => (
                <div key={i} className="sourceCard" style={{ "--i": i }}>
                  <div className="sourceMeta">
                    <span className="badge">{s.source_name}</span>
                    <span className="badge">chunk {s.chunk_id}</span>
                    <span className="badge">doc {s.doc_id.slice(0, 8)}…</span>
                  </div>
                  <div className="sourceText">{s.text}</div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      <footer className="footer">
        Tip: Once this works, add streaming responses + eval tests and you have a resume-grade RAG app.
      </footer>
    </div>
  );
}