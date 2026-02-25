import { useState } from "react";
import { api, apiRoutes } from "./api";
import "./App.css";

export default function RegisterFace({ onBack }) {
  const [name, setName]         = useState("");
  const [files, setFiles]       = useState([]);
  const [status, setStatus]     = useState("");
  const [training, setTraining] = useState(false);

  const handleRegister = async () => {
    if (!name.trim() || files.length === 0) {
      setStatus("⚠️ Enter a name and select at least one photo.");
      return;
    }
    setStatus(`Uploading ${files.length} photo(s) for "${name}"...`);
    for (let i = 0; i < files.length; i++) {
      const form = new FormData();
      form.append("file", files[i]);
      form.append("person_name", name.trim());
      await api.post(apiRoutes.registerFace, form);
    }
    setStatus(`✅ Saved ${files.length} photo(s) for "${name}". Add more people or click Retrain.`);
    setName("");
    setFiles([]);
  };

  const handleRetrain = async () => {
    setTraining(true);
    setStatus("⏳ Retraining model... (this takes 1–3 minutes, please wait)");
    try {
      const res = await api.post(apiRoutes.retrain);
      setStatus(`✅ Done! Model now knows ${res.data.total_people} people: ${res.data.people.join(", ")}`);
    } catch (e) {
      setStatus(`❌ Retrain failed: ${e.response?.data?.detail || e.message}`);
    }
    setTraining(false);
  };

  return (
    <div style={{ maxWidth: 480, margin: "60px auto", padding: 30, fontFamily: "sans-serif" }}>

      <button onClick={onBack} style={{ marginBottom: 20, background: "none", border: "none", cursor: "pointer", color: "#6c5ce7", fontSize: 14 }}>
        ← Back to Dashboard
      </button>

      <h2 style={{ marginBottom: 6 }}>Register a Person</h2>
      <p style={{ color: "#888", fontSize: 13, marginBottom: 24 }}>
        Upload 10–15 photos per person, then click Retrain.
      </p>

      <input
        placeholder="Person name (e.g. Mom, Dad, Priya)"
        value={name}
        onChange={e => setName(e.target.value)}
        style={{ width: "100%", padding: "10px 12px", marginBottom: 12, borderRadius: 8, border: "1px solid #ddd", fontSize: 14, boxSizing: "border-box" }}
      />

      <input
        type="file"
        accept="image/*"
        multiple
        onChange={e => setFiles(Array.from(e.target.files))}
        style={{ marginBottom: 6 }}
      />
      <p style={{ fontSize: 12, color: "#aaa", marginBottom: 16 }}>
        {files.length > 0 ? `${files.length} photo(s) selected` : "No photos selected"}
      </p>

      <div style={{ display: "flex", gap: 10, marginBottom: 20 }}>
        <button
          onClick={handleRegister}
          disabled={!name.trim() || files.length === 0}
          style={{ flex: 1, padding: "10px 0", background: "#6c5ce7", color: "#fff", border: "none", borderRadius: 8, cursor: "pointer", fontSize: 14 }}
        >
          Upload Photos
        </button>

        <button
          onClick={handleRetrain}
          disabled={training}
          style={{ flex: 1, padding: "10px 0", background: training ? "#aaa" : "#00b894", color: "#fff", border: "none", borderRadius: 8, cursor: "pointer", fontSize: 14 }}
        >
          {training ? "Training..." : "Retrain Model"}
        </button>
      </div>

      {status && (
        <div style={{ padding: 12, background: "#f8f8f8", borderRadius: 8, fontSize: 13, color: "#333", lineHeight: 1.5 }}>
          {status}
        </div>
      )}
    </div>
  );
}
