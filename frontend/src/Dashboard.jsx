import React, { useCallback, useEffect, useState } from 'react';
import { api, apiRoutes } from './api';
import './App.css';

function Dashboard({ username, setGlobalPhotos, onOpenEditorWithBook, onLogout }) {
  const [photos, setPhotos]             = useState({ clusters: {}, extras: [], extras_info: [] });
  const [loading, setLoading]           = useState(false);
  const [uploadProgress, setUploadProgress] = useState({ current: 0, total: 0, status: '' });
  const [uploadError, setUploadError]   = useState('');
  const [mlReady, setMlReady]           = useState(false);
  const [mlMessage, setMlMessage]       = useState('Checking...');

  // ── Multi-book state ─────────────────────────────────────────────────────
  const [books, setBooks]               = useState([]);
  const [booksLoading, setBooksLoading] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [modalBookName, setModalBookName]     = useState('');
  const [uploadBookId, setUploadBookId]       = useState('');   // which book to upload into
  const [uploadNewBookName, setUploadNewBookName] = useState(''); // name for new book on upload

  const MAX_PHOTOS = 20;

  const syncPhotos = useCallback((nextPhotos) => {
    setPhotos(nextPhotos);
    if (typeof setGlobalPhotos === 'function') setGlobalPhotos(nextPhotos);
  }, [setGlobalPhotos]);

  // ── Fetch books list ─────────────────────────────────────────────────────
  const fetchBooks = useCallback(async () => {
    setBooksLoading(true);
    try {
      const res = await api.get(apiRoutes.books(username));
      setBooks(res.data.books || []);
    } catch {
      setBooks([]);
    } finally {
      setBooksLoading(false);
    }
  }, [username]);

  useEffect(() => {
    let active = true;

    const fetchPhotos = async () => {
      try {
        const res = await api.get(apiRoutes.photos(username));
        if (active) syncPhotos(res.data);
      } catch { if (active) console.log('New user - no photos yet.'); }
    };

    const fetchModelInfo = async () => {
      try {
        const res = await api.get(apiRoutes.modelInfo);
        if (active) {
          setMlReady(Boolean(res.data?.ready));
          setMlMessage(res.data?.message || 'Outfit-based auto-clustering active.');
        }
      } catch {
        if (active) {
          setMlReady(false);
          setMlMessage('Could not reach backend. Please start the server.');
        }
      }
    };

    fetchPhotos();
    fetchModelInfo();
    fetchBooks();
    return () => { active = false; };
  }, [username, syncPhotos, fetchBooks]);

  // ── Create empty book ────────────────────────────────────────────────────
  const handleCreateBook = async (name) => {
    const bookName = (name || '').trim() || `Book ${books.length + 1}`;
    try {
      const res = await api.post(apiRoutes.createBook(username), { name: bookName });
      await fetchBooks();
      return res.data.book_id;
    } catch (err) {
      console.error('Create book error:', err);
      return null;
    }
  };

  // ── Delete book ──────────────────────────────────────────────────────────
  const handleDeleteBook = async (bookId, bookName) => {
    if (!window.confirm(`Delete "${bookName}"? This cannot be undone.`)) return;
    try {
      await api.delete(apiRoutes.deleteBook(username, bookId));
      await fetchBooks();
    } catch (err) { console.error('Delete error:', err); }
  };

  // ── Upload photos ────────────────────────────────────────────────────────
  const handleUpload = async (event) => {
    const files = event.target.files;
    if (!files.length) return;

    if (!mlReady) {
      setUploadError('Backend is not ready. Please start the server and refresh.');
      return;
    }
    if (files.length > MAX_PHOTOS) {
      alert(`Please upload maximum ${MAX_PHOTOS} photos at a time.`);
      return;
    }

    setLoading(true);
    setUploadError('');
    setUploadProgress({ current: 0, total: files.length, status: 'Uploading & Analyzing...' });

    const formData = new FormData();
    formData.append('username', username);

    if (uploadBookId) {
      // Upload into existing book
      formData.append('book_id', uploadBookId);
    } else {
      // Create a new book on upload (backend auto-names if empty)
      if (uploadNewBookName.trim()) {
        formData.append('book_name', uploadNewBookName.trim());
      }
    }

    for (let i = 0; i < files.length; i++) formData.append('files', files[i]);

    try {
      const res = await api.post(apiRoutes.uploadPhotos, formData);
      syncPhotos(res.data.data);
      setUploadProgress({ current: 0, total: 0, status: 'Success! Photos organized.' });
      setUploadNewBookName('');
      setUploadBookId('');
      await fetchBooks(); // Refresh books grid
      setTimeout(() => setUploadProgress({ current: 0, total: 0, status: '' }), 3000);
    } catch (err) {
      console.error('Upload error:', err);
      const detail = err.response?.data?.detail;
      const errorMsg = typeof detail === 'string' ? detail : detail?.message || 'Error processing photos.';
      setUploadError(errorMsg);
      alert(errorMsg);
    } finally {
      setLoading(false);
      event.target.value = '';
    }
  };

  const uploadDisabled = loading || !mlReady;
  const hasPhotos = Object.keys(photos.clusters).length > 0 || photos.extras.length > 0;

  return (
    <div className="dashboard-container">

      {/* ── Header ── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h1 style={{ margin: 0 }}>Welcome, {username.toUpperCase()}</h1>
        <button onClick={onLogout} style={{ padding: '8px 20px', background: '#e74c3c', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '0.9rem' }}>
          Logout
        </button>
      </div>

      {/* ── AI Status ── */}
      <div style={{
        marginBottom: '20px', padding: '14px', borderRadius: '8px',
        border: mlReady ? '1px solid #b7ebc6' : '1px solid #ffd591',
        background: mlReady ? '#f6ffed' : '#fff7e6',
      }}>
        <h3 style={{ marginTop: 0, marginBottom: '6px' }}>AI Status</h3>
        <p style={{ margin: 0, color: mlReady ? '#237804' : '#ad6800', fontSize: '0.95rem' }}>
          {mlReady ? '✅ ' : '⚠️ '}{mlMessage}
        </p>
        {mlReady && (
          <p style={{ margin: '4px 0 0', color: '#595959', fontSize: '0.85rem' }}>
            Upload your photos — AI will automatically group them by events and people.
          </p>
        )}
      </div>

      {/* ── Upload ── */}
      <div className="step-box">
        <h3>Upload Your Memories</h3>
        <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '12px' }}>
          Upload up to {MAX_PHOTOS} photos. AI will automatically detect people and group by events.
        </p>

        {/* Book target selector */}
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '12px', alignItems: 'center' }}>
          <select
            value={uploadBookId}
            onChange={(e) => { setUploadBookId(e.target.value); if (e.target.value) setUploadNewBookName(''); }}
            style={{ padding: '8px 12px', borderRadius: '6px', border: '1px solid #ddd', fontSize: '0.9rem', minWidth: '200px' }}
          >
            <option value="">➕ Create new book on upload</option>
            {books.map(b => (
              <option key={b.id} value={b.id}>
                📖 {b.name} ({b.total_photos} photos)
              </option>
            ))}
          </select>

          {!uploadBookId && (
            <input
              type="text"
              placeholder="New book name (optional)"
              value={uploadNewBookName}
              onChange={(e) => setUploadNewBookName(e.target.value)}
              style={{ padding: '8px 12px', borderRadius: '6px', border: '1px solid #ddd', fontSize: '0.9rem', minWidth: '200px' }}
            />
          )}
        </div>

        <input
          type="file"
          multiple
          accept="image/*,.heic"
          onChange={handleUpload}
          disabled={uploadDisabled}
        />

        {!mlReady && (
          <p style={{ marginTop: '8px', color: '#ad6800', fontSize: '0.85rem' }}>
            Upload disabled — backend not reachable.
          </p>
        )}

        {loading && (
          <div style={{ marginTop: '20px', padding: '15px', background: '#f0f0f7', borderRadius: '8px', textAlign: 'center' }}>
            <div className="spinner">Processing</div>
            <p><strong>{uploadProgress.status}</strong></p>
            <div style={{ width: '100%', height: '10px', background: '#ddd', borderRadius: '5px', marginTop: '10px' }}>
              <div style={{ width: '100%', height: '100%', background: '#6c5ce7', borderRadius: '5px', transition: '1s' }} />
            </div>
            <p style={{ fontSize: '0.8rem', marginTop: '8px' }}>
              Processing {uploadProgress.total} photos — detecting faces & grouping...
            </p>
          </div>
        )}

        {uploadProgress.status === 'Success! Photos organized.' && (
          <div style={{ marginTop: '15px', padding: '12px', background: '#d4edda', color: '#155724', borderRadius: '6px', border: '1px solid #c3e6cb' }}>
            ✅ Photos successfully organized into events!
          </div>
        )}
        {uploadError && (
          <div style={{ marginTop: '15px', padding: '12px', background: '#fff1f0', color: '#a8071a', borderRadius: '6px', border: '1px solid #ffa39e', whiteSpace: 'pre-line' }}>
            {uploadError}
          </div>
        )}
      </div>

      {/* ── MY BOOKS ─────────────────────────────────────────────────────── */}
      <div style={{ marginTop: '35px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ margin: 0 }}>📚 My Books ({books.length})</h3>
          <button
            onClick={() => setShowCreateModal(true)}
            style={{
              padding: '9px 20px',
              background: 'linear-gradient(135deg, #6c5ce7 0%, #5849c7 100%)',
              color: 'white', border: 'none', borderRadius: '8px',
              cursor: 'pointer', fontSize: '0.9rem', fontWeight: 'bold',
              boxShadow: '0 4px 12px rgba(108,92,231,0.3)',
            }}
          >
            ➕ New Book
          </button>
        </div>

        {booksLoading ? (
          <p style={{ color: '#888', textAlign: 'center', padding: '20px' }}>Loading books...</p>
        ) : books.length === 0 ? (
          <div style={{
            padding: '40px 20px', textAlign: 'center',
            background: '#fafafa', borderRadius: '12px',
            border: '2px dashed #e0e0e0', color: '#bbb',
          }}>
            <div style={{ fontSize: '2.5rem', marginBottom: '10px' }}>📖</div>
            <p style={{ fontSize: '1rem', margin: 0 }}>
              No books yet. Upload photos above or click <strong>New Book</strong> to get started!
            </p>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: '16px' }}>
            {books.map((book) => (
              <div
                key={book.id}
                style={{
                  background: 'white', borderRadius: '14px',
                  boxShadow: '0 2px 12px rgba(0,0,0,0.07)',
                  padding: '20px', border: '1px solid #f0f0f0',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  display: 'flex', flexDirection: 'column', gap: '6px',
                }}
                onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.12)'; }}
                onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 2px 12px rgba(0,0,0,0.07)'; }}
              >
                <div style={{ fontSize: '2.2rem' }}>📒</div>
                <h4 style={{ margin: 0, fontSize: '1rem', color: '#333', wordBreak: 'break-word' }}>
                  {book.name}
                </h4>
                <p style={{ margin: 0, fontSize: '0.82rem', color: '#888' }}>
                  {book.event_count} event{book.event_count !== 1 ? 's' : ''} &nbsp;·&nbsp; {book.total_photos} photo{book.total_photos !== 1 ? 's' : ''}
                </p>
                <p style={{ margin: '0 0 8px', fontSize: '0.78rem', color: '#ccc' }}>
                  {book.updated_at ? `Updated ${new Date(book.updated_at).toLocaleDateString()}` : ''}
                </p>
                <div style={{ display: 'flex', gap: '8px', marginTop: 'auto' }}>
                  <button
                    onClick={() => onOpenEditorWithBook(book.id)}
                    style={{
                      flex: 1, padding: '9px 0',
                      background: 'linear-gradient(135deg, #6c5ce7, #5849c7)',
                      color: 'white', border: 'none', borderRadius: '8px',
                      cursor: 'pointer', fontSize: '0.88rem', fontWeight: 'bold',
                    }}
                  >
                    Open Editor
                  </button>
                  <button
                    onClick={() => handleDeleteBook(book.id, book.name)}
                    title="Delete book"
                    style={{
                      padding: '9px 12px', background: '#fff5f5',
                      color: '#e74c3c', border: '1px solid #ffd6d6',
                      borderRadius: '8px', cursor: 'pointer', fontSize: '0.88rem',
                    }}
                  >
                    🗑
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── CREATE BOOK MODAL ── */}
      {showCreateModal && (
        <div style={{
          position: 'fixed', inset: 0,
          background: 'rgba(0,0,0,0.45)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          zIndex: 1000,
        }}>
          <div style={{
            background: 'white', borderRadius: '18px',
            padding: '36px', width: '380px',
            boxShadow: '0 24px 64px rgba(0,0,0,0.25)',
          }}>
            <h3 style={{ marginTop: 0, marginBottom: '8px' }}>📖 Create New Book</h3>
            <p style={{ color: '#888', fontSize: '0.9rem', marginBottom: '20px' }}>
              Give your book a name — you can add photos to it afterwards.
            </p>
            <input
              type="text"
              placeholder="e.g. Sri Lanka Trip 2025"
              value={modalBookName}
              onChange={(e) => setModalBookName(e.target.value)}
              onKeyDown={async (e) => {
                if (e.key === 'Enter') {
                  await handleCreateBook(modalBookName);
                  setModalBookName('');
                  setShowCreateModal(false);
                }
                if (e.key === 'Escape') {
                  setShowCreateModal(false);
                  setModalBookName('');
                }
              }}
              autoFocus
              style={{
                width: '100%', padding: '11px 14px',
                borderRadius: '8px', border: '1px solid #ddd',
                fontSize: '1rem', boxSizing: 'border-box', marginBottom: '20px',
              }}
            />
            <div style={{ display: 'flex', gap: '10px' }}>
              <button
                onClick={async () => {
                  await handleCreateBook(modalBookName);
                  setModalBookName('');
                  setShowCreateModal(false);
                }}
                style={{
                  flex: 1, padding: '11px',
                  background: 'linear-gradient(135deg, #6c5ce7, #5849c7)',
                  color: 'white', border: 'none', borderRadius: '8px',
                  cursor: 'pointer', fontSize: '1rem', fontWeight: 'bold',
                  boxShadow: '0 4px 12px rgba(108,92,231,0.3)',
                }}
              >
                Create Book
              </button>
              <button
                onClick={() => { setShowCreateModal(false); setModalBookName(''); }}
                style={{
                  padding: '11px 20px', background: '#f5f5f5',
                  color: '#666', border: '1px solid #e0e0e0',
                  borderRadius: '8px', cursor: 'pointer', fontSize: '1rem',
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Organized Events (latest upload preview) ── */}
      {hasPhotos && (
        <div style={{ marginTop: '35px' }}>
          <h3 style={{ marginBottom: '15px' }}>Your Organized Events</h3>

          {Object.keys(photos.clusters).map((key) => (
            <div key={key} style={{ marginBottom: '25px', background: 'white', padding: '15px', borderRadius: '12px', boxShadow: '0 2px 10px rgba(0,0,0,0.05)' }}>
              <h4 style={{ background: '#6c5ce7', color: 'white', padding: '8px 15px', borderRadius: '6px', marginBottom: '12px', display: 'inline-block' }}>
                {key} ({photos.clusters[key].length} photos)
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '10px' }}>
                {photos.clusters[key].slice(0, 6).map((url, i) => (
                  <img key={i} src={url} alt={`Event photo ${i + 1}`}
                    style={{ width: '100%', height: '150px', objectFit: 'cover', borderRadius: '8px', cursor: 'pointer', transition: 'transform 0.2s' }}
                    onMouseEnter={(e) => e.target.style.transform = 'scale(1.05)'}
                    onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
                  />
                ))}
                {photos.clusters[key].length > 6 && (
                  <div style={{ width: '100%', height: '150px', background: '#f0f0f0', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.2rem', color: '#666' }}>
                    +{photos.clusters[key].length - 6} more
                  </div>
                )}
              </div>
            </div>
          ))}

          {photos.extras?.length > 0 && (
            <div style={{ marginBottom: '25px', background: 'white', padding: '15px', borderRadius: '12px', boxShadow: '0 2px 10px rgba(0,0,0,0.05)' }}>
              <h4 style={{ background: '#95a5a6', color: 'white', padding: '8px 15px', borderRadius: '6px', marginBottom: '12px', display: 'inline-block' }}>
                Scenery / Extras ({photos.extras.length} photos)
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '10px' }}>
                {photos.extras.slice(0, 6).map((url, i) => (
                  <img key={i} src={url} alt={`Extra photo ${i + 1}`}
                    style={{ width: '100%', height: '150px', objectFit: 'cover', borderRadius: '8px', cursor: 'pointer', transition: 'transform 0.2s' }}
                    onMouseEnter={(e) => e.target.style.transform = 'scale(1.05)'}
                    onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
                  />
                ))}
                {photos.extras.length > 6 && (
                  <div style={{ width: '100%', height: '150px', background: '#f0f0f0', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.2rem', color: '#666' }}>
                    +{photos.extras.length - 6} more
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Dashboard;
