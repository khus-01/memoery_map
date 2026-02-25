import React, { useState } from 'react';
import Login from './Login';
import Signup from './Signup';
import Dashboard from './Dashboard';
import PhotoEditor from './Editor';
import './App.css';

function App() {
  const [view, setView]           = useState('login');
  const [username, setUsername]   = useState('');
  const [photos, setPhotos]       = useState({ clusters: {}, extras: [] });
  const [currentBookId, setCurrentBookId] = useState(null); // ← NEW

  if (view === 'login') {
    return (
      <Login
        onLoginSuccess={(user) => { setUsername(user); setView('dashboard'); }}
        onGoToSignup={() => setView('signup')}
      />
    );
  }

  if (view === 'signup') {
    return (
      <Signup
        onSignupSuccess={(user) => { setUsername(user); setView('dashboard'); }}
        onGoToLogin={() => setView('login')}
      />
    );
  }

  if (view === 'dashboard') {
    return (
      <Dashboard
        username={username}
        setGlobalPhotos={setPhotos}
        onOpenEditorWithBook={(bookId) => {   // ← CHANGED
          setCurrentBookId(bookId);
          setView('editor');
        }}
        onLogout={() => {
          setUsername('');
          setPhotos({ clusters: {}, extras: [] });
          setCurrentBookId(null);
          setView('login');
        }}
      />
    );
  }

  if (view === 'editor') {
    return (
      <PhotoEditor
        username={username}
        photos={photos}
        bookId={currentBookId}              // ← NEW: pass bookId to editor
        onBackToDashboard={() => setView('dashboard')}
        onLogout={() => {
          setUsername('');
          setPhotos({ clusters: {}, extras: [] });
          setCurrentBookId(null);
          setView('login');
        }}
      />
    );
  }

  return null;
}

export default App;
