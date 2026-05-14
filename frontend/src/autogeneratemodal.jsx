import React from 'react';

const AutoGenerateModal = ({ onGenerate, onSkip, generating }) => {
  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.65)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 99999,
    }}>
      <div style={{
        background: '#fff', borderRadius: '20px', padding: '40px 36px',
        maxWidth: '400px', width: '90%', textAlign: 'center',
        boxShadow: '0 24px 60px rgba(0,0,0,0.3)',
      }}>
        {generating ? (
          <>
            <div style={{ fontSize: '52px', marginBottom: '16px' }}>✨</div>
            <h2 style={{ marginBottom: '10px', fontSize: '20px', color: '#1e293b' }}>AI is building your book...</h2>
            <p style={{ color: '#777', fontSize: '14px', lineHeight: 1.6 }}>
              Detecting events · Picking best photos<br />Adding decorations &amp; captions
            </p>
            <div style={{ marginTop: '24px', height: '4px', background: '#f0f0f0', borderRadius: '2px', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: '40%', background: '#6c5ce7', borderRadius: '2px', animation: 'mmSlide 1.4s ease-in-out infinite' }} />
            </div>
            <style>{`@keyframes mmSlide { 0%{transform:translateX(-150%)} 100%{transform:translateX(400%)} }`}</style>
          </>
        ) : (
          <>
            <div style={{ fontSize: '52px', marginBottom: '16px' }}>📸</div>
            <h2 style={{ marginBottom: '10px', fontSize: '20px', color: '#1e293b' }}>Auto Generate Book?</h2>
            <p style={{ color: '#666', marginBottom: '24px', lineHeight: 1.7, fontSize: '14px' }}>
              AI will <strong>group your photos by event</strong>, pick the best shots, add <strong>captions &amp; decorations</strong> and build the full book automatically.
              <br /><br />You can still <strong>edit everything</strong> after!
            </p>
            <button onClick={onGenerate} style={{ width: '100%', padding: '13px', background: '#6c5ce7', color: '#fff', border: 'none', borderRadius: '10px', fontSize: '15px', fontWeight: 600, cursor: 'pointer', marginBottom: '10px' }}>
              ✨ Yes, Auto Generate!
            </button>
            <button onClick={onSkip} style={{ width: '100%', padding: '13px', background: 'transparent', color: '#888', border: '1px solid #e0e0e0', borderRadius: '10px', fontSize: '15px', cursor: 'pointer' }}>
              No thanks, I\'ll build manually
            </button>
          </>
        )}
      </div>
    </div>
  );
};

export default AutoGenerateModal;