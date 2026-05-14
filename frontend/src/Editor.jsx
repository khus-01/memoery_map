import React, { useState, useRef, useCallback, useEffect } from 'react';
import axios from 'axios';
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';
import AutoGenerateModal from './AutoGenerateModal';
import { api, apiRoutes } from './api';

const PAD = 3.3, GAP = 1.7;
const IW = 100 - PAD * 2, IH = 100 - PAD * 2;
const col2w = (IW - GAP) / 2;
const col3w = (IW - GAP * 2) / 3;
const row2h = (IH - GAP) / 2;

const LAYOUTS = {
  blank:         { name: 'Blank',        icon: '📄', zones: [] },
  heroFull:      { name: 'Full Page',     icon: '🖼️', zones: [{ x:PAD, y:PAD, w:IW, h:IH }] },
  twoVertical:   { name: 'Side by Side',  icon: '▯▯', zones: [
    { x:PAD,           y:PAD, w:col2w, h:IH },
    { x:PAD+col2w+GAP, y:PAD, w:col2w, h:IH },
  ]},
  twoHorizontal: { name: 'Top & Bottom',  icon: '▬▬', zones: [
    { x:PAD, y:PAD,           w:IW, h:row2h },
    { x:PAD, y:PAD+row2h+GAP, w:IW, h:row2h },
  ]},
  threeVertical: { name: 'Three Columns', icon: '▯▯▯', zones: [
    { x:PAD,               y:PAD, w:col3w, h:IH },
    { x:PAD+col3w+GAP,     y:PAD, w:col3w, h:IH },
    { x:PAD+col3w*2+GAP*2, y:PAD, w:col3w, h:IH },
  ]},
  fourGrid: { name: 'Four Grid', icon: '⊞', zones: [
    { x:PAD,           y:PAD,           w:col2w, h:row2h },
    { x:PAD+col2w+GAP, y:PAD,           w:col2w, h:row2h },
    { x:PAD,           y:PAD+row2h+GAP, w:col2w, h:row2h },
    { x:PAD+col2w+GAP, y:PAD+row2h+GAP, w:col2w, h:row2h },
  ]},
  sixGrid: { name: 'Six Grid', icon: '⊟', zones: [
    { x:PAD,               y:PAD,           w:col3w, h:row2h },
    { x:PAD+col3w+GAP,     y:PAD,           w:col3w, h:row2h },
    { x:PAD+col3w*2+GAP*2, y:PAD,           w:col3w, h:row2h },
    { x:PAD,               y:PAD+row2h+GAP, w:col3w, h:row2h },
    { x:PAD+col3w+GAP,     y:PAD+row2h+GAP, w:col3w, h:row2h },
    { x:PAD+col3w*2+GAP*2, y:PAD+row2h+GAP, w:col3w, h:row2h },
  ]},
  magazine: { name: 'Magazine', icon: '📰', zones: [
    { x:PAD,              y:PAD,           w:IW*0.6-GAP/2, h:IH },
    { x:PAD+IW*0.6+GAP/2, y:PAD,           w:IW*0.4-GAP/2, h:row2h },
    { x:PAD+IW*0.6+GAP/2, y:PAD+row2h+GAP, w:IW*0.4-GAP/2, h:row2h },
  ]},
  travel: { name: 'Travel Story', icon: '✈️', zones: [
    { x:PAD,               y:PAD,               w:IW,    h:IH*0.45-GAP/2 },
    { x:PAD,               y:PAD+IH*0.45+GAP/2, w:col3w, h:IH*0.55-GAP/2 },
    { x:PAD+col3w+GAP,     y:PAD+IH*0.45+GAP/2, w:col3w, h:IH*0.55-GAP/2 },
    { x:PAD+col3w*2+GAP*2, y:PAD+IH*0.45+GAP/2, w:col3w, h:IH*0.55-GAP/2 },
  ]},
  instagram: { name: 'Insta Grid', icon: '📱', zones: [
    { x:PAD,               y:PAD,               w:IW,    h:IH*0.65-GAP/2 },
    { x:PAD,               y:PAD+IH*0.65+GAP/2, w:col3w, h:IH*0.35-GAP/2 },
    { x:PAD+col3w+GAP,     y:PAD+IH*0.65+GAP/2, w:col3w, h:IH*0.35-GAP/2 },
    { x:PAD+col3w*2+GAP*2, y:PAD+IH*0.65+GAP/2, w:col3w, h:IH*0.35-GAP/2 },
  ]},
  scrapbook: { name: 'Scrapbook', icon: '✂️', zones: [
    { x:PAD,   y:PAD,   w:44, h:43 },
    { x:53,    y:PAD+2, w:40, h:38 },
    { x:PAD+2, y:50,    w:38, h:47 },
    { x:54,    y:52,    w:43, h:45 },
  ]},
};

const PAGE_W = 600, PAGE_H = 800;

const newPage = (id) => ({
  id, layoutKey: 'blank', photos: {}, texts: [], stickers: [], bgColor: '#ffffff',
});

const mapAILayout = (layoutStr) => {
  const map = {
    'single':       'heroFull',
    'two-column':   'twoVertical',
    'two-vertical': 'twoVertical',
    'three-mixed':  'magazine',
    'grid-2x2':     'fourGrid',
    'four-grid':    'fourGrid',
  };
  return map[layoutStr] || 'heroFull';
};

let _nextId = 1000;
const aiPageToEditorPage = (aiPage) => {
  const layoutKey = mapAILayout(aiPage.layout || 'single');
  const zones = LAYOUTS[layoutKey]?.zones || [];
  const photos = {};
  (aiPage.photos || []).forEach((url, i) => { if (i < zones.length) photos[i] = url; });
  const texts = (aiPage.texts || []).map(t => ({
    id: _nextId++, content: t.content || '', x: t.x || 10, y: t.y || 10,
    fontSize: t.fontSize || 18, color: t.color || '#333333',
    fontFamily: 'Georgia', bold: false, italic: false, align: 'left',
  }));
  const stickers = (aiPage.stickers || []).map((emoji, i) => ({
    id: _nextId++, content: emoji,
    x: 75 + (i % 2) * 10, y: 10 + (i * 15) % 60, size: 42, rotation: 0,
  }));
  return {
    id: _nextId++, layoutKey, photos, texts, stickers,
    bgColor: aiPage.bg_color || '#ffffff',
  };
};

const toolbarBtn = (extra = {}) => ({
  padding: '6px 12px', fontSize: 12, background: '#1a2744',
  border: '1px solid #0f3460', color: '#fff', borderRadius: 6,
  cursor: 'pointer', whiteSpace: 'nowrap', ...extra,
});
const miniBtn = (extra = {}) => ({
  background: 'transparent', border: 'none', color: '#a0aec0',
  cursor: 'pointer', fontSize: 12, padding: 0, ...extra,
});
const labelStyle = { display: 'block', fontSize: 10, color: '#a0aec0', marginBottom: 2 };
const selStyle = {
  width: '100%', padding: '4px 6px', fontSize: 11,
  background: '#0f3460', border: '1px solid #2d4a7a',
  color: '#fff', borderRadius: 4, marginBottom: 6,
};

const FONTS = [
  'Georgia', 'Playfair Display', 'Arial', 'Courier New',
  'Impact', 'Verdana', 'Trebuchet MS', 'Times New Roman',
];

const STICKER_CATEGORIES = {
  hearts:   { name: 'Hearts',   items: ['❤️','💕','💖','💗','💓','💝','💘','💞'] },
  flowers:  { name: 'Flowers',  items: ['🌸','🌺','🌻','🌼','🌷','🌹','💐','🌿','🍃','🌱','☘️','🍀','🌾','🌵','🌴','🌲'] },
  sparkles: { name: 'Sparkles', items: ['✨','⭐','🌟','💫','⚡','☀️','🌙','🌈','☁️','🌤️','🌞','🌝','🌛','🌜','🔆','🌠'] },
  cute:     { name: 'Cute',     items: ['🎀','🎁','🎈','🎊','🎉','🎂','🧁','🍰','🍓','🍒','🍑','🍊','🍋','🍌','🍉','🍇'] },
  animals:  { name: 'Animals',  items: ['🐱','🐶','🐰','🐻','🐼','🐨','🐯','🦁','🐮','🐷','🐸','🐵','🦊','🦋','🐝','🐞'] },
  travel:   { name: 'Travel',   items: ['✈️','🚗','🚢','🎡','🎢','🏰','🗼','🗽','🏖️','🏝️','⛺','🏕️','🗻','🏔️','🌋','⛰️'] },
  objects:  { name: 'Objects',  items: ['📷','📸','🎨','🖌️','✏️','📝','💌','📮','🎵','🎶','🎸','🎹','📚','📖','🔖','📎'] },
  symbols:  { name: 'Symbols',  items: ['💎','💍','👑','🔮','🎭','🎪','🎬','🎯','💝','🔥','💧','💨','🌊','🌀','🎆','🎇'] },
};

const BG_PATTERNS = [
  { label: 'White',     value: '#ffffff' },
  { label: 'Cream',     value: '#fdf8f0' },
  { label: 'Black',     value: '#1a1a1a' },
  { label: 'Navy',      value: '#1e3a5f' },
  { label: 'Blush',     value: '#fce4ec' },
  { label: 'Sage',      value: '#e8f5e9' },
  { label: 'Lavender',  value: '#f3e5f5' },
  { label: 'Warm Grey', value: '#f5f5f0' },
  { label: 'Slate',     value: '#546e7a' },
  { label: 'Rust',      value: '#bf360c' },
  { label: 'Gold',      value: '#f9a825' },
  { label: 'Forest',    value: '#2e7d32' },
];

export default function PhotoEditor({ username, bookId, onBackToDashboard }) {
  const [allPhotos,     setAllPhotos]     = useState({ clusters: {}, extras: [] });
  const [pages,         setPages]         = useState([newPage(Date.now())]);
  const [activePage,    setActivePage]    = useState(0);
  const [showLayouts,   setShowLayouts]   = useState(false);
  const [selectedText,  setSelectedText]  = useState(null);
  const [scale,         setScale]         = useState(1);
  const [showBgPanel,   setShowBgPanel]   = useState(false);
  const [showStickers,  setShowStickers]  = useState(false);
  const [showAutoModal, setShowAutoModal] = useState(false);
  const [generating,    setGenerating]    = useState(false);
  const [isSaving,      setIsSaving]      = useState(false);
  const [lastSaved,     setLastSaved]     = useState(null);

  const pageRef       = useRef(null);
  const canvasAreaRef = useRef(null);
  const nextId        = useRef(1);
  const activePageRef = useRef(0);

  useEffect(() => {
    const measure = () => {
      if (!canvasAreaRef.current) return;
      const h = canvasAreaRef.current.clientHeight - 60;
      const w = canvasAreaRef.current.clientWidth  - 60;
      setScale(Math.min(1, w / PAGE_W, h / PAGE_H));
    };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, []);

  useEffect(() => {
    const init = async () => {
      try {
        if (bookId) {
          const r = await axios.get(`http://127.0.0.1:8000/books/${username}/${bookId}`);
          const book = r.data?.book || {};
          setAllPhotos({ clusters: book.clusters || {}, extras: book.extras || [] });
        } else {
          const r = await axios.get(`http://127.0.0.1:8000/photos/${username}`);
          setAllPhotos(r.data);
        }
      } catch (e) { console.error('Photo load error', e); }
      try {
        const res = await api.get(apiRoutes.loadProgress(username, bookId));
        if (res.data.pages?.length > 0) { setPages(res.data.pages); return; }
      } catch (e) { /* no saved progress */ }
      setShowAutoModal(true);
    };
    init();
  }, [username, bookId]);

  useEffect(() => {
    if (!pages || pages.length === 0) return;
    const timer = setTimeout(async () => {
      setIsSaving(true);
      try {
        await api.post(apiRoutes.saveProgress, { username, book_id: bookId, pages });
        setLastSaved(new Date().toLocaleTimeString());
      } catch (e) { console.error('Autosave failed', e); }
      setIsSaving(false);
    }, 2000);
    return () => clearTimeout(timer);
  }, [pages]);

  const page = pages[activePage] || pages[0];
  activePageRef.current = activePage;

  const updatePage = useCallback((fn) => {
    setPages(prev => prev.map((p, i) => (i === activePageRef.current ? { ...fn(p) } : p)));
  }, []);

  const applyLayout = useCallback((key) => {
    updatePage(p => ({ ...p, layoutKey: key, photos: {} }));
    setShowLayouts(false);
  }, [updatePage]);

  const addPage = () => {
    const id = Date.now() + nextId.current++;
    setPages(prev => [...prev, newPage(id)]);
    setActivePage(prev => prev + 1);
  };

  const deletePage = (idx) => {
    if (pages.length === 1) return;
    setPages(prev => prev.filter((_, i) => i !== idx));
    setActivePage(prev => Math.min(prev, pages.length - 2));
  };

    const duplicatePage = (idx) => {
    const copy = { ...pages[idx], id: Date.now() + nextId.current++ };
    setPages(prev => {
      const next = [...prev];
      next.splice(idx + 1, 0, copy);
      return next;
    });
    setActivePage(idx + 1);
  };

  const handleDrop = useCallback((e, zoneIdx) => {
    e.preventDefault();
    const url = e.dataTransfer.getData('text/plain');
    if (!url) return;
    updatePage(p => ({ ...p, photos: { ...p.photos, [zoneIdx]: url } }));
  }, [updatePage]);

  const handleDragOver = (e) => { e.preventDefault(); };

  const removePhoto = useCallback((zoneIdx) => {
    updatePage(p => {
      const photos = { ...p.photos };
      delete photos[zoneIdx];
      return { ...p, photos };
    });
  }, [updatePage]);

  const addText = () => {
    const id = nextId.current++;
    updatePage(p => ({
      ...p,
      texts: [...(p.texts || []), {
        id, content: 'Double-click to edit', x: 10, y: 10,
        fontSize: 20, color: '#333333', fontFamily: 'Georgia',
        bold: false, italic: false, align: 'left',
      }],
    }));
    setSelectedText(id);
  };

  const updateText = useCallback((id, changes) => {
    updatePage(p => ({ ...p, texts: (p.texts || []).map(t => t.id === id ? { ...t, ...changes } : t) }));
  }, [updatePage]);

  const deleteText = useCallback((id) => {
    updatePage(p => ({ ...p, texts: (p.texts || []).filter(t => t.id !== id) }));
    setSelectedText(null);
  }, [updatePage]);

  const addSticker = (emoji) => {
    const id = nextId.current++;
    updatePage(p => ({
      ...p,
      stickers: [...(p.stickers || []), { id, content: emoji, x: 80, y: 5, size: 42, rotation: 0 }],
    }));
  };

  const updateSticker = useCallback((id, changes) => {
    updatePage(p => ({ ...p, stickers: (p.stickers || []).map(s => s.id === id ? { ...s, ...changes } : s) }));
  }, [updatePage]);

  const deleteSticker = useCallback((id) => {
    updatePage(p => ({ ...p, stickers: (p.stickers || []).filter(s => s.id !== id) }));
  }, [updatePage]);

  const setBgColor = (color) => { updatePage(p => ({ ...p, bgColor: color })); };

  const handleAutoGenerate = async () => {
    setGenerating(true);
    try {
      const res = await api.post(apiRoutes.autoGenerate(username, bookId));
      if (res.data?.pages?.length > 0) {
        setPages(res.data.pages.map(aiPageToEditorPage));
        setActivePage(0);
      }
    } catch (e) { console.error('Auto-generate failed', e); }
    setGenerating(false);
    setShowAutoModal(false);
  };

  const exportPNG = async () => {
    if (!pageRef.current) return;
    const el = pageRef.current;
    const orig = el.style.transform;
    
    // Hide delete buttons before export
    const deleteButtons = el.querySelectorAll('button');
    const origDisplays = Array.from(deleteButtons).map(btn => btn.style.display);
    deleteButtons.forEach(btn => btn.style.display = 'none');
    
    el.style.transform = 'none';
    const canvas = await html2canvas(el, { scale: 3, useCORS: true, allowTaint: true });
    el.style.transform = orig;
    
    // Restore delete buttons
    deleteButtons.forEach((btn, i) => btn.style.display = origDisplays[i]);
    
    const a = document.createElement('a');
    a.download = `page-${activePage + 1}.png`;
    a.href = canvas.toDataURL('image/png');
    a.click();
  };

  const exportPDF = async () => {
    const pdf = new jsPDF({ orientation: 'portrait', unit: 'px', format: [PAGE_W, PAGE_H] });
    const el = pageRef.current;
    if (!el) return;
    for (let i = 0; i < pages.length; i++) {
      setActivePage(i);
      await new Promise(r => setTimeout(r, 300));
      
      // Hide delete buttons before export
      const deleteButtons = el.querySelectorAll('button');
      const origDisplays = Array.from(deleteButtons).map(btn => btn.style.display);
      deleteButtons.forEach(btn => btn.style.display = 'none');
      
      const orig = el.style.transform;
      el.style.transform = 'none';
      const canvas = await html2canvas(el, { scale: 2, useCORS: true, allowTaint: true });
      el.style.transform = orig;
      
      // Restore delete buttons
      deleteButtons.forEach((btn, idx) => btn.style.display = origDisplays[idx]);
      
      if (i > 0) pdf.addPage();
      pdf.addImage(canvas.toDataURL('image/jpeg', 0.92), 'JPEG', 0, 0, PAGE_W, PAGE_H);
    }
    pdf.save('memorymap-book.pdf');
  };

  const textEditingRef = useRef(null);

  const makeDraggable = (item, onMove) => {
    let startX, startY, startIX, startIY;
    const onMouseMove = (e) => {
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      onMove(
        Math.max(0, Math.min(95, startIX + (dx / (PAGE_W * scale)) * 100)),
        Math.max(0, Math.min(95, startIY + (dy / (PAGE_H * scale)) * 100))
      );
    };
    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    return (e) => {
      if (textEditingRef.current === e.currentTarget) return; // Don't drag while editing
      e.preventDefault();
      startX = e.clientX; startY = e.clientY;
      startIX = item.x;   startIY = item.y;
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    };
  };

  const selText = (page?.texts || []).find(x => x.id === selectedText);

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'sans-serif', background: '#1a1a2e', color: '#fff' }}>

      {showAutoModal && (
        <AutoGenerateModal
          onGenerate={handleAutoGenerate}
          onSkip={() => setShowAutoModal(false)}
          generating={generating}
        />
      )}

      {/* LEFT SIDEBAR */}
      <div style={{ width: 220, background: '#16213e', borderRight: '1px solid #0f3460', display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
        <div style={{ padding: '14px 12px 8px', borderBottom: '1px solid #0f3460' }}>
          <button onClick={onBackToDashboard} style={{ background: 'transparent', color: '#a0aec0', border: 'none', cursor: 'pointer', fontSize: 13, marginBottom: 8, padding: 0 }}>← Back</button>
          <div style={{ fontWeight: 700, fontSize: 15 }}>📸 Photo Library</div>
        </div>
        {Object.entries(allPhotos.clusters || {}).map(([event, clusterData]) => {
          // Handle both old format (array) and new format (object with photos)
          const urls = Array.isArray(clusterData) ? clusterData : clusterData?.photos || [];
          if (!urls || urls.length === 0) return null;
          
          return (
          <div key={event}>
            <div style={{ padding: '8px 12px', fontSize: 12, fontWeight: 600, color: '#63b3ed', borderBottom: '1px solid #0f3460', background: '#1a2744' }}>{event}</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, padding: 8 }}>
              {urls.map((url, i) => (
                <img key={i} src={url} alt="" draggable
                  onDragStart={e => e.dataTransfer.setData('text/plain', url)}
                  style={{ width: 88, height: 66, objectFit: 'cover', borderRadius: 4, cursor: 'grab', border: '2px solid transparent' }}
                  onMouseOver={e => e.currentTarget.style.borderColor = '#63b3ed'}
                  onMouseOut={e  => e.currentTarget.style.borderColor = 'transparent'}
                />
              ))}
            </div>
          </div>
          );
        })}
        {(allPhotos.extras || []).length > 0 && (
          <div>
            <div style={{ padding: '8px 12px', fontSize: 12, fontWeight: 600, color: '#a0aec0', borderBottom: '1px solid #0f3460', background: '#1a2744' }}>📦 Extras</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, padding: 8 }}>
              {allPhotos.extras.map((url, i) => (
                <img key={i} src={url} alt="" draggable
                  onDragStart={e => e.dataTransfer.setData('text/plain', url)}
                  style={{ width: 88, height: 66, objectFit: 'cover', borderRadius: 4, cursor: 'grab', border: '2px solid transparent' }}
                  onMouseOver={e => e.currentTarget.style.borderColor = '#a0aec0'}
                  onMouseOut={e  => e.currentTarget.style.borderColor = 'transparent'}
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* CENTER */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        {/* TOOLBAR */}
        <div style={{ background: '#16213e', borderBottom: '1px solid #0f3460', padding: '8px 16px', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>

          <div style={{ position: 'relative' }}>
            <button onClick={() => setShowLayouts(v => !v)} style={toolbarBtn()}>🔲 Layout</button>
            {showLayouts && (
              <div style={{ position: 'absolute', top: '100%', left: 0, zIndex: 100, background: '#1a2744', border: '1px solid #0f3460', borderRadius: 8, padding: 8, display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 4, width: 260 }}>
                {Object.entries(LAYOUTS).map(([key, lay]) => (
                  <button key={key} onClick={() => applyLayout(key)}
                    style={{ background: page?.layoutKey === key ? '#0f3460' : 'transparent', border: '1px solid #0f3460', borderRadius: 6, color: '#fff', cursor: 'pointer', padding: '6px 4px', fontSize: 11, textAlign: 'center' }}>
                    {lay.icon} {lay.name}
                  </button>
                ))}
              </div>
            )}
          </div>

          <button onClick={addText} style={toolbarBtn()}>✏️ Text</button>

          <div style={{ position: 'relative' }}>
            <button onClick={() => setShowStickers(v => !v)} style={toolbarBtn()}>😊 Stickers</button>
            {showStickers && (
              <div style={{ position: 'absolute', top: '100%', left: 0, zIndex: 100, background: '#1a2744', border: '1px solid #0f3460', borderRadius: 8, padding: 10, width: 320, maxHeight: 360, overflowY: 'auto' }}>
                {Object.entries(STICKER_CATEGORIES).map(([catKey, cat]) => (
                  <div key={catKey} style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 11, color: '#a0aec0', marginBottom: 4 }}>{cat.name}</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                      {cat.items.map(emoji => (
                        <button key={emoji} onClick={() => { addSticker(emoji); setShowStickers(false); }}
                          style={{ background: 'transparent', border: 'none', fontSize: 22, cursor: 'pointer', padding: 2, borderRadius: 4 }}
                          onMouseOver={e => e.currentTarget.style.background = '#0f3460'}
                          onMouseOut={e  => e.currentTarget.style.background = 'transparent'}
                        >{emoji}</button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div style={{ position: 'relative' }}>
            <button onClick={() => setShowBgPanel(v => !v)} style={toolbarBtn()}>🎨 Background</button>
            {showBgPanel && (
              <div style={{ position: 'absolute', top: '100%', left: 0, zIndex: 100, background: '#1a2744', border: '1px solid #0f3460', borderRadius: 8, padding: 10, width: 240 }}>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
                  {BG_PATTERNS.map(p => (
                    <button key={p.value} onClick={() => { setBgColor(p.value); setShowBgPanel(false); }} title={p.label}
                      style={{ width: 32, height: 32, background: p.value, border: page?.bgColor === p.value ? '2px solid #63b3ed' : '2px solid rgba(255,255,255,0.2)', borderRadius: 6, cursor: 'pointer' }} />
                  ))}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                  <span>Custom:</span>
                  <input type="color" value={page?.bgColor || '#ffffff'} onChange={e => setBgColor(e.target.value)}
                    style={{ width: 40, height: 28, border: 'none', cursor: 'pointer' }} />
                </div>
              </div>
            )}
          </div>

          <div style={{ flex: 1 }} />
          <span style={{ fontSize: 11, color: isSaving ? '#f6e05e' : '#68d391' }}>
            {isSaving ? '💾 Saving…' : lastSaved ? `✓ Saved ${lastSaved}` : ''}
          </span>
          <button onClick={() => setShowAutoModal(true)} style={toolbarBtn({ background: '#553c9a', borderColor: '#805ad5' })}>🤖 AI Generate</button>
          <button onClick={exportPNG} style={toolbarBtn()}>⬇ PNG</button>
          <button onClick={exportPDF} style={toolbarBtn({ background: '#2d3748' })}>📄 PDF</button>
        </div>

        {/* CANVAS */}
        <div ref={canvasAreaRef}
          style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', background: '#0d1117' }}
          onClick={() => { setShowLayouts(false); setShowBgPanel(false); setShowStickers(false); }}
        >
          <div ref={pageRef}
            style={{ width: PAGE_W, height: PAGE_H, background: page?.bgColor || '#ffffff', position: 'relative', overflow: 'hidden', boxShadow: '0 8px 40px rgba(0,0,0,0.6)', transform: `scale(${scale})`, transformOrigin: 'center center', flexShrink: 0 }}
          >
            {(LAYOUTS[page?.layoutKey]?.zones || []).map((zone, zi) => (
              <div key={zi} onDrop={e => handleDrop(e, zi)} onDragOver={handleDragOver}
                style={{ position: 'absolute', left: `${zone.x}%`, top: `${zone.y}%`, width: `${zone.w}%`, height: `${zone.h}%`, border: page?.photos?.[zi] ? 'none' : '2px dashed #ccc', background: page?.photos?.[zi] ? 'transparent' : 'rgba(200,200,200,0.08)', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              >
                {page?.photos?.[zi] ? (
                  <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                    <img src={page.photos[zi]} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
                    <button onClick={() => removePhoto(zi)}
                      style={{ position: 'absolute', top: 4, right: 4, background: 'rgba(0,0,0,0.6)', color: '#fff', border: 'none', borderRadius: '50%', width: 20, height: 20, cursor: 'pointer', fontSize: 12, lineHeight: 1 }}
                      onMouseOver={e => e.currentTarget.style.opacity = '1'}
                      onMouseOut={e  => e.currentTarget.style.opacity = '0.6'}
                    >×</button>
                  </div>
                ) : (
                  <span style={{ color: '#aaa', fontSize: 13, userSelect: 'none' }}>Drop photo</span>
                )}
              </div>
            ))}

            {(page?.texts || []).map(t => (
              <div key={t.id}
                ref={selectedText === t.id ? textEditingRef : null}
                onMouseDown={makeDraggable(t, (x, y) => updateText(t.id, { x, y }))}
                onClick={e => { e.stopPropagation(); setSelectedText(t.id); }}
                onDoubleClick={(e) => {
                  e.stopPropagation();
                  setSelectedText(t.id);
                  textEditingRef.current = e.currentTarget;
                  setTimeout(() => e.currentTarget?.focus(), 0);
                }}
                contentEditable suppressContentEditableWarning
                onFocus={() => { textEditingRef.current = null; }}
                onBlur={(e) => {
                  updateText(t.id, { content: e.currentTarget.innerText });
                  textEditingRef.current = null;
                }}
                style={{ position: 'absolute', left: `${t.x}%`, top: `${t.y}%`, fontSize: t.fontSize, color: t.color, fontFamily: t.fontFamily, fontWeight: t.bold ? 'bold' : 'normal', fontStyle: t.italic ? 'italic' : 'normal', textAlign: t.align, cursor: selectedText === t.id ? 'text' : 'move', userSelect: 'text', outline: selectedText === t.id ? '2px dashed #63b3ed' : 'none', padding: 2, maxWidth: '80%' }}
              >{t.content}</div>
            ))}

            {(page?.stickers || []).map(s => (
              <div key={s.id}
                onMouseDown={makeDraggable(s, (x, y) => updateSticker(s.id, { x, y }))}
                onDoubleClick={() => deleteSticker(s.id)}
                title="Double-click to remove"
                style={{ position: 'absolute', left: `${s.x}%`, top: `${s.y}%`, fontSize: s.size, cursor: 'move', userSelect: 'none', transform: `rotate(${s.rotation}deg)`, lineHeight: 1 }}
              >{s.content}</div>
            ))}
          </div>
        </div>
      </div>

      {/* RIGHT PANEL */}
      <div style={{ width: 200, background: '#16213e', borderLeft: '1px solid #0f3460', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {selText && (
          <div style={{ padding: 12, borderBottom: '1px solid #0f3460' }}>
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: '#63b3ed' }}>✏️ Text</div>
            <label style={labelStyle}>Font</label>
            <select value={selText.fontFamily} onChange={e => updateText(selText.id, { fontFamily: e.target.value })} style={selStyle}>
              {FONTS.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
            <label style={labelStyle}>Size</label>
            <input type="range" min={8} max={80} value={selText.fontSize} onChange={e => updateText(selText.id, { fontSize: +e.target.value })} style={{ width: '100%', marginBottom: 6 }} />
            <label style={labelStyle}>Color</label>
            <input type="color" value={selText.color} onChange={e => updateText(selText.id, { color: e.target.value })} style={{ width: '100%', height: 28, border: 'none', cursor: 'pointer', marginBottom: 6 }} />
            <label style={labelStyle}>Align</label>
            <div style={{ display: 'flex', gap: 4, marginBottom: 6 }}>
              {['left', 'center', 'right'].map(a => (
                <button key={a} onClick={() => updateText(selText.id, { align: a })}
                  style={{ flex: 1, padding: '3px 0', fontSize: 12, background: selText.align === a ? '#0f3460' : 'transparent', border: '1px solid #0f3460', color: '#fff', borderRadius: 4, cursor: 'pointer' }}>
                  {a[0].toUpperCase()}
                </button>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
              <button onClick={() => updateText(selText.id, { bold: !selText.bold })}
                style={{ flex: 1, padding: '3px 0', fontSize: 12, fontWeight: 'bold', background: selText.bold ? '#0f3460' : 'transparent', border: '1px solid #0f3460', color: '#fff', borderRadius: 4, cursor: 'pointer' }}>B</button>
              <button onClick={() => updateText(selText.id, { italic: !selText.italic })}
                style={{ flex: 1, padding: '3px 0', fontSize: 12, fontStyle: 'italic', background: selText.italic ? '#0f3460' : 'transparent', border: '1px solid #0f3460', color: '#fff', borderRadius: 4, cursor: 'pointer' }}>I</button>
            </div>
            <button onClick={() => deleteText(selText.id)}
              style={{ width: '100%', padding: '4px 0', fontSize: 12, background: '#742a2a', border: 'none', color: '#fff', borderRadius: 4, cursor: 'pointer' }}>
              🗑 Delete Text
            </button>
          </div>
        )}
        <div style={{ flex: 1, overflowY: 'auto', padding: 8 }}>
          <div style={{ fontSize: 12, color: '#a0aec0', marginBottom: 8, fontWeight: 600 }}>Pages ({pages.length})</div>
          {pages.map((p, i) => (
            <div key={p.id} onClick={() => { setActivePage(i); setSelectedText(null); }}
              style={{ marginBottom: 8, cursor: 'pointer', borderRadius: 6, border: i === activePage ? '2px solid #63b3ed' : '2px solid #0f3460', background: '#1a2744', overflow: 'hidden' }}
            >
              <div style={{ width: '100%', aspectRatio: `${PAGE_W}/${PAGE_H}`, background: p.bgColor || '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {Object.values(p.photos || {}).length > 0
                  ? <img src={Object.values(p.photos)[0]} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  : <span style={{ color: '#888', fontSize: 10 }}>Pg {i + 1}</span>
                }
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 6px', fontSize: 10, color: '#a0aec0' }}>
                <span>{i + 1}</span>
                <div style={{ display: 'flex', gap: 4 }}>
                  <button onClick={e => { e.stopPropagation(); duplicatePage(i); }} style={miniBtn()} title="Duplicate">⧉</button>
                  <button onClick={e => { e.stopPropagation(); deletePage(i); }} style={miniBtn({ color: '#fc8181' })} title="Delete">✕</button>
                </div>
              </div>
            </div>
          ))}
          <button onClick={addPage}
            style={{ width: '100%', padding: '8px 0', fontSize: 13, background: '#1a2744', border: '1px dashed #0f3460', color: '#63b3ed', borderRadius: 6, cursor: 'pointer' }}>
            + Add Page
          </button>
        </div>
      </div>
    </div>
  );
}