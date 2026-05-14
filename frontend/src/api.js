import axios from 'axios';

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export const api = axios.create({ baseURL: API_BASE_URL });

export const apiRoutes = {
  signup:       '/signup/',
  login:        '/login/',
  uploadPhotos: '/upload-photos/',
  photos:       (username)           => `/photos/${encodeURIComponent(username)}`,
  modelInfo:    '/model-info',
  registerFace: '/register-face/',
  retrain:      '/retrain/',
  knownPeople:  '/known-people/',
  books:        (username)           => `/books/${encodeURIComponent(username)}`,
  createBook:   (username)           => `/books/${encodeURIComponent(username)}/create`,
  getBook:      (username, bookId)   => `/books/${encodeURIComponent(username)}/${bookId}`,
  deleteBook:   (username, bookId)   => `/books/${encodeURIComponent(username)}/${bookId}`,
  autoGenerate: (u, b) => `/auto-generate?username=${u}&book_id=${b}`,
  saveProgress: '/save-progress',
  loadProgress: (username, bookId)   => `/load-progress/${encodeURIComponent(username)}/${bookId}`,
};