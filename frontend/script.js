// ─── Config ──────────────────────────────────────────────────────
const API_URL = 'http://localhost:5500/predict';
// ─── State ───────────────────────────────────────────────────────
let isThinking = false;
let sessionHistory = [];
let chatSessions  = [];
let currentSessionId = null;

// ─── DOM refs ─────────────────────────────────────────────────────
const messagesContainer = document.getElementById('messagesContainer');
const messagesList      = document.getElementById('messagesList');
const emptyState        = document.getElementById('emptyState');
const userInput         = document.getElementById('userInput');
const sendBtn           = document.getElementById('sendBtn');
const statusDot         = document.getElementById('statusDot');
const statusText        = document.getElementById('statusText');
const chatHistory       = document.getElementById('chatHistory');

// ─── Init ─────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  newChat();
  userInput.focus();
});

// ─── Send message ─────────────────────────────────────────────────
async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isThinking) return;

  hideEmptyState();
  appendMessage('user', text);
  userInput.value = '';
  autoResize(userInput);
  sessionHistory.push({ role: 'user', content: text });
  updateSessionPreview(text);

  setThinking(true);
  const typingId = showTyping();

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);

    const data = await response.json();
    const reply = data.prediction || data.response || 'No response received.';

    removeTyping(typingId);
    appendMessage('bot', reply);
    sessionHistory.push({ role: 'bot', content: reply });

  } catch (err) {
    removeTyping(typingId);
    appendMessage('bot', `⚠️ Could not reach the model. Make sure your backend is running.\n\n<code>${err.message}</code>`);
    setStatus('error', 'Disconnected');
  } finally {
    setThinking(false);
  }
}

// ─── Append message bubble ────────────────────────────────────────
function appendMessage(role, text) {
  const msg = document.createElement('div');
  msg.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'bot' ? 'AI' : 'You';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = formatText(text);

  const time = document.createElement('span');
  time.className = 'message-time';
  time.textContent = now();
  bubble.appendChild(time);

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  messagesList.appendChild(msg);
  scrollToBottom();
}

// ─── Typing indicator ─────────────────────────────────────────────
function showTyping() {
  const id = 'typing-' + Date.now();
  const msg = document.createElement('div');
  msg.className = 'message bot';
  msg.id = id;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'bubble typing-bubble';
  bubble.innerHTML = `
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
  `;

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  messagesList.appendChild(msg);
  scrollToBottom();
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ─── Status bar ───────────────────────────────────────────────────
function setThinking(val) {
  isThinking = val;
  sendBtn.disabled = val;

  if (val) {
    setStatus('thinking', 'Thinking…');
  } else {
    setStatus('ready', 'Ready');
  }
}

function setStatus(type, label) {
  statusDot.className = 'status-dot' + (type !== 'ready' ? ` ${type}` : '');
  statusText.textContent = label;
}

// ─── Keyboard handler ─────────────────────────────────────────────
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

// ─── Auto-resize textarea ─────────────────────────────────────────
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

// ─── Suggestion chips ─────────────────────────────────────────────
function useChip(btn) {
  userInput.value = btn.textContent;
  autoResize(userInput);
  userInput.focus();
}

// ─── Sidebar ──────────────────────────────────────────────────────
function toggleSidebar() {
  document.querySelector('.sidebar').classList.toggle('collapsed');
}

// ─── New chat ─────────────────────────────────────────────────────
function newChat() {
  // Save current session if it has messages
  if (sessionHistory.length > 0 && currentSessionId) {
    const existing = chatSessions.find(s => s.id === currentSessionId);
    if (existing) existing.history = [...sessionHistory];
  }

  // Start fresh
  currentSessionId = 'session-' + Date.now();
  sessionHistory   = [];
  messagesList.innerHTML = '';
  showEmptyState();
  userInput.value = '';
  autoResize(userInput);

  // Add to sidebar history
  const item = document.createElement('div');
  item.className = 'history-item active';
  item.textContent = 'New conversation';
  item.dataset.id  = currentSessionId;
  item.onclick = () => switchSession(currentSessionId);

  // Deactivate others
  chatHistory.querySelectorAll('.history-item').forEach(i => i.classList.remove('active'));
  chatHistory.prepend(item);

  chatSessions.unshift({ id: currentSessionId, preview: 'New conversation', history: [] });
  userInput.focus();
}

function switchSession(id) {
  const session = chatSessions.find(s => s.id === id);
  if (!session) return;

  currentSessionId = id;
  sessionHistory   = [...session.history];
  messagesList.innerHTML = '';

  if (sessionHistory.length === 0) {
    showEmptyState();
  } else {
    hideEmptyState();
    sessionHistory.forEach(m => appendMessage(m.role, m.content));
  }

  chatHistory.querySelectorAll('.history-item').forEach(i => {
    i.classList.toggle('active', i.dataset.id === id);
  });
}

function updateSessionPreview(text) {
  const preview = text.length > 32 ? text.slice(0, 32) + '…' : text;
  const item = chatHistory.querySelector(`[data-id="${currentSessionId}"]`);
  if (item) item.textContent = preview;

  const session = chatSessions.find(s => s.id === currentSessionId);
  if (session) session.preview = preview;
}

// ─── Helpers ──────────────────────────────────────────────────────
function showEmptyState() {
  emptyState.style.display = 'flex';
  messagesList.style.display = 'none';
}

function hideEmptyState() {
  emptyState.style.display = 'none';
  messagesList.style.display = 'flex';
}

function scrollToBottom() {
  messagesContainer.scrollTo({
    top: messagesContainer.scrollHeight,
    behavior: 'smooth',
  });
}

function now() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatText(text) {
  // Basic formatting — escape HTML then restore code blocks
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.15);padding:1px 5px;border-radius:4px;font-size:0.85em">$1</code>')
    .replace(/\n/g, '<br>');
}