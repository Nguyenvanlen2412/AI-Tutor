const API = '';
let currentSessionId = null;
let currentSessionName = null;
let mediaRecorder = null;
let audioChunks = [];
let recInterval = null;
let recSeconds = 0;
let isBusy = false;

const $ = id => document.getElementById(id);
const qs = s => document.querySelector(s);

/* ── Helpers ───────────────────────────────────────────────── */
function toast(msg, dur=2500){
  const el=$('toast'); el.textContent=msg; el.classList.add('show');
  setTimeout(()=>el.classList.remove('show'), dur);
}
function setStatus(s){ 
  const pill = $('status-pill');
  pill.textContent=s;
  if(s === 'Ready') {
    pill.style.borderColor = 'var(--border-2)';
    pill.style.color = 'var(--text-2)';
  } else {
    pill.style.borderColor = 'var(--accent-mid)';
    pill.style.color = 'var(--accent)';
  }
}
function fmtTime(d){
  const dt=new Date(d*1000);
  const now=new Date();
  const diff=(now-dt)/1000;
  if(diff<60) return 'just now';
  if(diff<3600) return Math.floor(diff/60)+'m ago';
  if(diff<86400) return Math.floor(diff/3600)+'h ago';
  return dt.toLocaleDateString('en-US',{month:'short',day:'numeric'});
}
function autoGrow(el){
  el.style.height='auto';
  el.style.height=Math.min(el.scrollHeight,150)+'px';
}
function scrollToBottom(){
  const m=$('messages');
  m.scrollTop=m.scrollHeight;
}

/* ── Session list ──────────────────────────────────────────── */
async function loadSessions(){
  try{
    const r=await fetch(API+'/api/sessions?limit=30');
    const d=await r.json();
    renderSessionList(d.sessions||[]);
  }catch(e){console.warn('Sessions load failed',e)}
}

function renderSessionList(sessions){
  const list=$('sessions-list');
  list.innerHTML='';
  if(!sessions.length){
    list.innerHTML='<p style="font-size:13px;color:var(--text-3);padding:16px 14px;font-family:\'Inter\',sans-serif">No sessions yet</p>';
    return;
  }
  sessions.forEach(s=>{
    const div=document.createElement('div');
    div.className='session-item'+(s.session_id===currentSessionId?' active':'');
    div.dataset.id=s.session_id;
    div.innerHTML=`
      <div class="session-name">${escHtml(s.name)}</div>
      <div class="session-meta">${fmtTime(s.updated_at)}</div>
      <button class="session-del" data-id="${s.session_id}" title="Delete session">
        <svg viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>
      </button>`;
    div.addEventListener('click', e=>{
      if(!e.target.closest('.session-del')) loadSession(s.session_id, s.name);
    });
    div.querySelector('.session-del').addEventListener('click', e=>{
      e.stopPropagation(); deleteSession(s.session_id);
    });
    list.appendChild(div);
  });
}

function escHtml(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ── Create session ────────────────────────────────────────── */
async function createSession(name){
  try{
    const r=await fetch(API+'/api/sessions',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:name||null,user_id:'user'})
    });
    const d=await r.json();
    await loadSessions();
    loadSession(d.session.session_id, d.session.name);
  }catch(e){ toast('Failed to create session'); console.error(e); }
}

/* ── Load existing session ─────────────────────────────────── */
async function loadSession(sid, name){
  currentSessionId=sid;
  currentSessionName=name;
  $('chat-title').textContent=name;
  $('chat-session-id').textContent=sid.slice(0,18)+'…';
  $('no-session').style.display='none';
  $('chat-view').style.display='flex';
  clearMessages();
  setStatus('Loading…');

  try{
    const r=await fetch(API+'/api/sessions/'+sid);
    const d=await r.json();
    const history=d.history||[];
    if(history.length){
      $('empty-state').style.display='none';
      // Replay history pairs
      for(let i=0;i<history.length;i+=2){
        const u=history[i], a=history[i+1];
        if(u) appendMessage('user', u.content);
        if(a) appendMessage('ai', a.content, null, null, null, []);
      }
    }
    setStatus('Ready');
    updateActiveSession();
    scrollToBottom();
  }catch(e){
    setStatus('Ready');
    updateActiveSession();
  }
}

function updateActiveSession(){
  document.querySelectorAll('.session-item').forEach(el=>{
    el.classList.toggle('active', el.dataset.id===currentSessionId);
  });
}

/* ── Delete session ────────────────────────────────────────── */
async function deleteSession(sid){
  if(!confirm('Delete this session?')) return;
  try{
    await fetch(API+'/api/sessions/'+sid,{method:'DELETE'});
    if(sid===currentSessionId){
      currentSessionId=null;
      $('no-session').style.display='flex';
      $('chat-view').style.display='none';
    }
    await loadSessions();
    toast('Session deleted');
  }catch(e){ toast('Delete failed'); }
}

/* ── Message rendering ─────────────────────────────────────── */
function clearMessages(){
  const m=$('messages');
  m.innerHTML='<div id="empty-state">' +
    '<div class="empty-icon"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div>' +
    '<h2>Start the conversation</h2>' +
    '<p>Type a question or use the microphone. I\'ll respond with text and audio.</p>' +
    '</div>';
}

function hideEmpty(){
  const e=$('empty-state');
  if(e) e.style.display='none';
}

function appendMessage(role, text, audioB64=null, sources=[], cacheHit=false, latencyMs=null){
  hideEmpty();
  const m=$('messages');
  const div=document.createElement('div');
  div.className='msg '+role;
  const now=new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'});

  let html=`<div class="msg-header">
    <span class="msg-role">${role==='user'?'You':'AI Tutor'}</span>
    <span style="color:var(--text-3);">${now}</span>
    ${cacheHit?'<span class="badge-cache">cache hit</span>':''}
    ${latencyMs!==null?`<span class="badge-latency">${latencyMs}ms</span>`:''}
  </div>`;

  html+=`<div class="bubble">${formatContent(text)}</div>`;

  if(audioB64){
    html+=buildAudioPlayer(audioB64);
  }
  if(sources&&sources.length){
    html+=`<div class="sources">${sources.map(s=>`<span class="source-tag">📄 ${escHtml(s)}</span>`).join('')}</div>`;
  }

  div.innerHTML=html;
  m.appendChild(div);

  if(audioB64){
    initAudioPlayer(div.querySelector('.audio-player'), audioB64);
  }

  scrollToBottom();
  return div;
}

function formatContent(text){
  let html=escHtml(text);
  html=html.replace(/```([\s\S]*?)```/g,'<pre><code>$1</code></pre>');
  html=html.replace(/`([^`]+)`/g,'<code>$1</code>');
  html=html.replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>');
  html=html.replace(/\*(.*?)\*/g,'<em>$1</em>');
  html=html.replace(/\n/g,'<br>');
  return `<p>${html}</p>`;
}

/* ── Audio player ──────────────────────────────────────────── */
function buildAudioPlayer(b64){
  const bars=Array.from({length:24},(_,i)=>{
    const h=6+Math.floor(Math.random()*18);
    return `<div class="wbar" style="height:${h}px"></div>`;
  }).join('');
  return `<div class="audio-player">
    <button class="play-btn" title="Play response">
      <svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
    </button>
    <div class="audio-info">
      <div class="audio-waveform">${bars}</div>
      <div class="audio-time">Audio response</div>
    </div>
  </div>`;
}

function initAudioPlayer(playerEl, b64){
  if(!playerEl) return;
  const btn=playerEl.querySelector('.play-btn');
  const timeEl=playerEl.querySelector('.audio-time');
  const bars=playerEl.querySelectorAll('.wbar');

  const bytes=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));
  const blob=new Blob([bytes],{type:'audio/wav'});
  const url=URL.createObjectURL(blob);
  const audio=new Audio(url);

  let playing=false;
  let animFrame;

  function animateBars(on){
    cancelAnimationFrame(animFrame);
    if(!on){bars.forEach(b=>b.style.background='var(--text-3)');return;}
    function frame(){
      bars.forEach(b=>{
        const h=6+Math.floor(Math.random()*18);
        b.style.height=h+'px';
        b.style.background='var(--accent)';
      });
      animFrame=requestAnimationFrame(frame);
    }
    frame();
  }

  function fmtDur(s){
    if(isNaN(s)) return '0:00';
    return Math.floor(s/60)+':'+(Math.floor(s%60)+'').padStart(2,'0');
  }

  audio.addEventListener('loadedmetadata',()=>{
    timeEl.textContent=fmtDur(audio.duration);
  });
  audio.addEventListener('timeupdate',()=>{
    timeEl.textContent=fmtDur(audio.currentTime)+' / '+fmtDur(audio.duration);
  });
  audio.addEventListener('ended',()=>{
    playing=false;
    btn.innerHTML=`<svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
    animateBars(false);
    timeEl.textContent=fmtDur(audio.duration);
  });

  btn.addEventListener('click',()=>{
    if(!playing){
      audio.play();
      playing=true;
      btn.innerHTML=`<svg viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/></svg>`;
      animateBars(true);
    } else {
      audio.pause();
      playing=false;
      btn.innerHTML=`<svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
      animateBars(false);
    }
  });

  // Auto-play
  setTimeout(()=>btn.click(), 300);
}

/* ── Thinking bubble ───────────────────────────────────────── */
function showThinking(){
  hideEmpty();
  const m=$('messages');
  const div=document.createElement('div');
  div.id='thinking-bubble';
  div.className='msg ai';
  div.innerHTML=`<div class="msg-header"><span class="msg-role">AI Tutor</span></div>
    <div class="bubble thinking">
      <div class="dot-flashing"></div>
      <div class="dot-flashing"></div>
      <div class="dot-flashing"></div>
    </div>`;
  m.appendChild(div);
  scrollToBottom();
}
function hideThinking(){
  const t=$('thinking-bubble');
  if(t) t.remove();
}

/* ── Send text ─────────────────────────────────────────────── */
async function sendText(){
  if(isBusy||!currentSessionId) return;
  const input=$('text-input');
  const text=input.value.trim();
  if(!text) return;
  input.value='';
  input.style.height='auto';
  isBusy=true; $('btn-send').disabled=true; setStatus('Thinking…');

  appendMessage('user', text);
  showThinking();

  try{
    const fd=new FormData();
    fd.append('session_id', currentSessionId);
    fd.append('user_id','user');
    fd.append('text', text);

    const r=await fetch(API+'/api/chat',{method:'POST',body:fd});
    const d=await r.json();
    hideThinking();

    if(d.error){ appendMessage('ai','⚠ '+d.error); }
    else {
      appendMessage('ai', d.response, d.audio_b64, d.sources, d.is_cache_hit, d.latency_ms);
      await loadSessions();
    }
  }catch(e){
    hideThinking();
    appendMessage('ai','⚠ Connection error. Is the server running?');
  }

  isBusy=false; $('btn-send').disabled=false; setStatus('Ready');
}

/* ── Voice recording ───────────────────────────────────────── */
async function toggleRecording(){
  if(mediaRecorder&&mediaRecorder.state==='recording'){
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording(){
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    audioChunks=[];
    mediaRecorder=new MediaRecorder(stream,{mimeType:'audio/webm'});
    mediaRecorder.ondataavailable=e=>{ if(e.data.size>0) audioChunks.push(e.data); };
    mediaRecorder.onstop=()=>{ handleRecordingStop(stream); };
    mediaRecorder.start(100);

    $('btn-mic').classList.add('recording');
    $('rec-indicator').classList.add('visible');
    recSeconds=0;
    updateRecTimer();
    recInterval=setInterval(()=>{ recSeconds++; updateRecTimer(); },1000);
    setStatus('Recording…');
  }catch(e){
    toast('Microphone access denied');
  }
}

function stopRecording(){
  if(mediaRecorder&&mediaRecorder.state==='recording'){
    mediaRecorder.stop();
  }
  $('btn-mic').classList.remove('recording');
  $('rec-indicator').classList.remove('visible');
  clearInterval(recInterval);
  setStatus('Processing audio…');
}

function updateRecTimer(){
  const m=Math.floor(recSeconds/60);
  const s=(recSeconds%60+'').padStart(2,'0');
  $('rec-timer').textContent=m+':'+s;
}

async function handleRecordingStop(stream){
  stream.getTracks().forEach(t=>t.stop());
  if(!audioChunks.length||!currentSessionId){ setStatus('Ready'); return; }

  const blob=new Blob(audioChunks,{type:'audio/webm'});
  isBusy=true; $('btn-send').disabled=true;

  appendMessage('user','🎤 Voice message');
  showThinking();

  try{
    const fd=new FormData();
    fd.append('session_id', currentSessionId);
    fd.append('user_id','user');
    fd.append('audio', blob, 'recording.webm');

    const r=await fetch(API+'/api/chat',{method:'POST',body:fd});
    const d=await r.json();
    hideThinking();

    // Replace placeholder with transcript if available
    if(d.user_query&&d.user_query!=='🎤 Voice message'){
      const msgs=$('messages').querySelectorAll('.msg.user');
      const last=msgs[msgs.length-1];
      if(last) last.querySelector('.bubble').innerHTML=formatContent(d.user_query);
    }

    if(d.error){ appendMessage('ai','⚠ '+d.error); }
    else {
      appendMessage('ai', d.response, d.audio_b64, d.sources, d.is_cache_hit, d.latency_ms);
      await loadSessions();
    }
  }catch(e){
    hideThinking();
    appendMessage('ai','⚠ Connection error or audio processing failed.');
  }

  isBusy=false; $('btn-send').disabled=false; setStatus('Ready');
}

/* ── Event listeners ───────────────────────────────────────── */
$('btn-new-session').addEventListener('click',()=>{
  const name=prompt('Session name (optional):','');
  createSession(name||null);
});
$('btn-start').addEventListener('click',()=>createSession(null));
$('btn-send').addEventListener('click', sendText);
$('btn-mic').addEventListener('click', toggleRecording);

$('text-input').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); sendText(); }
});
$('text-input').addEventListener('input',()=>autoGrow($('text-input')));

/* ── Init ──────────────────────────────────────────────────── */
loadSessions();