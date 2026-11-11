"""Conversation dashboard: live view of user speech, LLM input, tool calls and latencies.

Served at http://127.0.0.1:8900 (DASHBOARD_PORT). Runs in a daemon thread with its own
event loop; events arrive from the pipecat loop via a thread-safe EventBus.

Endpoints: GET / (dashboard), GET /events (SSE), GET /api/events (JSON history).
"""
import asyncio
import json
import logging
import queue
import threading
from pathlib import Path

from agent.config import Config
from agent.observability import EventBus

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>voice-agent</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{font:14px/1.5 var(--x,system-ui,Segoe UI,sans-serif);margin:0;background:#0f1117;color:#e6e6e6;height:100vh;display:flex;flex-direction:column}
header{padding:10px 16px;background:#161a23;border-bottom:1px solid #2a2f3c;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header h1{font-size:15px;margin:0;color:#9ecbff}
.dot{width:8px;height:8px;border-radius:50%;background:#666;display:inline-block;margin-right:5px}
.dot.live{background:#5fbf6f}.dot.err{background:#e06c5f}
#status{font-size:12px;color:#8b93a7}
#tabs{display:none;margin-left:auto;gap:6px}
#tabs button{background:#1c2130;color:#aab;padding:4px 10px;border:1px solid #2a2f3c;border-radius:6px;font-size:12px}
#tabs button.on{color:#e6e6e6;border-color:#4a5a8a}
main{flex:1;display:flex;min-height:0;padding:10px;gap:10px}
.panel{background:#161a23;border:1px solid #2a2f3c;border-radius:10px;display:flex;flex-direction:column;min-height:0;flex:1}
.panel h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#79839a;margin:0;padding:8px 12px;border-bottom:1px solid #2a2f3c;display:flex;justify-content:space-between;align-items:center}
#conv{flex:1.6;min-width:0}#act{flex:1;min-width:0}
#conv.hide,#act.hide{display:none}
.scroll{flex:1;overflow-y:auto;padding:10px 12px;min-height:0}
.row{display:flex;margin:6px 0}
.row.me{justify-content:flex-end}.row.bot{justify-content:flex-start}
.bubble{max-width:78%;padding:8px 12px;border-radius:14px;white-space:pre-wrap;word-wrap:break-word}
.me .bubble{background:#2f5fa8;border-bottom-right-radius:4px}
.bot .bubble{background:#232937;border-bottom-left-radius:4px}
.bubble.err{background:#3a2026;color:#ff9a8a}
.bubble.ann{background:#3a3320;color:#f0c674}
.typing{color:#79839a;font-size:12px;padding:4px 12px}
.sys-note{font-size:11px;color:#6a7386;margin:4px 0}
/* activity */
.evt{background:#1c2130;border:1px solid #2a2f3c;border-radius:8px;margin:6px 0;font-size:12px;overflow:hidden}
.evt.root{background:#1d2330;border-color:#3a4258}
.evt.root .nm{color:#e6e6e6;font-weight:600;white-space:normal}
.evt.root .root-meta{padding:2px 9px 6px;color:#79839a;font-size:11px}
.evt .kids{margin:0 6px 6px 10px;border-left:2px solid #2a2f3c;padding-left:4px}
.evt.kid{margin:4px 0}
.evt .t{display:flex;gap:8px;align-items:center;padding:6px 9px}
.evt .ic{width:18px;text-align:center}
.evt .nm{color:#f0c674;font-weight:600}
.evt .sum{color:#c3c8d4;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.evt .tm{color:#6a7386;font-size:11px}
.chip{font-size:10px;padding:1px 7px;border-radius:8px;letter-spacing:.04em}
.chip.run{background:#2a3a55;color:#9ecbff}.chip.ok{background:#1f3a26;color:#8fd694}
.chip.fail{background:#3a2026;color:#ff9a8a}.chip.unk{background:#2a2f3c;color:#8b93a7}
.evt details{border-top:1px solid #2a2f3c}
.evt summary{cursor:pointer;padding:4px 9px;color:#79839a;font-size:11px;list-style:none}
.evt pre{margin:0;padding:4px 9px 8px;white-space:pre-wrap;word-break:break-word;color:#aab;font-size:11px;max-height:200px;overflow:auto}
.time{font-size:10px;color:#5a6070;text-align:center;margin:8px 0}
.errline{color:#ff9a8a;font-size:12px;padding:2px 9px}
/* composer */
.composer{display:flex;gap:8px;padding:8px 10px;border-top:1px solid #2a2f3c}
.composer input{flex:1;background:#12151d;border:1px solid #2a2f3c;color:#e6e6e6;border-radius:8px;padding:8px 10px;font:inherit}
.composer button{background:#2f5fa8;border:0;color:#fff;border-radius:8px;padding:8px 16px;font:inherit}
.composer button:disabled{opacity:.5}
/* music player */
#music .scroll{display:flex;flex-direction:column;gap:10px;padding:14px}
#mArt{width:180px;height:180px;border-radius:12px;background:#1c2130;border:1px solid #2a2f3c;object-fit:cover;align-self:center}
#mTitle{font-size:16px;font-weight:600;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#mArtists,#mAlbum{color:#8b93a7;font-size:13px;text-align:center;margin:0}
#mProg{width:100%}
#mSeek{width:100%;accent-color:#4a7dd6}
#mTimes{display:flex;justify-content:space-between;color:#6a7386;font-size:11px}
#mCtl{display:flex;gap:16px;align-items:center;justify-content:center}
#mCtl button{background:#1c2130;border:1px solid #2a2f3c;color:#e6e6e6;border-radius:50%;width:44px;height:44px;font-size:15px;cursor:pointer}
#mCtl #mPlay{background:#2f5fa8;border-color:#2f5fa8;width:56px;height:56px;font-size:20px}
#mVol{display:flex;gap:8px;align-items:center;color:#8b93a7}
#mVol input{flex:1;accent-color:#4a7dd6}
#mDev{background:#12151d;color:#e6e6e6;border:1px solid #2a2f3c;border-radius:8px;padding:8px;font:inherit}
#mSearch{display:flex;gap:6px}
#mSearch input{flex:1;background:#12151d;border:1px solid #2a2f3c;color:#e6e6e6;border-radius:8px;padding:8px 10px;font:inherit}
#mSearch select{background:#12151d;color:#e6e6e6;border:1px solid #2a2f3c;border-radius:8px;font:inherit}
#mSearch button{background:#2f5fa8;border:0;color:#fff;border-radius:8px;padding:8px 14px;cursor:pointer}
#mMsg{color:#8fd694;font-size:12px;min-height:16px;text-align:center;word-break:break-word}
#mErr{min-height:0}
@media (max-width:900px){main{flex-direction:column}#tabs{display:flex}#conv.hide,#act.hide{display:none}#conv,#act{flex:1}}
</style></head>
<body>
<header><h1>voice-agent</h1>
 <span id="status"><span class="dot" id="dot"></span>connecting…</span>
 <div id="tabs"><button data-p="conv" class="on">Chat</button><button data-p="act">Activity</button><button data-p="music">Music</button></div>
</header>
<main>
 <section class="panel" id="conv"><h2>Conversation <span id="agentState"></span></h2>
  <div class="scroll" id="feed"></div>
  <div class="composer"><input id="inp" placeholder="Type a message (voice mode: just speak)"><button id="send">Send</button></div>
 </section>
 <section class="panel hide-on-narrow" id="act"><h2>Agent activity</h2><div class="scroll" id="feed2"></div></section>
 <section class="panel" id="music"><h2>Music player</h2>
  <div class="scroll">
   <div class="errline" id="mErr"></div>
   <img id="mArt" alt="album art">
   <div id="mTitle">Nothing playing</div>
   <div id="mArtists"></div>
   <div id="mAlbum"></div>
   <div id="mProg">
    <input id="mSeek" type="range" min="0" max="1000" value="0" disabled>
    <div id="mTimes"><span id="mPos">0:00</span><span id="mDur">0:00</span></div>
   </div>
   <div id="mCtl">
    <button id="mPrev" title="previous">⏮</button>
    <button id="mPlay" title="play or pause">▶</button>
    <button id="mNext" title="next">⏭</button>
   </div>
   <div id="mVol"><span>🔊</span><input id="mVolR" type="range" min="0" max="100" value="50"></div>
   <select id="mDev"><option value="">device…</option></select>
   <div id="mSearch">
    <input id="mQ" placeholder="search and play…">
    <select id="mKind"><option value="track">song</option><option value="album">album</option><option value="artist">artist</option><option value="playlist">playlist</option></select>
    <button id="mGo" title="play">▶</button>
   </div>
   <div id="mMsg"></div>
  </div>
 </section>
</main>
<script>
const feed=document.getElementById('feed'),feed2=document.getElementById('feed2');
const dot=document.getElementById('dot'),statusEl=document.getElementById('status');
const es=new EventSource('/events');
es.onopen=()=>{dot.className='dot live';statusEl.innerHTML='<span class=\"dot live\"></span>live';};
es.onerror=()=>{dot.className='dot err';statusEl.innerHTML='<span class=\"dot err\"></span>reconnecting…';};
/* ---- conversation ---- */
const seen=new Set();let lastUserKey='';let typingEl=null;
function now(){return new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'});}
function ts(){let d=document.createElement('div');d.className='time';d.textContent=now();feed.appendChild(d);}
function bubble(cls,text,key){key=key||cls+':'+text;if(seen.has(key))return null;seen.add(key);
 if(typingEl){typingEl.remove();typingEl=null;}
 const r=document.createElement('div');r.className='row '+cls;
 const b=document.createElement('div');b.className='bubble'+(cls==='bot'||cls==='me'?'':' '+cls);
 b.textContent=text;r.appendChild(b);feed.appendChild(r);feed.scrollTop=feed.scrollHeight;return b;}
function setTyping(on){if(on&&!typingEl){typingEl=document.createElement('div');typingEl.className='typing';
 typingEl.textContent='agent is working…';feed.appendChild(typingEl);}
 if(!on&&typingEl){typingEl.remove();typingEl=null;}}
/* ---- activity ---- */
const evts=new Map();let evtSeq=0;
function redact(s){return String(s).replace(/(sk-[A-Za-z0-9_-]{8})[\\w-]*/g,'$1…').replace(/(gh[pousr]_[A-Za-z0-9]{6})[\\w]*/g,'$1…').replace(/((?:api[_-]?key|token|password|secret)[\"':=\\s]+)[^\\s,\"']+/gi,'$1***');}
function meta(n){
 const m={search_web:['🔎','web search'],get_weather:['⛅','weather'],get_time:['🕐','time'],
 create_file:['📄','create file'],edit_file:['✏️','edit file'],read_file:['📖','read file'],
 run_command:['▶️','command'],submit_task:['🗂️','background task'],task_status:['🗂️','task status'],
 task_cancel:['🗑️','task cancel'],build_website:['🌐','build website']};
 return m[n]||['🔧','tool: '+n];}
function msT(v){return v==null?'—':(v>=1000?(v/1000).toFixed(2)+'s':Math.round(v)+'ms');}
function tstamp(ev){if(ev.t_ms!=null)return new Date(ev.t_ms).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit',fractionalSecondDigits:3});return now();}
function evtCard(ev,summary,ok){
 const id=ev.id||('t'+(evtSeq++));
 let card=evts.get(id);
 if(!card){card=document.createElement('div');card.className='evt kid';evts.set(id,card);
  const [ic,label]=meta(ev.name||'');
  const tm=ev.t_ms?new Date(ev.t_ms).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit',fractionalSecondDigits:3}):now();
  card.innerHTML='<div class="t"><span class="ic">'+ic+'</span><span class="nm">'+label+'</span>'+
   '<span class="sum"></span><span class="tm">'+tm+'</span><span class="chip run">RUNNING</span></div>'+
   '<details><summary>details</summary><pre></pre></details>';
  card.dataset.sum=summary||'';
  (window._curKids||feed2).appendChild(card);
  card.querySelector('.sum').textContent=((ev.elapsed_ms!=null?'[+'+ev.elapsed_ms+'ms] ':'')+(summary||'')).slice(0,90);}
 const chip=card.querySelector('.chip');
 if(ok===true){const d=ev.duration?msT(ev.duration*1000):'';chip.className='chip ok';chip.textContent='COMPLETED'+(d?' · '+d:'');}
 else if(ok===false){chip.className='chip fail';chip.textContent='FAILED';}
 return card.querySelector('pre');}
function toolDetails(ev,ok){
 const lines=[];
 if(ev.id)lines.push('call_id: '+ev.id);
 if(ev.start_ms)lines.push('started: '+new Date(ev.start_ms).toLocaleTimeString([],{fractionalSecondDigits:3}));
 if(ev.t_ms&&ev.start_ms)lines.push('ended:   '+new Date(ev.t_ms).toLocaleTimeString([],{fractionalSecondDigits:3}));
 if(ev.duration!=null)lines.push('duration: '+msT(ev.duration*1000));
 lines.push('status: '+(ok===true?'completed':(ok===false?'failed':'running')));
 if(ev.arguments)lines.push('args: '+ev.arguments);
 if(ev.result)lines.push('result:\\n'+ev.result);
 return redact(lines.join('\\n'));}
function kid(root,icon,label,detail,chipHtml,expandText){
 const d=document.createElement('div');d.className='evt kid';
 let inner='<div class="t"><span class="ic">'+icon+'</span><span class="nm">'+label+'</span>'+
  '<span class="sum"></span><span class="tm">'+now()+'</span>'+(chipHtml||'')+'</div>';
 if(expandText)inner+='<details><summary>details</summary><pre></pre></details>';
 d.innerHTML=inner;
 d.querySelector('.sum').textContent=detail||'';
 if(expandText)d.querySelector('pre').textContent=redact(expandText);
 root.querySelector('.kids').appendChild(d);feed2.scrollTop=feed2.scrollHeight;return d;}
function turnRoot(ev,text){
 const card=document.createElement('div');card.className='evt root';
 const tm=tstamp(ev);
 card.innerHTML='<div class="t"><span class="ic">🗣️</span><span class="nm root-text"></span>'+
  '<span class="tm">'+tm+'</span><span class="chip run" id="tot-'+ev.turn+'">IN PROGRESS</span></div>'+
  '<div class="root-meta"></div><div class="kids"></div>';
 card.querySelector('.root-text').textContent=text;
 feed2.appendChild(card);feed2.scrollTop=feed2.scrollHeight;return card;}
function evtInfo(ev,ok){const pre=evtCard(ev,ev.arguments||'',ok);pre.textContent=toolDetails(ev,ok);feed2.scrollTop=feed2.scrollHeight;}
/* tabs (mobile) */
const tabs=document.getElementById('tabs');
const PANELS=['conv','act','music'];
function showOnlyMobile(p){
 PANELS.forEach(id=>{document.getElementById(id).style.display=id===p?'flex':'none';});
 tabs.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.p===p));}
tabs.querySelectorAll('button').forEach(b=>b.onclick=()=>{showOnlyMobile(b.dataset.p);
 if(b.dataset.p==='music'){mPoll();mLoadDevices();}});
if(innerWidth>900){document.getElementById('act').style.display='flex';}
else{showOnlyMobile('conv');}
/* agent state */
function state(s){document.getElementById('agentState').textContent=s?'· '+s:'';}
/* ---- events ---- */
window._curKids=null;window._roots=new Map();window._lastLlmReq={};window._speech={};
es.onmessage=(e)=>{const ev=JSON.parse(e.data);
 const rootOf=(id,txt)=>window._roots.get(id)||(txt!=null?(()=>{const r=turnRoot({turn:id,t_ms:ev.t_ms},txt);window._roots.set(id,r);window._curKids=r.querySelector('.kids');return r;})():window._curKids);
 switch(ev.type){
  case 'speech_start':break;
  case 'speech_end':window._speech[ev.turn]={speech_ms:ev.speech_ms,t_ms:ev.t_ms};break;
  case 'transcript':{ts();
   // A new user utterance closes any still-open reply bubble from a previous
   // turn, so the next reply always starts a fresh bubble.
   if(window._live){window._live.removeAttribute('id');window._live=null;}
   // Deepgram endpointing / turn analysis can split ONE spoken utterance into
   // several final transcripts -> several bubbles. Merge consecutive user
   // transcripts into one bubble when the agent hasn't replied in between.
   const last=window._lastMe;
   const merge=last&&!window._botSpoke&&last.el&&document.body.contains(last.el)
     &&ev.turn!==last.turn&&(ev.t_ms-last.t_ms)<3000;
   const key='u:'+ev.turn+':'+ev.text;
   let el;
   if(merge){last.el.textContent=(last.el.textContent?last.el.textContent+' ':'')+ev.text;
    seen.add(key);window._lastMe={turn:last.turn,t_ms:ev.t_ms,el:last.el};}
   else{el=bubble('me',ev.text,key);window._lastMe={turn:ev.turn,t_ms:ev.t_ms,el:el};}
   window._botSpoke=false;
   state('listening done');
   const r=turnRoot(ev,ev.text);window._roots.set(ev.turn,r);window._curKids=r.querySelector('.kids');
   const sp=window._speech[ev.turn];
   if(sp&&sp.speech_ms!=null){const stt=ev.t_ms-sp.t_ms;
    r.querySelector('.root-meta').textContent='speech '+msT(sp.speech_ms)+' · STT '+msT(stt);}
   break;}
  case 'llm_request':window._lastLlmReq[ev.turn]=ev.t_ms;break;
  case 'first_token':{const t0=window._lastLlmReq[ev.turn];
   const think=t0&&ev.t_ms-t0>0?msT(ev.t_ms-t0):null;
   kid(rootOf(ev.turn),'🧠','LLM think',think,
    null,think?('model: '+(ev.model||'?')+'\\nthink time: '+think):null);break;}
  case 'llm_delta':setTyping(false);state('replying');
   if(!window._live||window._liveTurn!==ev.turn){const live=document.createElement('div');live.className='row bot';
    live.innerHTML='<div class=\"bubble\" id=\"liveb\"></div>';feed.appendChild(live);window._live=live.querySelector('#liveb');window._liveTurn=ev.turn;}
   window._live.textContent+=ev.delta;feed.scrollTop=feed.scrollHeight;break;
  case 'bot_speech_start':
   window._botSpoke=true;
   if(window._live){window._live.removeAttribute('id');window._live=null;}
   {const r=rootOf(ev.turn);
    const lts=ev.latencies&&ev.latencies.llm_request_to_bot_speech!=null?msT(ev.latencies.llm_request_to_bot_speech*1000):null;
    kid(r,'🔊','TTS response',lts?lts+' after last LLM call':'',
     null,'model: '+(ev.model||'?')+(lts?'\\nLLM→TTS: '+lts:''));
    const tot=document.getElementById('tot-'+ev.turn);
    if(tot&&ev.latencies&&ev.latencies.speech_end_to_bot_speech!=null){
     tot.className='chip ok';tot.textContent='TOTAL '+msT(ev.latencies.speech_end_to_bot_speech*1000);}
    state('');}
   break;
  case 'turn_end':{const r=window._roots.get(ev.turn);
   if(r){const tot=document.getElementById('tot-'+ev.turn);
    if(tot&&tot.classList.contains('run')){tot.className='chip unk';tot.textContent='NO RESPONSE';}
    window._curKids=null;}
   if(window._live){window._live.removeAttribute('id');window._live=null;}break;}
  case 'tool_running':setTyping(true);state('running '+ev.name);evtInfo(ev,null);break;
  case 'tool_result':evtInfo(ev,ev.ok);break;
  case 'tool_calls_started':break;
  case 'announcement':{const r=rootOf(ev.turn,ev.text);
   kid(r,'🔊','announcement',ev.text.slice(0,60),null,'spoken announcement\\ntext: '+ev.text);
   ts();bubble('ann',ev.text,'a:'+ev.turn+':'+ev.text.slice(0,40));break;}
  case 'interruption':ts();bubble('bot err','[interrupted by user]','i:'+ev.turn);break;
  case 'error':{const pre=evtCard({name:'error',id:'err'+ev.turn},ev.error,false);pre.textContent=redact(ev.error);break;}
 }};
/* composer (text mode convenience; voice mode answers spoken input) */
const inp=document.getElementById('inp');
async function send(){const text=inp.value.trim();if(!text)return;inp.value='';
 document.getElementById('send').disabled=true;
 try{const r=await fetch('/say',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
  if(!r.ok)console.warn('say failed',r.status);}catch(e){console.warn(e);}
 document.getElementById('send').disabled=false;}
document.getElementById('send').onclick=send;inp.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
/* ---- music player ---- */
const $m=id=>document.getElementById(id);
let mSeeking=false,mVolDrag=false,mDurMs=0;
function fmtMs(v){v=Math.round(v/1000);if(!isFinite(v))v=0;return Math.floor(v/60)+':'+String(v%60).padStart(2,'0');}
async function mApi(path,body){
 try{const r=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
  return await r.json();}catch(err){return{ok:false,error:String(err)};}}
function mMsg(t,bad){$m('mMsg').textContent=t||'';$m('mMsg').style.color=bad?'#ff9a8a':'#8fd694';}
function mApply(p){
 const empty=!p.title;
 $m('mTitle').textContent=empty?'Nothing playing':p.title;
 $m('mArtists').textContent=empty?'':(p.artists||'');
 $m('mAlbum').textContent=empty?'':(p.album||'');
 if(p.art)$m('mArt').src=p.art;
 $m('mPlay').textContent=p.playing?'⏸':'▶';
 mDurMs=p.duration_ms||0;
 const pos=p.progress_ms||0;
 if(!mSeeking){$m('mSeek').disabled=!mDurMs;$m('mSeek').value=mDurMs?Math.round(pos/mDurMs*1000):0;}
 $m('mPos').textContent=fmtMs(pos);$m('mDur').textContent=fmtMs(mDurMs);
 if(!mVolDrag&&p.volume!=null)$m('mVolR').value=p.volume;}
async function mPoll(){
 if($m('music').style.display==='none')return;
 const d=await mApi('/api/now_playing');
 if(!d.ok){$m('mErr').textContent='player: '+(d.error||'unavailable');return;}
 $m('mErr').textContent='';
 mApply(d.playing||{});}
async function mAct(action){mMsg('');
 const d=await mApi('/api/player',{action:action});
 mMsg(d.ok?(d.message||''):d.error,!d.ok);setTimeout(mPoll,900);}
$m('mPlay').onclick=()=>{mAct($m('mPlay').textContent==='⏸'?'pause':'resume');};
$m('mNext').onclick=()=>mAct('next');
$m('mPrev').onclick=()=>mAct('previous');
$m('mSeek').addEventListener('input',()=>{mSeeking=true;
 $m('mPos').textContent=fmtMs(mDurMs*$m('mSeek').value/1000);});
$m('mSeek').addEventListener('change',async()=>{mSeeking=false;
 const d=await mApi('/api/seek',{position_ms:Math.round(mDurMs*$m('mSeek').value/1000)});
 mMsg(d.ok?'':d.error,!d.ok);setTimeout(mPoll,700);});
$m('mVolR').addEventListener('input',()=>{mVolDrag=true;});
$m('mVolR').addEventListener('change',async()=>{mVolDrag=false;
 const d=await mApi('/api/volume',{level:+$m('mVolR').value});
 mMsg(d.ok?(d.message||''):d.error,!d.ok);});
async function mLoadDevices(){
 const d=await mApi('/api/devices');
 if(!d.ok)return;
 const dd=$m('mDev');dd.innerHTML='';
 if(!d.devices.length){const o=document.createElement('option');o.value='';o.text='no devices';dd.appendChild(o);return;}
 d.devices.forEach(x=>{const o=document.createElement('option');o.value=x.id;
  o.text=x.name+(x.active?' · active':'');dd.appendChild(o);});}
async function mPlayQuery(){
 const q=$m('mQ').value.trim();if(!q)return;mMsg('');
 const d=await mApi('/api/play',{query:q,kind:$m('mKind').value});
 mMsg(d.ok?(d.message||'Now playing.'):d.error,!d.ok);setTimeout(mPoll,1500);}
$m('mGo').onclick=mPlayQuery;
$m('mQ').addEventListener('keydown',e=>{if(e.key==='Enter')mPlayQuery();});
$m('mDev').addEventListener('change',async()=>{
 const id=$m('mDev').value;if(!id)return;
 const d=await mApi('/api/transfer',{device_id:id});
 mMsg(d.ok?'Playback moved.':d.error,!d.ok);setTimeout(mPoll,1200);});
setInterval(mPoll,3000);
mPoll();mLoadDevices();
</script></body></html>"""


class DashboardServer:
    def __init__(self, bus: EventBus, port: int = None, say_handler=None):
        self.bus = bus
        self.say_handler = say_handler  # callable(text) -> async, injects user text into the agent
        self.port = port or int(Config.DASHBOARD_PORT or 8900)
        self._thread: threading.Thread = None

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="dashboard")
        self._thread.start()

    def _run(self) -> None:
        try:
            from starlette.applications import Starlette
            from starlette.responses import HTMLResponse, StreamingResponse
            from starlette.routing import Route
            import uvicorn

            async def index(request):
                return HTMLResponse(HTML, headers={"Cache-Control": "no-store"})

            async def events(request):
                q = self.bus.subscribe()

                async def gen():
                    try:
                        while True:
                            try:
                                ev = q.get_nowait()
                                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                            except queue.Empty:
                                await asyncio.sleep(0.08)
                    finally:
                        self.bus.unsubscribe(q)

                return StreamingResponse(gen(), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache"})

            async def history(request):
                return StreamingResponse(
                    json.dumps({"events": list(self.bus._history)}, ensure_ascii=False),
                    media_type="application/json")

            from starlette.responses import JSONResponse

            async def say(request):
                if not self.say_handler:
                    return JSONResponse({"error": "text input not available"}, status_code=400)
                try:
                    body = await request.json()
                except ValueError:
                    return JSONResponse({"error": "invalid json"}, status_code=400)
                text = str(body.get("text", "")).strip()[:2000]
                if not text:
                    return JSONResponse({"error": "empty text"}, status_code=400)
                try:
                    await self.say_handler(text)
                except Exception as e:
                    return JSONResponse({"error": str(e)}, status_code=500)
                return JSONResponse({"ok": True})

            # ---- music player API (wraps services.spotify_client) ----
            from services import spotify_client as spc

            async def api_now_playing(request):
                try:
                    data = await asyncio.to_thread(spc.now_playing_raw)
                    return JSONResponse({"ok": True, "playing": data})
                except Exception as e:
                    return JSONResponse({"ok": False, "error": str(e)})

            async def api_player(request):
                try:
                    body = await request.json()
                except ValueError:
                    return JSONResponse({"ok": False, "error": "invalid json"}, status_code=400)
                action = str(body.get("action", ""))
                device = str(body.get("device", "") or "")
                try:
                    if action == "pause":
                        msg = await asyncio.to_thread(spc.pause)
                    elif action == "resume":
                        msg = await asyncio.to_thread(spc.resume, device or None)
                    elif action == "next":
                        msg = await asyncio.to_thread(spc.next_track)
                    elif action == "previous":
                        msg = await asyncio.to_thread(spc.previous_track)
                    else:
                        return JSONResponse({"ok": False, "error": f"unknown action {action!r}"}, status_code=400)
                    return JSONResponse({"ok": True, "message": msg})
                except Exception as e:
                    return JSONResponse({"ok": False, "error": str(e)})

            async def api_volume(request):
                try:
                    body = await request.json()
                    level = int(body.get("level", 50))
                    msg = await asyncio.to_thread(spc.set_volume, level)
                    return JSONResponse({"ok": True, "message": msg})
                except Exception as e:
                    return JSONResponse({"ok": False, "error": str(e)})

            async def api_seek(request):
                try:
                    body = await request.json()
                    pos = int(body.get("position_ms", 0))
                    device = str(body.get("device", "") or "")
                    msg = await asyncio.to_thread(spc.seek, pos, device or None)
                    return JSONResponse({"ok": True, "message": msg})
                except Exception as e:
                    return JSONResponse({"ok": False, "error": str(e)})

            async def api_play(request):
                try:
                    body = await request.json()
                    query = str(body.get("query", "")).strip()[:300]
                    kind = str(body.get("kind", "track") or "track")
                    device = str(body.get("device", "") or "")
                    if not query:
                        return JSONResponse({"ok": False, "error": "empty query"}, status_code=400)
                    if kind in ("album", "artist", "playlist"):
                        msg = await asyncio.to_thread(spc.play_context, query, kind, device or None)
                    else:
                        msg = await asyncio.to_thread(spc.search_and_play, query, device or None)
                    return JSONResponse({"ok": True, "message": msg})
                except Exception as e:
                    return JSONResponse({"ok": False, "error": str(e)})

            async def api_devices(request):
                try:
                    devs = await asyncio.to_thread(spc.list_devices_raw)
                    return JSONResponse({"ok": True, "devices": devs})
                except Exception as e:
                    return JSONResponse({"ok": False, "error": str(e)})

            async def api_transfer(request):
                try:
                    body = await request.json()
                    dev_id = str(body.get("device_id", ""))
                    if not dev_id:
                        return JSONResponse({"ok": False, "error": "missing device_id"}, status_code=400)
                    msg = await asyncio.to_thread(spc.transfer, dev_id)
                    return JSONResponse({"ok": True, "message": msg})
                except Exception as e:
                    return JSONResponse({"ok": False, "error": str(e)})

            app = Starlette(routes=[
                Route("/", index),
                Route("/events", events),
                Route("/api/events", history),
                Route("/say", say, methods=["POST"]),
                Route("/api/now_playing", api_now_playing),
                Route("/api/player", api_player, methods=["POST"]),
                Route("/api/volume", api_volume, methods=["POST"]),
                Route("/api/seek", api_seek, methods=["POST"]),
                Route("/api/play", api_play, methods=["POST"]),
                Route("/api/devices", api_devices),
                Route("/api/transfer", api_transfer, methods=["POST"]),
            ])
            config = uvicorn.Config(app, host=Config.DASHBOARD_HOST, port=self.port,
                                    log_level="warning", loop="asyncio")
            uvicorn.Server(config).run()
        except Exception:
            import logging
            logging.getLogger("dashboard").exception("dashboard server failed")

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"