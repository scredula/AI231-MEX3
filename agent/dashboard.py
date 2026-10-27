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
@media (max-width:900px){main{flex-direction:column}#tabs{display:flex}#conv.hide,#act.hide{display:none}#conv,#act{flex:1}}
</style></head>
<body>
<header><h1>voice-agent</h1>
 <span id="status"><span class="dot" id="dot"></span>connecting…</span>
 <div id="tabs"><button data-p="conv" class="on">Chat</button><button data-p="act">Activity</button></div>
</header>
<main>
 <section class="panel" id="conv"><h2>Conversation <span id="agentState"></span></h2>
  <div class="scroll" id="feed"></div>
  <div class="composer"><input id="inp" placeholder="Type a message (voice mode: just speak)"><button id="send">Send</button></div>
 </section>
 <section class="panel hide-on-narrow" id="act"><h2>Agent activity</h2><div class="scroll" id="feed2"></div></section>
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
function bubble(cls,text,key){key=key||cls+':'+text;if(seen.has(key))return;seen.add(key);
 if(typingEl){typingEl.remove();typingEl=null;}
 const r=document.createElement('div');r.className='row '+cls;
 const b=document.createElement('div');b.className='bubble'+(cls==='bot'||cls==='me'?'':' '+cls);
 b.textContent=text;r.appendChild(b);feed.appendChild(r);feed.scrollTop=feed.scrollHeight;}
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
 if(!card){card=document.createElement('div');card.className='evt';evts.set(id,card);
  const [ic,label]=meta(ev.name||'');
  const tm=ev.t_ms?new Date(ev.t_ms).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit',fractionalSecondDigits:3}):now();
  card.innerHTML='<div class="t"><span class="ic">'+ic+'</span><span class="nm">'+label+'</span>'+
   '<span class="sum"></span><span class="tm">'+tm+'</span><span class="chip run">RUNNING</span></div>'+
   '<details><summary>details</summary><pre></pre></details>';
  card.dataset.sum=summary||'';feed2.appendChild(card);
  card.querySelector('.sum').textContent=((ev.elapsed_ms!=null?'[+'+ev.elapsed_ms+'ms] ':'')+(summary||'')).slice(0,90);}
 const chip=card.querySelector('.chip');
 if(ok===true){const d=ev.duration?msT(ev.duration*1000):'';chip.className='chip ok';chip.textContent='COMPLETED'+(d?' · '+d:'');}
 else if(ok===false){chip.className='chip fail';chip.textContent='FAILED';}
 return card.querySelector('pre');}
function tlRow(ev,icon,label,extra){
 const d=document.createElement('div');d.className='evt';
 const tm=tstamp(ev);
 d.innerHTML='<div class="t"><span class="ic">'+icon+'</span><span class="nm">'+label+'</span>'+
  '<span class="sum"></span><span class="tm">'+tm+'</span>'+(extra||'')+'</div>';
 d.querySelector('.sum').textContent=(ev.elapsed_ms!=null?'[+'+ev.elapsed_ms+'ms]':'');
 feed2.appendChild(d);feed2.scrollTop=feed2.scrollHeight;return d;}
function evtInfo(ev,ok){const pre=evtCard(ev,ev.arguments||ev.query||ev.location||'',ok);
 let s=(ev.arguments||'')+(ev.result?('\\n→ '+ev.result):'');pre.textContent=redact(s);feed2.scrollTop=feed2.scrollHeight;}
/* tabs (mobile) */
const tabs=document.getElementById('tabs');
tabs.querySelectorAll('button').forEach(b=>b.onclick=()=>{
 tabs.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
 document.getElementById('conv').style.display=b.dataset.p==='conv'?'flex':'none';
 document.getElementById('act').style.display=b.dataset.p==='act'?'flex':'none';});
if(innerWidth>900){document.getElementById('act').style.display='flex';}
/* agent state */
function state(s){document.getElementById('agentState').textContent=s?'· '+s:'';}
/* ---- events ---- */
es.onmessage=(e)=>{const ev=JSON.parse(e.data);
 switch(ev.type){
  case 'speech_start':tlRow(ev,'🎤','speech start');break;
  case 'speech_end':tlRow(ev,'🎤','speech end','<span class="chip unk">'+msT(ev.speech_ms)+'</span>');break;
  case 'transcript':tlRow(ev,'📝','transcript ready');ts();bubble('me',ev.text,'u:'+ev.turn+':'+ev.text);state('listening done');break;
  case 'llm_request':tlRow(ev,'🧠','sent to LLM ('+ev.messages.length+' msgs)');break;
  case 'first_token':tlRow(ev,'⚡','first LLM token');break;
  case 'llm_delta':setTyping(false);state('replying');
   if(!window._live){const live=document.createElement('div');live.className='row bot';
    live.innerHTML='<div class=\"bubble\" id=\"liveb\"></div>';feed.appendChild(live);window._live=live.querySelector('#liveb');}
   window._live.textContent+=ev.delta;feed.scrollTop=feed.scrollHeight;break;
  case 'bot_speech_start':
   if(window._live){window._live.removeAttribute('id');window._live=null;}
   if(ev.kind==='filler'){tlRow(ev,'🔊','TTS filler spoken');}
   else {tlRow(ev,'🔊','TTS response spoken');
    if(ev.latencies){const l=ev.latencies;state('');
     let d=document.createElement('div');d.className='sys-note';
     d.textContent='latency — speech→transcript '+(l.speech_to_transcript==null?'—':(l.speech_to_transcript*1000|0)+'ms')
      +', LLM→first token '+(l.llm_request_to_first_token==null?'—':(l.llm_request_to_first_token*1000|0)+'ms')
      +', LLM→spoken '+(l.llm_request_to_bot_speech==null?'—':(l.llm_request_to_bot_speech*1000|0)+'ms')
      +(l.speech_end_to_bot_speech==null?'':(', TOTAL speech-end→spoken '+(l.speech_end_to_bot_speech*1000|0)+'ms'));
     feed.appendChild(d);}}
   break;
  case 'turn_end':break;
  case 'tool_running':setTyping(true);state('running '+ev.name);evtInfo(ev,null);break;
  case 'tool_result':evtInfo(ev,ev.ok);break;
  case 'tool_calls_started':break;
  case 'announcement':ts();bubble('ann',ev.text,'a:'+ev.turn+':'+ev.text.slice(0,40));break;
  case 'interruption':ts();bubble('bot err','[interrupted by user]','i:'+ev.turn);break;
  case 'error':evtCard({name:'error',id:'err'+ev.turn},ev.error,false).textContent=redact(ev.error);break;
 }};
/* composer (text mode convenience; voice mode answers spoken input) */
const inp=document.getElementById('inp');
async function send(){const text=inp.value.trim();if(!text)return;inp.value='';
 document.getElementById('send').disabled=true;
 try{const r=await fetch('/say',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
  if(!r.ok)console.warn('say failed',r.status);}catch(e){console.warn(e);}
 document.getElementById('send').disabled=false;}
document.getElementById('send').onclick=send;inp.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
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
                return HTMLResponse(HTML)

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

            app = Starlette(routes=[
                Route("/", index),
                Route("/events", events),
                Route("/api/events", history),
                Route("/say", say, methods=["POST"]),
            ])
            config = uvicorn.Config(app, host="127.0.0.1", port=self.port,
                                    log_level="warning", loop="asyncio")
            uvicorn.Server(config).run()
        except Exception:
            import logging
            logging.getLogger("dashboard").exception("dashboard server failed")

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"