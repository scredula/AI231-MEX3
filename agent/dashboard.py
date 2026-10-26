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
<html><head><meta charset="utf-8"><title>voice-agent dashboard</title>
<style>
:root{color-scheme:dark}
body{font:14px/1.5 var(--x,system-ui,Segoe UI,sans-serif);margin:0;background:#0f1117;color:#e6e6e6}
header{padding:10px 16px;background:#161a23;border-bottom:1px solid #2a2f3c;position:sticky;top:0;display:flex;gap:16px;align-items:baseline}
header h1{font-size:15px;margin:0;color:#9ecbff}
#status{font-size:12px;color:#8b93a7}
main{padding:14px 16px;max-width:1100px}
.turn{background:#161a23;border:1px solid #2a2f3c;border-radius:10px;margin-bottom:12px;overflow:hidden}
.turn .head{display:flex;gap:10px;padding:8px 12px;background:#1c2130;font-size:12px;color:#8b93a7}
.turn .body{padding:8px 12px}
.sec{margin:6px 0}
.sec .lbl{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#79839a;margin-right:6px}
.user{color:#9ecbff;white-space:pre-wrap}
.bot{color:#b7e0a0;white-space:pre-wrap}
.sys{color:#8b93a7;white-space:pre-wrap;font-size:12px}
.tool{background:#1c2130;border-radius:6px;padding:6px 8px;margin:4px 0;font-size:12px}
.tool .name{color:#f0c674;font-weight:600}
.tool pre{margin:2px 0 0;white-space:pre-wrap;color:#c3c8d4}
.lat{display:inline-block;background:#1c2130;border-radius:6px;padding:2px 8px;margin:2px 4px 2px 0;font-size:12px}
.lat b{color:#c3c8d4}
.err{color:#ff8a8a}
.ann{color:#f0c674;font-size:12px}
details summary{cursor:pointer;color:#9ecbff;font-size:12px}
details pre{white-space:pre-wrap;font-size:12px;color:#8b93a7;max-height:260px;overflow:auto;background:#12151d;padding:6px;border-radius:6px}
.delta{color:#5a6070;font-size:11px}
</style></head>
<body>
<header><h1>voice-agent conversation</h1><span id="status">connecting…</span></header>
<main id="feed"></main>
<script>
const feed=document.getElementById('feed');
const es=new EventSource('/events');
es.onopen=()=>status('live');
es.onerror=()=>status('reconnecting…');
function status(s){document.getElementById('status').textContent=s;}
let cur=null;
function turn(id){if(!cur||cur.dataset.id!==String(id)){cur=document.createElement('div');cur.className='turn';cur.dataset.id=String(id);
 cur.innerHTML='<div class="head"><b>turn '+id+'</b><span class="t">'+new Date().toLocaleTimeString()+'</span></div><div class="body"></div>';
 feed.appendChild(cur);}return cur.querySelector('.body');}
function sec(body,lbl,cls,text){let d=document.createElement('div');d.className='sec';
 d.innerHTML='<span class="lbl">'+lbl+'</span><span class="'+cls+'"></span>';d.querySelector('span:last-child').textContent=text;body.appendChild(d);return d;}
function lats(body,l){if(!l)return;const b=document.createElement('div');b.className='sec';
 const items=[['speech→transcript',l.speech_to_transcript],['transcript→LLM',l.transcript_to_llm_request],['LLM→first token',l.llm_request_to_first_token],['LLM→bot speech',l.llm_request_to_bot_speech],['speech→bot speech',l.speech_to_bot_speech]];
 b.innerHTML=items.map(([k,v])=>'<span class="lat">'+k+' <b>'+(v==null?'—':(v*1000).toFixed(0)+'ms')+'</b></span>').join('');
 if(l.interrupted)b.innerHTML+='<span class="lat err">interrupted by user</span>';
 body.appendChild(b);}
function toolcard(body,name,details,cls){let t=document.createElement('div');t.className='tool';
 t.innerHTML='<span class="name">'+name+'</span><pre></pre>';t.querySelector('pre').textContent=details||'';body.appendChild(t);}
es.onmessage=(e)=>{const ev=JSON.parse(e.data);const body=turn(ev.turn??0);
 switch(ev.type){
  case 'turn_start':break;
  case 'transcript':sec(body,'you said','user',ev.text);break;
  case 'llm_request':{let d=document.createElement('details');d.innerHTML='<summary>LLM input ('+ev.messages.length+' msgs'+(ev.model?' · '+ev.model:'')+')</summary><pre></pre>';
   d.querySelector('pre').textContent=ev.messages.map(m=>'['+m.role+'] '+m.content).join('\\n');body.appendChild(d);break;}
  case 'llm_delta':{let d=body.querySelector('.bot.live');if(!d){d=sec(body,'assistant','bot live','');}d.querySelector('span:last-child').textContent+=ev.delta;break;}
  case 'tool_calls_started':ev.calls.forEach(c=>toolcard(body,c.name,c.arguments));break;
  case 'tool_running':toolcard(body,ev.name,ev.arguments);break;
  case 'tool_result':toolcard(body,ev.name+' →',ev.result);break;
  case 'bot_speech_start':{let d=body.querySelector('.bot.live');if(d){d.classList.remove('live');d.querySelector('.lbl').textContent='assistant (spoken)';}lats(body,ev.latencies);break;}
  case 'turn_end':break;
  case 'interruption':sec(body,'interruption','err','user interrupted the assistant');break;
  case 'announcement':sec(body,'announcement','ann',ev.text);break;
  case 'error':sec(body,'error','err',ev.error);break;
 }};
</script></body></html>"""


class DashboardServer:
    def __init__(self, bus: EventBus, port: int = None):
        self.bus = bus
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

            app = Starlette(routes=[
                Route("/", index),
                Route("/events", events),
                Route("/api/events", history),
            ])
            config = uvicorn.Config(app, host="127.0.0.1", port=self.port,
                                    log_level="warning", loop="asyncio")
            uvicorn.Server(config).run()
        except Exception:
            import logging
            logging.getLogger("dashboard").exception("dashboard server failed")

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"