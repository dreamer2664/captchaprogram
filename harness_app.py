"""
CAPTCHA test harness — the rebuilt test page.

Serves: generated image, CASE SENSITIVE text box, 15 s countdown, submit,
scoreboard. Modes: manual play, AUTOPILOT (solver types live), 100-round test.

Run:  python3 harness_app.py [port]
"""
import io
import random
import threading
import time

from flask import Flask, Response, jsonify, request

import solver
from generator import generate

app = Flask(__name__)

ROUND_TIME = 15.0
KEY_DELAY = 0.09

_lock = threading.Lock()
_rounds: dict = {}
_stats = {"rounds": 0, "correct": 0, "history": []}


def _new_round() -> dict:
    img, truth = generate()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    rid = f"r{int(time.time()*1000)}{random.randint(100,999)}"
    with _lock:
        _rounds[rid] = {"truth": truth, "png": buf.getvalue(),
                        "born": time.time(),
                        "deadline": time.time() + ROUND_TIME,
                        "closed": False, "autopilot": None}
    return {"round_id": rid, "truth_len": len(truth)}


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/round/new")
def round_new():
    return jsonify(_new_round())


@app.route("/round/<rid>.png")
def round_png(rid):
    with _lock:
        st = _rounds.get(rid)
    if not st:
        return Response("no such round", status=404)
    return Response(st["png"], mimetype="image/png")


@app.route("/autopilot/start", methods=["POST"])
def autopilot_start():
    rid = request.json.get("round_id")
    with _lock:
        st = _rounds.get(rid)
        if not st or st["closed"]:
            return jsonify(error="bad round"), 400
    from PIL import Image
    img = Image.open(io.BytesIO(st["png"]))   # exactly what a screen grab sees
    t0 = time.perf_counter()
    guess = solver.solve(img, expected_len=len(st["truth"]))
    dt = time.perf_counter() - t0
    with _lock:
        st["autopilot"] = {"keys": list(guess), "start": time.time(),
                           "idx": 0, "solve_s": dt}
    return jsonify(solve_seconds=round(dt, 3), chars=len(guess))


@app.route("/autopilot/key")
def autopilot_key():
    rid = request.args.get("round_id")
    with _lock:
        st = _rounds.get(rid)
        ap = st["autopilot"] if st else None
        if not ap:
            return jsonify(error="not started"), 400
        due = min(len(ap["keys"]),
                  int((time.time() - ap["start"]) / KEY_DELAY) + 1)
        if ap["idx"] < due:
            ch = ap["keys"][ap["idx"]]
            ap["idx"] += 1
            return jsonify(char=ch)
        if ap["idx"] >= len(ap["keys"]):
            return jsonify(done=True)
        return jsonify(wait=True)


@app.route("/submit", methods=["POST"])
def submit():
    rid = request.json.get("round_id")
    answer = (request.json.get("answer") or "").strip()
    with _lock:
        st = _rounds.get(rid)
        if not st or st["closed"]:
            return jsonify(error="bad or already-closed round"), 400
        st["closed"] = True
        timed_out = time.time() > st["deadline"]
        correct = (not timed_out) and answer == st["truth"]   # CASE SENSITIVE
        _stats["rounds"] += 1
        _stats["correct"] += int(correct)
        _stats["history"].append({"truth": st["truth"], "answer": answer,
                                  "correct": correct, "timeout": timed_out})
        _stats["history"] = _stats["history"][-200:]
        snap = dict(_stats)
    return jsonify(correct=correct, truth=st["truth"], timeout=timed_out,
                   rounds=snap["rounds"], score=snap["correct"],
                   pct=round(100 * snap["correct"] / snap["rounds"], 1))


@app.route("/stats")
def stats():
    with _lock:
        return jsonify(dict(_stats))


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>CAPTCHA typer — test harness</title>
<style>
  body{font-family:ui-monospace,Menlo,Consolas,monospace;background:#14161a;color:#e8e8e8;
       display:flex;flex-direction:column;align-items:center;padding:28px;margin:0}
  h1{font-size:20px;letter-spacing:2px;margin:0 0 4px}
  .sub{color:#8a8f98;font-size:12px;margin-bottom:18px}
  #stage{background:#fff;padding:10px;border-radius:8px;min-height:90px;
         display:flex;align-items:center;justify-content:center;box-shadow:0 4px 24px #0008}
  #cap{image-rendering:pixelated;max-width:80vw}
  #row{display:flex;gap:10px;margin-top:16px;align-items:center}
  #answer{font:22px ui-monospace,monospace;padding:8px 12px;width:220px;border-radius:6px;
          border:1px solid #444;background:#1d2026;color:#fff;letter-spacing:4px}
  button{font:14px ui-monospace,monospace;padding:9px 14px;border-radius:6px;cursor:pointer;
         border:1px solid #444;background:#232730;color:#e8e8e8}
  button:hover{background:#2d3340}
  button.on{background:#1d4d2b;border-color:#2f7a45;color:#baf5c8}
  #barwrap{width:460px;max-width:86vw;height:8px;background:#262a31;border-radius:4px;
           margin-top:14px;overflow:hidden}
  #bar{height:100%;width:100%;background:#3fa650;transition:width .1s linear}
  #bar.low{background:#c0392b}
  #score{margin-top:14px;font-size:15px;color:#9fd8a9}
  #msg{height:22px;margin-top:8px;font-size:15px;font-weight:bold}
  #msg.ok{color:#5ddb7a} #msg.bad{color:#ff6b5e}
  #log{margin-top:14px;font-size:12px;color:#7d838c;max-width:520px;text-align:center}
</style></head><body>
<h1>CAPTCHA TYPER</h1>
<div class="sub">read the image &middot; type it &middot; submit before the bar empties</div>
<div id="stage"><img id="cap" alt="captcha"></div>
<div class="sub" style="margin-top:10px">You have to solve the image in
  <b id="secs">15</b> seconds &middot; Text: <span style="color:#e05a4e;font-weight:bold">CASE SENSITIVE</span></div>
<div id="barwrap"><div id="bar"></div></div>
<div id="row">
  <input id="answer" autocomplete="off" spellcheck="false" placeholder="type here">
  <button id="go">SUBMIT</button>
  <button id="ap">AUTOPILOT: OFF</button>
  <button id="loop">RUN 100-ROUND TEST</button>
</div>
<div id="msg"></div>
<div id="score">rounds 0 &middot; correct 0 &middot; 0%</div>
<div id="log"></div>
<script>
let round=null, deadline=0, autopilot=false, looping=false, loopLeft=0, busy=false;
const $=id=>document.getElementById(id);
const img=$('cap'), box=$('answer'), bar=$('bar');

async function newRound(){
  box.value=''; box.disabled=false; $('msg').textContent='';
  const r=await (await fetch('/round/new')).json();
  round=r; deadline=Date.now()+15000;
  img.src='/round/'+r.round_id+'.png?t='+Date.now();
  if(autopilot){ box.disabled=true;
    const s=await (await fetch('/autopilot/start',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({round_id:r.round_id})})).json();
    $('log').textContent='solver read it in '+s.solve_seconds+'s → typing '+s.chars+' chars…';
    typeLoop();
  } else { box.focus(); }
}

async function typeLoop(){
  if(!round) return;
  const r=await (await fetch('/autopilot/key?round_id='+round.round_id)).json();
  if(r.char!==undefined){ box.value+=r.char; setTimeout(typeLoop, 90); }
  else if(r.wait){ setTimeout(typeLoop, 40); }
  else if(r.done){ submit(); }
}

async function submit(){
  if(!round||busy) return; busy=true;
  const res=await (await fetch('/submit',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({round_id:round.round_id, answer:box.value})})).json();
  if(res.error){ busy=false; return; }
  const m=$('msg');
  m.textContent=res.correct?('✓ '+res.truth):((res.timeout?'TIMEOUT — ':'✗ got "'+
    (box.value||'—')+'" — ')+'was '+res.truth);
  m.className=res.correct?'ok':'bad';
  $('score').textContent='rounds '+res.rounds+' · correct '+res.score+' · '+res.pct+'%';
  busy=false;
  if(looping){ loopLeft--;
    if(loopLeft<=0){ looping=false; $('loop').classList.remove('on');
      $('log').textContent='TEST COMPLETE — final: '+res.score+'/'+res.rounds+' ('+res.pct+'%)'; }
    else setTimeout(newRound, 500);
  }
}

setInterval(()=>{ if(!round) return;
  const left=Math.max(0,deadline-Date.now());
  document.getElementById('secs').textContent=Math.ceil(left/1000);
  bar.style.width=(100*left/15000)+'%';
  bar.classList.toggle('low', left<4000);
  if(left<=0 && !busy && !round.closed){ box.disabled=true; submit(); }
},100);

$('go').onclick=submit;
box.addEventListener('keydown',e=>{ if(e.key==='Enter') submit(); });
$('ap').onclick=()=>{ autopilot=!autopilot;
  $('ap').textContent='AUTOPILOT: '+(autopilot?'ON':'OFF');
  $('ap').classList.toggle('on',autopilot); newRound(); };
$('loop').onclick=()=>{ looping=!looping;
  $('loop').classList.toggle('on',looping);
  if(looping){ loopLeft=100; newRound(); } };

newRound();
</script></body></html>
"""


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(host="0.0.0.0", port=port, threaded=True)
