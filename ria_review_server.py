"""Local review UI before Slack delivery.

Run:
  python ria_review_server.py

Then open:
  http://127.0.0.1:8765
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = int(os.environ.get("RIA_REVIEW_PORT", "8765"))
ALERT = None


def load_alert_module():
    path = ROOT / "18_run_alert_v3.py"
    spec = importlib.util.spec_from_file_location("ria_alert_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ask_runtime_config():
    """Collect optional runtime keys without storing them in source files."""
    if not sys.stdin.isatty():
        print("[setup] 대화형 터미널이 아니므로 환경변수/캐시 설정을 그대로 사용합니다.")
        return

    print("\nRIA 실행 설정")
    print("- 입력하지 않고 Enter를 누르면 캐시/mock 또는 dry-run으로 실행합니다.")
    print("- 키는 코드에 저장하지 않고 현재 터미널 세션 환경변수로만 사용합니다.\n")

    if not os.environ.get("KMA_KEY"):
        key = input("KMA_KEY 입력 (빈 값이면 캐시/mock 사용): ").strip()
        if key:
            os.environ["KMA_KEY"] = key
            os.environ["USE_KMA_LIVE"] = "1"
        else:
            os.environ.setdefault("USE_KMA_LIVE", "0")
    else:
        os.environ.setdefault("USE_KMA_LIVE", "1")
        print("[setup] KMA_KEY 환경변수 감지: KMA live 사용")

    if not os.environ.get("SLACK_WEBHOOK"):
        webhook = input("Slack Webhook URL 입력 (빈 값이면 전송 비활성): ").strip()
        if webhook:
            os.environ["SLACK_WEBHOOK"] = webhook
            os.environ["USE_SLACK_LIVE"] = "1"
        else:
            os.environ.setdefault("USE_SLACK_LIVE", "0")
    else:
        os.environ.setdefault("USE_SLACK_LIVE", "1")
        print("[setup] SLACK_WEBHOOK 환경변수 감지: Slack 전송 사용")


def pick_port(start_port):
    port = start_port
    while port < start_port + 20:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
            except OSError:
                port += 1
                continue
            return port
    raise RuntimeError(f"사용 가능한 포트를 찾지 못했습니다: {start_port}-{start_port + 19}")


_SCENARIOS = [
    ("집중호우",     lambda c: c["누적강수_7d"] >= 80 or c["강수_연속_7d"] >= 3),
    ("장기가뭄·폭염", lambda c: c["폭염_연속_7d"] >= 2),
    ("태풍",         lambda c: c["풍속"] >= 6),
    ("한파",         lambda c: c["한파_연속_7d"] >= 2),
]

_FEAT_KEYS = ["일강수량", "누적강수_7d", "Δtavg", "기온변동성_7d", "풍속"]
_FEAT_MAX  = [50.0, 150.0, 12.0, 8.0, 10.0]
_HEADS     = ["중대재해", "다중사상", "외국인피해", "고령자피해"]

# Static W matrix: feature[5] × head[4]  (도메인 지식 기반)
_W = [
    [0.82, 0.75, 0.52, 0.38],  # 강수량 이탈
    [0.78, 0.80, 0.48, 0.42],  # 누적강수
    [0.60, 0.32, 0.62, 0.85],  # 기온 이탈
    [0.52, 0.38, 0.28, 0.65],  # 기온변동
    [0.75, 0.55, 0.45, 0.30],  # 풍속
]


def _detect_scenario(cumf):
    for name, cond in _SCENARIOS:
        if cond(cumf):
            return name
    return "정상"


def _feature_vec(cumf):
    return [round(min(abs(cumf[k]) / m, 1.0), 3) for k, m in zip(_FEAT_KEYS, _FEAT_MAX)]


def _w_matrix(fv):
    return [[round(_W[fi][hi] * fv[fi], 3) for hi in range(4)] for fi in range(5)]


def _issue_scores(evals):
    br = ALERT.base_rates
    scores = {h: 0.0 for h in _HEADS}
    for ev in evals:
        for h in _HEADS:
            lift = ev["pred"][h] / max(br.get(h, 1e-9), 1e-9)
            scores[h] = max(scores[h], round(min(lift / 3.0, 1.0), 3))
    return scores


def make_alert_payload():
    if ALERT is None:
        raise RuntimeError("alert module is not loaded")
    today_row = ALERT.get_today()
    today_date = ALERT.pd.Timestamp(today_row["일시"]).normalize()
    cumf = ALERT.compute_cumulative(today_date)
    scenario = ALERT.detect_scenario(cumf)
    evals = [ALERT.evaluate_env(e, cumf, today_date, scenario) for e in ALERT.ENVIRONMENTS]

    fv = _feature_vec(cumf)
    sites = []
    for ev in evals:
        env = ev["env"]
        categories = [
            {"name": cat["name"], "source": cat["source"], "actions": list(cat["actions"])}
            for cat in ev["cats"]
        ]
        sites.append({
            "id": env["id"],
            "highRisk": bool(env["고위험"]),
            "categories": categories,
        })

    return {
        "date": str(today_date.date()),
        "weather": cumf,
        "scenario": scenario,
        "features": fv,
        "matrix": _w_matrix(fv),
        "issueScores": _issue_scores(evals),
        "sites": sites,
        "slackLive": bool(ALERT.USE_SLACK_LIVE),
        "kmaLive": bool(ALERT.USE_KMA_LIVE),
    }


def build_slack_message(payload):
    weather = payload["weather"]
    lines = []
    lines.append(f"🚨 *RIA Risk Instinct Alert — {payload['date']}*")
    lines.append("")
    lines.append("*오늘의 전국 기상*")
    lines.append(
        f"• 평균 {weather['평균기온']:.1f}°C / 최고 {weather['최고기온']:.1f} / "
        f"최저 {weather['최저기온']:.1f}"
    )
    lines.append(
        f"• 강수 {weather['일강수량']:.1f}mm / 풍속 {weather['풍속']:.1f}m/s / "
        f"7일 누적강수 {weather['누적강수_7d']:.0f}mm"
    )
    lines.append("")
    lines.append("*오늘의 안전 이슈 체크리스트*")

    has_any = False
    for site in payload["sites"]:
        enabled_categories = [c for c in site["categories"] if c.get("enabled", True)]
        if not site.get("enabled", True) or not enabled_categories:
            continue
        has_any = True
        tags = []
        if site.get("highRisk"):
            tags.append("고위험공종")
        if site.get("small"):
            tags.append("소규모현장")
        tag_text = f" ({', '.join(tags)})" if tags else ""
        lines.append("")
        lines.append(f"▣ *{site['id']}* — 공정율 {site['progress']}% · {site['hour']}시 작업{tag_text}")
        for cat in enabled_categories:
            actions = [a for a in cat["actions"] if a.get("enabled", True) and a.get("text", "").strip()]
            if not actions:
                continue
            icon = "🔴" if cat.get("source") == "model" else "🟡"
            lines.append(f"  {icon} *{cat['name']}*")
            for action in actions:
                lines.append(f"    ☐ {action['text'].strip()}")

    if not has_any:
        lines.append("_오늘은 모든 공종이 안정 범위입니다._")

    footer = payload.get("note", "").strip()
    if footer:
        lines.append("")
        lines.append(f"_메모: {footer}_")
    return "\n".join(lines)


def send_slack(message):
    webhook = os.environ.get("SLACK_WEBHOOK", "")
    if not webhook:
        return {"ok": False, "error": "SLACK_WEBHOOK 미설정"}
    body = json.dumps({"text": message}).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        text = resp.read().decode("utf-8")
    return {"ok": text.strip() == "ok", "response": text}


REVIEW_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RIA Review Console</title>
<style>
:root{
  --bg:#0c0f14;--surface:#131720;--panel:#1a2030;--border:#252d3d;
  --txt:#e8ecf4;--muted:#7b8aaa;--accent:#4af0b0;
  --danger:#f25c5c;--warn:#f5a623;--blue:#4a8fcc;
  --orange:#e8724a;--teal:#5bc8b0;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,"Pretendard","Segoe UI",sans-serif;font-size:14px;line-height:1.5}
header{height:52px;background:var(--surface);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 24px;gap:12px;position:sticky;top:0;z-index:10}
.logo{font-weight:800;color:var(--accent);font-size:17px;letter-spacing:-.3px}
.header-sub{color:var(--muted);font-size:12px}
.header-right{margin-left:auto;display:flex;gap:8px;align-items:center}
.main{padding:20px 24px;max-width:1160px;margin:0 auto}
.section-label{font-size:11px;color:var(--muted);font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px}
.scenarios{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:22px}
.s-btn{padding:8px 18px;border-radius:999px;border:1.5px solid var(--border);background:transparent;color:var(--muted);font-size:13px;font-weight:600;transition:.2s;cursor:default}
.s-btn.active-bad{border-color:var(--orange);background:rgba(232,114,74,.15);color:var(--orange)}
.s-btn.active-ok{border-color:var(--teal);background:rgba(91,200,176,.12);color:var(--teal)}
.content-grid{display:grid;grid-template-columns:340px 1fr;gap:20px;align-items:start}
.viz-col{display:flex;flex-direction:column;gap:12px}
.output-col{display:flex;flex-direction:column;gap:0}
.viz-panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px}
.viz-title{font-size:10px;color:var(--muted);font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:14px}
.fb-row{display:flex;align-items:center;gap:9px;margin-bottom:11px}
.fb-label{font-size:11px;color:var(--muted);width:68px;flex-shrink:0;text-align:right;line-height:1.2}
.fb-track{flex:1;height:7px;background:#1b2334;border-radius:4px;overflow:hidden}
.fb-fill{height:100%;border-radius:4px;transition:width .6s ease}
.hm-head{display:grid;grid-template-columns:70px repeat(4,1fr);margin-bottom:4px}
.hm-head-cell{font-size:10px;color:var(--muted);text-align:center;padding:2px 0}
.hm-row{display:grid;grid-template-columns:70px repeat(4,1fr);gap:4px;margin-bottom:4px}
.hm-rlabel{font-size:10px;color:var(--muted);text-align:right;padding-right:6px;display:flex;align-items:center;justify-content:flex-end}
.hm-cell{height:32px;border-radius:5px;min-width:0}
.hm-note{font-size:10px;color:var(--muted);text-align:center;margin-top:8px}
.ib-row{display:flex;align-items:center;gap:8px;margin-bottom:14px}
.ib-label{font-size:12px;width:68px;flex-shrink:0;line-height:1.2}
.ib-track{flex:1;height:7px;background:#1b2334;border-radius:4px;overflow:hidden}
.ib-fill{height:100%;border-radius:4px;transition:width .6s ease}
.ib-score{font-size:13px;font-weight:700;width:38px;text-align:right;flex-shrink:0}
.ib-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.output-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.meta{font-size:12px;color:var(--muted)}
.stable-msg{color:var(--muted);padding:32px 0;text-align:center}
.site-card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:12px}
.site-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.site-name{font-weight:800;font-size:15px}
.site-meta{font-size:12px;color:var(--muted)}
.chip{font-size:11px;border-radius:999px;padding:2px 8px;border:1px solid}
.chip-danger{color:var(--danger);border-color:rgba(242,92,92,.4)}
.chip-warn{color:var(--warn);border-color:rgba(245,166,35,.4)}
.item{display:flex;align-items:flex-start;gap:10px;padding:8px 0;border-bottom:1px solid #1b2538}
.item:last-child{border:none}
.item-tag{font-size:11px;font-weight:700;padding:2px 7px;border-radius:4px;flex-shrink:0;margin-top:1px}
.tag-danger{background:rgba(242,92,92,.15);color:var(--danger)}
.tag-warn{background:rgba(245,166,35,.12);color:var(--warn)}
.item-text{font-size:13px;flex:1}
.item-cat{font-size:11px;color:#3d4f6a;flex-shrink:0;margin-top:2px;max-width:130px;text-align:right;line-height:1.3}
.bottom{display:flex;align-items:center;gap:10px;margin-top:16px;padding-top:16px;border-top:1px solid var(--border);flex-wrap:wrap}
.note-inp{flex:1;min-width:180px;background:#10151f;border:1px solid var(--border);border-radius:7px;color:var(--txt);padding:8px 12px;font:inherit;font-size:13px}
.btn{border:0;border-radius:8px;padding:9px 16px;font-weight:700;cursor:pointer;font-size:13px}
.btn-refresh{background:transparent;color:var(--muted);border:1px solid var(--border)}
.btn-slack{background:#4a154b;color:#fff}
.status{font-size:13px}.ok{color:var(--accent)}.err{color:var(--danger)}
</style>
</style>
</head>
<body>
<header>
  <span class="logo">RIA</span>
  <span class="header-sub">Review Console</span>
  <div class="header-right">
    <span class="status" id="status">loading...</span>
    <button class="btn btn-refresh" onclick="loadAlert()">새로고침</button>
  </div>
</header>

<div class="main">
  <div class="section-label">시나리오 감지</div>
  <div class="scenarios" id="scenarios">
    <div class="s-btn" data-s="집중호우">집중호우</div>
    <div class="s-btn" data-s="장기가뭄·폭염">장기가뭄·폭염</div>
    <div class="s-btn" data-s="태풍">태풍</div>
    <div class="s-btn" data-s="한파">한파</div>
    <div class="s-btn" data-s="정상">정상</div>
  </div>

  <div class="content-grid">
    <div class="viz-col">
      <div class="section-label">온라인 추론 — 이탈 감지 × 가중치</div>
      <div class="viz-panel">
        <div class="viz-title">피처 이탈 벡터</div>
        <div id="feat-bars"></div>
      </div>
      <div class="viz-panel">
        <div class="viz-title">W [FEATURE × ISSUE]</div>
        <div id="heatmap"></div>
        <div class="hm-note">셀 밝기 = 가중치 × 이탈량</div>
      </div>
      <div class="viz-panel">
        <div class="viz-title">이슈 활성화</div>
        <div id="issue-bars"></div>
      </div>
    </div>
    <div class="output-col">
      <div class="output-header">
        <div class="section-label" style="margin:0">OUTPUT — 이슈 체크리스트</div>
        <div class="meta" id="meta"></div>
      </div>
      <div id="sites"></div>
    </div>
  </div>

  <div class="bottom">
    <input class="note-inp" id="note" placeholder="추가 메모 (선택)">
    <button class="btn btn-slack" onclick="sendSlack()">Slack 전송</button>
  </div>
</div>

<script>
let DATA = null;
const FEAT_LABELS = ['강수량 이탈','누적 초과량','기온 이탈','기온변동','풍속'];
const FEAT_SHORT  = ['강수이탈','누적초과','기온이탈','기온변동','풍속변화'];
const ISSUE_LABELS = ['중대재해','다중사상','외국인피해','고령자피해'];
const ISSUE_SHORT  = ['중대재해','다중사상','외국인','고령자'];

function lerp(a,b,t){ return a+(b-a)*t }
function cellColor(v){
  const r=Math.round(lerp(91,232,v)), g=Math.round(lerp(200,114,v)), b=Math.round(lerp(176,74,v));
  return `rgba(${r},${g},${b},${0.15+v*0.85})`;
}

async function api(path,opts){
  const res=await fetch(path,opts);
  const data=await res.json();
  if(!res.ok) throw new Error(data.error||res.statusText);
  return data;
}

async function loadAlert(){
  setStatus('불러오는 중...');
  DATA=await api('/api/alert');
  DATA.sites.forEach(site=>site.categories.forEach(cat=>{
    cat.actions=cat.actions.map(a=>typeof a==='string'?{text:a,enabled:true}:a);
  }));
  document.getElementById('meta').textContent=
    `${DATA.date} · KMA=${DATA.kmaLive?'live':'cache'} · Slack=${DATA.slackLive?'live':'dry'}`;
  renderScenario(); renderFeatBars(); renderHeatmap(); renderIssueBars(); renderSites();
  setStatus('완료','ok');
}

function renderScenario(){
  document.querySelectorAll('.s-btn').forEach(btn=>{
    btn.classList.remove('active-bad','active-ok');
    if(btn.dataset.s===DATA.scenario)
      btn.classList.add(DATA.scenario==='정상'?'active-ok':'active-bad');
  });
}

function renderFeatBars(){
  document.getElementById('feat-bars').innerHTML=FEAT_LABELS.map((label,i)=>{
    const v=DATA.features[i], pct=Math.round(v*100);
    const color=v>0.6?'var(--orange)':v>0.3?'#d4946a':'var(--teal)';
    return `<div class="fb-row">
      <div class="fb-label">${label}</div>
      <div class="fb-track"><div class="fb-fill" style="width:${pct}%;background:${color}"></div></div>
    </div>`;
  }).join('');
}

function renderHeatmap(){
  const m=DATA.matrix;
  const head=`<div class="hm-head"><div></div>${ISSUE_SHORT.map(l=>`<div class="hm-head-cell">${l}</div>`).join('')}</div>`;
  const rows=FEAT_SHORT.map((label,fi)=>
    `<div class="hm-row"><div class="hm-rlabel">${label}</div>${ISSUE_LABELS.map((_,hi)=>
      `<div class="hm-cell" style="background:${cellColor(m[fi][hi])}"></div>`).join('')}</div>`
  ).join('');
  document.getElementById('heatmap').innerHTML=head+rows;
}

function renderIssueBars(){
  const scores=DATA.issueScores;
  document.getElementById('issue-bars').innerHTML=ISSUE_LABELS.map(label=>{
    const v=scores[label]||0, pct=Math.round(v*100), alert=v>=0.5;
    const color=alert?'var(--danger)':'var(--blue)';
    return `<div class="ib-row">
      <div class="ib-label">${label}</div>
      <div class="ib-track"><div class="ib-fill" style="width:${pct}%;background:${color}"></div></div>
      <div class="ib-score" style="color:${color}">${(v*3).toFixed(2)}</div>
      <div class="ib-dot" style="background:${alert?color:'transparent'};border:1.5px solid ${color}"></div>
    </div>`;
  }).join('');
}

function renderSites(){
  const triggered=DATA.sites.filter(s=>s.categories&&s.categories.length>0);
  if(!triggered.length){
    document.getElementById('sites').innerHTML='<div class="stable-msg">오늘은 모든 공종이 안정 범위입니다.</div>';
    return;
  }
  document.getElementById('sites').innerHTML=triggered.map(site=>{
    const chips=site.highRisk?'<span class="chip chip-danger">고위험 환경</span>':'';
    const items=site.categories.flatMap(cat=>
      cat.actions.filter(a=>a.enabled!==false).map(a=>{
        const sev=cat.source==='model'?'danger':'warn';
        return `<div class="item">
          <span class="item-tag tag-${sev}">${sev==='danger'?'위험':'주의'}</span>
          <span class="item-text">${a.text||a}</span>
          <span class="item-cat">${cat.name}</span>
        </div>`;
      })
    ).join('');
    return `<div class="site-card">
      <div class="site-head">
        <span class="site-name">${site.id}</span>
        ${chips}
      </div>${items}</div>`;
  }).join('');
}

async function sendSlack(){
  try{
    DATA.note=document.getElementById('note').value||'';
    const res=await api('/api/send-slack',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(DATA)});
    setStatus(res.ok?'Slack 전송 완료':res.error||'전송 실패',res.ok?'ok':'err');
  }catch(e){ setStatus(e.message,'err'); }
}

function setStatus(t,c=''){ const el=document.getElementById('status'); el.textContent=t; el.className='status '+c; }
loadAlert().catch(e=>setStatus(e.message,'err'));
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def do_GET(self):
        try:
            if self.path == "/" or self.path == "/review":
                self._send(200, REVIEW_HTML, "text/html; charset=utf-8")
            elif self.path == "/legacy":
                self._send(200, (ROOT / "RIA_webapp.html").read_text(encoding="utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/alert":
                self._send(200, make_alert_payload())
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def do_POST(self):
        try:
            if self.path == "/api/preview":
                payload = self._read_json()
                self._send(200, {"message": build_slack_message(payload)})
            elif self.path == "/api/send-slack":
                payload = self._read_json()
                message = build_slack_message(payload)
                result = send_slack(message)
                self._send(200, {**result, "message": message})
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def log_message(self, fmt, *args):
        print(f"[review] {self.address_string()} - {fmt % args}")


def main():
    global ALERT
    ask_runtime_config()
    ALERT = load_alert_module()

    port = pick_port(PORT)
    if port != PORT:
        print(f"[setup] {PORT} 포트가 사용 중이라 {port} 포트로 실행합니다.")
    server = ThreadingHTTPServer((HOST, port), Handler)
    print(f"RIA review server: http://{HOST}:{port}")
    print(f"Legacy static page: http://{HOST}:{port}/legacy")
    server.serve_forever()


if __name__ == "__main__":
    main()
