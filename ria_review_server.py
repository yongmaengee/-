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


def make_alert_payload():
    if ALERT is None:
        raise RuntimeError("alert module is not loaded")
    today_row = ALERT.get_today()
    today_date = ALERT.pd.Timestamp(today_row["일시"]).normalize()
    cumf = ALERT.compute_cumulative(today_date)
    evals = [ALERT.evaluate_site(s, cumf, today_date) for s in ALERT.SITES]

    sites = []
    for ev in evals:
        site = ev["site"]
        categories = []
        for cat in ev["cats"]:
            categories.append(
                {
                    "name": cat["name"],
                    "source": cat["source"],
                    "actions": list(cat["actions"]),
                }
            )
        sites.append(
            {
                "id": site["id"],
                "trade": site["공종"],
                "progress": site["공정율"],
                "hour": site["시간"],
                "small": bool(site["소규모"]),
                "highRisk": bool(site["고위험"]),
                "categories": categories,
            }
        )

    return {
        "date": str(today_date.date()),
        "weather": cumf,
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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RIA Review Console</title>
<style>
:root{--bg:#0c0f14;--surface:#131720;--panel:#1a2030;--border:#2f3a52;--txt:#e8ecf4;--muted:#7b8aaa;--accent:#4af0b0;--danger:#f25c5c;--warn:#f5a623}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,"Pretendard","Segoe UI",sans-serif}
header{height:56px;background:var(--surface);border-bottom:1px solid #252d3d;display:flex;align-items:center;padding:0 24px;gap:16px}
.logo{font-weight:800;color:var(--accent);font-size:18px}.sub{color:var(--muted);font-size:13px}.wrap{padding:24px;display:grid;grid-template-columns:minmax(520px,1fr) 420px;gap:18px}
.panel{background:var(--panel);border:1px solid #252d3d;border-radius:10px;padding:18px}.row{display:flex;gap:10px;align-items:center;justify-content:space-between}
button{border:0;border-radius:8px;padding:9px 14px;font-weight:700;cursor:pointer}.primary{background:var(--accent);color:#07100d}.ghost{background:transparent;color:var(--muted);border:1px solid var(--border)}.slack{background:#4a154b;color:white}
.weather{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}.metric{background:#0c0f14;border:1px solid #252d3d;border-radius:8px;padding:10px}.metric span{display:block;color:var(--muted);font-size:12px}.metric b{font-size:17px}
.site{background:#0c0f14;border:1px solid #252d3d;border-radius:10px;padding:14px;margin-top:12px}.site.off{opacity:.45}.site-head{display:flex;align-items:center;gap:10px;margin-bottom:10px}.site-title{font-weight:800}.tag{font-size:11px;color:var(--warn);border:1px solid #65461b;border-radius:999px;padding:2px 7px}
.cat{border-top:1px solid #252d3d;padding-top:10px;margin-top:10px}.cat-title{display:flex;align-items:center;gap:8px;font-weight:700}.actions{display:flex;flex-direction:column;gap:7px;margin-top:8px}.action{display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center}
input[type=text],textarea{width:100%;background:#10151f;border:1px solid var(--border);border-radius:7px;color:var(--txt);padding:8px;font:inherit}
textarea{height:520px;resize:vertical;line-height:1.45;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}.small{color:var(--muted);font-size:12px}.status{font-size:13px}.ok{color:var(--accent)}.err{color:var(--danger)}
</style>
</head>
<body>
<header><div class="logo">RIA</div><div class="sub">Review Console · 모델 결과 수정 후 Slack 전송</div></header>
<main class="wrap">
  <section>
    <div class="panel">
      <div class="row">
        <div><h2 style="margin:0">오늘의 알림 검토</h2><div class="small" id="meta">loading...</div></div>
        <div><button class="ghost" onclick="loadAlert()">새로고침</button> <button class="primary" onclick="renderPreview()">미리보기 갱신</button></div>
      </div>
      <div class="weather" id="weather"></div>
    </div>
    <div id="sites"></div>
  </section>
  <aside class="panel">
    <div class="row"><h3 style="margin:0">Slack 전송 전 메시지</h3><button class="slack" onclick="sendSlack()">전송</button></div>
    <p class="small">왼쪽에서 현장/카테고리/체크항목을 제외하거나 문구를 수정한 뒤 전송하세요. Webhook은 서버의 SLACK_WEBHOOK 환경변수를 사용합니다.</p>
    <textarea id="preview"></textarea>
    <div class="row" style="margin-top:10px"><input id="note" type="text" placeholder="추가 메모"><span class="status" id="status"></span></div>
  </aside>
</main>
<script>
let DATA = null;

async function api(path, options){
  const res = await fetch(path, options);
  const data = await res.json();
  if(!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function loadAlert(){
  setStatus('불러오는 중...');
  DATA = await api('/api/alert');
  DATA.sites.forEach(site => {
    site.enabled = true;
    site.categories.forEach(cat => {
      cat.enabled = true;
      cat.actions = cat.actions.map(text => ({text, enabled:true}));
    });
  });
  document.getElementById('meta').textContent = `${DATA.date} · KMA=${DATA.kmaLive?'live':'mock'} · Slack=${DATA.slackLive?'live':'dry/server'}`;
  renderWeather();
  renderSites();
  renderPreview();
  setStatus('불러오기 완료','ok');
}

function renderWeather(){
  const w = DATA.weather;
  document.getElementById('weather').innerHTML = [
    ['평균기온', `${w['평균기온'].toFixed(1)}°C`],
    ['최고/최저', `${w['최고기온'].toFixed(1)} / ${w['최저기온'].toFixed(1)}`],
    ['강수/풍속', `${w['일강수량'].toFixed(1)}mm / ${w['풍속'].toFixed(1)}m/s`],
    ['7일 누적강수', `${w['누적강수_7d'].toFixed(0)}mm`],
  ].map(([k,v]) => `<div class="metric"><span>${k}</span><b>${v}</b></div>`).join('');
}

function renderSites(){
  document.getElementById('sites').innerHTML = DATA.sites.map((site, si) => `
    <div class="site ${site.enabled?'':'off'}">
      <div class="site-head">
        <input type="checkbox" ${site.enabled?'checked':''} onchange="DATA.sites[${si}].enabled=this.checked;renderSites();renderPreview()">
        <div class="site-title">${site.id}</div>
        <span class="small">${site.trade} · 공정율 ${site.progress}% · ${site.hour}시</span>
        ${site.highRisk?'<span class="tag">고위험</span>':''}${site.small?'<span class="tag">소규모</span>':''}
      </div>
      ${site.categories.map((cat, ci) => `
        <div class="cat">
          <div class="cat-title">
            <input type="checkbox" ${cat.enabled?'checked':''} onchange="DATA.sites[${si}].categories[${ci}].enabled=this.checked;renderPreview()">
            ${cat.source==='model'?'🔴':'🟡'} <input type="text" value="${escapeHtml(cat.name)}" oninput="DATA.sites[${si}].categories[${ci}].name=this.value;renderPreview()">
          </div>
          <div class="actions">
            ${cat.actions.map((a, ai) => `
              <div class="action">
                <input type="checkbox" ${a.enabled?'checked':''} onchange="DATA.sites[${si}].categories[${ci}].actions[${ai}].enabled=this.checked;renderPreview()">
                <input type="text" value="${escapeHtml(a.text)}" oninput="DATA.sites[${si}].categories[${ci}].actions[${ai}].text=this.value;renderPreview()">
                <button class="ghost" onclick="DATA.sites[${si}].categories[${ci}].actions.splice(${ai},1);renderSites();renderPreview()">삭제</button>
              </div>`).join('')}
            <button class="ghost" onclick="DATA.sites[${si}].categories[${ci}].actions.push({text:'새 체크 항목',enabled:true});renderSites();renderPreview()">항목 추가</button>
          </div>
        </div>`).join('')}
    </div>`).join('');
}

async function renderPreview(){
  if(!DATA) return;
  DATA.note = document.getElementById('note').value || '';
  const data = await api('/api/preview', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(DATA)});
  document.getElementById('preview').value = data.message;
}

async function sendSlack(){
  try{
    DATA.note = document.getElementById('note').value || '';
    const data = await api('/api/send-slack', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(DATA)});
    if(data.ok) setStatus('Slack 전송 완료','ok');
    else setStatus(data.error || '전송 실패','err');
  }catch(e){ setStatus(e.message,'err'); }
}

function setStatus(text, cls=''){ const el=document.getElementById('status'); el.textContent=text; el.className='status '+cls; }
function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
document.getElementById('note').addEventListener('input', renderPreview);
loadAlert().catch(e => setStatus(e.message,'err'));
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
