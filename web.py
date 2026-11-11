"""Страница бота: состояние, сделки, кривая баланса, мысли и графики сети, монеты, Старт/Стоп."""
import collections
import hmac
import json
import logging
import os
import re
import threading
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

BOT = None
LOCK = threading.Lock()
STATUS = {"state": "запускаюсь", "error": "", "started": time.time()}
LOGS = collections.deque(maxlen=200)
CONTROL = {"paused": os.getenv("START_PAUSED", "") == "1", "msg": "", "symbols": [], "close_all": False}
CONTROL_KEY = os.getenv("CONTROL_KEY", "")
SYMBOLS_PATH = ""


def update(**kw):
    with LOCK:
        STATUS.update(kw)


def load_symbols(path):
    global SYMBOLS_PATH
    SYMBOLS_PATH = path
    try:
        with open(path, encoding="utf-8") as f:
            CONTROL["symbols"] = [str(x) for x in json.load(f)]
    except Exception:  # noqa: BLE001
        CONTROL["symbols"] = []


def parse_symbols(text):
    out = []
    for w in re.split(r"[\s,;]+", text.upper()):
        w = re.sub(r"[^A-Z0-9]", "", w)
        if not w or w == "USDT":
            continue
        if not w.endswith("USDT"):
            w += "USDT"
        if w not in out:
            out.append(w)
    return out[:20]


class Ring(logging.Handler):
    def emit(self, rec):
        LOGS.append(time.strftime("%H:%M:%S ", time.gmtime(rec.created)) + rec.getMessage())


# ---------------------------------------------------------------- маленькие графики (SVG)

def spark(series, w=520, h=120, colors=("#2a78d6", "#898781"), zero=False, fmt="{:.2f}"):
    """series: список рядов [(x, y), ...]; None в y — пропуск."""
    pts = [p for s in series for p in s if p[1] is not None]
    if len(pts) < 2:
        return '<p class="muted small">Данных пока нет</p>'
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts] + ([0] if zero else [])
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if y1 - y0 < 1e-9:
        y1 += 1
        y0 -= 1
    pad = (y1 - y0) * 0.08
    y0, y1 = y0 - pad, y1 + pad
    X = lambda x: 44 + (x - x0) / max(x1 - x0, 1e-9) * (w - 52)
    Y = lambda y: 8 + (y1 - y) / (y1 - y0) * (h - 24)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for k in range(3):
        yy = y0 + (y1 - y0) * k / 2
        out.append(f'<line x1="44" x2="{w - 8}" y1="{Y(yy):.1f}" y2="{Y(yy):.1f}" class="grid"/>'
                   f'<text x="40" y="{Y(yy) + 4:.1f}" class="ax" text-anchor="end">{escape(fmt.format(yy))}</text>')
    if zero and y0 < 0 < y1:
        out.append(f'<line x1="44" x2="{w - 8}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="zero"/>')
    for s, col in zip(series, colors):
        seg, path = [], []
        for x, y in s:
            if y is None:
                if len(seg) > 1:
                    path.append(seg)
                seg = []
            else:
                seg.append(f"{X(x):.1f},{Y(y):.1f}")
        if len(seg) > 1:
            path.append(seg)
        for sg in path:
            out.append(f'<polyline points="{" ".join(sg)}" fill="none" stroke="{col}" stroke-width="2" '
                       f'stroke-linejoin="round" stroke-linecap="round"/>')
    out.append("</svg>")
    return "".join(out)


def bufs(r):
    return "—" if r.get("best_buf") is None else f'{r["best_buf"]:g} ATR'


def bavg(r):
    return "—" if r.get("best_avg") is None else f'{r["best_avg"]:+.3f}R'


def best_r(o):
    r0 = (o["entry"] - o.get("stop0", o["stop"])) * o["dir"]
    return (o.get("best", o["entry"]) - o["entry"]) * o["dir"] / r0 if r0 > 0 else 0.0


def stop_r(o):
    r0 = (o["entry"] - o.get("stop0", o["stop"])) * o["dir"]
    return (o["stop"] - o["entry"]) * o["dir"] / r0 if r0 > 0 else 0.0


def tm(ts):
    return time.strftime("%d.%m %H:%M", time.gmtime(ts)) if ts else "—"


def render():
    b = BOT
    with LOCK:
        S = dict(STATUS)
    paused = CONTROL["paused"]
    key_in = ('<input name="key" type="password" placeholder="ключ управления" autocomplete="current-password">'
              if CONTROL_KEY else "")
    msg = f'<span class="cmsg">{escape(CONTROL["msg"])}</span>' if CONTROL["msg"] else ""
    syms = ", ".join(CONTROL["symbols"])
    body = []
    if b is None:
        body.append(f'<div class="panel"><p>{escape(S["state"])}</p><p class="err">{escape(S["error"])}</p></div>')
    else:
        a, st, br = b.a, b.st, b.brain.snapshot()
        try:
            bal = b.balance()
        except Exception:  # noqa: BLE001
            bal = None
        closed = st.closed
        n = len(closed)
        wins = sum(1 for x in closed if x["r"] > 0)
        tot_r = sum(x["r"] for x in closed)
        tot_usd = sum(x["pnl"] for x in closed)
        mode = "РЕАЛЬНЫЙ СЧЁТ" if a.live else "БУМАЖНАЯ ТОРГОВЛЯ"
        kv = [("Режим", mode), ("Состояние", S["state"]),
              ("Баланс", f"{bal:.2f} $" if bal is not None else "—"),
              ("Сегодня (UTC)", f'{st.day["r"]:+.2f}R · {st.day["usd"]:+.2f}$ · {st.day["n"]} сд.'),
              ("Всего сделок", f"{n} · в плюсе {wins / n:.0%}" if n else "0"),
              ("Итог", f"{tot_r:+.2f}R · {tot_usd:+.2f}$" if n else "—"),
              ("Средняя сделка", f"{tot_r / n:+.2f}R" if n else "—"),
              ("Риск / выход", f"{a.risk_pct:g}% баланса · трейлинг {a.trail_r:g}R с +{a.trail_start:g}R"),
              ("Сеть", ("ТОРГУЕТ" if br["active"] else "УЧИТСЯ") + " — " + br["why"])]
        if time.time() < st.pause_until:
            kv.append(("Пауза", "после серии стопов до " + tm(st.pause_until)))
        body.append('<div class="panel"><div class="kv">' + "".join(
            f'<div><span>{escape(k)}</span><b>{escape(str(v))}</b></div>' for k, v in kv) + "</div>"
            + (f'<p class="err">{escape(S["error"])}</p>' if S["error"] else "") + "</div>")
        # кривая баланса
        eq = st.equity[-500:]
        body.append('<div class="panel"><h2>Баланс после сделок</h2>' + spark([[(t, v) for t, v in eq]], fmt="{:.2f}") + "</div>")
        # открытые сделки
        rows = "".join(
            f'<tr><td>{escape(o["sym"])}</td><td class="{"pos" if o["dir"] > 0 else "neg"}">{"лонг" if o["dir"] > 0 else "шорт"}</td>'
            f'<td>{escape(o["setup"])} {o.get("tf", 1)}м</td><td>{best_r(o):+.2f}R</td>'
            f'<td>{o["entry"]:.6g}</td><td>{o["stop"]:.6g}</td><td>{stop_r(o):+.2f}R</td>'
            f'<td>{o.get("risk_usd", 0):.2f}$</td><td>{int((time.time() - o.get("opened", o.get("placed", time.time()))) / 60)} мин</td></tr>'
            for o in st.open)
        body.append('<div class="panel"><h2>Открытые сделки (выход по трейлингу)</h2><div class="scroll"><table><tr><th>Монета</th><th></th>'
                    '<th>Сетап</th><th>Лучшее</th><th>Вход</th><th>Стоп</th><th>Стоп сейчас</th><th>Риск</th><th>Идёт</th></tr>'
                    + (rows or '<tr><td colspan="9" class="muted">Нет</td></tr>') + "</table></div></div>")
        # исследование: какие сетапы реально в плюсе
        rs = br.get("research") or []
        rows = "".join(
            f'<tr><td>{escape(r["setup"])}</td><td>{r["tf"]}м</td><td>{r["n"]:,}</td>'
            f'<td class="{"pos" if r["avg"] > 0 else "neg"}">{r["avg"]:+.3f}R</td>'
            f'<td>{bufs(r)}</td><td class="{"pos" if (r["best_avg"] or 0) > 0 else "neg"}">{bavg(r)}</td>'
            f'<td>{r["tp"]:.0%}</td><td class="{"pos" if r["on"] else "muted"}">{"торгую" if r["on"] else "нет"}</td></tr>'
            for r in rs)
        body.append('<div class="panel"><h2>Какие сетапы в плюсе (реальная история, с комиссиями)</h2><div class="scroll"><table>'
                    '<tr><th>Сетап</th><th>ТФ</th><th>Сетапов</th><th>Средний R</th><th>Лучший запас стопа</th><th>R с ним</th>'
                    '<th>Дошли до +1R</th><th>Торговать</th></tr>'
                    + (rows or '<tr><td colspan="8" class="muted">Посчитаю после первого обучения</td></tr>') + "</table></div>"
                    f'<p class="muted small">Торгую сочетание, если на лучшем запасе стопа в среднем больше +{a.research_min_r:g}R '
                    f'на {a.research_min}+ сетапах. Вход по рынку, выход: стоп за структурой, безубыток при +{a.be_r:g}R, трейлинг {a.trail_r:g}R '
                    f'с +{a.trail_start:g}R, нет +{a.stale_r:g}R за {a.stale_bars} баров — выход. Комиссии тейкера и проскальзывание учтены.</p></div>')
        # сигналы этой минуты
        rows = "".join(
            f'<tr><td>{escape(x["sym"])}</td><td>{escape(x["setup"])} {x.get("tf", 1)}м {"↑" if x["dir"] > 0 else "↓"}</td>'
            f'<td class="{"pos" if x["pred"] >= 0 else "neg"}">{x["pred"]:+.2f}R</td><td>{x["risk_pct"]:.2f}%</td>'
            f'<td>{x["buf"]:g} ATR</td><td>{min(x["room"], 9):.1f}R</td>'
            f'<td class="{"pos" if x["take"] else "muted"}">{escape(x["note"])}</td></tr>' for x in b.signals[:15])
        body.append('<div class="panel"><h2>Сетапы этой минуты</h2><div class="scroll"><table><tr><th>Монета</th><th>Сетап</th>'
                    '<th>Сеть ждёт</th><th>Стоп</th><th>Запас</th><th>До уровня</th><th>Решение</th></tr>'
                    + (rows or '<tr><td colspan="7" class="muted">Сетапов нет — жду</td></tr>') + "</table></div></div>")
        # графики сети
        oh = br["online_hist"]
        body.append('<div class="grid2"><div class="panel"><h2>Онлайн-проверка: отбор сети vs все сетапы, R</h2>'
                    + spark([[(t, s) for t, s, _ in oh], [(t, al) for t, _, al in oh]], zero=True)
                    + '<p class="small"><span class="sw" style="background:#2a78d6"></span>сеть отобрала '
                      '<span class="sw" style="background:#898781"></span>все сетапы</p></div>'
                    '<div class="panel"><h2>Память сети, примеров</h2>'
                    + spark([[(t, v) for t, v in br["mem_hist"]]], fmt="{:.0f}") + "</div>"
                    '<div class="panel"><h2>Переобучение: ошибка по эпохам</h2>'
                    + spark([[(k, tr) for k, (tr, _) in enumerate(br["train_hist"])],
                             [(k, va) for k, (_, va) in enumerate(br["train_hist"])]], colors=("#1baf7a", "#eda100"), fmt="{:.3f}")
                    + '<p class="small"><span class="sw" style="background:#1baf7a"></span>обучение '
                      '<span class="sw" style="background:#eda100"></span>проверка</p></div>'
                    '<div class="panel"><h2>На что сеть опирается</h2>'
                    + ("".join(f'<div class="bar"><span>{escape(nm)}</span><i style="width:{max(4, v / max(br["importance"][0][1], 1e-9) * 100):.0f}%"></i></div>'
                               for nm, v in br["importance"]) or '<p class="muted small">После первого обучения</p>')
                    + "</div></div>")
        on, v = br["online"], br["val"]
        body.append('<div class="panel"><h2>Сеть</h2><div class="kv">' + "".join(
            f'<div><span>{escape(k)}</span><b>{escape(str(x))}</b></div>' for k, x in [
                ("Примеров в памяти", f'{br["samples"]:,} (своих сделок {br["real"]})'),
                ("Ждут исхода", f"{len(b.pending):,}"),
                (f"Онлайн {a.online_hours:g} ч: отобранные", f'{on["sel"]} · ' + (f'{on["sel_avg"]:+.2f}R, в плюсе {on["wins"]:.0%}' if on["sel"] else "—")),
                (f"Онлайн {a.online_hours:g} ч: все сетапы", f'{on["n"]} · ' + (f'{on["all_avg"]:+.2f}R' if on["n"] else "—")),
                ("Проверка на свежем куске", f'{v.get("sel", 0)} из {v.get("n", 0)} · {v.get("sel_avg", 0):+.2f}R против {v.get("base", 0):+.2f}R' if v else "—"),
                ("Последнее переобучение", tm(br["last_train"]))]) + "</div></div>")
        # мысли
        th = "".join(
            f'<div class="th"><div><span class="tag {escape(x["kind"])}">{escape(x["kind"])}</span><b>{escape(x["title"])}</b>'
            f'<em>{tm(x["time"])}</em></div><ul>' + "".join(f"<li>{escape(l)}</li>" for l in x["lines"] if l) + "</ul></div>"
            for x in reversed(br["thoughts"]))
        body.append(f'<div class="panel"><h2>Мысли сети (новое сверху)</h2><div class="thoughts">{th or "—"}</div></div>')
        # монеты
        uni = ", ".join(b.universe) or "—"
        vol = "".join(f'<tr><td>{escape(x["symbol"])}</td><td>{x["atr"]:.2f}%</td><td>{x["chg"]:+.1f}%</td>'
                      f'<td>{x["turnover"] / 1e6:.0f}M</td></tr>' for x in b.uni_info[:15])
        body.append(f'<div class="panel"><h2>Монеты</h2><p>Торгую: <b>{escape(uni)}</b>'
                    f'{" (со страницы)" if CONTROL["symbols"] else " (самые волатильные)"}</p>'
                    '<div class="scroll"><table><tr><th>Самые волатильные</th><th>Размах минуты</th><th>За 24ч</th><th>Оборот</th></tr>'
                    + (vol or '<tr><td colspan="4" class="muted">—</td></tr>') + "</table></div></div>")
        # закрытые сделки
        rows = "".join(
            f'<tr><td>{tm(x["closed"])}</td><td>{escape(x["sym"])}</td><td>{"лонг" if x["dir"] > 0 else "шорт"} · {escape(x["setup"])} {x.get("tf", 1)}м</td>'
            f'<td>{escape(x["why"])}</td><td class="{"pos" if x["r"] > 0 else "neg"}">{x["r"]:+.2f}R</td>'
            f'<td class="{"pos" if x["pnl"] > 0 else "neg"}">{x["pnl"]:+.2f}$</td><td>{x["pred"]:+.2f}R</td></tr>'
            for x in reversed(closed[-40:]))
        body.append('<div class="panel"><h2>Закрытые сделки</h2><div class="scroll"><table><tr><th>Время UTC</th><th>Монета</th>'
                    '<th>Сделка</th><th>Выход</th><th>R</th><th>$</th><th>Сеть ждала</th></tr>'
                    + (rows or '<tr><td colspan="7" class="muted">Пока нет</td></tr>') + "</table></div></div>")
    logs = escape("\n".join(reversed(LOGS)))
    controls = f"""<form class="ctrl" method="post">{key_in}
<button formaction="/api/start" class="start" {'disabled' if not paused else ''}>Старт</button>
<button formaction="/api/stop" class="stop" {'disabled' if paused else ''}>Стоп</button>
<span class="cstate {'off' if paused else 'on'}">{'На паузе' if paused else 'Работает'}</span>{msg}
<div class="syms"><input name="symbols" value="{escape(syms)}" placeholder="пусто — самые волатильные; или: SOL, DOGE" autocomplete="off">
<button formaction="/api/symbols" class="save">Сохранить монеты</button></div></form>
<p class="muted small">«Стоп» закрывает открытые сделки и не даёт открывать новые (сеть продолжает учиться). Поле монет пустое — бот сам берёт
самые волатильные; вписал монеты — торгует только их.</p>"""
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Price Action бот</title><style>
:root{{--bg:#0f1312;--panel:#161b1a;--line:#26302e;--fg:#e7ecea;--muted:#8b9692;--pos:#34c58f;--neg:#ef6b5e}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}}.wrap{{max-width:1100px;margin:0 auto;padding:16px}}
h1{{font-size:20px;margin:0 0 12px}}h2{{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:0 0 14px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}}.grid2 .panel{{margin:0}}.grid2{{margin-bottom:14px}}
.kv{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}}.kv span{{display:block;color:var(--muted);font-size:12px}}.kv b{{font:600 15px ui-monospace,monospace}}
table{{width:100%;border-collapse:collapse;font:13px ui-monospace,monospace}}td,th{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
td:first-child,th:first-child{{text-align:left}}th{{font:600 11px system-ui;color:var(--muted);text-transform:uppercase}}.scroll{{overflow-x:auto}}
.pos{{color:var(--pos)}}.neg{{color:var(--neg)}}.muted{{color:var(--muted)}}.small{{font-size:12px}}.err{{color:var(--neg)}}
.chart{{width:100%;height:auto}}.chart .grid{{stroke:var(--line)}}.chart .zero{{stroke:#5b6663;stroke-dasharray:4 4}}.chart .ax{{fill:var(--muted);font:10px ui-monospace,monospace}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px;vertical-align:-1px}}
.bar{{display:grid;grid-template-columns:160px 1fr;gap:8px;align-items:center;margin:5px 0;font-size:13px}}.bar i{{display:block;height:8px;border-radius:4px;background:#34c58f}}
.thoughts{{max-height:420px;overflow:auto}}.th{{border-left:3px solid var(--line);padding:4px 10px;margin:0 0 10px}}.th b{{margin-left:8px}}.th em{{float:right;color:var(--muted);font:12px ui-monospace,monospace}}
.th ul{{margin:4px 0 0;padding-left:18px;color:#c9d1ce}}.tag{{font:700 10px system-ui;text-transform:uppercase;padding:2px 6px;border-radius:4px;background:#2a2f45;color:#a9a3ff}}
.tag.trade{{background:#173a2c;color:#34c58f}}.tag.result{{background:#3a2a17;color:#eda100}}
.ctrl{{display:flex;flex-wrap:wrap;gap:10px;align-items:center}}.ctrl input{{font:14px ui-monospace,monospace;padding:8px 10px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg);width:180px}}
.ctrl button{{font:700 14px system-ui;padding:9px 22px;border-radius:6px;border:0;cursor:pointer}}.ctrl button:disabled{{opacity:.35}}
.start{{background:#34c58f;color:#06231a}}.stop{{background:#ef6b5e;color:#2a0905}}.save{{background:#3b82f6;color:#fff}}
.syms{{display:flex;gap:8px;width:100%}}.syms input{{flex:1;width:auto;text-transform:uppercase}}.cstate.on{{color:var(--pos)}}.cstate.off{{color:var(--neg)}}.cmsg{{color:var(--muted)}}
pre{{max-height:340px;overflow:auto;font:12px/1.45 ui-monospace,monospace;white-space:pre-wrap;margin:0}}
</style></head><body><div class="wrap"><h1>Price Action бот</h1>
<div class="panel">{controls}</div>{"".join(body)}
<div class="panel"><h2>Лог</h2><pre>{logs}</pre></div>
<p class="muted small">Страница обновляется каждые 10 сек. Время — UTC.</p></div>
<script>
const keyIn=()=>document.querySelector('input[name=key]');
try{{const k=keyIn();if(k)k.value=localStorage.getItem('paKey')||''}}catch(e){{}}
document.addEventListener('submit',()=>{{const k=keyIn();try{{if(k)localStorage.setItem('paKey',k.value)}}catch(e){{}}}});
setInterval(async()=>{{
 if(document.hidden)return;const ae=document.activeElement;if(ae&&ae.tagName==='INPUT')return;
 const sel=window.getSelection&&window.getSelection();if(sel&&sel.toString())return;
 let h;try{{h=await(await fetch('/',{{cache:'no-store'}})).text()}}catch(e){{return}}
 const n=new DOMParser().parseFromString(h,'text/html').querySelector('.wrap'),c=document.querySelector('.wrap');
 if(!n||!c)return;const k=keyIn(),kv=k?k.value:'',y=window.scrollY;c.replaceWith(n);
 const k2=keyIn();if(k2)k2.value=kv;window.scrollTo(0,y);
}},10000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            try:
                body = render().encode()
            except Exception as e:  # noqa: BLE001
                body = f"<pre>Ошибка страницы: {escape(str(e))}</pre>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        elif self.path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(min(n, 4096)).decode("utf-8", "ignore"))
        key = (form.get("key") or [""])[0]
        lg = logging.getLogger("pa")
        if CONTROL_KEY and not hmac.compare_digest(key, CONTROL_KEY):
            CONTROL["msg"] = "Неверный ключ управления"
        elif self.path.startswith("/api/stop"):
            CONTROL["paused"], CONTROL["close_all"] = True, True
            CONTROL["msg"] = "Остановлен: сделки закрываются, новых не будет"
            lg.info("СТОП со страницы")
        elif self.path.startswith("/api/start"):
            CONTROL["paused"] = False
            CONTROL["msg"] = "Запущен"
            lg.info("СТАРТ со страницы")
        elif self.path.startswith("/api/symbols"):
            CONTROL["symbols"] = parse_symbols((form.get("symbols") or [""])[0])
            CONTROL["msg"] = ("Монеты: " + ", ".join(CONTROL["symbols"])) if CONTROL["symbols"] else "Монеты: самые волатильные"
            lg.info("Монеты со страницы: %s", ", ".join(CONTROL["symbols"]) or "авто (самые волатильные)")
            if BOT is not None:
                BOT.uni_time = 0
            if SYMBOLS_PATH:
                try:
                    with open(SYMBOLS_PATH, "w", encoding="utf-8") as f:
                        json.dump(CONTROL["symbols"], f)
                except Exception:  # noqa: BLE001
                    pass
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, *_):
        pass


def start():
    logging.getLogger("pa").addHandler(Ring())
    port = int(os.getenv("PORT") or 8080)
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    logging.getLogger("pa").info("Страница на порту %d", port)
