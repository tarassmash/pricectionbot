"""Price Action скальпер для Bybit (USDT-фьючерсы) с самообучающейся сетью.

Каждую минуту:
  1. берёт самые волатильные монеты (или монеты со страницы),
  2. дочитывает минутные свечи и ищет сетапы price action (pa.py),
  3. для каждого сетапа сеть оценивает варианты умного стопа и выбирает лучший,
  4. все сетапы (взятые и нет) ждут своего исхода и идут в обучение сети,
  5. если сеть «в форме» (онлайн-проверка в плюсе) и ждёт от сетапа ≥ --min-edge R — входит:
     риск --risk-pct % баланса, стоп за структурой, тейк RR (по умолчанию 1:2).

По умолчанию — БУМАЖНАЯ торговля (реальные цены, без ордеров). Реальный счёт: переменная LIVE=1.
"""
import argparse
import json
import logging
import os
import shlex
import signal
import sys
import threading
import time
from decimal import ROUND_DOWN, Decimal as D

import numpy as np
from pybit.unified_trading import HTTP

import brain as brain_mod
import pa
import web

CAT = "linear"
log = logging.getLogger("pa")
STOP = threading.Event()


# ---------------------------------------------------------------- параметры

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true", help="реальный счёт (то же, что LIVE=1)")
    p.add_argument("--paper-balance", type=float, default=200, help="стартовый бумажный баланс, USDT")
    p.add_argument("--risk-pct", type=float, default=0.5, help="риск на сделку, %% баланса")
    p.add_argument("--rr", type=float, default=2.0, help="тейк = RR × риск (1:2)")
    p.add_argument("--max-open", type=int, default=3, help="сделок одновременно (по одной на монету)")
    p.add_argument("--leverage", type=float, default=10, help="плечо на бирже (влияет только на маржу)")
    p.add_argument("--max-notional-x", type=float, default=3, help="объём сделки не больше X × баланс")
    p.add_argument("--day-loss-r", type=float, default=4, help="за сутки UTC −X R — новых сделок до завтра")
    p.add_argument("--loss-streak", type=int, default=4, help="столько стопов подряд — пауза --streak-pause мин")
    p.add_argument("--streak-pause", type=int, default=60)
    p.add_argument("--maker-pct", type=float, default=0.02, help="комиссия мейкера (лимитка), %%")
    p.add_argument("--taker-pct", type=float, default=0.055, help="комиссия тейкера (стоп, выход по времени), %%")
    p.add_argument("--tfs", default="1,3,5", help="таймфреймы сетапов, мин")
    p.add_argument("--research-min", type=int, default=150, help="сетапов, чтобы судить о сетапе/ТФ")
    p.add_argument("--research-min-r", type=float, default=0.03, help="торговать сетап/ТФ, если в среднем > X R")
    # монеты
    p.add_argument("--symbols", default="", help="монеты через запятую (иначе — самые волатильные; страница главнее)")
    p.add_argument("--top", type=int, default=10, help="сколько самых волатильных монет торговать")
    p.add_argument("--learn-top", type=int, default=25, help="на скольких монетах учиться")
    p.add_argument("--min-turnover", type=float, default=20e6, help="мин. оборот за 24ч, USDT")
    p.add_argument("--max-24h", type=float, default=80, help="не брать монеты, улетевшие за 24ч больше ±X %%")
    # стопы и сетапы
    p.add_argument("--min-risk-atr", type=float, default=0.6, help="стоп не ближе X ATR (иначе выбьет шумом)")
    p.add_argument("--max-risk-atr", type=float, default=4.0)
    p.add_argument("--min-risk-pct", type=float, default=0.5, help="стоп не ближе X %% (иначе съедят комиссии)")
    p.add_argument("--min-room", type=float, default=1.2, help="до встречного уровня минимум X R")
    # сеть
    p.add_argument("--min-edge", type=float, default=0.15, help="входить, если сеть ждёт ≥ X R")
    p.add_argument("--history-days", type=float, default=3, help="история свечей для первого обучения, дней")
    p.add_argument("--memory-days", type=float, default=7)
    p.add_argument("--max-samples", type=int, default=150_000)
    p.add_argument("--min-samples", type=int, default=800)
    p.add_argument("--half-life-h", type=float, default=24, help="вес примера вдвое меньше каждые X часов")
    p.add_argument("--retrain-min", type=int, default=30)
    p.add_argument("--online-hours", type=float, default=6)
    p.add_argument("--min-online", type=int, default=20, help="отобранных онлайн-итогов, чтобы им верить")
    p.add_argument("--min-online-r", type=float, default=0.05, help="онлайн-отбор должен давать > X R")
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR") or ("/data" if os.path.isdir("/data") else "data"))
    argv = sys.argv[1:]
    env = os.getenv("BOT_ARGS", "").strip()
    if env:
        for w in shlex.split(env.replace("—", "--")):
            argv += w.split("=", 1) if w.startswith("--") and "=" in w else [w]
    a, unknown = p.parse_known_args(argv)
    if unknown:
        log.warning("Непонятные параметры пропущены: %s", " ".join(unknown))
    a.live = a.live or os.getenv("LIVE", "") in ("1", "true", "yes", "YES")
    a.tfs = [int(x) for x in str(a.tfs).replace(" ", "").split(",") if x and int(x) in pa.TFS] or [1]
    return a


# ---------------------------------------------------------------- биржа

def api(fn, **kw):
    for k in range(4):
        try:
            r = fn(**kw)
            if r.get("retCode", 0) not in (0, 110043, 34040):
                raise RuntimeError(f"{r.get('retCode')}: {r.get('retMsg')}")
            return r
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if any(x in msg for x in ("110043", "34040", "not modified")):
                return {"retCode": 0, "result": {}}
            if k == 3 or STOP.is_set():
                raise
            time.sleep(1.5 * (k + 1))


def load_instruments(s):
    out, cursor = {}, ""
    while True:
        r = api(s.get_instruments_info, category=CAT, limit=1000, cursor=cursor)["result"]
        for x in r["list"]:
            if x.get("quoteCoin") != "USDT" or x.get("status") != "Trading" or x.get("contractType") != "LinearPerpetual":
                continue
            lf, pf = x["lotSizeFilter"], x["priceFilter"]
            out[x["symbol"]] = {"tick": D(pf["tickSize"]), "step": D(lf["qtyStep"]), "min_qty": D(lf["minOrderQty"]),
                                "min_notional": D(lf.get("minNotionalValue") or "5"),
                                "max_lev": float(x.get("leverageFilter", {}).get("maxLeverage") or 10)}
        cursor = r.get("nextPageCursor") or ""
        if not cursor:
            return out


def q_down(x, step):
    return (D(str(x)) / step).to_integral_value(ROUND_DOWN) * step


def q_price(x, tick):
    return (D(str(x)) / tick).quantize(D(1)) * tick


class Candles:
    """Минутные свечи по монетам: первая загрузка — история, дальше дочитываем хвост."""

    def __init__(self, s, keep=1500):
        self.s, self.keep = s, keep
        self.data = {}      # sym -> np.array [[t,o,h,l,c,v], ...] только закрытые свечи

    def _fetch(self, sym, limit, end=None):
        kw = {"category": CAT, "symbol": sym, "interval": "1", "limit": limit}
        if end:
            kw["end"] = int(end)
        rows = api(self.s.get_kline, **kw)["result"]["list"]
        arr = np.array([[float(x) for x in r[:6]] for r in rows][::-1])
        return arr

    def history(self, sym, minutes):
        """Длинная история (для первого обучения), постранично по 1000."""
        out, end = [], None
        while minutes > 0 and not STOP.is_set():
            arr = self._fetch(sym, min(1000, minutes + 1), end)
            if not len(arr):
                break
            out.insert(0, arr)
            minutes -= len(arr)
            end = arr[0, 0] - 1
            if len(arr) < 2:
                break
        if not out:
            return np.zeros((0, 6))
        arr = np.vstack(out)
        _, idx = np.unique(arr[:, 0], return_index=True)
        arr = arr[idx]
        now_min = int(time.time() // 60) * 60_000
        return arr[arr[:, 0] < now_min]            # без текущей незакрытой

    def update(self, sym):
        now_min = int(time.time() // 60) * 60_000
        cur = self.data.get(sym)
        need = 1000 if cur is None or not len(cur) else min(1000, int((now_min - cur[-1, 0]) // 60_000) + 2)
        if cur is not None and len(cur) and cur[-1, 0] >= now_min - 60_000:
            return cur
        arr = self._fetch(sym, max(need, 3))
        arr = arr[arr[:, 0] < now_min] if len(arr) else arr
        if cur is not None and len(cur):
            arr = np.vstack([cur, arr[arr[:, 0] > cur[-1, 0]]]) if len(arr) else cur
        self.data[sym] = arr[-self.keep:]
        return self.data[sym]


# ---------------------------------------------------------------- состояние (сделки, бумажный баланс)

class State:
    def __init__(self, path, a):
        self.path, self.a = path, a
        self.paper_balance = a.paper_balance
        self.open = []          # открытые сделки
        self.closed = []        # закрытые (последние 500)
        self.equity = []        # (время, баланс)
        self.day = {"key": "", "r": 0.0, "usd": 0.0, "n": 0}
        self.streak = 0
        self.pause_until = 0.0
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("live") == a.live:
                self.paper_balance = d.get("paper_balance", a.paper_balance)
                self.open, self.closed = d.get("open", []), d.get("closed", [])
                self.equity, self.day = d.get("equity", []), d.get("day", self.day)
                self.streak, self.pause_until = d.get("streak", 0), d.get("pause_until", 0.0)
        except Exception:  # noqa: BLE001
            pass

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"live": self.a.live, "paper_balance": self.paper_balance, "open": self.open,
                           "closed": self.closed[-500:], "equity": self.equity[-3000:], "day": self.day,
                           "streak": self.streak, "pause_until": self.pause_until}, f)
            os.replace(self.path + ".tmp", self.path)
        except Exception as e:  # noqa: BLE001
            log.warning("Не сохранил состояние: %s", e)

    def roll_day(self):
        k = time.strftime("%Y-%m-%d", time.gmtime())
        if self.day["key"] != k:
            self.day = {"key": k, "r": 0.0, "usd": 0.0, "n": 0}


# ---------------------------------------------------------------- бот

class Bot:
    def __init__(self, a):
        self.a = a
        key, sec = os.getenv("BYBIT_API_KEY", ""), os.getenv("BYBIT_API_SECRET", "")
        self.market = HTTP(testnet=False)
        self.s = HTTP(testnet=False, api_key=key, api_secret=sec) if a.live else None
        if a.live and not (key and sec):
            raise SystemExit("LIVE=1, но нет BYBIT_API_KEY / BYBIT_API_SECRET")
        self.inst = load_instruments(self.market)
        self.cd = Candles(self.market)
        self.st = State(os.path.join(a.data_dir, "state.json"), a)
        self.brain = brain_mod.Brain(os.path.join(a.data_dir, "brain"), len(pa.FEAT_NAMES), a, pa.FEAT_NAMES)
        self.pending = []           # сетапы, ждущие исхода: {sym, t, plan, pred}
        self.universe, self.learn_set, self.uni_time, self.uni_info = [], [], 0.0, []
        self.signals = []           # последние сетапы для страницы
        self.pos_mode = {}
        self.lev_set = set()
        self.last_minute = 0
        self.last_eval = {}         # (монета, ТФ) -> время последнего разобранного бара
        self.training = False
        web.BOT = self

    # ------------------------------------------------ монеты
    def pick_universe(self):
        manual = list(web.CONTROL["symbols"])
        if time.time() - self.uni_time < 900 and self.learn_set and not manual:
            return
        tick = api(self.market.get_tickers, category=CAT)["result"]["list"]
        cand = []
        for t in tick:
            sym = t["symbol"]
            if sym not in self.inst:
                continue
            to = float(t.get("turnover24h") or 0)
            chg = float(t.get("price24hPcnt") or 0) * 100
            if to >= self.a.min_turnover and abs(chg) <= self.a.max_24h:
                cand.append((sym, to, chg))
        cand.sort(key=lambda x: -x[1])
        cand = cand[:45]
        vol = []
        for sym, to, chg in cand:
            try:
                arr = self.cd._fetch(sym, 61)
            except Exception:  # noqa: BLE001
                continue
            if len(arr) < 30:
                continue
            h, l, c = arr[:, 2], arr[:, 3], arr[:, 4]
            atrp = float(np.mean((h - l) / c) * 100)          # средний размах минутки за час, %
            vol.append({"symbol": sym, "atr": atrp, "turnover": to, "chg": chg})
        vol.sort(key=lambda x: -x["atr"])
        self.uni_info = vol[:self.a.learn_top]
        auto = [x["symbol"] for x in vol[:self.a.top]]
        self.universe = [m for m in manual if m in self.inst] if manual else auto
        learn = list(dict.fromkeys(self.universe + [x["symbol"] for x in vol[:self.a.learn_top]]))
        if "BTCUSDT" not in learn:
            learn.append("BTCUSDT")
        self.learn_set = learn
        self.uni_time = time.time()
        log.info("Торгую: %s", ", ".join(self.universe))

    # ------------------------------------------------ первое обучение на истории
    def bootstrap(self):
        if len(self.brain.y) >= self.a.min_samples:
            return
        minutes = int(self.a.history_days * 1440)
        self.brain.think("data", "Беру историю для первого обучения",
                         [f"{self.a.history_days:g} дн. минутных свечей по {len(self.learn_set)} монетам"])
        btc = self.cd.history("BTCUSDT", minutes)
        btc_map = {int(r[0]): r[4] for r in btc}
        total = 0
        for sym in self.learn_set:
            if STOP.is_set():
                return
            try:
                arr = btc if sym == "BTCUSDT" else self.cd.history(sym, minutes)
            except Exception as e:  # noqa: BLE001
                log.info("%s: история не загрузилась: %s", sym, str(e)[:80])
                continue
            X, y, t = self.label_history(arr, btc_map)
            self.brain.add(X, y, t)
            total += len(y)
            web.update(state=f"учусь на истории: {sym}, примеров {total:,}")
        self.brain.think("data", "История размечена", [f"Сетапов с известным исходом: {total:,}"])
        self.brain.train()

    def label_history(self, arr, btc_map):
        X, y, t = [], [], []
        a = self.a
        for tf in a.tfs:
            agg = pa.aggregate(arr, tf)
            cfg = pa.TFS[tf]
            if len(agg) < 120:
                continue
            T, O, H, L, C, V = (agg[:, k] for k in range(6))
            cx = pa.Ctx(T, O, H, L, C, V, btc_map, tf)
            for i in range(80, len(C) - cfg["hold"] - cfg["wait"] - 1):
                for pl in pa.plans(cx, a, i):
                    res = pa.outcome(pl, H, L, C, i + 1, cfg["hold"], cfg["wait"], a.maker_pct, a.taker_pct)
                    if res:
                        X.append(pl["feats"])
                        y.append(res[0])
                        t.append(T[i])
        return X, y, t

    # ------------------------------------------------ раз в минуту: свечи, исходы, сетапы, решения
    def on_minute(self):
        self.pick_universe()
        btc = self.cd.update("BTCUSDT")
        btc_map = {int(r[0]): r[4] for r in btc}
        fresh = []
        for sym in self.learn_set:
            try:
                arr = self.cd.update(sym)
            except Exception as e:  # noqa: BLE001
                log.info("%s: свечи: %s", sym, str(e)[:80])
                continue
            for tf in self.a.tfs:
                agg = pa.aggregate(arr, tf)
                if len(agg) < 100:
                    continue
                key = (sym, tf)
                if self.last_eval.get(key) == agg[-1, 0]:      # бар этого ТФ ещё не закрылся
                    continue
                self.last_eval[key] = agg[-1, 0]
                T, O, H, L, C, V = (agg[:, k] for k in range(6))
                cx = pa.Ctx(T, O, H, L, C, V, btc_map, tf)
                i = cx.n - 1
                pls = pa.plans(cx, self.a, i)
                if not pls:
                    continue
                preds = self.brain.predict([p["feats"] for p in pls])
                for k, p in enumerate(pls):
                    p["pred"] = None if preds is None else float(preds[k])
                    p["t_bar"] = float(T[i])
                    self.pending.append({"sym": sym, "tf": tf, "t": float(T[i]), "plan": p})
                fresh += [(sym, p) for p in pls]
        self.resolve()
        self.decide(fresh)
        self.brain.mem_hist.append((time.time(), len(self.brain.y)))
        on = self.brain.online_stats()
        self.brain.online_hist.append((time.time(), on["sel_avg"], on["all_avg"]))
        if time.time() - self.brain.last_train > self.a.retrain_min * 60 and not self.training:
            self.training = True
            threading.Thread(target=self._train_bg, daemon=True).start()

    def _train_bg(self):
        try:
            self.brain.train()
        except Exception as e:  # noqa: BLE001
            log.error("Обучение: %s", e)
        finally:
            self.training = False

    def resolve(self):
        """Сетапы, у которых исход уже известен, — в обучение и в онлайн-проверку."""
        keep, X, y, t, n_on = [], [], [], [], 0
        aggs = {}
        a = self.a
        for p in self.pending:
            k = (p["sym"], p["tf"])
            if k not in aggs:
                arr = self.cd.data.get(p["sym"])
                aggs[k] = pa.aggregate(arr, p["tf"]) if arr is not None else None
            agg = aggs[k]
            if agg is None or not len(agg):
                continue
            cfg = pa.TFS[p["tf"]]
            start = int(np.searchsorted(agg[:, 0], p["t"], side="right"))
            res = pa.outcome(p["plan"], agg[:, 2], agg[:, 3], agg[:, 4], start, cfg["hold"], cfg["wait"],
                             a.maker_pct, a.taker_pct) if start < len(agg) else None
            if res is None:
                if time.time() * 1000 - p["t"] < ((cfg["hold"] + cfg["wait"]) * p["tf"] + 30) * 60_000:
                    keep.append(p)
                continue
            X.append(p["plan"]["feats"])
            y.append(res[0])
            t.append(p["t"])
            if p["plan"].get("pred") is not None and p["t"] > self.brain.trained_until:
                self.brain.add_online(p["plan"]["pred"], res[0])
                n_on += 1
        self.pending = keep[-30000:]
        self.brain.add(X, y, t)
        if y:
            self.brain.think("learn", "Учусь на ходу", [
                f"Новых исходов: {len(y)} (тейков {sum(1 for r in y if r > 1)}), в онлайн-проверку: {n_on}",
                f"В памяти {len(self.brain.y):,} примеров, ждут исхода {len(self.pending)}"])

    def decide(self, fresh):
        self.signals = []
        best = {}
        for sym, p in fresh:          # по каждому сетапу — лучший вариант стопа
            k = (sym, p["tf"], p["setup"], p["dir"])
            if p.get("pred") is not None and (k not in best or p["pred"] > best[k]["pred"]):
                best[k] = p
        ok, why = self.brain.active()
        enabled = self.brain.enabled
        st = self.st
        st.roll_day()
        for (sym, tf, setup, d), p in sorted(best.items(), key=lambda x: -x[1]["pred"]):
            take = sym in self.universe and p["pred"] >= self.a.min_edge
            note = ""
            if take and enabled is not None and (tf, setup) not in enabled:
                take, note = False, "этот сетап/ТФ в минусе на истории"
            elif take and not ok:
                take, note = False, "сеть учится"
            elif take and any(o["sym"] == sym for o in st.open):
                take, note = False, "уже в сделке"
            elif take and len(st.open) >= self.a.max_open:
                take, note = False, f"уже {self.a.max_open} сделки"
            elif take and st.day["r"] <= -self.a.day_loss_r:
                take, note = False, "дневной лимит"
            elif take and time.time() < st.pause_until:
                take, note = False, "пауза после серии стопов"
            elif take and web.CONTROL["paused"]:
                take, note = False, "на паузе"
            self.signals.append({"sym": sym, "tf": tf, "setup": setup, "dir": d, "pred": p["pred"],
                                 "risk_pct": p["risk"] / p["entry"] * 100, "buf": p["buf"], "room": p["room"],
                                 "take": take, "note": note or ("беру" if take else "слабый")})
            if take:
                self.enter(sym, p)

    # ------------------------------------------------ сделки
    def enter(self, sym, p):
        """Вход лимиткой по цене сигнала (мейкер). Ждёт заливки wait баров ТФ, потом отменяется."""
        a, ins = self.a, self.inst[sym]
        bal = self.balance()
        qty = min(bal * a.risk_pct / 100 / p["risk"], bal * a.max_notional_x / p["entry"])
        qty = q_down(qty, ins["step"])
        if qty < ins["min_qty"] or qty * D(str(p["entry"])) < ins["min_notional"]:
            log.info("%s: объём %s меньше минимума биржи — пропускаю", sym, qty)
            return
        d, tf = p["dir"], p["tf"]
        cfg = pa.TFS[tf]
        price = float(q_price(p["entry"], ins["tick"]))
        tr = {"sym": sym, "setup": p["setup"], "tf": tf, "dir": d, "qty": float(qty), "entry": price,
              "stop": float(q_price(p["stop"], ins["tick"])), "pred": p["pred"], "buf": p["buf"], "room": p["room"],
              "feats": p["feats"], "status": "order", "placed": time.time(),
              "deadline": time.time() + cfg["wait"] * tf * 60 + 20, "hold_min": cfg["hold"] * tf}
        tr["tp"] = float(q_price(price + d * a.rr * (price - tr["stop"]) * d, ins["tick"]))
        if (price - tr["stop"]) * d <= 0:
            return
        if a.live:
            idx = self.mode(sym)
            if sym not in self.lev_set:
                lev = str(min(a.leverage, ins["max_lev"]))
                api(self.s.set_leverage, category=CAT, symbol=sym, buyLeverage=lev, sellLeverage=lev)
                self.lev_set.add(sym)
            try:
                r = api(self.s.place_order, category=CAT, symbol=sym, side="Buy" if d > 0 else "Sell",
                        orderType="Limit", qty=str(qty), price=str(price), timeInForce="PostOnly", positionIdx=idx(d))
                tr["order_id"] = r["result"]["orderId"]
            except Exception as e:  # noqa: BLE001
                log.info("%s: лимитка на вход не встала: %s", sym, str(e)[:100])
                return
        tr["risk_usd"] = (price - tr["stop"]) * d * tr["qty"]
        self.st.open.append(tr)
        self.st.save()
        log.info("ЛИМИТКА %s %s %s %dм: %s @ %.6g, стоп %.6g, тейк %.6g, риск %.2f$ (сеть ждёт %+.2fR)",
                 sym, "ЛОНГ" if d > 0 else "ШОРТ", p["setup"], tf, tr["qty"], price, tr["stop"], tr["tp"],
                 tr["risk_usd"], p["pred"])
        self.brain.think("trade", f"Ставлю лимитку: {sym} {'лонг' if d > 0 else 'шорт'} ({p['setup']}, {tf}м)", [
            f"Вход {price:.6g}, стоп {tr['stop']:.6g} (за структурой + {p['buf']:g} ATR), тейк {tr['tp']:.6g} (1:{a.rr:g})",
            f"Жду {p['pred']:+.2f}R; до встречного уровня {p['room']:.1f}R; риск {tr['risk_usd']:.2f}$; "
            f"лимитка ждёт {cfg['wait'] * tf} мин"])

    def balance(self):
        if not self.a.live:
            return self.st.paper_balance
        r = api(self.s.get_wallet_balance, accountType="UNIFIED", coin="USDT")["result"]["list"]
        for c in (r[0].get("coin") or []) if r else []:
            if c["coin"] == "USDT":
                return float(c.get("walletBalance") or 0)
        return 0.0

    def price(self, sym):
        t = api(self.market.get_tickers, category=CAT, symbol=sym)["result"]["list"][0]
        return float(t["lastPrice"])

    def mode(self, sym):
        if sym not in self.pos_mode:
            lst = api(self.s.get_positions, category=CAT, symbol=sym)["result"]["list"]
            self.pos_mode[sym] = any(int(x.get("positionIdx", 0)) in (1, 2) for x in lst)
        hedge = self.pos_mode[sym]
        return lambda d: (1 if d > 0 else 2) if hedge else 0

    def position(self, sym, d):
        lst = api(self.s.get_positions, category=CAT, symbol=sym)["result"]["list"]
        for x in lst:
            side = x.get("side")
            if float(x.get("size") or 0) > 0 and ((side == "Buy") == (d > 0)):
                return float(x["size"]), float(x.get("avgPrice") or 0)
        return 0.0, 0.0

    def close_live(self, sym, d, size):
        ins = self.inst[sym]
        idx = self.mode(sym)
        for _ in range(6):
            px = self.price(sym)
            lim = q_price(px * (1 - 0.002 * d), ins["tick"])
            api(self.s.place_order, category=CAT, symbol=sym, side="Sell" if d > 0 else "Buy", orderType="Limit",
                qty=str(q_down(size, ins["step"])), price=str(lim), timeInForce="IOC", reduceOnly=True,
                positionIdx=idx(d))
            time.sleep(1.0)
            size, _ = self.position(sym, d)
            if size <= 0:
                return True
        return False

    def cancel(self, sym, oid):
        if not oid:
            return
        try:
            api(self.s.cancel_order, category=CAT, symbol=sym, orderId=oid)
        except Exception:  # noqa: BLE001
            pass

    def arm_live(self, tr, size, avg):
        """Лимитка залилась: стоп на бирже (по рынку при касании), тейк — лимиткой reduce-only (мейкер)."""
        sym, d, ins = tr["sym"], tr["dir"], self.inst[tr["sym"]]
        idx = self.mode(sym)
        tr.update(status="open", opened=time.time(), qty=size, entry=avg or tr["entry"])
        tr["risk_usd"] = (tr["entry"] - tr["stop"]) * d * size
        if tr["risk_usd"] <= 0:
            log.info("%s: цена уже за стопом — выхожу", sym)
            self.close_live(sym, d, size)
            return
        api(self.s.set_trading_stop, category=CAT, symbol=sym, tpslMode="Full", positionIdx=idx(d),
            stopLoss=str(q_price(tr["stop"], ins["tick"])), slTriggerBy="LastPrice")
        r = api(self.s.place_order, category=CAT, symbol=sym, side="Sell" if d > 0 else "Buy", orderType="Limit",
                qty=str(q_down(size, ins["step"])), price=str(q_price(tr["tp"], ins["tick"])), timeInForce="GTC",
                reduceOnly=True, positionIdx=idx(d))
        tr["tp_id"] = r["result"].get("orderId")
        log.info("ВХОД %s: залилось %s @ %.6g, стоп %.6g на бирже, тейк-лимитка %.6g", sym, size, tr["entry"],
                 tr["stop"], tr["tp"])

    def manage(self):
        """Каждые пару секунд: заливка лимиток, тейк/стоп/время по открытым сделкам."""
        for tr in list(self.st.open):
            sym, d = tr["sym"], tr["dir"]
            try:
                if self.a.live:
                    self.manage_live(tr)
                else:
                    self.manage_paper(tr)
            except Exception as e:  # noqa: BLE001
                log.error("%s: сопровождение: %s", sym, str(e)[:120])

    def drop(self, tr, why):
        self.st.open = [x for x in self.st.open if x is not tr]
        self.st.save()
        log.info("%s: %s", tr["sym"], why)

    def manage_live(self, tr):
        sym, d = tr["sym"], tr["dir"]
        size, avg = self.position(sym, d)
        if tr["status"] == "order":
            if size > 0:
                self.cancel(sym, tr.get("order_id"))          # остаток лимитки (частичная заливка)
                size, avg = self.position(sym, d)
                self.arm_live(tr, size, avg)
                self.st.save()
            elif time.time() > tr["deadline"]:
                self.cancel(sym, tr.get("order_id"))
                size, avg = self.position(sym, d)
                if size > 0:
                    self.arm_live(tr, size, avg)
                    self.st.save()
                else:
                    self.drop(tr, "лимитка на вход не залилась — отменил")
            return
        if size <= 0:
            self.cancel(sym, tr.get("tp_id"))
            self.finish_live(tr)
            return
        if time.time() - tr["opened"] > tr["hold_min"] * 60:
            self.cancel(sym, tr.get("tp_id"))
            self.close_live(sym, d, size)
            self.finish_live(tr, "время")

    def manage_paper(self, tr):
        sym, d = tr["sym"], tr["dir"]
        px = self.price(sym)
        arr = self.cd.data.get(sym)
        since = tr["opened"] if tr["status"] == "open" else tr["placed"]
        hi = lo = px
        if arr is not None and len(arr):          # хвосты закрытых минуток после входа
            m = arr[:, 0] >= (since // 60) * 60_000 + 60_000
            if m.any():
                hi, lo = max(px, arr[m, 2].max()), min(px, arr[m, 3].min())
        if tr["status"] == "order":
            if (lo <= tr["entry"]) if d > 0 else (hi >= tr["entry"]):
                tr.update(status="open", opened=time.time())
                log.info("ВХОД %s (бумага): лимитка %.6g залилась", sym, tr["entry"])
                self.st.save()
            elif time.time() > tr["deadline"]:
                self.drop(tr, "лимитка на вход не залилась — отменил")
            return
        hit_sl = lo <= tr["stop"] if d > 0 else hi >= tr["stop"]
        hit_tp = hi >= tr["tp"] if d > 0 else lo <= tr["tp"]
        if hit_sl:
            self.finish_paper(tr, tr["stop"] - d * tr["stop"] * 0.0003, "стоп")
        elif hit_tp:
            self.finish_paper(tr, tr["tp"], "тейк")
        elif time.time() - tr["opened"] > tr["hold_min"] * 60:
            self.finish_paper(tr, px * (1 - 0.0002 * d), "время")

    def finish_paper(self, tr, exit_px, why):
        d = tr["dir"]
        exit_fee = self.a.maker_pct if why == "тейк" else self.a.taker_pct
        fee = (tr["entry"] * self.a.maker_pct + exit_px * exit_fee) * tr["qty"] / 100
        pnl = (exit_px - tr["entry"]) * d * tr["qty"] - fee
        self.st.paper_balance += pnl
        self.record(tr, exit_px, pnl, why)

    def finish_live(self, tr, why=None):
        pnl, exit_px = 0.0, 0.0
        try:
            r = api(self.s.get_closed_pnl, category=CAT, symbol=tr["sym"], limit=5)["result"]["list"]
            mine = [x for x in r if float(x.get("updatedTime", 0)) / 1000 >= tr["opened"] - 5]
            pnl = sum(float(x.get("closedPnl") or 0) for x in mine)
            exit_px = float(mine[0].get("avgExitPrice") or 0) if mine else 0.0
        except Exception as e:  # noqa: BLE001
            log.warning("%s: closed pnl: %s", tr["sym"], e)
        if why is None:
            why = "тейк" if pnl > 0 else "стоп"
        self.record(tr, exit_px, pnl, why)

    def record(self, tr, exit_px, pnl, why):
        st = self.st
        r = pnl / tr["risk_usd"] if tr.get("risk_usd") else 0.0
        st.open = [x for x in st.open if x is not tr]
        st.roll_day()
        st.day["r"] += r
        st.day["usd"] += pnl
        st.day["n"] += 1
        st.streak = st.streak + 1 if r < 0 else 0
        if st.streak >= self.a.loss_streak:
            st.pause_until = time.time() + self.a.streak_pause * 60
            st.streak = 0
            log.info("%d стопов подряд — пауза %d мин", self.a.loss_streak, self.a.streak_pause)
        st.closed.append({"sym": tr["sym"], "setup": tr["setup"], "tf": tr.get("tf", 1), "dir": tr["dir"],
                          "entry": tr["entry"], "exit": exit_px, "stop": tr["stop"], "tp": tr["tp"], "pnl": pnl,
                          "r": r, "why": why, "pred": tr["pred"], "opened": tr["opened"], "closed": time.time()})
        try:
            st.equity.append((time.time(), self.balance()))
        except Exception:  # noqa: BLE001
            pass
        st.save()
        self.brain.add([tr["feats"]], [r], [tr["opened"] * 1000], real=True)
        log.info("ВЫХОД %s (%s): %+.2f$ = %+.2fR, сеть ждала %+.2fR", tr["sym"], why, pnl, r, tr["pred"])
        self.brain.think("result", f"{tr['sym']}: {why} {r:+.2f}R ({pnl:+.2f}$)",
                         [f"Ждала {tr['pred']:+.2f}R — {'лучше' if r > tr['pred'] else 'хуже'}, учу с весом ×3",
                          f"За сутки: {st.day['r']:+.2f}R, {st.day['usd']:+.2f}$ в {st.day['n']} сделках"])

    # ------------------------------------------------ главный цикл
    def run(self):
        log.info("Режим: %s | риск %.2g%% на сделку, тейк 1:%g, до %d сделок",
                 "РЕАЛЬНЫЙ СЧЁТ" if self.a.live else "БУМАЖНЫЙ (без ордеров)", self.a.risk_pct, self.a.rr,
                 self.a.max_open)
        self.pick_universe()
        threading.Thread(target=self.bootstrap, daemon=True).start()
        last_manage = 0.0
        while not STOP.is_set():
            now = time.time()
            minute = int(now // 60)
            if minute != self.last_minute and now % 60 >= 2:
                self.last_minute = minute
                try:
                    web.update(state="читаю рынок")
                    self.on_minute()
                    ok, why = self.brain.active()
                    web.update(state=("торгую" if ok else "учусь") + f" — {why}", error="")
                except Exception as e:  # noqa: BLE001
                    log.exception("Минутный цикл: %s", e)
                    web.update(error=str(e)[:200])
            if now - last_manage >= 2:
                last_manage = now
                if web.CONTROL["paused"] and web.CONTROL.get("close_all"):
                    self.close_all()
                    web.CONTROL["close_all"] = False
                self.manage()
            STOP.wait(0.5)
        self.brain.save()
        self.st.save()

    def close_all(self):
        for tr in list(self.st.open):
            try:
                if tr["status"] == "order":
                    if self.a.live:
                        self.cancel(tr["sym"], tr.get("order_id"))
                        size, avg = self.position(tr["sym"], tr["dir"])
                        if size > 0:
                            self.arm_live(tr, size, avg)
                    if tr["status"] == "order":
                        self.drop(tr, "лимитка отменена со страницы")
                        continue
                if self.a.live:
                    self.cancel(tr["sym"], tr.get("tp_id"))
                    size, _ = self.position(tr["sym"], tr["dir"])
                    if size > 0:
                        self.close_live(tr["sym"], tr["dir"], size)
                    self.finish_live(tr, "стоп со страницы")
                else:
                    self.finish_paper(tr, self.price(tr["sym"]), "стоп со страницы")
            except Exception as e:  # noqa: BLE001
                log.error("%s: закрытие: %s", tr["sym"], e)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    a = parse_args()
    os.makedirs(a.data_dir, exist_ok=True)
    web.load_symbols(os.path.join(a.data_dir, "symbols.json"))
    if a.symbols and not web.CONTROL["symbols"]:
        web.CONTROL["symbols"] = web.parse_symbols(a.symbols)
    web.start()
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    while not STOP.is_set():
        try:
            Bot(a).run()
            return
        except SystemExit as e:
            web.update(error=str(e), state="ошибка настроек")
            log.error("%s", e)
            STOP.wait(60)
        except Exception as e:  # noqa: BLE001
            log.exception("Запуск: %s", e)
            web.update(error=str(e)[:200], state="ошибка запуска, повтор через 30 сек")
            STOP.wait(30)


if __name__ == "__main__":
    main()
