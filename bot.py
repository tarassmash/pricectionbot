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
    p.add_argument("--maker-pct", type=float, default=0.02, help="комиссия мейкера, %%")
    p.add_argument("--slip-pct", type=float, default=0.05, help="проскальзывание выхода догоняющей лимиткой, %% (в разметке)")
    p.add_argument("--fee-mult", type=float, default=10, help="до стопа минимум X × комиссия туда-обратно")
    p.add_argument("--chase-sec", type=float, default=12, help="сколько секунд догонять лимиткой вход")
    p.add_argument("--chase-max-r", type=float, default=0.15, help="вход не дальше X R от цены сигнала")
    p.add_argument("--backup-r", type=float, default=0.5, help="запасной стоп-лимит на бирже на X R дальше стопа бота")
    p.add_argument("--be-r", type=float, default=1.0, help="при +X R стоп в безубыток")
    p.add_argument("--trail-start", type=float, default=1.5, help="с +X R стоп тянется за ценой")
    p.add_argument("--trail-r", type=float, default=1.0, help="трейлинг: стоп в X R от лучшей цены")
    p.add_argument("--stale-bars", type=int, default=10, help="за столько баров нет +--stale-r R — выход (режем убыток)")
    p.add_argument("--stale-r", type=float, default=0.5)
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
    p.add_argument("--rule-trade", type=int, default=1,
                   help="1 — пока сеть учится, торговать сетапы из таблицы «в плюсе» (сеть только отсекает минус)")
    p.add_argument("--veto-r", type=float, default=0.0, help="по таблице: не входить, если сеть ждёт меньше X R")
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
        self.brain = brain_mod.Brain(os.path.join(a.data_dir, "brain"), len(pa.FEAT_NAMES), a, pa.FEAT_NAMES,
                                     pa.LABEL_VERSION)
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
            for i in range(80, len(C) - cfg["hold"] - 1):
                for pl in pa.plans(cx, a, i):
                    res = pa.outcome(pl, H, L, C, i + 1, cfg["hold"], a)
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
            res = pa.outcome(p["plan"], agg[:, 2], agg[:, 3], agg[:, 4], start, cfg["hold"], a) \
                if start < len(agg) else None
            if res is None:
                if time.time() * 1000 - p["t"] < ((cfg["hold"] + 1) * p["tf"] + 30) * 60_000:
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
            # сеть в форме — входим, если она ждёт ≥ --min-edge; учится — входим по таблице сетапов,
            # а сеть только отсекает то, где ждёт минус (--veto-r)
            rule = (not ok) and self.a.rule_trade and enabled is not None
            edge = self.a.min_edge if ok else self.a.veto_r
            take = sym in self.universe and (ok or rule) and p["pred"] >= edge
            note = "" if (ok or rule or sym not in self.universe) else "сеть учится"
            if sym in self.universe and (ok or rule) and p["pred"] < edge:
                note = "сеть ждёт минус" if rule else "слабый"
            if take and enabled is not None:
                if (tf, setup) not in enabled:
                    take, note = False, "этот сетап/ТФ в минусе на истории"
                elif abs(p["buf"] - enabled[(tf, setup)]) > 1e-6:
                    # сеть выбрала другой запас стопа — берём тот, что в плюсе на истории
                    alt = [q for s_, q in fresh if s_ == sym and q["tf"] == tf and q["setup"] == setup
                           and q["dir"] == d and abs(q["buf"] - enabled[(tf, setup)]) < 1e-6 and q.get("pred") is not None]
                    if alt and alt[0]["pred"] >= edge:
                        p = alt[0]
                    else:
                        take, note = False, f"с запасом {enabled[(tf, setup)]:g} ATR сеть ждёт мало"
            if not take:
                pass
            elif any(o["sym"] == sym for o in st.open):
                take, note = False, "уже в сделке"
            elif len(st.open) >= self.a.max_open:
                take, note = False, f"уже {self.a.max_open} сделки"
            elif st.day["r"] <= -self.a.day_loss_r:
                take, note = False, "дневной лимит"
            elif time.time() < st.pause_until:
                take, note = False, "пауза после серии стопов"
            elif web.CONTROL["paused"]:
                take, note = False, "на паузе"
            self.signals.append({"sym": sym, "tf": tf, "setup": setup, "dir": d, "pred": p["pred"],
                                 "risk_pct": p["risk"] / p["entry"] * 100, "buf": p["buf"], "room": p["room"],
                                 "take": take, "note": note or ("беру" if take else "слабый")})
            if take:
                self.enter(sym, p)

    # ------------------------------------------------ сделки
    def book(self, sym):
        ob = api(self.market.get_orderbook, category=CAT, symbol=sym, limit=1)["result"]
        return float(ob["b"][0][0]), float(ob["a"][0][0])

    def chase(self, sym, d, qty, reduce, seconds, cap_px=None):
        """Догоняющая PostOnly-лимитка по лучшей цене стакана: каждые ~1.5 с переставляем, пока не зальётся,
        не выйдет время или цена не уйдёт за cap_px. Только лимитки, без маркета. -> сколько исполнено."""
        ins = self.inst[sym]
        idx = self.mode(sym)
        buy = (d > 0) != reduce              # вход в лонг / выход из шорта — покупка
        side = "Buy" if buy else "Sell"
        size0, _ = self.position(sym, d)
        t_end = time.time() + seconds
        done = 0.0
        while time.time() < t_end and not STOP.is_set():
            left = q_down(float(qty) - done, ins["step"])
            if left < ins["min_qty"]:
                break
            bid, ask = self.book(sym)
            px = bid if buy else ask
            if cap_px is not None and ((buy and px > cap_px) or (not buy and px < cap_px)):
                break
            try:
                r = api(self.s.place_order, category=CAT, symbol=sym, side=side, orderType="Limit", qty=str(left),
                        price=str(q_price(px, ins["tick"])), timeInForce="PostOnly", reduceOnly=reduce,
                        positionIdx=idx(d))
                oid = r["result"].get("orderId")
            except Exception as e:  # noqa: BLE001
                log.info("%s: лимитка не встала: %s", sym, str(e)[:80])
                time.sleep(0.5)
                continue
            time.sleep(1.5)
            self.cancel(sym, oid)
            size, _ = self.position(sym, d)
            done = (size - size0) if not reduce else (size0 - size)
            if reduce and size <= 0:
                return float(qty)
        return max(done, 0.0)

    def exit_live(self, sym, d, size, urgent):
        """Выход лимитками: догоняем по лучшей цене; если срочно (стоп) и не залилось за 6 с —
        лимитка IOC через спред с потолком 0.3% (всё равно лимитка, не маркет)."""
        self.chase(sym, d, size, True, 6 if urgent else 20)
        size, _ = self.position(sym, d)
        ins, idx = self.inst[sym], self.mode(sym)
        for _ in range(8):
            if size <= 0:
                return True
            bid, ask = self.book(sym)
            px = (bid * (1 - 0.003)) if d > 0 else (ask * (1 + 0.003))
            try:
                api(self.s.place_order, category=CAT, symbol=sym, side="Sell" if d > 0 else "Buy", orderType="Limit",
                    qty=str(q_down(size, ins["step"])), price=str(q_price(px, ins["tick"])), timeInForce="IOC",
                    reduceOnly=True, positionIdx=idx(d))
            except Exception as e:  # noqa: BLE001
                log.info("%s: выход: %s", sym, str(e)[:80])
            time.sleep(1.0)
            size, _ = self.position(sym, d)
        return size <= 0

    def backup_stop(self, tr):
        """Запасной стоп-ЛИМИТ на бирже чуть дальше стопа бота — сработает, только если бот не успел (упал)."""
        sym, d, ins = tr["sym"], tr["dir"], self.inst[tr["sym"]]
        risk0 = (tr["entry"] - tr["stop0"]) * d
        trig = tr["stop"] - d * self.a.backup_r * risk0
        lim = trig * (1 - d * 0.004)
        try:
            api(self.s.set_trading_stop, category=CAT, symbol=sym, tpslMode="Partial", positionIdx=self.mode(sym)(d),
                stopLoss=str(q_price(trig, ins["tick"])), slTriggerBy="LastPrice", slOrderType="Limit",
                slLimitPrice=str(q_price(lim, ins["tick"])), slSize=str(q_down(tr["qty"], ins["step"])))
            tr["backup"] = trig
        except Exception as e:  # noqa: BLE001
            log.info("%s: запасной стоп-лимит не встал: %s", sym, str(e)[:100])

    def enter(self, sym, p):
        """Вход догоняющей лимиткой (мейкер) по лучшей цене стакана; дальше стоп и трейлинг ведёт бот."""
        a, ins = self.a, self.inst[sym]
        bal = self.balance()
        d, tf = p["dir"], p["tf"]
        cap = p["entry"] + d * a.chase_max_r * p["risk"]          # дальше этого не догоняем
        px = self.price(sym)
        if (px - cap) * d > 0:
            log.info("%s: цена уже ушла дальше %.2fR от сигнала — пропускаю", sym, a.chase_max_r)
            return
        risk = (px - p["stop"]) * d
        if risk <= 0:
            return
        qty = min(bal * a.risk_pct / 100 / risk, bal * a.max_notional_x / px)
        qty = q_down(qty, ins["step"])
        if qty < ins["min_qty"] or qty * D(str(px)) < ins["min_notional"]:
            log.info("%s: объём %s меньше минимума биржи (мин. %s шт., %s$) — пропускаю", sym, qty, ins["min_qty"],
                     ins["min_notional"])
            return
        cfg = pa.TFS[tf]
        tr = {"sym": sym, "setup": p["setup"], "tf": tf, "dir": d, "qty": float(qty), "entry": px,
              "stop": float(q_price(p["stop"], ins["tick"])), "stop0": p["stop"], "pred": p["pred"], "buf": p["buf"],
              "room": p["room"], "feats": p["feats"], "status": "open", "opened": time.time(), "placed": time.time(),
              "hold_min": cfg["hold"] * tf, "tp": None}
        if a.live:
            if sym not in self.lev_set:
                lev = str(min(a.leverage, ins["max_lev"]))
                api(self.s.set_leverage, category=CAT, symbol=sym, buyLeverage=lev, sellLeverage=lev)
                self.lev_set.add(sym)
            got = self.chase(sym, d, qty, False, a.chase_sec, cap)
            size, avg = self.position(sym, d)
            if size <= 0 or got <= 0:
                log.info("%s: лимитка на вход не залилась за %g с — пропускаю", sym, a.chase_sec)
                return
            tr.update(qty=size, entry=avg or px)
            if (tr["entry"] - tr["stop"]) * d <= 0:
                log.info("%s: цена уже за стопом — выхожу", sym)
                self.exit_live(sym, d, size, True)
                return
            self.backup_stop(tr)
        else:
            tr["entry"] = p["entry"] if (px - p["entry"]) * d <= 0 else px     # бумага: лимитка по лучшей цене
        tr["risk_usd"] = (tr["entry"] - tr["stop"]) * d * tr["qty"]
        tr["best"] = tr["entry"]
        tr["last_check"] = time.time()
        self.st.open.append(tr)
        self.st.save()
        log.info("ВХОД %s %s %s %dм: %s @ %.6g лимиткой, стоп %.6g, риск %.2f$, дальше трейлинг (сеть ждёт %+.2fR)",
                 sym, "ЛОНГ" if d > 0 else "ШОРТ", p["setup"], tf, tr["qty"], tr["entry"], tr["stop"], tr["risk_usd"],
                 p["pred"])
        self.brain.think("trade", f"Вхожу: {sym} {'лонг' if d > 0 else 'шорт'} ({p['setup']}, {tf}м)", [
            f"Вход {tr['entry']:.6g} лимиткой, стоп {tr['stop']:.6g} (за структурой + {p['buf']:g} ATR), риск {tr['risk_usd']:.2f}$",
            f"При +{a.be_r:g}R — безубыток, с +{a.trail_start:g}R стоп тянется в {a.trail_r:g}R за ценой; "
            f"нет +{a.stale_r:g}R за {a.stale_bars} баров — выхожу; выход тоже лимиткой",
            f"Жду {p['pred']:+.2f}R; до встречного уровня {p['room']:.1f}R"])

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
        return self.exit_live(sym, d, size, True)

    def _close_live_old(self, sym, d, size):
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

    def _extremes(self, tr, px):
        """Мин/макс цены с прошлой проверки: текущая цена + закрытые минутки после неё."""
        arr = self.cd.data.get(tr["sym"])
        hi = lo = px
        if arr is not None and len(arr):
            m = arr[:, 0] >= (tr["last_check"] // 60) * 60_000
            if m.any():
                hi, lo = max(px, arr[m, 2].max()), min(px, arr[m, 3].min())
        tr["last_check"] = time.time()
        return hi, lo

    def _trail(self, tr, hi, lo):
        """Двигаем стоп: безубыток / трейлинг. -> новый стоп (или старый)."""
        d = tr["dir"]
        risk0 = (tr["entry"] - tr["stop0"]) * d
        tr["best"] = max(tr["best"], hi) if d > 0 else min(tr["best"], lo)
        return pa.trail_stop(d, tr["entry"], risk0, tr["stop"], tr["best"], self.a)

    def _stale(self, tr):
        d = tr["dir"]
        risk0 = (tr["entry"] - tr["stop0"]) * d
        bars = (time.time() - tr["opened"]) / (tr.get("tf", 1) * 60)
        return bars >= self.a.stale_bars and (tr["best"] - tr["entry"]) * d < self.a.stale_r * risk0

    def manage_live(self, tr):
        sym, d = tr["sym"], tr["dir"]
        size, _ = self.position(sym, d)
        if size <= 0:
            self.finish_live(tr)
            return
        px = self.price(sym)
        hi, lo = self._extremes(tr, px)
        why = None
        if (lo <= tr["stop"]) if d > 0 else (hi >= tr["stop"]):            # стоп бота — выходим лимитками
            why = self._why(tr, tr["stop"])
        else:
            new = self._trail(tr, hi, lo)
            if (new - tr["stop"]) * d > 0:
                tr["stop"] = new
                risk0 = (tr["entry"] - tr["stop0"]) * d
                if (new - tr.get("backup", tr["stop0"])) * d > 0.5 * risk0 + self.a.backup_r * risk0:
                    self.backup_stop(tr)                                       # подтягиваем запасной стоп-лимит
                self.st.save()
            if self._stale(tr):
                why = "без движения"
            elif time.time() - tr["opened"] > tr["hold_min"] * 60:
                why = "время"
        if why:
            self.exit_live(sym, d, size, why in ("стоп", "безубыток", "трейлинг"))
            self.finish_live(tr, why)

    def manage_paper(self, tr):
        d = tr["dir"]
        px = self.price(tr["sym"])
        hi, lo = self._extremes(tr, px)
        if (lo <= tr["stop"]) if d > 0 else (hi >= tr["stop"]):         # сначала стоп, что стоял до движения
            self.finish_paper(tr, tr["stop"] * (1 - d * self.a.slip_pct / 100), self._why(tr, tr["stop"]))
            return
        new = self._trail(tr, hi, lo)
        if (new - tr["stop"]) * d > 0:
            tr["stop"] = new
            self.st.save()
        if self._stale(tr):
            self.finish_paper(tr, px * (1 - d * self.a.slip_pct / 100), "без движения")
        elif time.time() - tr["opened"] > tr["hold_min"] * 60:
            self.finish_paper(tr, px * (1 - d * self.a.slip_pct / 100), "время")

    def _why(self, tr, exit_px):
        d = tr["dir"]
        risk0 = (tr["entry"] - tr["stop0"]) * d
        x = (exit_px - tr["entry"]) * d
        return "стоп" if x < -0.05 * risk0 else ("безубыток" if x < 0.5 * risk0 else "трейлинг")

    def finish_paper(self, tr, exit_px, why):
        d = tr["dir"]
        fee = (tr["entry"] + exit_px) * self.a.maker_pct * tr["qty"] / 100
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
            why = self._why(tr, exit_px) if exit_px else ("трейлинг" if pnl > 0 else "стоп")
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
