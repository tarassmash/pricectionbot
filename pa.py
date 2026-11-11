"""Price action на минутных свечах: структура, сетапы, умные стопы, разметка исхода.

Свечи — numpy-массивы одинаковой длины: t (мс), o, h, l, c, v. Последняя свеча в массиве — ЗАКРЫТАЯ.

Сетапы (вход по закрытию сигнальной свечи):
  sweep    — снятие ликвидности: свеча прокалывает недавний swing-уровень хвостом и закрывается обратно.
             Вход против прокола, стоп за хвостом.
  retest   — пробой swing-уровня закрытием, откат к уровню и удержание (свеча закрылась от уровня в сторону пробоя).
             Стоп за минимумом/максимумом отката.
  pullback — тренд на 5м (EMA20 > EMA50 и наклон), откат к EMA20 на 1м с отбоем. Стоп за локальным экстремумом отката.

Умный стоп: не на самом экстремуме, а за ним с запасом (buffer × ATR); если рядом за стопом есть ещё
один swing-экстремум (кластер стопов толпы) — стоп уносится за него. Тейк = RR × риск, и только если до
ближайшего встречного уровня есть место (room).
"""
import math

import numpy as np

SETUPS = ("sweep", "retest", "pullback")
BUFFERS = (0.25, 0.5, 1.0)          # варианты запаса за экстремумом, × ATR — сеть выбирает лучший


def ema(x, n):
    a = 2.0 / (n + 1)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def atr(h, l, c, n=14):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return ema(tr, n)


def swings(h, l, k=3):
    """Индексы подтверждённых swing high / swing low (фрактал: экстремум среди k свечей слева и справа)."""
    n = len(h)
    sh, sl = [], []
    for i in range(k, n - k):
        if h[i] == h[i - k:i + k + 1].max() and h[i] > h[i - k:i].max():
            sh.append(i)
        if l[i] == l[i - k:i + k + 1].min() and l[i] < l[i - k:i].min():
            sl.append(i)
    return sh, sl


def er(c, n):
    """Efficiency ratio за n свечей: 0 — боковик, 1 — прямая."""
    if len(c) <= n:
        return 0.5
    seg = c[-n - 1:]
    path = np.abs(np.diff(seg)).sum()
    return float(abs(seg[-1] - seg[0]) / path) if path > 0 else 0.0


class Ctx:
    """Всё, что нужно сетапам и признакам, считается один раз на свечи монеты."""

    def __init__(self, t, o, h, l, c, v, btc=None):
        """btc — {время свечи (мс): close BTC} для контекста рынка (движение BTC за 15 мин)."""
        self.t, self.o, self.h, self.l, self.c, self.v = t, o, h, l, c, v
        self.n = len(c)
        self.atr = atr(h, l, c, 14)
        self.atr60 = atr(h, l, c, 60)
        self.ema20 = ema(c, 20)
        self.sh, self.sl = swings(h, l, 3)
        # тренд 5м на каждую минуту — только по уже закрытым 5-минуткам (без заглядывания вперёд)
        g = (t // 300_000).astype(np.int64)
        idx = np.flatnonzero(np.diff(g)) + 1
        ends = np.concatenate([idx, [len(t)]])
        C5 = c[ends - 1]
        e20, e50 = ema(C5, 20), ema(C5, 50)
        k5 = np.searchsorted(ends, np.arange(self.n) + 1, side="right") - 1
        # последняя 5-минутка закрыта, только если минута i — её последняя (конец кратен 5 минутам)
        self.trend_a = np.zeros(self.n, dtype=int)
        self.slope_a = np.zeros(self.n)
        for i in range(self.n):
            k = k5[i]
            if k < 4:
                continue
            sl = (e20[k] - e20[k - 3]) / max(self.atr[i] * 3, 1e-12)
            self.slope_a[i] = sl
            self.trend_a[i] = 1 if (e20[k] > e50[k] and sl > 0.05) else (-1 if (e20[k] < e50[k] and sl < -0.05) else 0)
        self.btc_a = np.zeros(self.n)
        if btc:
            for i in range(self.n):
                b0, b1 = btc.get(int(t[i])), btc.get(int(t[i]) - 15 * 60_000)
                if b0 and b1:
                    self.btc_a[i] = (b0 / b1 - 1) * 100


def _levels(cx, i, kind, lookback=60):
    """Swing-уровни, подтверждённые до свечи i (не позже i-3), за последние lookback свечей."""
    src = cx.sh if kind == "high" else cx.sl
    arr = cx.h if kind == "high" else cx.l
    return [(j, float(arr[j])) for j in src if i - lookback <= j <= i - 4]


def _smart_stop(cx, i, d, extreme, buf_atr):
    """Стоп за экстремумом с запасом; если в 0.4 ATR дальше лежит другой swing — уносим за него."""
    a = cx.atr[i]
    if d > 0:   # лонг: стоп ниже
        stop = extreme - buf_atr * a
        for _, lv in _levels(cx, i, "low", 90):
            if stop - 0.4 * a <= lv < stop + 0.05 * a and lv < extreme:
                stop = min(stop, lv - 0.15 * a)
    else:
        stop = extreme + buf_atr * a
        for _, lv in _levels(cx, i, "high", 90):
            if stop - 0.05 * a < lv <= stop + 0.4 * a and lv > extreme:
                stop = max(stop, lv + 0.15 * a)
    return stop


def _room(cx, i, d, entry, risk):
    """Сколько R до ближайшего встречного swing-уровня (куда цена может упереться до тейка)."""
    if d > 0:
        lv = [p for _, p in _levels(cx, i, "high", 120) if p > entry]
        return (min(lv) - entry) / risk if lv else 9.0
    lv = [p for _, p in _levels(cx, i, "low", 120) if p < entry]
    return (entry - max(lv)) / risk if lv else 9.0


def detect(cx, i=None):
    """Сетапы на свече i (по умолчанию последняя закрытая). -> [(setup, dir, extreme, extra)]"""
    if i is None:
        i = cx.n - 1
    if i < 70:
        return []
    o, h, l, c = cx.o[i], cx.h[i], cx.l[i], cx.c[i]
    a = cx.atr[i]
    rng = max(h - l, 1e-12)
    out = []
    # 1) снятие ликвидности
    for j, lv in _levels(cx, i, "high", 60):
        if h > lv + 0.05 * a and c < lv and (h - max(o, c)) / rng >= 0.45 and cx.h[j + 1:i].max(initial=-1e18) <= lv:
            out.append(("sweep", -1, h, {"depth": (h - lv) / a, "lvl_age": i - j}))
            break
    for j, lv in _levels(cx, i, "low", 60):
        if l < lv - 0.05 * a and c > lv and (min(o, c) - l) / rng >= 0.45 and cx.l[j + 1:i].min(initial=1e18) >= lv:
            out.append(("sweep", 1, l, {"depth": (lv - l) / a, "lvl_age": i - j}))
            break
    # 2) пробой и ретест
    for j, lv in _levels(cx, i, "high", 60):
        win = range(max(j + 1, i - 15), i)
        brk = [k for k in win if cx.c[k] > lv + 0.1 * cx.atr[k]]
        if brk and cx.l[brk[0] + 1:i + 1].min(initial=1e18) <= lv + 0.15 * a and c > lv and c > o \
                and cx.c[brk[0] + 1:i + 1].min() > lv - 0.3 * a:
            ext = cx.l[brk[0] + 1:i + 1].min()
            out.append(("retest", 1, ext, {"depth": (c - lv) / a, "lvl_age": i - j}))
            break
    for j, lv in _levels(cx, i, "low", 60):
        win = range(max(j + 1, i - 15), i)
        brk = [k for k in win if cx.c[k] < lv - 0.1 * cx.atr[k]]
        if brk and cx.h[brk[0] + 1:i + 1].max(initial=-1e18) >= lv - 0.15 * a and c < lv and c < o \
                and cx.c[brk[0] + 1:i + 1].max() < lv + 0.3 * a:
            ext = cx.h[brk[0] + 1:i + 1].max()
            out.append(("retest", -1, ext, {"depth": (lv - c) / a, "lvl_age": i - j}))
            break
    # 3) откат по тренду 5м к EMA20 1м
    e = cx.ema20[i]
    tr = cx.trend_a[i]
    if tr > 0 and l <= e + 0.1 * a and c > e and c > o and (min(o, c) - l) >= 0.3 * rng:
        out.append(("pullback", 1, cx.l[i - 5:i + 1].min(), {"depth": (c - e) / a, "lvl_age": 0}))
    if tr < 0 and h >= e - 0.1 * a and c < e and c < o and (h - max(o, c)) >= 0.3 * rng:
        out.append(("pullback", -1, cx.h[i - 5:i + 1].max(), {"depth": (e - c) / a, "lvl_age": 0}))
    return out


def plans(cx, a, i=None):
    """Варианты сделок по сетапам свечи i: для каждого сетапа — по варианту стопа на каждый буфер.
    -> [dict(setup, dir, entry, stop, tp, risk, buf, room, feats)]"""
    if i is None:
        i = cx.n - 1
    res = []
    entry = float(cx.c[i])
    at = float(cx.atr[i])
    for setup, d, ext, extra in detect(cx, i):
        for buf in BUFFERS:
            stop = _smart_stop(cx, i, d, ext, buf)
            risk = (entry - stop) * d
            if risk <= 0:
                continue
            risk_atr = risk / at
            risk_pct = risk / entry * 100
            if not (a.min_risk_atr <= risk_atr <= a.max_risk_atr) or risk_pct < a.min_risk_pct:
                continue
            room = _room(cx, i, d, entry, risk)
            if room < a.min_room:
                continue
            tp = entry + d * a.rr * risk
            res.append({"setup": setup, "dir": d, "entry": entry, "stop": float(stop), "tp": float(tp),
                        "risk": float(risk), "buf": buf, "room": float(room), "i": i,
                        "feats": features(cx, i, setup, d, risk, buf, room, extra, a)})
    return res


FEAT_NAMES = ["sweep", "retest", "pullback", "по тренду 5м", "против тренда 5м", "стоп, ATR", "стоп, %",
              "запас стопа", "место до уровня, R", "хвост сигнальной", "тело сигнальной", "объём к среднему",
              "волатильность, %", "ATR к часовому", "боковик 20м (ER)", "наклон 5м", "от EMA20, ATR",
              "глубина прокола/отбоя", "возраст уровня", "движение за 15м, %", "BTC за 15м, %", "час (sin)",
              "час (cos)", "лонг"]


def features(cx, i, setup, d, risk, buf, room, extra, a):
    o, h, l, c, v = cx.o[i], cx.h[i], cx.l[i], cx.c[i], cx.v[i]
    at = max(cx.atr[i], 1e-12)
    rng = max(h - l, 1e-12)
    wick = (h - max(o, c)) / rng if d < 0 else (min(o, c) - l) / rng
    vavg = cx.v[max(0, i - 20):i].mean() if i > 0 else v
    move15 = (c / cx.c[max(0, i - 15)] - 1) * 100 * d
    hr = (cx.t[i] // 3_600_000) % 24
    return [
        float(setup == "sweep"), float(setup == "retest"), float(setup == "pullback"),
        float(cx.trend_a[i] == d), float(cx.trend_a[i] == -d), risk / at, risk / c * 100, buf, min(room, 9.0),
        wick, abs(c - o) / rng, min(v / max(vavg, 1e-12), 10.0), at / c * 100,
        at / max(cx.atr60[i], 1e-12), er(cx.c[:i + 1], 20), cx.slope_a[i] * d, (c - cx.ema20[i]) / at * d,
        min(extra.get("depth", 0.0), 5.0), min(extra.get("lvl_age", 0) / 60.0, 2.0), move15,
        cx.btc_a[i] * d, math.sin(hr / 24 * 2 * math.pi), math.cos(hr / 24 * 2 * math.pi), float(d > 0),
    ]


def outcome(plan, h, l, c, start, max_hold, fee_pct):
    """Исход сделки на свечах после сигнала (h,l,c — массивы, start — индекс первой свечи после входа).
    Если в одной свече задеты и стоп, и тейк — считаем стоп (осторожно). -> (R с комиссиями, свечей, итог)
    или None, если свечей ещё не хватает."""
    d, entry, stop, tp, risk = plan["dir"], plan["entry"], plan["stop"], plan["tp"], plan["risk"]
    fee_r = fee_pct / 100 * 2 * entry / risk
    end = min(len(c), start + max_hold)
    for k in range(start, end):
        hit_sl = l[k] <= stop if d > 0 else h[k] >= stop
        hit_tp = h[k] >= tp if d > 0 else l[k] <= tp
        if hit_sl:
            return -1.0 - fee_r, k - start + 1, "стоп"
        if hit_tp:
            return (tp - entry) * d / risk - fee_r, k - start + 1, "тейк"
    if end - start < max_hold:
        return None
    return (c[end - 1] - entry) * d / risk - fee_r, max_hold, "время"
