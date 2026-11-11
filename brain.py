"""Самообучающаяся сеть: по признакам сетапа предсказывает исход сделки в R (с комиссиями).

Учится на ВСЕХ сетапах, которые бот видит на рынке (взятых и не взятых): каждый вариант стопа каждого
сетапа размечается тем, что цена сделала потом — тейк (+RR), стоп (−1) или выход по времени.
Память — последние --memory-days дней, свежие примеры весят больше. Полное переобучение раз в 30 мин,
с отложенным самым свежим куском (проверка «как в жизни»: сделки, которые сеть взяла бы).

Онлайн-проверка: каждый новый сетап сеть оценивает ДО того, как известен исход; когда исход приходит,
он попадает в проверку. Торговать сеть разрешает себе, только если за последние часы её отобранные
сетапы по факту в плюсе.
"""
import json
import logging
import os
import threading
import time

import numpy as np


class MLP:
    def __init__(self, nin, hidden=(48, 24), seed=0):
        rng = np.random.default_rng(seed)
        sizes = [nin, *hidden, 1]
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.mu = np.zeros(nin)
        self.sd = np.ones(nin)

    def _fwd(self, X):
        hs = [X]
        for k in range(len(self.W) - 1):
            hs.append(np.maximum(0, hs[-1] @ self.W[k] + self.b[k]))
        return hs, (hs[-1] @ self.W[-1] + self.b[-1])[:, 0]

    def predict(self, X):
        X = (np.asarray(X, float) - self.mu) / self.sd
        return self._fwd(X)[1]

    def fit(self, X, y, w, Xv, yv, epochs=40, lr=2e-3, l2=1e-4, batch=256, seed=0, log=None):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        Xn, Xvn = (X - self.mu) / self.sd, (Xv - self.mu) / self.sd
        rng = np.random.default_rng(seed)
        m = [np.zeros_like(p) for p in self.W + self.b]
        v = [np.zeros_like(p) for p in self.W + self.b]
        best, best_state, bad, step, hist = 1e18, None, 0, 0, []
        for ep in range(epochs):
            idx = rng.permutation(len(Xn))
            for s in range(0, len(idx), batch):
                bi = idx[s:s + batch]
                xb, yb, wb = Xn[bi], y[bi], w[bi]
                hs, out = self._fwd(xb)
                g = (2 * (out - yb) * wb / wb.sum())[:, None]
                gW, gb = [None] * len(self.W), [None] * len(self.b)
                for k in range(len(self.W) - 1, -1, -1):
                    gW[k] = hs[k].T @ g + l2 * self.W[k]
                    gb[k] = g.sum(0)
                    if k:
                        g = (g @ self.W[k].T) * (hs[k] > 0)
                step += 1
                params, grads = self.W + self.b, gW + gb
                for q, (p, gr) in enumerate(zip(params, grads)):
                    m[q] = 0.9 * m[q] + 0.1 * gr
                    v[q] = 0.999 * v[q] + 0.001 * gr * gr
                    p -= lr * (m[q] / (1 - 0.9 ** step)) / (np.sqrt(v[q] / (1 - 0.999 ** step)) + 1e-8)
            tr = float(np.average((self._fwd(Xn)[1] - y) ** 2, weights=w))
            va = float(np.mean((self._fwd(Xvn)[1] - yv) ** 2)) if len(yv) else tr
            hist.append((tr, va))
            if va < best - 1e-5:
                best, bad = va, 0
                best_state = ([p.copy() for p in self.W], [p.copy() for p in self.b], ep + 1)
            else:
                bad += 1
                if bad >= 6:
                    break
        if best_state:
            self.W, self.b = best_state[0], best_state[1]
        return hist, (best_state[2] if best_state else len(hist))

    def state(self):
        return {"W": [p.tolist() for p in self.W], "b": [p.tolist() for p in self.b],
                "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    def load(self, st):
        self.W = [np.array(p) for p in st["W"]]
        self.b = [np.array(p) for p in st["b"]]
        self.mu, self.sd = np.array(st["mu"]), np.array(st["sd"])


class Brain:
    def __init__(self, path, nfeat, a, names, label_version=1):
        self.path, self.a, self.names, self.nfeat = path, a, names, nfeat
        self.label_version = label_version
        self.lock = threading.Lock()
        self.X = np.zeros((0, nfeat))
        self.y = np.zeros(0)
        self.t = np.zeros(0)            # время сигнала, мс
        self.real = np.zeros(0)         # 1 — реальная/бумажная сделка бота (вес ×3)
        self.model = None
        self.trained_until = 0          # примеры новее — модель не видела (честная проверка)
        self.last_train = 0.0
        self.online = []                # [(время исхода, прогноз, R)] — оценено до исхода
        self.train_hist = []            # ошибка по эпохам последнего переобучения
        self.val = {}                   # итоги проверки на отложенном куске
        self.importance = []
        self.thoughts = []              # журнал «мыслей» для страницы
        self.mem_hist = []              # (время, примеров) для графика
        self.online_hist = []           # (время, ср. R отобранных, ср. R всех) для графика
        self.research = []              # таблица: сетап × ТФ на реальной истории
        self.enabled = None             # {(ТФ, сетап): запас стопа}, которые в плюсе; None — ещё не посчитано
        self._load()

    # ---------------------------------------------------------------- память
    def _load(self):
        try:
            d = np.load(self.path + ".npz")
            self.X, self.y, self.t, self.real = d["X"], d["y"], d["t"], d["real"]
            if self.X.shape[1] != self.nfeat:
                raise ValueError("другие признаки")
            with open(self.path + ".json", encoding="utf-8") as f:
                st = json.load(f)
            if st.get("label_version") != self.label_version:
                raise ValueError("старая разметка исходов")
            if st.get("model"):
                self.model = MLP(self.nfeat)
                self.model.load(st["model"])
            self.trained_until = st.get("trained_until", 0)
            self.online = [tuple(x) for x in st.get("online", [])]
            self.val = st.get("val", {})
            self.think("plan", "Загрузила память", [f"Примеров: {len(self.y):,}",
                                                     "Модель: " + ("есть" if self.model else "нет, обучу")])
        except Exception:  # noqa: BLE001
            self.X, self.y = np.zeros((0, self.nfeat)), np.zeros(0)
            self.t, self.real = np.zeros(0), np.zeros(0)

    def save(self):
        with self.lock:
            X, y, t, real = self.X.copy(), self.y.copy(), self.t.copy(), self.real.copy()
            st = {"model": self.model.state() if self.model else None, "trained_until": self.trained_until,
                  "label_version": self.label_version,
                  "online": self.online[-5000:], "val": self.val}
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            np.savez_compressed(self.path + ".tmp.npz", X=X, y=y, t=t, real=real)
            os.replace(self.path + ".tmp.npz", self.path + ".npz")
            with open(self.path + ".json.tmp", "w", encoding="utf-8") as f:
                json.dump(st, f)
            os.replace(self.path + ".json.tmp", self.path + ".json")
        except Exception:  # noqa: BLE001
            pass

    def add(self, X, y, t, real=False):
        if not len(y):
            return
        with self.lock:
            self.X = np.vstack([self.X, np.asarray(X, float)])
            self.y = np.concatenate([self.y, np.asarray(y, float)])
            self.t = np.concatenate([self.t, np.asarray(t, float)])
            self.real = np.concatenate([self.real, np.full(len(y), 1.0 if real else 0.0)])
            keep = self.t >= (time.time() - self.a.memory_days * 86400) * 1000
            keep |= self.real > 0
            if keep.sum() > self.a.max_samples:
                cut = np.sort(self.t[keep])[-self.a.max_samples]
                keep &= self.t >= cut
            self.X, self.y, self.t, self.real = self.X[keep], self.y[keep], self.t[keep], self.real[keep]

    # ---------------------------------------------------------------- журнал
    def think(self, kind, title, lines=()):
        if kind != "learn":                     # «учусь на ходу» каждую минуту — только на странице
            logging.getLogger("pa").info("Сеть [%s]: %s%s", kind, title,
                                         ("; " + "; ".join(l for l in lines if l)) if lines else "")
        self.thoughts.append({"time": time.time(), "kind": kind, "title": title, "lines": list(lines)})
        self.thoughts = self.thoughts[-60:]

    # ---------------------------------------------------------------- исследование: какие сетапы в плюсе
    def do_research(self):
        """По всей памяти (реальная история рынка): средний R каждого сетапа на каждом ТФ и по каждому
        запасу стопа. Торгуются только сочетания, где лучший запас стопа в среднем > --research-min-r."""
        with self.lock:
            X, y = self.X.copy(), self.y.copy()
        if not len(y):
            return
        names = self.names
        ci = {n: k for k, n in enumerate(names)}
        rows, enabled = [], {}
        for tf, col in ((1, "ТФ 1м"), (3, "ТФ 3м"), (5, "ТФ 5м")):
            for st in ("sweep", "retest", "pullback"):
                m = (X[:, ci[col]] > 0.5) & (X[:, ci[st]] > 0.5)
                n = int(m.sum())
                if not n:
                    continue
                yy, buf = y[m], X[m, ci["запас стопа"]]
                best = None
                for b in np.unique(np.round(buf, 2)):
                    mb = np.abs(buf - b) < 1e-6
                    if mb.sum() >= max(20, self.a.research_min // 3):
                        v = float(yy[mb].mean())
                        if best is None or v > best[1]:
                            best = (float(b), v, int(mb.sum()))
                filled = yy[yy != 0]
                row = {"tf": tf, "setup": st, "n": n, "avg": float(yy.mean()),
                       "tp": float((filled > 1).mean()) if len(filled) else 0.0,
                       "fill": float(len(filled) / n), "best_buf": best[0] if best else None,
                       "best_avg": best[1] if best else None}
                row["on"] = bool(best and n >= self.a.research_min and best[1] > self.a.research_min_r)
                if row["on"]:
                    enabled[(tf, st)] = best[0]
                rows.append(row)
        rows.sort(key=lambda r: -(r["best_avg"] if r["best_avg"] is not None else -9))
        with self.lock:
            self.research, self.enabled = rows, enabled
        self.think("research", "Посчитала сетапы на истории", [
            f"{r['setup']} {r['tf']}м: {r['n']} сетапов, в среднем {r['avg']:+.2f}R, лучший запас стопа "
            f"{r['best_buf']} ATR → {r['best_avg']:+.2f}R, тейков {r['tp']:.0%}, заливка {r['fill']:.0%}"
            + (" — ТОРГУЮ" if r["on"] else " — не торгую") for r in rows if r["best_avg"] is not None])

    # ---------------------------------------------------------------- обучение
    def train(self, log=None):
        with self.lock:
            X, y, t, real = self.X.copy(), self.y.copy(), self.t.copy(), self.real.copy()
        if len(y) < self.a.min_samples:
            self.think("train", "Мало примеров для обучения", [f"{len(y)} из {self.a.min_samples}"])
            return False
        self.do_research()
        order = np.argsort(t)
        X, y, t, real = X[order], y[order], t[order], real[order]
        cut = int(len(y) * 0.8)
        age_h = (t.max() - t) / 3_600_000
        w = 0.5 ** (age_h / self.a.half_life_h) * np.where(real > 0, 3.0, 1.0)
        yc = np.clip(y, -1.5, self.a.rr + 0.5)
        m = MLP(X.shape[1], seed=int(time.time()) % 1000)
        hist, best_ep = m.fit(X[:cut], yc[:cut], w[:cut], X[cut:], yc[cut:])
        pv = m.predict(X[cut:])
        yv = y[cut:]
        sel = pv >= self.a.min_edge
        base = float(yv.mean()) if len(yv) else 0.0
        sel_avg = float(yv[sel].mean()) if sel.sum() else 0.0
        wins = float((yv[sel] > 0).mean()) if sel.sum() else 0.0
        # важность признаков: насколько растёт ошибка, если признак перемешать
        imp = []
        if len(yv) > 50:
            base_err = float(np.mean((pv - np.clip(yv, -1.5, self.a.rr + 0.5)) ** 2))
            rng = np.random.default_rng(0)
            for k in range(X.shape[1]):
                Xp = X[cut:].copy()
                Xp[:, k] = rng.permutation(Xp[:, k])
                imp.append(float(np.mean((m.predict(Xp) - np.clip(yv, -1.5, self.a.rr + 0.5)) ** 2)) - base_err)
        # дообучаем на всём (включая свежий кусок) с найденным числом эпох
        m2 = MLP(X.shape[1], seed=1)
        m2.fit(X, yc, w, X[cut:], yc[cut:], epochs=max(best_ep, 3))
        with self.lock:
            self.model = m2
            self.trained_until = float(t.max())
            self.last_train = time.time()
            self.train_hist = hist
            self.val = {"n": int(len(yv)), "sel": int(sel.sum()), "sel_avg": sel_avg, "base": base,
                        "wins": wins, "best_ep": best_ep, "time": time.time()}
            self.importance = sorted(zip(self.names, imp), key=lambda x: -x[1])[:8] if imp else []
        self.think("train", "Полное переобучение готово", [
            f"Примеров: {len(y):,}, лучшая эпоха {best_ep} из {len(hist)}",
            f"На свежем куске взяла бы {int(sel.sum())} из {len(yv)} сетапов: в среднем {sel_avg:+.2f}R "
            f"(тейков {wins:.0%}) против {base:+.2f}R у всех сетапов подряд",
            "Больше всего опираюсь на: " + ", ".join(n for n, _ in self.importance[:4]) if self.importance else "",
        ])
        self.save()
        return True

    def predict(self, X):
        with self.lock:
            m = self.model
        if m is None:
            return None
        return m.predict(np.asarray(X, float))

    # ---------------------------------------------------------------- онлайн-проверка
    def add_online(self, pred, r):
        self.online.append((time.time(), float(pred), float(r)))
        self.online = [x for x in self.online if x[0] > time.time() - 48 * 3600]

    def online_stats(self, hours=None):
        hours = hours or self.a.online_hours
        W = [x for x in self.online if x[0] > time.time() - hours * 3600]
        sel = [x[2] for x in W if x[1] >= self.a.min_edge]
        return {"n": len(W), "sel": len(sel), "sel_avg": float(np.mean(sel)) if sel else None,
                "all_avg": float(np.mean([x[2] for x in W])) if W else None,
                "wins": float(np.mean([r > 0 for r in sel])) if sel else None}

    def active(self):
        """Можно ли торговать: модель есть и её отбор по факту в плюсе (онлайн; пока онлайна мало — проверка)."""
        if self.model is None:
            return False, "модели ещё нет"
        on = self.online_stats()
        if on["sel"] >= self.a.min_online:
            ok = on["sel_avg"] > self.a.min_online_r
            return ok, (f"онлайн за {self.a.online_hours} ч: {on['sel']} отобранных, в среднем {on['sel_avg']:+.2f}R"
                        + ("" if ok else " — мало, учусь"))
        v = self.val
        if v.get("sel", 0) >= 30 and v.get("sel_avg", -1) > self.a.min_online_r and v["sel_avg"] > v.get("base", 0):
            return True, (f"онлайн-итогов пока {on['sel']} — по проверке на свежем куске "
                          f"{v['sel_avg']:+.2f}R на {v['sel']} сетапах")
        return False, f"онлайн-итогов {on['sel']} из {self.a.min_online}, проверка не в плюсе — учусь"

    def snapshot(self):
        on = self.online_stats()
        ok, why = self.active()
        return {"samples": int(len(self.y)), "real": int(self.real.sum()), "active": ok, "why": why,
                "online": on, "val": self.val, "train_hist": self.train_hist[-60:],
                "importance": self.importance, "thoughts": self.thoughts[-40:],
                "mem_hist": self.mem_hist[-720:], "online_hist": self.online_hist[-720:],
                "research": self.research,
                "last_train": self.last_train}
