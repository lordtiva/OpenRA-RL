# -*- coding: utf-8 -*-
"""Carrera económica: series de riqueza y RECAUDACIÓN por bando.

Dos preguntas distintas, dos métricas:
  - Recaudación bruta (`earned`, del motor): cuánto mineral se extrajo.
    Siempre creciente, inmune a las muertes — "quién produce".
  - Riqueza total (cash + unidades + edificios): patrimonio neto.
    Sube con cosecha/gasto, baja con muertes — "quién conserva".

Ambas salen del GlobalSummary espectador (datos exactos de ambos bandos,
recaudación desde PlayerResources.Earned). Puro instrumental: no toca el
motor ni el reward.

Detalle de honestidad: antes del primer GlobalSummary del episodio el
enemigo es niebla (y la propia recaudación aún no fue leída), así que las
pendientes del rival y de recaudación se computan SOLO desde el primer
dato conocido — arrastrar ceros inflaría las estimaciones.
"""


class EconomyRace:
    """Muestras (tick, cash_propio, riqueza_propia, cash_rival,
    riqueza_rival, earned_propio, earned_rival)."""

    def __init__(self, max_samples: int = 400):
        self.max_samples = max_samples
        self.samples = []
        self.known = []    # ¿hay dato real del rival/recaudación en esta muestra?
        self._last_ec = 0
        self._last_ew = 0
        self._last_oe = None
        self._last_ee = 0
        self._ew_known = False

    def add(self, tick, own_cash=None, own_wealth=None,
            enemy_cash=None, enemy_wealth=None, own_earned=None,
            enemy_earned=None):
        try:
            tick = int(tick)
        except (TypeError, ValueError):
            return
        if tick < 0:
            return
        # Decimar manteniendo uniformidad si excedemos el presupuesto
        if len(self.samples) >= self.max_samples:
            self.samples = self.samples[::2]
            self.known = self.known[::2]
        # Sin retroceder ni duplicar ticks (step de cierre repite tick)
        if self.samples and self.samples[-1][0] >= tick:
            return
        # Desconocido -> arrastrar el último valor real (None ≠ cero)
        if enemy_cash is not None:
            self._last_ec = int(enemy_cash)
        if enemy_wealth is not None:
            self._last_ew = int(enemy_wealth)
            self._ew_known = True
        if own_earned is not None:
            self._last_oe = int(own_earned)
        if enemy_earned is not None:
            self._last_ee = int(enemy_earned)
        oe = self._last_oe if self._last_oe is not None else 0
        self.samples.append((tick,
                             int(own_cash or 0), int(own_wealth or 0),
                             self._last_ec, self._last_ew,
                             oe, self._last_ee))
        self.known.append(self._ew_known)

    def add_global_summary(self, tick: int, gs) -> None:
        """Muestrea desde un bloque global_summary del modo espectador."""
        if not isinstance(gs, dict):
            return
        own, ene = gs.get("own") or {}, gs.get("enemy") or {}
        self.add(tick,
                 own_cash=own.get("cash"), enemy_cash=ene.get("cash"),
                 own_wealth=(own.get("cash", 0) + own.get("unit_value", 0)
                             + own.get("building_value", 0)),
                 enemy_wealth=(ene.get("cash", 0) + ene.get("unit_value", 0)
                               + ene.get("building_value", 0)),
                 own_earned=own.get("earned"),
                 enemy_earned=ene.get("earned"))

    def _delta_1k(self, idx: int, desde: int = 0) -> float:
        """(fin - inicio) / Δticks × 1000 sobre samples[desde:] (col. idx).

        F7 (auditoría): para la RECAUDACIÓN bruta (contador monótono del
        motor que sube y luego platanea) la pendiente OLS daba valores
        imposibles (negativos). La tasa correcta de un acumulativo es el
        delta de extremos.
        """
        ss = self.samples[desde:]
        n = len(ss)
        if n < 2:
            return 0.0
        dt = ss[-1][0] - ss[0][0]
        if dt <= 0:
            return 0.0
        return (ss[-1][idx] - ss[0][idx]) / dt * 1000.0

    def _slope_1k(self, idx: int, desde: int = 0) -> float:
        """Pendiente OLS por 1000 ticks (SOLO para riqueza, que NO es
        monótona y sí tiene tendencia con ruido)."""
        xs = [s[0] for s in self.samples[desde:]]
        ys = [s[idx] for s in self.samples[desde:]]
        n = len(xs)
        if n < 2:
            return 0.0
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den == 0:
            return 0.0
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return (num / den) * 1000.0

    def resumen(self):
        """Agregados livianos para metrics.jsonl (None sin datos)."""
        if len(self.samples) < 2:
            return None
        t0, oc0, ow0, ec0, ew0, oe0, ee0 = self.samples[0]
        t1, oc1, ow1, ec1, ew1, oe1, ee1 = self.samples[-1]
        first_known = self.known.index(True) if True in self.known else None
        own_inc = self._slope_1k(2)
        enemy_inc = (self._slope_1k(4, desde=first_known)
                     if first_known is not None else 0.0)
        # F7: recaudación bruta por DELTA de extremos (no OLS), solo desde
        # el primer dato conocido.
        own_harv = (self._delta_1k(5, desde=first_known)
                    if first_known is not None else 0.0)
        enemy_harv = (self._delta_1k(6, desde=first_known)
                      if first_known is not None else 0.0)
        leads = [ow - ew for (_, _, ow, _, ew, _, _), k
                 in zip(self.samples, self.known) if k]
        return {
            "samples": len(self.samples),
            "own_cash_end": oc1,
            "enemy_cash_end": ec1,
            "own_wealth_start": ow0, "own_wealth_end": ow1,
            "enemy_wealth_start": ew0, "enemy_wealth_end": ew1,
            "own_income_per_1k": round(own_inc, 1),
            "enemy_income_per_1k": round(enemy_inc, 1),
            "income_edge": round(own_inc - enemy_inc, 1),
            "own_harvest_per_1k": round(own_harv, 1),
            "enemy_harvest_per_1k": round(enemy_harv, 1),
            "harvest_edge": round(own_harv - enemy_harv, 1),
            "own_earned_total": oe1, "enemy_earned_total": ee1,
            "peak_lead": round(max(leads), 0) if leads else 0,
            "worst_deficit": round(min(leads), 0) if leads else 0,
        }

    def series(self) -> dict:
        """Series completas para el archivo por-episodio."""
        return {
            "ticks": [s[0] for s in self.samples],
            "own_cash": [s[1] for s in self.samples],
            "own_wealth": [s[2] for s in self.samples],
            "enemy_cash": [s[3] for s in self.samples],
            "enemy_wealth": [s[4] for s in self.samples],
            "own_earned": [s[5] for s in self.samples],
            "enemy_earned": [s[6] for s in self.samples],
        }
