"""Reward shaping denso calculado del lado del AGENTE (no del server).

Presets disponibles (elegir UNO por run: un cambio de regimen por vez):
    legacy      -- shaper historico (SimCity + combate simetrico + margen
                   al truncar). Control A/B, no el objetivo de Fase 2.
    eradicate   -- curriculum militar (Escenario A): la victoria es el objetivo.
    eradicate_v2 -- "hold & raze": gradiente denso del espectador (raze por
                   valor, defense_loss, hold_zero, spread, produce). Requiere
                   modo macro (advance() trae global_summary).
    eradicate_v3 -- v2 + economia real (refinería/recolectoras/DeltaEarned),
                   combate 3:1, sin el produce farmeable y castigo al bucle
                   build/train -> cancel_production.
"""

import math


PRESETS = ("legacy", "eradicate", "eradicate_v2", "eradicate_v3", "eradicate_v4")


def _preset_kwargs(preset: str) -> dict:
    if preset == "legacy":
        return dict(
            w_kills=0.01, w_deaths=0.01, combat_scale=10.0,
            w_assets=0.003, w_building=0.15, w_new_type=0.5,
            w_refinery=1.0, w_harvester=0.25, harvester_cap=4,
            w_raze=0.0, raze_cap=0,
            w_margin=1.0, margin_scale=3000.0, margin_on_truncate=True,
            w_win=8.0, w_lose=4.0, w_timeout=0.0,
        )
    if preset == "eradicate":
        # Combate ASIMETRICO: $1000 destruidos = +0.5; un rifle propio muerto
        # = -0.005. Marchar y perder e1 casi no cuesta; razar un edificio paga.
        return dict(
            w_kills=0.5, w_deaths=0.05, combat_scale=1000.0,
            w_assets=0.0, w_building=0.0, w_new_type=0.0,
            w_refinery=0.0, w_harvester=0.0, harvester_cap=4,
            w_raze=1.0, raze_cap=6,
            w_margin=1.0, margin_scale=3000.0, margin_on_truncate=False,
            w_win=8.0, w_lose=4.0, w_timeout=0.0,
        )
    if preset == "eradicate_v2":
        # "hold & raze": gradiente DENSO hacia la victoria. Componentes
        # espectador (requieren modo macro/advance):
        #   combat simetrico por evento, raze por VALOR, defense_loss,
        #   hold_zero, spread, produce. win +8 de colofon; truncar paga 0.
        return dict(
            w_kills=0.5, w_deaths=0.05, combat_scale=1000.0,
            w_assets=0.0, w_building=0.0, w_new_type=0.0,
            w_refinery=0.0, w_harvester=0.0, harvester_cap=4,
            w_raze=1.0, raze_cap=0, raze_value_scale=2000.0,
            w_defense_loss=0.25, defense_value_scale=2000.0, w_defense_first=3.0,
            w_hold_zero=0.06, w_spread=0.002, spread_scale=1000.0, w_produce=0.0008,
            w_margin=1.0, margin_scale=3000.0, margin_on_truncate=False,
            w_win=8.0, w_lose=4.0, w_timeout=0.0,
        )
    if preset == "eradicate_v3":
        # "hold & raze + ECONOMIA": v2 + la pieza que faltaba (recoleccion).
        # En OpenRA la recoleccion base es AUTOMATICA (construyes proc/harv y
        # minan solos); el comando harvest es control manual (proteger o
        # redirigir), NO el interruptor. Por eso el reward apunta a INFRAS +
        # DeltaEarned (la cosecha real, no farmeable):
        #   refinery +1.0 unico por tener proc
        #   harvester_up +0.25 por recolectora (cap 4)
        #   mining_rate +0.03*DeltaEarned/1000 (cosecha automatica RESULTANTE)
        #   harvester_idle -0.01/bloque al cerrar macro con harv sin ganancia
        # CORRECCION (audit 2026-08-27): el produce anterior se ELIMINA
        # (w_produce=0) porque era farmeable con el bucle build/train ->
        # cancel_production; ademas se castiga cada cancel (w_cancel=0.15).
        # Combate 3:1 (kills 0.15 / deaths 0.05): el 10:1 fomentaba zerg-rush.
        # w_lose 4->2.5 para mitigar el "pesimismo aprendido" del critico.
        return dict(
            w_kills=0.15, w_deaths=0.05, combat_scale=1000.0,
            w_assets=0.0, w_building=0.0, w_new_type=0.0,
            w_refinery=1.0, w_harvester=0.25, harvester_cap=4,
            w_raze=1.0, raze_cap=0, raze_value_scale=2000.0,
            w_defense_loss=0.25, defense_value_scale=2000.0, w_defense_first=3.0,
            w_hold_zero=0.06, w_spread=0.002, spread_scale=1000.0, w_produce=0.0,
                        w_cancel=0.15, w_refinery_early=2.0, refinery_target_tick=6000,
                        w_first_ore=1.5,
                        w_garrison=0.005, w_naked_base=0.005,
                        w_mining_rate=0.04, mining_rate_scale=1000.0, w_harvester_idle=0.01,
                        w_margin=1.0, margin_scale=3000.0, margin_on_truncate=False,
                        w_win=8.0, w_lose=2.5, w_timeout=0.0,
                    )
    if preset == "eradicate_v4":
        # Run3 — Coloso con remate: v3 + gradiente ofensivo calibrado + full-stack infra
        # w_raze 1.0->2.0 (2x, no 3x para no tapar minería), w_timeout 0->1.0
        # (rankea win > incomplete > lose sin reintroducir pesimismo).
        # SCALAR 21 (military_ratio + tech_tier) y auto_support van por fuera
        # del preset pero se entrenan juntos en este run (congelado acá).
        return dict(
            w_kills=0.15, w_deaths=0.05, combat_scale=1000.0,
            w_assets=0.0, w_building=0.0, w_new_type=0.0,
            w_refinery=1.0, w_harvester=0.25, harvester_cap=4,
            w_raze=2.0, raze_cap=0, raze_value_scale=2000.0,
            w_defense_loss=0.25, defense_value_scale=2000.0, w_defense_first=3.0,
            w_hold_zero=0.06, w_spread=0.002, spread_scale=1000.0, w_produce=0.0,
                        w_cancel=0.15, w_refinery_early=2.0, refinery_target_tick=6000,
                        w_first_ore=1.5,
                        w_garrison=0.005, w_naked_base=0.005,
                        w_mining_rate=0.04, mining_rate_scale=1000.0, w_harvester_idle=0.01,
                        w_margin=1.0, margin_scale=3000.0, margin_on_truncate=False,
                        w_win=8.0, w_lose=2.5, w_timeout=1.0,
                    )
    raise ValueError(f"preset desconocido: {preset!r} (validos: {PRESETS})")


class ShapedReward:
    def __init__(self, preset: str = "eradicate", **overrides):
        if preset not in PRESETS:
            raise ValueError(f"preset desconocido: {preset!r} (validos: {PRESETS})")
        self.preset = preset
        cfg = _preset_kwargs(preset)
        cfg.update(overrides)
        self.w_kills = cfg["w_kills"]
        self.w_deaths = cfg["w_deaths"]
        self.combat_scale = cfg["combat_scale"]
        self.w_assets = cfg["w_assets"]
        self.w_building = cfg["w_building"]
        self.w_new_type = cfg["w_new_type"]
        self.w_refinery = cfg["w_refinery"]
        self.w_harvester = cfg["w_harvester"]
        self.harvester_cap = cfg["harvester_cap"]
        self.w_raze = cfg["w_raze"]
        self.raze_cap = cfg["raze_cap"]
        self.w_margin = cfg["w_margin"]
        self.margin_scale = cfg["margin_scale"]
        self.margin_on_truncate = cfg["margin_on_truncate"]
        self.w_win = cfg["w_win"]
        self.w_lose = cfg["w_lose"]
        self.w_timeout = cfg["w_timeout"]
        # ex[eradicate_v2] parametros del gradiente denso del espectador
        self.raze_value_scale = cfg.get("raze_value_scale", 2000.0)
        self.w_defense_loss = cfg.get("w_defense_loss", 0.0)
        self.defense_value_scale = cfg.get("defense_value_scale", 2000.0)
        self.w_defense_first = cfg.get("w_defense_first", 0.0)
        self.w_hold_zero = cfg.get("w_hold_zero", 0.0)
        self.w_spread = cfg.get("w_spread", 0.0)
        self.spread_scale = cfg.get("spread_scale", 1000.0)
        self.w_produce = cfg.get("w_produce", 0.0)
        self.w_cancel = cfg.get("w_cancel", 0.0)
        # ex[eradicate_v3] economia: refinería/recolectoras/DeltaEarned
        self.w_mining_rate = cfg.get("w_mining_rate", 0.0)
        self.mining_rate_scale = cfg.get("mining_rate_scale", 1000.0)
        self.w_harvester_idle = cfg.get("w_harvester_idle", 0.0)
        self.w_refinery_early = cfg.get("w_refinery_early", 0.0)
        self.refinery_target_tick = cfg.get("refinery_target_tick", 6000)
        self.w_first_ore = cfg.get("w_first_ore", 0.0)
        self.w_garrison = cfg.get("w_garrison", 0.0)
        self.w_naked_base = cfg.get("w_naked_base", 0.0)
        self._first_ore_paid = False
        # en v2/v3 el raze se paga por VALOR del global_summary (no counting)
        self._raze_by_value = preset in ("eradicate_v2", "eradicate_v3")
        # Compat: tests/docs viejos leen w_combat como el peso simetrico.
        self.w_combat = self.w_kills

        self._margin_paid = False
        self._prev_kills_cost = 0
        self._prev_deaths_cost = 0
        self._prev_assets = 0
        self._prev_n_buildings = 0
        self._prev_buildings_killed = 0
        self._prev_enemy_n_buildings = None
        self._seen_types = set()
        self._refinery_paid = False
        self._harvesters_paid = 0
        self._raze_paid = 0.0
        self._last_mil = None
        # estado del gradiente denso del espectador (None = baseline pendiente)
        self._prev_ene_bv = None
        self._prev_own_bv = None
        self._prev_own_n = None
        self._prev_diff = None
        self._defense_first_paid = False
        self._prev_earned = None
        self.last_components = {
            "combat": 0.0, "assets": 0.0,
            "buildings": 0.0, "new_types": 0.0, "margin": 0.0,
            "mining": 0.0, "win": 0.0, "raze": 0.0, "timeout": 0.0,
            "spread": 0.0, "defense_loss": 0.0, "hold_zero": 0.0, "produce": 0.0,
            "cancel_penalty": 0.0,
            "garrison": 0.0, "early_refinery": 0.0, "first_ore": 0.0,
        }

    def reset(self, obs):
        mil = obs.military
        self._prev_kills_cost = mil.kills_cost
        self._prev_deaths_cost = mil.deaths_cost
        self._prev_assets = mil.assets_value
        self._prev_n_buildings = len(obs.buildings)
        self._prev_buildings_killed = int(
            getattr(mil, "buildings_killed", 0) or 0)
        self._prev_enemy_n_buildings = _enemy_n_buildings(obs)
        self._last_mil = mil
        self._seen_types = {b.type for b in obs.buildings}
        self._refinery_paid = "proc" in self._seen_types
        self._harvesters_paid = min(
            getattr(obs.economy, "harvester_count", 0), self.harvester_cap)
        self._raze_paid = 0.0
        for k in self.last_components:
            self.last_components[k] = 0.0
        self._margin_paid = False
        # reset del gradiente denso del espectador (baseline en el proximo gs)
        self._prev_ene_bv = None
        self._prev_own_bv = None
        self._prev_own_n = None
        self._prev_diff = None
        self._defense_first_paid = False
        self._prev_earned = None
        self._first_ore_paid = False

    def step(self, obs, done: bool, gs=None, action_type=None, closing=False) -> float:
        """Reward conformado por el delta de estado desde el paso anterior.

        gs: global_summary espectador (exacto, ambos bandos) que en modo
            macro trae advance(). Solo lo usa v2/v3 para el gradiente denso.
        action_type: str de la accion efectiva (para castigar cancel_production).
        closing: True si este step cierra un bloque macro (el ocio economico
            solo se castiga al agregar un DeltaEarned significativo, no en el
            micro-step de 2 ticks en el que el harvester aun no descarga).
        """
        mil = obs.military
        d_kills = mil.kills_cost - self._prev_kills_cost
        d_deaths = mil.deaths_cost - self._prev_deaths_cost
        d_assets = mil.assets_value - self._prev_assets
        scale = self.combat_scale if self.combat_scale else 1.0

        r_combat = (self.w_kills * d_kills - self.w_deaths * d_deaths) / scale
        r_assets = self.w_assets * d_assets / 100.0
        n_new_bldgs = max(0, len(obs.buildings) - self._prev_n_buildings)
        r_building = self.w_building * n_new_bldgs
        r = r_combat + r_assets + r_building

        n_new_types = 0
        for b in obs.buildings:
            if b.type not in self._seen_types:
                self._seen_types.add(b.type)
                n_new_types += 1
        r += self.w_new_type * n_new_types

        n_harv = getattr(obs.economy, "harvester_count", 0)
        r_mining = 0.0
        if not self._refinery_paid and "proc" in {b.type for b in obs.buildings}:
            self._refinery_paid = True
            r_mining += self.w_refinery
        nuevos_harv = min(n_harv, self.harvester_cap) - self._harvesters_paid
        if nuevos_harv > 0:
            r_mining += self.w_harvester * nuevos_harv
            self._harvesters_paid += nuevos_harv
        r += r_mining

        r_raze = self._raze_delta(obs, mil)
        r += r_raze

        # gradiente denso del espectador (v2/v3). Reemplaza al raze counting.
        r_v2 = 0.0
        if self._raze_by_value:
            r_v2 = self._v2_dense(gs)
            if self.preset == "eradicate_v3":
                r_v2 += self._v3_econ(obs, gs, action_type, closing)

        r += r_v2

        self._last_mil = mil

        # Termino terminal por RESULTADO DECLARADO. Truncamiento NO paga
        # win/lose (excepto margen en legacy). Pago UNICO: el juego puede
        # terminar dentro de advance() y el cierre step(NO_OP) trae done=True.
        r_win = 0.0
        r_margin = 0.0
        if done and not self._margin_paid:
            r_win, r_margin = self._pay_declared(obs, mil)
            r += r_win + r_margin

        self.last_components["combat"] += r_combat
        self.last_components["assets"] += r_assets
        self.last_components["buildings"] += r_building
        self.last_components["new_types"] += self.w_new_type * n_new_types
        self.last_components["mining"] += r_mining
        self.last_components["raze"] += r_raze
        self.last_components["win"] += r_win
        self.last_components["margin"] += r_margin

        self._prev_kills_cost = mil.kills_cost
        self._prev_deaths_cost = mil.deaths_cost
        self._prev_assets = mil.assets_value
        self._prev_n_buildings = len(obs.buildings)
        self._prev_buildings_killed = int(
            getattr(mil, "buildings_killed", 0) or 0)
        ene_n = _enemy_n_buildings(obs)
        if ene_n is not None:
            self._prev_enemy_n_buildings = ene_n
        return r

    def finalize(self, truncated: bool, result: str = "") -> float:
        """Termino terminal UNA vez por episodio (lo llama el rollout).

        win -> +w_win [+ margen tanh, desempate de "ganar mas"]
        lose -> -w_lose
        truncate -> 0 en eradicate (la muleta del margen se APAGO: truncar con
            ventaja material era el optimo local medido). legacy todavia paga
            el margen. w_timeout, si >0, es una urgencia constante que solo
            rankea cuando ALGUN episodio termina; si todos truncan, GAE lo
            centrado lo anula.
        """
        if self._margin_paid:
            return 0.0
        res = (result or "").lower()
        if res == "win":
            r_win, r_margin = self._win_terms(self._last_mil)
            self.last_components["win"] += r_win
            self.last_components["margin"] += r_margin
            self._margin_paid = True
            return r_win + r_margin
        if res == "lose":
            self.last_components["margin"] += -self.w_lose
            self._margin_paid = True
            return -self.w_lose
        if truncated:
            r = 0.0
            if self.margin_on_truncate and self._last_mil is not None:
                margin = (self._last_mil.kills_cost
                          - self._last_mil.deaths_cost)
                r_margin = self.w_margin * math.tanh(margin / self.margin_scale)
                self.last_components["margin"] += r_margin
                r += r_margin
            if self.w_timeout:
                self.last_components["timeout"] += -self.w_timeout
                r -= self.w_timeout
            self._margin_paid = True
            return r
        return 0.0

    def _pay_declared(self, obs, mil):
        res = (getattr(obs, "result", "") or "").lower()
        if res == "win":
            r_win, r_margin = self._win_terms(mil)
            self._margin_paid = True
            return r_win, r_margin
        if res == "lose":
            self._margin_paid = True
            return 0.0, -self.w_lose
        # done sin result (no deberia ocurrir): nada, para no reintroducir el
        # margen-al-truncar por la puerta de atras.
        self._margin_paid = True
        return 0.0, 0.0

    def _win_terms(self, mil):
        if mil is None:
            return self.w_win, 0.0
        margin = mil.kills_cost - mil.deaths_cost
        r_margin = self.w_margin * math.tanh(margin / self.margin_scale)
        return self.w_win, r_margin

    def _v2_dense(self, gs) -> float:
        """Gradiente denso espectador para eradicate_v2 (por bloque ~80 ticks).

        Lee gs (dict) own/enemy {cash, unit_value, building_value, n_buildings}
        -- exacto, sin niebla (lo trae advance() en modo macro):
          raze        = +Delta(building_value_enemigo) por valor (sin cap duro)
          defense_loss = -Delta(building_value_propio) + penalizacion por el
                        primer edificio propio perdido (paranoia temprana)
          hold_zero   = -w_hold_zero por bloque sin NINGUN edificio propio
          spread      = +/-Delta(diff material exacto) (abrir brecha paga)
        Sin gs (obs en niebla, run no-macro) devuelve 0 y no rompe el run.
        """
        if not isinstance(gs, dict):
            return 0.0
        gs_own = gs.get("own") or {}
        gs_ene = gs.get("enemy") or {}
        obv = float(gs_own.get("building_value", 0) or 0)
        ebv = float(gs_ene.get("building_value", 0) or 0)
        own_n = int(gs_own.get("n_buildings", 0) or 0)
        diff = (float(gs_own.get("cash", 0) or 0)
                + float(gs_own.get("unit_value", 0) or 0) + obv
                - float(gs_ene.get("cash", 0) or 0)
                - float(gs_ene.get("unit_value", 0) or 0) - ebv)

        # Primer global_summary del episodio: solo fija baseline (un delta
        # desde nada pagaria movimiento falso).
        if self._prev_ene_bv is None:
            self._prev_ene_bv = ebv
            self._prev_own_bv = obv
            self._prev_own_n = own_n
            self._prev_diff = diff
            return 0.0

        r_raze = max(0.0, self._prev_ene_bv - ebv) * self.w_raze / self.raze_value_scale
        r_defense = (-max(0.0, self._prev_own_bv - obv)
                     * self.w_defense_loss / self.defense_value_scale)
        if not self._defense_first_paid and own_n < self._prev_own_n:
            self._defense_first_paid = True
            r_defense -= self.w_defense_first
        r_hold = -self.w_hold_zero if own_n <= 0 else 0.0
        r_spread = self.w_spread * (diff - self._prev_diff) / self.spread_scale

        self._prev_ene_bv = ebv
        self._prev_own_bv = obv
        self._prev_own_n = own_n
        self._prev_diff = diff

        for k, v in (("raze", r_raze), ("defense_loss", r_defense),
                     ("hold_zero", r_hold), ("spread", r_spread)):
            self.last_components[k] += v
        return r_raze + r_defense + r_hold + r_spread

    def _v3_econ(self, obs, gs, action_type=None, closing=False) -> float:
        """Economia + guarnición (eradicate_v3): infraestructura + DeltaEarned +
        bonos de lección del bot.

        La recolección en OpenRA es AUTOMÁTICA (construyes proc/harv y minan
        solos); el comando harvest es CONTROL manual y NO lleva reward.
        Premiamos TENER la infraestructura y la cosecha RESULTANTE (Delta earned).
        Además:
          - early_refinery: PBRS decaído por tick para proc temprana
          - first_ore: bono único por el primer camión que entrega
          - garrison/naked_base: ancla defensiva por mantener guardias cerca
        El castigo por recolector ocioso solo se evalúa al CERRAR el bloque
        macro (closing=True): en micro-step de 2 ticks el harvester aún no
        descargó y no debe penalizarse.
        """
        r_mining = 0.0
        n_harv = sum(1 for u in obs.units
                     if getattr(u, "type", "") in ("harv", "harvester"))
        btypes = {getattr(b, "type", "") for b in obs.buildings}

        # 1. Refinería operativa (bonus único) + early bonus decaído
        if not self._refinery_paid and "proc" in btypes:
            self._refinery_paid = True
            r_mining += self.w_refinery
            if self.w_refinery_early:
                early_mult = max(0.0, 1.0 - obs.tick / float(self.refinery_target_tick))
                r_early = self.w_refinery_early * early_mult
                r_mining += r_early
                self.last_components["early_refinery"] += r_early

        # 2. Recolectoras (cap)
        nuevos_harv = min(n_harv, self.harvester_cap) - self._harvesters_paid
        if nuevos_harv > 0:
            r_mining += self.w_harvester * nuevos_harv
            self._harvesters_paid += nuevos_harv

        # 3. Cosecha RESULTANTE (Delta earned, espectador exacto) + primer camión + ocio
        earned = None
        if isinstance(gs, dict):
            earned = (gs.get("own") or {}).get("earned")
        if earned is not None:
            earned = int(earned)
            if self._prev_earned is not None:
                delta = max(0, earned - self._prev_earned)
                r_mining += self.w_mining_rate * delta / self.mining_rate_scale
                # Bono primer camión entregado (salto cuántico)
                if not self._first_ore_paid and earned > 0:
                    self._first_ore_paid = True
                    r_mining += self.w_first_ore
                    self.last_components["first_ore"] += self.w_first_ore
                if self.w_harvester_idle and closing and n_harv > 0 and delta == 0:
                    r_mining -= self.w_harvester_idle
            self._prev_earned = earned
        self.last_components["mining"] += r_mining

        # 4. Guarnición de base (anti-suicidio): >=1 guardia cerca = +w_garrison,
        #    0 guardias con base viva = -w_naked_base
        r_defense_posture = 0.0
        if obs.buildings and obs.units:
            b_coords = [(b.cell_x, b.cell_y) for b in obs.buildings]
            guards = sum(1 for u in obs.units
                         if getattr(u, "can_attack", True) and "harv" not in getattr(u, "type", "").lower()
                         and min(abs(u.cell_x - bx) + abs(u.cell_y - by) for bx, by in b_coords) <= 10)
            if guards >= 1 and self.w_garrison:
                r_defense_posture += self.w_garrison
            elif guards == 0 and self.w_naked_base:
                r_defense_posture -= self.w_naked_base
        self.last_components["garrison"] += r_defense_posture

        # 5. Producción activa (w_produce=0 en v3, queda por compat)
        prod_active = sum(1 for p in getattr(obs, "production", []) if not getattr(p, "paused", False))
        r_produce = self.w_produce * prod_active
        self.last_components["produce"] += r_produce

        # 6. Corrección bucle parásito: castigo por cancel_production
        r_cancel = 0.0
        if self.w_cancel and action_type == "cancel_production":
            r_cancel = -self.w_cancel
            self.last_components["cancel_penalty"] += r_cancel

        return r_mining + r_defense_posture + r_produce + r_cancel

    def _raze_delta(self, obs, mil) -> float:
        """Pago por edificio enemigo NETO destruido, tope por episodio.

        Prefiere n_buildings del espectador (exacto, baja al razar y sube al
        reconstruir: no se farmea una fabrica que respawnea salvo que vuelva a
        caer). Si no hay summary, cae a buildings_killed acumulado.
        """
        if self.w_raze <= 0 or self._raze_paid >= self.raze_cap:
            return 0.0
        dropped = 0
        ene_n = _enemy_n_buildings(obs)
        if ene_n is not None and self._prev_enemy_n_buildings is not None:
            dropped = max(0, self._prev_enemy_n_buildings - ene_n)
        else:
            killed = int(getattr(mil, "buildings_killed", 0) or 0)
            dropped = max(0, killed - self._prev_buildings_killed)
        if dropped <= 0:
            return 0.0
        room = self.raze_cap - self._raze_paid
        n_pay = min(dropped, room / self.w_raze if self.w_raze else 0)
        r = self.w_raze * n_pay
        self._raze_paid += r
        return r


def _enemy_n_buildings(obs):
    gs = getattr(obs, "global_summary", None)
    if not isinstance(gs, dict):
        return None
    ene = gs.get("enemy") or {}
    n = ene.get("n_buildings")
    if n is None:
        return None
    try:
        return int(n)
    except (TypeError, ValueError):
        return None