"""Red política estilo AlphaStar-lite para OpenRA.

Arquitectura:
    - Encoder espacial: CNN+ResBlocks sobre el tensor 9×H×W
    - Encoder de escalares: MLP sobre economía/militar
    - Encoder de unidades: MLP por-slot + pooling enmascarado (set encoding)
    - Core: GRU (memoria de parcialmente-observable; hidden se guarda entre
      steps y se DESACOPLA del gradiente — sin BPTT, simplificación deliberada)
    - Cabezas AUTORREGRESIVAS con máscaras de acciones legales:
        1) tipo de acción (21 tipos)
        2) slot de unidad (condicionada al tipo elegido)
        3) celda objetivo H×W (conv 1×1 sobre el mapa + emb del tipo)
        4) ítem de producción (embedding de tipos de actor)

El log_prob total es la suma SOLO de las cabezas que el tipo elegido usa
(cadena p(tipo)·[p(unidad|tipo)]·[p(celda|tipo)]·[p(ítem|tipo)]) — auditoría
2026-08-24 (F3): antes se sumaban unidad+celda SIEMPRE y ~6000 logits de
celda metían ruido en acciones que ni los miraban.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.obs_encoding import MAX_UNITS, SCALAR_DIM

ACTION_TYPES = [
    "no_op", "move", "attack_move", "attack", "stop", "harvest",
    "build", "train", "deploy", "sell", "repair", "place_building",
    "cancel_production", "set_rally_point", "guard", "set_stance",
    "enter_transport", "unload", "power_down", "set_primary", "surrender",
    "army_attack_move",
]
N_ACTION_TYPES = len(ACTION_TYPES)
TYPE_TO_IDX = {t: i for i, t in enumerate(ACTION_TYPES)}

HIDDEN_DIM = 416

# Qué cabeza usa cada tipo de acción (FUENTE ÚNICA para log_prob condicional;
# action_adapter debe ser coherente con estos conjuntos). Auditoría 2026-08-24.
TYPES_USE_UNIT = {"move", "attack_move", "attack", "stop", "set_stance",
                  "harvest", "deploy"}
TYPES_USE_CELL = {"move", "attack_move", "attack", "place_building",
                  "army_attack_move"}
TYPES_USE_ITEM = {"train", "build", "place_building", "cancel_production"}


def _heads_used(t_idx: torch.Tensor, device) -> tuple:
    """Mascaras [B] bool: usa_unidad, usa_celda, usa_item para cada tipo."""
    names = [ACTION_TYPES[int(t)] for t in t_idx]
    use_u = torch.tensor([n in TYPES_USE_UNIT for n in names],
                         dtype=torch.bool, device=device)
    use_c = torch.tensor([n in TYPES_USE_CELL for n in names],
                         dtype=torch.bool, device=device)
    use_i = torch.tensor([n in TYPES_USE_ITEM for n in names],
                         dtype=torch.bool, device=device)
    return use_u, use_c, use_i


def build_type_masks(obs) -> torch.Tensor:
    """Máscara booleana [N_ACTION_TYPES] de tipos legales para esta observación.

    Aproximaciones documentadas (refinar con feedback real del engine):
      - place_building requiere producción de edificio completada (progress>=1)
      - deploy requiere una unidad tipo MCV propia
      - enter_transport/unload deshabilitados en v0
      - surrender nunca legal (la política no aprende a rendirse)
    """
    m = np.zeros(N_ACTION_TYPES, dtype=bool)
    m[TYPE_TO_IDX["no_op"]] = True

    have_units = len(obs.units) > 0
    have_buildings = len(obs.buildings) > 0
    if have_units:
        for t in ("move", "attack_move", "attack", "stop", "guard",
                  "harvest", "set_stance", "army_attack_move"):
            m[TYPE_TO_IDX[t]] = True
        m[TYPE_TO_IDX["deploy"]] = any("mcv" in u.type.lower() for u in obs.units)
    if have_buildings:
        for t in ("sell", "repair", "power_down", "set_primary", "set_rally_point"):
            m[TYPE_TO_IDX[t]] = True

    building_done = any(
        p.queue_type == "Building" and p.progress >= 1.0 for p in obs.production
    )
    if building_done:
        m[TYPE_TO_IDX["place_building"]] = True
    if obs.available_production:
        m[TYPE_TO_IDX["train"]] = True   # unidades
        m[TYPE_TO_IDX["build"]] = True   # edificios
        # Combat TRAIN (e1 / infantry_basic / tanks / ...) is gated later in
        # ActionIndex until the player owns a proc + harvester. BUILD/PLACE
        # of proc stay legal so this cannot deadlock the economy.
    if obs.production:
        m[TYPE_TO_IDX["cancel_production"]] = True
    return torch.from_numpy(m)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + F.relu(self.conv2(F.relu(self.conv1(x))))


class AlphaLiteNet(nn.Module):
    def __init__(self, n_item_types: int = 128, item_embed_dim: int = 64):
        super().__init__()
        # Agnóstica al tamaño de mapa: las head convolucionales operan sobre
        # cualquier H×W (adaptive pool para el vector global)

        ch = 96
        # Encoder espacial con CoordConv (canales 9-10: x,y normalizados a
        # [-1,1]) y U-Net lite (2 niveles down/up con skip) para ampliar el
        # campo receptivo (~9 celdas -> ~35-45) sin perder resolución en el
        # fmap final (sigue [B,ch,H,W]).
        self.spatial_in = nn.Sequential(
            nn.Conv2d(9 + 2, ch, 3, padding=1), nn.ReLU())
        self.enc1 = nn.Sequential(
            ResBlock(ch), nn.Conv2d(ch, ch, 3, stride=2, padding=1),
            nn.ReLU(), ResBlock(ch))
        self.enc2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.ReLU(), ResBlock(ch))
        self.bott = ResBlock(ch)
        self.dec1 = nn.Sequential(
            nn.Conv2d(2 * ch, ch, 3, padding=1), nn.ReLU(), ResBlock(ch))
        self.dec0 = nn.Sequential(
            nn.Conv2d(2 * ch, ch, 3, padding=1), nn.ReLU(), ResBlock(ch))
        self.scalar_mlp = nn.Sequential(
            nn.Linear(SCALAR_DIM, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
        )
        self.unit_mlp = nn.Sequential(
            nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
        )
        fused = ch + 128 + 128
        self.core = nn.GRUCell(fused, HIDDEN_DIM)

        # Embeddings de condicionamiento autoregresivo
        self.type_embedding = nn.Embedding(N_ACTION_TYPES, 64)
        self.item_embedding = nn.Embedding(n_item_types, item_embed_dim)
        # entrada: hidden + emb tipo + emb ítem
        self.item_scorer = nn.Linear(HIDDEN_DIM + 64 + item_embed_dim, 1)

        # Cabezas
        self.head_type = nn.Linear(HIDDEN_DIM, N_ACTION_TYPES)
        # Scorer PER-SLOT: cada unidad es puntuada con SUS features propias
        # (HP, tipo implícito por velocidad/rango, posición...) junto al
        # estado global y el tipo de acción elegido.
        self.unit_scorer = nn.Sequential(
            nn.Linear(10 + HIDDEN_DIM + 64, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.cell_head = nn.Conv2d(ch + 64 + 64, 1, 1)  # fmap + emb tipo + hidden(gru)
        # Proyección del estado global del GRU para broadcast espacial en la
        # cabeza de celda: la decisión (x,y) "ve" hacia dónde va el plan.
        self.hidden_proj = nn.Linear(HIDDEN_DIM, 64)
        self.value_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 256), nn.ReLU(), nn.Linear(256, 1),
        )

    def _coord_conv(self, spatial):
        """CoordConv: canales 9-10 con x,y normalizados a [-1,1]."""
        B, _, H, W = spatial.shape
        ys = torch.linspace(-1.0, 1.0, H, device=spatial.device).view(1, 1, H, 1)
        xs = torch.linspace(-1.0, 1.0, W, device=spatial.device).view(1, 1, 1, W)
        yy = ys.expand(B, 1, H, W)
        xx = xs.expand(B, 1, H, W)
        return torch.cat([spatial, xx, yy], dim=1)

    def _enc_spatial(self, x):
        """U-Net lite: 2 niveles down/up con skip, devuelve fmap [B,ch,H,W]."""
        x = self.spatial_in(x)          # [B,ch,H,W]
        s1 = x
        x = self.enc1(x)                # [B,ch,H/2,W/2]
        s2 = x
        x = self.enc2(x)                # [B,ch,H/4,W/4]
        x = self.bott(x)
        x = self.dec1(torch.cat(
            [F.interpolate(x, size=s2.shape[2:], mode="bilinear", align_corners=False),
             s2], dim=1))
        x = self.dec0(torch.cat(
            [F.interpolate(x, size=s1.shape[2:], mode="bilinear", align_corners=False),
             s1], dim=1))
        return x

    def encode(self, spatial, scalars, unit_feats, unit_valid, hidden):
        """spatial [B,9,H,W], scalars [B,S], units [B,U,F], valid [B,U] bool."""
        fmap = self._enc_spatial(self._coord_conv(spatial))
        feat_map_flat = fmap.flatten(1)
        spatial_vec = F.adaptive_avg_pool2d(fmap, 1).flatten(1)

        u = self.unit_mlp(unit_feats)
        valid_f = unit_valid.float().unsqueeze(-1)
        denom = valid_f.sum(1).clamp(min=1.0)
        unit_vec = (u * valid_f).sum(1) / denom

        s = self.scalar_mlp(scalars)
        fused = torch.cat([spatial_vec, s, unit_vec], dim=-1)
        new_hidden = self.core(fused, hidden)
        return fmap, feat_map_flat, new_hidden

    @staticmethod
    def _categorical(logits):
        """Categorical that does not crash on NaN/all-masked rows (PPO 947)."""
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        dead = (logits <= -1e8).all(dim=-1)
        if dead.any():
            logits = logits.clone()
            logits[dead, 0] = 0.0
        return torch.distributions.Categorical(logits=logits)

    def dist_type(self, hidden, type_mask):
        logits = self._logits_type(hidden, type_mask)
        return self._categorical(logits)

    def _logits_type(self, hidden, type_mask):
        return self.head_type(hidden).masked_fill(~type_mask, -1e9)

    def _scores_unit(self, hidden, chosen_type, unit_feats, unit_valid):
        """Logits crudos de la cabeza de unidades (para temperatura real)."""
        B, U, _ = unit_feats.shape
        h = hidden.unsqueeze(1).expand(-1, U, -1)
        t = self.type_embedding(chosen_type).unsqueeze(1).expand(-1, U, -1)
        scores = self.unit_scorer(
            torch.cat([unit_feats, h, t], dim=-1)).squeeze(-1)
        return scores.masked_fill(~unit_valid, -1e9)

    def dist_unit(self, hidden, chosen_type, unit_feats, unit_valid):
        """Puntúa CADA slot con sus propias features (+ estado global + tipo).

        Unidades idénticas reciben puntajes casi idénticos (elección
        indistinta, correcto); unidades distintas (MCV vs rifleman herido)
        son distinguibles desde las features.
        """
        logits = self._scores_unit(hidden, chosen_type, unit_feats, unit_valid)
        return self._categorical(logits)

    def _logits_cell(self, fmap, chosen_type, cell_mask, hidden):
        """Logits crudos [B, H*W] de la cabeza de celda.

        Recibe fmap (U-Net/CoordConv, campo receptivo amplio) + embedding del
        tipo + el estado global del GRU proyectado (broadcast): la decisión
        (x,y) conoce hacia dónde va el plan, no solo el parche local.
        """
        emb = self.type_embedding(chosen_type)  # [B,64]
        b, _, H, W = fmap.shape
        emb_map = emb[:, :, None, None].expand(-1, -1, H, W)
        hd = F.relu(self.hidden_proj(hidden))[:, :, None, None].expand(-1, -1, H, W)
        logits_map = self.cell_head(
            torch.cat([fmap, emb_map, hd], dim=1)).squeeze(1)
        logits_map = logits_map.masked_fill(~cell_mask.view(b, H, W), -1e9)
        return logits_map.reshape(b, -1)

    def dist_cell(self, fmap, chosen_type, cell_mask, hidden):
        logits = self._logits_cell(fmap, chosen_type, cell_mask, hidden)
        return self._categorical(logits)

    def _scores_item(self, hidden, chosen_type, item_indices, item_mask):
        """Logits crudos de la cabeza de ítems. item_indices: [B,V]
        ids de vocabulario (pad = 0); item_mask: [B,V]."""
        emb = self.item_embedding(item_indices)  # [B,V,E]
        h = hidden.unsqueeze(1).expand(-1, emb.size(1), -1)
        t = self.type_embedding(chosen_type).unsqueeze(1).expand(-1, emb.size(1), -1)
        scores = self.item_scorer(torch.cat([h, t, emb], dim=-1)).squeeze(-1)
        return scores.masked_fill(~item_mask, -1e9)

    def dist_item(self, hidden, chosen_type, item_indices, item_mask):
        logits = self._scores_item(hidden, chosen_type, item_indices, item_mask)
        return self._categorical(logits)

    def _item_cat_mask(self, batch, t_idx):
        """Máscara de ítems ESTRICTAMENTE condicional a la cabeza de tipo.

        Autorregresivo: si el tipo muestreado es 'train', SOLO habilitamos los
        slots de train_roles; si 'build', SOLO los de build_roles. Así es
        matemáticamente imposible muestrear un ítem de categoría equivocada y
        desaparece la coerción post-hoc del adapter (sin sesgo off-policy).
        Para tipos sin ítems se devuelve la máscara base (use_i es False ahí).
        """
        trm = (batch.get("train_slot_mask") if isinstance(batch, dict)
               else getattr(batch, "train_slot_mask", None))
        bum = (batch.get("build_slot_mask") if isinstance(batch, dict)
               else getattr(batch, "build_slot_mask", None))
        base = (batch.get("item_mask") if isinstance(batch, dict)
                else getattr(batch, "item_mask", None))
        if base is None or trm is None or bum is None:
            return base
        trm = trm.bool().to(base.device)
        bum = bum.bool().to(base.device)
        is_tr = (t_idx == ACTION_TYPES.index("train")).unsqueeze(-1)
        is_bu = (t_idx == ACTION_TYPES.index("build")).unsqueeze(-1)
        cat = (is_tr & trm) | (is_bu & bum)          # [B,V]
        have = cat.any(dim=-1, keepdim=True)
        return torch.where(have, base & cat, base)

    @torch.no_grad()
    def act(self, batch, hidden, temperature: float = 1.0):
        """Muestrea UNA acción completa para cada elemento del batch.

        F4 (auditoría 2026-08-24): temperature divide los logits de TODAS las
        cabezas. Antes solo T<=0 hacia argmax en tipo y cualquier otro valor
        era T=1.0 disfrazado (el diagnóstico a 0.35 muestreaba igual).
        T=0 -> argmax total (greedy verdadero en las 4 cabezas).

        El log_prob devuelto es el de la POLÍTICA (T=1) sobre lo muestreado:
        es la referencia contra la que PPO mide el drift.
        """
        fmap, _, new_hidden = self.encode(
            batch["spatial"], batch["scalars"],
            batch["unit_feats"], batch["unit_valid"], hidden,
        )
        greedy = temperature <= 0.0

        lt = self._logits_type(new_hidden, batch["type_mask"])
        dist_t = self._categorical(lt)
        t_idx = lt.argmax(dim=-1) if greedy else \
            self._categorical(lt / temperature).sample()

        ls_u = self._scores_unit(new_hidden, t_idx, batch["unit_feats"],
                                 batch["unit_valid"])
        dist_u = self._categorical(ls_u)
        u_idx = ls_u.argmax(dim=-1) if greedy else \
            self._categorical(ls_u / temperature).sample()

        lc = self._logits_cell(fmap, t_idx, batch["cell_mask"], new_hidden)
        dist_c = self._categorical(lc)
        c_idx = lc.argmax(dim=-1) if greedy else \
            self._categorical(lc / temperature).sample()

        has_items = batch["item_mask"].any(dim=-1)
        safe_item_mask = self._item_cat_mask(batch, t_idx).clone()
        safe_item_mask[~has_items] = True  # fila dummy si no hay items
        li = self._scores_item(new_hidden, t_idx, batch["item_indices"],
                               safe_item_mask)
        dist_i = self._categorical(li)
        # SAMPLEAR (no argmax) salvo greedy: argmax mataba la exploración
        # entre edificios y el agente colapsaba en producir siempre el mismo
        i_sampled = li.argmax(dim=-1) if greedy else \
            self._categorical(li / temperature).sample()
        i_idx = torch.where(has_items, i_sampled, torch.zeros_like(t_idx))

        # F3 (auditoría 2026-08-24): log_prob SOLO de las cabezas que el
        # tipo usa. Antes unidad+celda sumaban siempre (~6000 logits de
        # celda dominando con ruido puro en no_op/train/build).
        use_u, use_c, use_i = _heads_used(t_idx, t_idx.device)
        zero = torch.zeros_like(dist_t.log_prob(t_idx))
        lp = (dist_t.log_prob(t_idx)
              + torch.where(use_u, dist_u.log_prob(u_idx), zero)
              + torch.where(use_c, dist_c.log_prob(c_idx), zero)
              + torch.where(use_i & has_items, dist_i.log_prob(i_idx.clamp(
                  min=0, max=safe_item_mask.size(1) - 1)), zero))

        value = self.value_head(new_hidden).squeeze(-1)
        return {
            "hidden": new_hidden,
            "type": t_idx, "unit_slot": u_idx, "cell_flat": c_idx,
            "item_slot": i_idx,
            "log_prob": lp, "value": value,
        }

    def evaluate_actions(self, batch, hidden, actions):
        """Recalcula log_prob/valor/entropía para PPO (misma semilla de hidden).

        Nota: el hidden entrante se trata como constante (detach en rollout);
        el gradiente fluye solo por los parámetros del step actual.

        F3 (auditoría 2026-08-24): el log_prob suma SOLO las cabezas que el
        tipo usa (mismos conjuntos que act()). Debe llamarse con los índices
        EFECTIVOS de la acción ejecutada.
        """
        fmap, _, new_hidden = self.encode(
            batch["spatial"], batch["scalars"],
            batch["unit_feats"], batch["unit_valid"], hidden,
        )
        t_idx = actions["type"]
        dist_t = self.dist_type(new_hidden, batch["type_mask"])
        dist_u = self.dist_unit(new_hidden, t_idx, batch["unit_feats"],
                                batch["unit_valid"])
        dist_c = self.dist_cell(fmap, t_idx, batch["cell_mask"], new_hidden)

        has_items = batch["item_mask"].any(dim=-1)
        safe_item_mask = self._item_cat_mask(batch, t_idx).clone()
        safe_item_mask[~has_items] = True
        dist_i = self.dist_item(new_hidden, t_idx, batch["item_indices"],
                                safe_item_mask)

        use_u, use_c, use_i = _heads_used(t_idx, t_idx.device)
        zero = torch.zeros_like(dist_t.log_prob(t_idx))
        lp = (dist_t.log_prob(t_idx)
              + torch.where(use_u, dist_u.log_prob(actions["unit_slot"]), zero)
              + torch.where(use_c, dist_c.log_prob(actions["cell_flat"]), zero)
              + torch.where(use_i & has_items & actions["had_item"],
                            dist_i.log_prob(actions["item_slot"].clamp(
                                min=0, max=batch["item_mask"].size(1) - 1)),
                            zero))

        # Entropía TOTAL de las cabezas activas (revisión externa 2026-08-24):
        # antes solo la del tipo ("proxy regularizador"), lo que permitía
        # colapso determinista prematuro en unidad/celda/ítem. La cabeza de
        # celda va amortiguada (x0.25): su espacio es ~6000 veces el del tipo
        # y sin freno dominaría el bono. Promedio ponderado mantiene la escala
        # del coeficiente de entropía existente.
        h_t = dist_t.entropy()
        h_u = dist_u.entropy()
        h_c = dist_c.entropy() * 0.25
        h_i = torch.where(has_items, dist_i.entropy(),
                          torch.zeros_like(h_t)) * 0.5
        # Entropía ENMASCARADA por cabeza activa (mismo criterio que en el
        # entrenamiento por segmentos): no regularizar cabezas no actuantes.
        zero_h = torch.zeros_like(h_t)
        use_u_f = use_u.float()
        use_c_f = use_c.float() * 0.25
        use_i_f = (use_i & has_items).float() * 0.5
        entropy = ((h_t
                    + torch.where(use_u, h_u, zero_h)
                    + torch.where(use_c, h_c, zero_h)
                    + torch.where(use_i & has_items, h_i, zero_h))
                   / (1.0 + use_u_f + use_c_f + use_i_f))
        value = self.value_head(new_hidden).squeeze(-1)
        return lp, entropy, value

    def evaluate_actions_seq(self, seg, device):
        """PPO recurrente con BPTT truncado sobre UN segmento (lista de steps).

        Propaga el estado oculto del GRU a lo largo del segmento SI
        n detach: el gradiente fluye h_t → h_{t+1} y la red aprende a guardar
        memoria temporal (POMDP / niebla de guerra). El h_in del PRIMER step
        entra como constante (arranque del segmento); los siguientes usan el
        hidden PROPAGADO por el GRU, no el guardado del rollout.

        Devuelve (log_prob, entropy, value) concatenados: un valor por step.

        (Fijación de memoria temporal — sin BPTT el GRUCell se entrenaba como
        un MLP no-recurrente y no aprendía qué guardar para el futuro.)
        """
        lp_list, ent_list, val_list = [], [], []
        h = seg[0]["h_in"].to(device)  # [1, HIDDEN] arranque (constante)
        for s in seg:
            b = {k: s["batch"][k].to(device)
                 for k in ("spatial", "scalars", "unit_feats", "unit_valid",
                           "type_mask", "cell_mask", "item_indices",
                           "item_mask")}
            # máscara jerárquica por categoría (opcional: muestras viejas sin
            # ella caen a la máscara global base)
            if s["batch"].get("train_slot_mask") is not None:
                b["train_slot_mask"] = s["batch"]["train_slot_mask"].to(device)
                b["build_slot_mask"] = s["batch"]["build_slot_mask"].to(device)
            else:
                b["train_slot_mask"] = b["item_mask"]
                b["build_slot_mask"] = b["item_mask"]
            fmap, _, h = self.encode(
                b["spatial"], b["scalars"], b["unit_feats"],
                b["unit_valid"], h)  # SIN detach → gradiente a través del time
            t_idx = s["action"]["type"].to(device)
            dist_t = self.dist_type(h, b["type_mask"])
            dist_u = self.dist_unit(h, t_idx, b["unit_feats"], b["unit_valid"])
            dist_c = self.dist_cell(fmap, t_idx, b["cell_mask"], h)
            has_items = b["item_mask"].any(dim=-1)
            safe_item = self._item_cat_mask(b, t_idx).clone()
            safe_item[~has_items] = True
            dist_i = self.dist_item(h, t_idx, b["item_indices"], safe_item)

            use_u, use_c, use_i = _heads_used(t_idx, t_idx.device)
            zero = torch.zeros_like(dist_t.log_prob(t_idx))
            u_idx = s["action"]["unit_slot"].to(device)
            c_idx = s["action"]["cell_flat"].to(device)
            i_idx = s["action"]["item_slot"].to(device)
            had_item = bool(s["action"]["had_item"])
            lp = (dist_t.log_prob(t_idx)
                  + torch.where(use_u, dist_u.log_prob(u_idx), zero)
                  + torch.where(use_c, dist_c.log_prob(c_idx), zero)
                  + torch.where(use_i & has_items & had_item,
                                dist_i.log_prob(i_idx.clamp(
                                    min=0, max=safe_item.size(1) - 1)), zero))

            ht = dist_t.entropy()
            hu = dist_u.entropy()
            hc = dist_c.entropy() * 0.25
            hi = torch.where(has_items, dist_i.entropy(), torch.zeros_like(ht)) * 0.5

            # Entropía ENMASCARADA por cabeza activa: no inyectar gradientes de
            # exploración en cabezas que NO participaron en la acción del paso
            # (p.ej. no_op/train no deben regularizar cell_head/unit_scorer).
            zero = torch.zeros_like(ht)
            use_u_f = use_u.float()
            use_c_f = use_c.float() * 0.25
            use_i_f = (use_i & has_items).float() * 0.5
            entropy = ((ht
                        + torch.where(use_u, hu, zero)
                        + torch.where(use_c, hc, zero)
                        + torch.where(use_i & has_items, hi, zero))
                       / (1.0 + use_u_f + use_c_f + use_i_f))
            value = self.value_head(h).squeeze(-1)
            lp_list.append(lp)
            ent_list.append(entropy)
            val_list.append(value)
        return (torch.cat(lp_list), torch.cat(ent_list), torch.cat(val_list))
