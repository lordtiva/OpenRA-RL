"""Red política estilo AlphaStar-lite para OpenRA.

Arquitectura:
    - Encoder espacial: CNN+ResBlocks sobre el tensor 9×H×W
    - Encoder de escalares: MLP sobre economía/militar
    - Encoder de unidades: MLP por-slot + transformer 2×4h d=64 (Capa 2)
      residual gate=0 al init → GRU ve el mean 922
    - Core: GRU (memoria de parcialmente-observable; hidden se guarda entre
      steps y se DESACOPLA del gradiente — sin BPTT, simplificación deliberada)
    - Cabezas AUTORREGRESIVAS con máscaras de acciones legales:
        1) tipo de acción (21 tipos)
        2) slot de unidad (condicionada al tipo elegido)
        3) celda objetivo H×W (conv 1×1: fmap + scatter + tipo + GRU + unidad)
        4) ítem de producción (embedding de tipos de actor)
        5) slot enemigo (solo `attack`: pointer, no el más cercano)

El log_prob total es la suma SOLO de las cabezas que el tipo elegido usa
(cadena p(tipo)·[p(unidad|tipo)]·[p(celda|tipo)]·[p(ítem|tipo)]·[p(ene|attack)])
— auditoría 2026-08-24 (F3): antes se sumaban unidad+celda SIEMPRE y ~6000
logits de celda metían ruido en acciones que ni los miraban.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.obs_encoding import (
    MAX_ENEMIES, MAX_TOKENS, MAX_UNITS, SCALAR_DIM, UNIT_FEAT_DIM,
    ready_place_items,
)
from rl.roles import N_ROLES

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
# Capa 2 (doc 12) + 2c-A 96 + 2c-B role/team/32 ene + 2c-C attack pointer.
# Residual/zero-init para Net2Net desde 922 (GRU y U-Net se conservan).
ROLE_EMB_DIM = 8
UNIT_MLP_IN = UNIT_FEAT_DIM + ROLE_EMB_DIM  # 11 + 8 = 19
XF_DIM = 64
XF_HEADS = 4
XF_LAYERS = 2
XF_FF = 128
SCATTER_CH = 8
UNIT_COND_DIM = 64
SPATIAL_CH = 96
CELL_HEAD_OLD_IN = SPATIAL_CH + 64 + 64  # fmap + tipo + hidden (pre Capa 2)

# Qué cabeza usa cada tipo de acción (FUENTE ÚNICA para log_prob condicional;
# action_adapter debe ser coherente con estos conjuntos). Auditoría 2026-08-24.
TYPES_USE_UNIT = {"move", "attack_move", "attack", "stop", "set_stance",
                  "harvest", "deploy"}
TYPES_USE_CELL = {"move", "attack_move", "attack", "place_building",
                  "army_attack_move"}
TYPES_USE_ITEM = {"train", "build", "place_building", "cancel_production"}
# Capa 2c-C: pointer de ataque. army_attack_move / attack_move no: AutoTarget.
TYPES_USE_ENEMY = {"attack"}
TOKEN_DIM = 128  # unit_mlp / xf residual
ENEMY_SCORER_IN = TOKEN_DIM + TOKEN_DIM + HIDDEN_DIM + 64  # ene+own+h+type


def _heads_used(t_idx: torch.Tensor, device) -> tuple:
    """Mascaras [B] bool: usa_unidad, usa_celda, usa_item, usa_enemigo."""
    names = [ACTION_TYPES[int(t)] for t in t_idx]
    use_u = torch.tensor([n in TYPES_USE_UNIT for n in names],
                         dtype=torch.bool, device=device)
    use_c = torch.tensor([n in TYPES_USE_CELL for n in names],
                         dtype=torch.bool, device=device)
    use_i = torch.tensor([n in TYPES_USE_ITEM for n in names],
                         dtype=torch.bool, device=device)
    use_e = torch.tensor([n in TYPES_USE_ENEMY for n in names],
                         dtype=torch.bool, device=device)
    return use_u, use_c, use_i, use_e


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

    if ready_place_items(obs):
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

        ch = SPATIAL_CH
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
        self.role_emb = nn.Embedding(N_ROLES, ROLE_EMB_DIM, padding_idx=0)
        nn.init.zeros_(self.role_emb.weight)
        self.unit_mlp = nn.Sequential(
            nn.Linear(UNIT_MLP_IN, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
        )
        # Transformer de entidades: residual * scale, scale=0 → mean 922.
        xf_layer = nn.TransformerEncoderLayer(
            d_model=XF_DIM, nhead=XF_HEADS, dim_feedforward=XF_FF,
            dropout=0.0, batch_first=True, norm_first=True,
            activation="relu")
        self.unit_xf_in = nn.Linear(128, XF_DIM)
        self.unit_xf = nn.TransformerEncoder(
            xf_layer, num_layers=XF_LAYERS, enable_nested_tensor=False)
        self.unit_xf_out = nn.Linear(XF_DIM, 128)
        self.unit_xf_scale = nn.Parameter(torch.zeros(1))
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
            nn.Linear(UNIT_MLP_IN + HIDDEN_DIM + 64, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        # Capa 2c-C: pointer de ataque. Last layer 0 → uniforme entre visibles.
        self.enemy_scorer = nn.Sequential(
            nn.Linear(ENEMY_SCORER_IN, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        nn.init.zeros_(self.enemy_scorer[-1].weight)
        nn.init.zeros_(self.enemy_scorer[-1].bias)
        # Capa 2: fmap + scatter + tipo + GRU + unidad elegida (o pool).
        self.scatter_proj = nn.Linear(UNIT_MLP_IN, SCATTER_CH)
        nn.init.zeros_(self.scatter_proj.weight)
        nn.init.zeros_(self.scatter_proj.bias)
        self.unit_cond_proj = nn.Linear(128, UNIT_COND_DIM)
        nn.init.zeros_(self.unit_cond_proj.weight)
        nn.init.zeros_(self.unit_cond_proj.bias)
        self.cell_head = nn.Conv2d(
            ch + SCATTER_CH + 64 + 64 + UNIT_COND_DIM, 1, 1)
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

    def _entity_tokens(self, u, unit_valid):
        """MLP tokens + transformer residual (scale=0 → identidad 922)."""
        pad = ~unit_valid
        all_pad = pad.all(dim=-1)
        pad = pad.clone()
        pad[all_pad, 0] = False
        tok = self.unit_xf(self.unit_xf_in(u), src_key_padding_mask=pad)
        tok = tok.masked_fill(all_pad[:, None, None], 0.0)
        return u + self.unit_xf_scale * self.unit_xf_out(tok)

    def _unit_in(self, unit_feats, role_ids):
        """cat(feats11, role_emb8) → 19-d para mlp/scatter/scorer."""
        if role_ids is None:
            role_ids = torch.zeros(unit_feats.shape[:2], dtype=torch.long,
                                   device=unit_feats.device)
        emb = self.role_emb(role_ids.clamp(0, N_ROLES - 1))
        return torch.cat([unit_feats, emb], dim=-1)

    def _unit_ctx(self, batch):
        feats = batch["unit_feats"]
        valid = batch["unit_valid"]
        role_ids = batch.get("unit_role_ids")
        if role_ids is None:
            role_ids = torch.zeros(feats.shape[:2], dtype=torch.long,
                                   device=feats.device)
        own = batch.get("unit_own_mask")
        if own is None:
            own = valid
        return feats, valid, role_ids, own

    def _scatter_units(self, unit_feats, unit_valid, hw, role_ids=None):
        """Pinta cada slot (HP/idle/xy/team/rol) en el fmap. Pesos 0 al init."""
        B, U, _ = unit_feats.shape
        H, W = int(hw[0]), int(hw[1])
        u_in = self._unit_in(unit_feats, role_ids)
        paint = self.scatter_proj(u_in) * unit_valid.unsqueeze(-1).float()
        xs = (unit_feats[..., 7] * 128.0).round().long().clamp(0, W - 1)
        ys = (unit_feats[..., 8] * 128.0).round().long().clamp(0, H - 1)
        idx = (ys * W + xs).unsqueeze(1).expand(-1, SCATTER_CH, -1)
        out = paint.new_zeros(B, SCATTER_CH, H * W)
        out.scatter_add_(2, idx, paint.transpose(1, 2))
        return out.view(B, SCATTER_CH, H, W)

    def encode(self, spatial, scalars, unit_feats, unit_valid, hidden,
               unit_role_ids=None, unit_own_mask=None):
        """spatial [B,9,H,W], scalars [B,S], units [B,U,F], valid [B,U] bool.

        Devuelve (fmap, feat_map_flat, new_hidden, tokens). tokens [B,U,128]
        alimentan dist_cell (Capa 2). feat_map_flat se conserva por firma.
        GRU pool: solo propias (unit_own_mask). El xf ve own++ene.
        """
        fmap = self._enc_spatial(self._coord_conv(spatial))
        feat_map_flat = fmap.flatten(1)
        spatial_vec = F.adaptive_avg_pool2d(fmap, 1).flatten(1)

        u = self.unit_mlp(self._unit_in(unit_feats, unit_role_ids))
        tokens = self._entity_tokens(u, unit_valid)
        pool_m = unit_own_mask if unit_own_mask is not None else unit_valid
        own_f = pool_m.float().unsqueeze(-1)
        denom = own_f.sum(1).clamp(min=1.0)
        unit_vec = (tokens * own_f).sum(1) / denom

        s = self.scalar_mlp(scalars)
        fused = torch.cat([spatial_vec, s, unit_vec], dim=-1)
        new_hidden = self.core(fused, hidden)
        return fmap, feat_map_flat, new_hidden, tokens

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

    def _scores_unit(self, hidden, chosen_type, unit_feats, unit_legal,
                     role_ids=None):
        """Logits crudos de la cabeza de unidades (para temperatura real).

        unit_legal = propias (own_mask). Enemigos van −1e9: no se eligen.
        """
        B, U, _ = unit_feats.shape
        u_in = self._unit_in(unit_feats, role_ids)
        h = hidden.unsqueeze(1).expand(-1, U, -1)
        t = self.type_embedding(chosen_type).unsqueeze(1).expand(-1, U, -1)
        scores = self.unit_scorer(
            torch.cat([u_in, h, t], dim=-1)).squeeze(-1)
        return scores.masked_fill(~unit_legal, -1e9)

    def dist_unit(self, hidden, chosen_type, unit_feats, unit_legal,
                  role_ids=None):
        """Puntúa CADA slot con sus propias features (+ estado global + tipo).

        Unidades idénticas reciben puntajes casi idénticos (elección
        indistinta, correcto); unidades distintas (MCV vs rifleman herido)
        son distinguibles desde las features. Slots enemigos ilegales.
        """
        logits = self._scores_unit(
            hidden, chosen_type, unit_feats, unit_legal, role_ids)
        return self._categorical(logits)

    def _unit_cond_map(self, tokens, unit_valid, unit_slot, chosen_type, hw,
                       unit_own_mask=None):
        """Broadcast del slot elegido, o del pool propio si el tipo no usa unidad."""
        B, U, _ = tokens.shape
        H, W = int(hw[0]), int(hw[1])
        slot = unit_slot.clamp(0, max(U - 1, 0))
        chosen = tokens[torch.arange(B, device=tokens.device), slot]
        pool_m = unit_own_mask if unit_own_mask is not None else unit_valid
        valid_f = pool_m.float().unsqueeze(-1)
        pooled = (tokens * valid_f).sum(1) / valid_f.sum(1).clamp(min=1.0)
        use_u = _heads_used(chosen_type, tokens.device)[0]
        cond = torch.where(use_u.unsqueeze(-1), chosen, pooled)
        emb = F.relu(self.unit_cond_proj(cond))
        return emb[:, :, None, None].expand(-1, -1, H, W)

    def _logits_cell(self, fmap, chosen_type, cell_mask, hidden,
                     tokens, unit_feats, unit_valid, unit_slot,
                     role_ids=None, unit_own_mask=None):
        """Logits crudos [B, H*W] de la cabeza de celda.

        fmap U-Net + scatter de unidades + emb tipo + GRU + slot (Capa 2).
        Un rifle y un MCV ya no ven el mismo heatmap.
        """
        b, _, H, W = fmap.shape
        scatter = self._scatter_units(
            unit_feats, unit_valid, (H, W), role_ids=role_ids)
        emb = self.type_embedding(chosen_type)
        emb_map = emb[:, :, None, None].expand(-1, -1, H, W)
        hd = F.relu(self.hidden_proj(hidden))[:, :, None, None].expand(-1, -1, H, W)
        unit_map = self._unit_cond_map(
            tokens, unit_valid, unit_slot, chosen_type, (H, W),
            unit_own_mask=unit_own_mask)
        logits_map = self.cell_head(
            torch.cat([fmap, scatter, emb_map, hd, unit_map], dim=1)).squeeze(1)
        logits_map = logits_map.masked_fill(~cell_mask.view(b, H, W), -1e9)
        return logits_map.reshape(b, -1)

    def dist_cell(self, fmap, chosen_type, cell_mask, hidden,
                  tokens, unit_feats, unit_valid, unit_slot,
                  role_ids=None, unit_own_mask=None):
        logits = self._logits_cell(
            fmap, chosen_type, cell_mask, hidden,
            tokens, unit_feats, unit_valid, unit_slot,
            role_ids=role_ids, unit_own_mask=unit_own_mask)
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

    def _enemy_legal(self, unit_valid, unit_own_mask=None):
        """Máscara [B, MAX_ENEMIES] de slots enemigos visibles (no propias)."""
        B, U = unit_valid.shape
        legal = unit_valid.new_zeros(B, MAX_ENEMIES)
        if U > MAX_UNITS:
            n = min(MAX_ENEMIES, U - MAX_UNITS)
            sl = unit_valid[:, MAX_UNITS:MAX_UNITS + n]
            if unit_own_mask is not None:
                sl = sl & ~unit_own_mask[:, MAX_UNITS:MAX_UNITS + n]
            legal[:, :n] = sl
        return legal

    def _scores_enemy(self, hidden, chosen_type, tokens, unit_slot,
                      enemy_legal):
        """Logits [B, MAX_ENEMIES]: token ene + token propio + hidden + type."""
        B = tokens.size(0)
        d = tokens.size(-1)
        ene = tokens.new_zeros(B, MAX_ENEMIES, d)
        if tokens.size(1) > MAX_UNITS:
            n = min(MAX_ENEMIES, tokens.size(1) - MAX_UNITS)
            ene[:, :n] = tokens[:, MAX_UNITS:MAX_UNITS + n]
        slot = unit_slot.clamp(0, max(tokens.size(1) - 1, 0))
        own = tokens[torch.arange(B, device=tokens.device), slot]
        own = own.unsqueeze(1).expand(-1, MAX_ENEMIES, -1)
        h = hidden.unsqueeze(1).expand(-1, MAX_ENEMIES, -1)
        t = self.type_embedding(chosen_type).unsqueeze(1).expand(
            -1, MAX_ENEMIES, -1)
        scores = self.enemy_scorer(
            torch.cat([ene, own, h, t], dim=-1)).squeeze(-1)
        return scores.masked_fill(~enemy_legal, -1e9)

    def dist_enemy(self, hidden, chosen_type, tokens, unit_slot, enemy_legal):
        logits = self._scores_enemy(
            hidden, chosen_type, tokens, unit_slot, enemy_legal)
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
        feats, valid, role_ids, own = self._unit_ctx(batch)
        fmap, _, new_hidden, tokens = self.encode(
            batch["spatial"], batch["scalars"],
            feats, valid, hidden,
            unit_role_ids=role_ids, unit_own_mask=own,
        )
        greedy = temperature <= 0.0

        lt = self._logits_type(new_hidden, batch["type_mask"])
        dist_t = self._categorical(lt)
        t_idx = lt.argmax(dim=-1) if greedy else \
            self._categorical(lt / temperature).sample()

        ls_u = self._scores_unit(new_hidden, t_idx, feats, own, role_ids)
        dist_u = self._categorical(ls_u)
        u_idx = ls_u.argmax(dim=-1) if greedy else \
            self._categorical(ls_u / temperature).sample()

        lc = self._logits_cell(
            fmap, t_idx, batch["cell_mask"], new_hidden,
            tokens, feats, valid, u_idx,
            role_ids=role_ids, unit_own_mask=own)
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

        enemy_legal = self._enemy_legal(valid, own)
        ls_e = self._scores_enemy(new_hidden, t_idx, tokens, u_idx, enemy_legal)
        dist_e = self._categorical(ls_e)
        e_idx = ls_e.argmax(dim=-1) if greedy else \
            self._categorical(ls_e / temperature).sample()
        has_ene = enemy_legal.any(dim=-1)

        # F3 (auditoría 2026-08-24): log_prob SOLO de las cabezas que el
        # tipo usa. Antes unidad+celda sumaban siempre (~6000 logits de
        # celda dominando con ruido puro en no_op/train/build).
        use_u, use_c, use_i, use_e = _heads_used(t_idx, t_idx.device)
        zero = torch.zeros_like(dist_t.log_prob(t_idx))
        lp = (dist_t.log_prob(t_idx)
              + torch.where(use_u, dist_u.log_prob(u_idx), zero)
              + torch.where(use_c, dist_c.log_prob(c_idx), zero)
              + torch.where(use_i & has_items, dist_i.log_prob(i_idx.clamp(
                  min=0, max=safe_item_mask.size(1) - 1)), zero)
              + torch.where(use_e & has_ene, dist_e.log_prob(e_idx), zero))

        value = self.value_head(new_hidden).squeeze(-1)
        return {
            "hidden": new_hidden,
            "type": t_idx, "unit_slot": u_idx, "cell_flat": c_idx,
            "item_slot": i_idx, "enemy_slot": e_idx,
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
        feats, valid, role_ids, own = self._unit_ctx(batch)
        fmap, _, new_hidden, tokens = self.encode(
            batch["spatial"], batch["scalars"],
            feats, valid, hidden,
            unit_role_ids=role_ids, unit_own_mask=own,
        )
        t_idx = actions["type"]
        dist_t = self.dist_type(new_hidden, batch["type_mask"])
        dist_u = self.dist_unit(new_hidden, t_idx, feats, own, role_ids)
        dist_c = self.dist_cell(
            fmap, t_idx, batch["cell_mask"], new_hidden,
            tokens, feats, valid, actions["unit_slot"],
            role_ids=role_ids, unit_own_mask=own)

        has_items = batch["item_mask"].any(dim=-1)
        safe_item_mask = self._item_cat_mask(batch, t_idx).clone()
        safe_item_mask[~has_items] = True
        dist_i = self.dist_item(new_hidden, t_idx, batch["item_indices"],
                                safe_item_mask)

        enemy_legal = self._enemy_legal(valid, own)
        e_idx = actions.get("enemy_slot")
        if e_idx is None:
            e_idx = torch.zeros_like(t_idx)
        else:
            e_idx = e_idx.to(t_idx.device).long().clamp(0, MAX_ENEMIES - 1)
        dist_e = self.dist_enemy(
            new_hidden, t_idx, tokens, actions["unit_slot"], enemy_legal)
        has_ene = enemy_legal.any(dim=-1)

        use_u, use_c, use_i, use_e = _heads_used(t_idx, t_idx.device)
        zero = torch.zeros_like(dist_t.log_prob(t_idx))
        lp = (dist_t.log_prob(t_idx)
              + torch.where(use_u, dist_u.log_prob(actions["unit_slot"]), zero)
              + torch.where(use_c, dist_c.log_prob(actions["cell_flat"]), zero)
              + torch.where(use_i & has_items & actions["had_item"],
                            dist_i.log_prob(actions["item_slot"].clamp(
                                min=0, max=batch["item_mask"].size(1) - 1)),
                            zero)
              + torch.where(use_e & has_ene, dist_e.log_prob(e_idx), zero))

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
        h_e = torch.where(has_ene, dist_e.entropy(), torch.zeros_like(h_t))
        # Entropía ENMASCARADA por cabeza activa (mismo criterio que en el
        # entrenamiento por segmentos): no regularizar cabezas no actuantes.
        zero_h = torch.zeros_like(h_t)
        use_u_f = use_u.float()
        use_c_f = use_c.float() * 0.25
        use_i_f = (use_i & has_items).float() * 0.5
        use_e_f = (use_e & has_ene).float()
        entropy = ((h_t
                    + torch.where(use_u, h_u, zero_h)
                    + torch.where(use_c, h_c, zero_h)
                    + torch.where(use_i & has_items, h_i, zero_h)
                    + torch.where(use_e & has_ene, h_e, zero_h))
                   / (1.0 + use_u_f + use_c_f + use_i_f + use_e_f))
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
            if s["batch"].get("unit_role_ids") is not None:
                b["unit_role_ids"] = s["batch"]["unit_role_ids"].to(device)
            if s["batch"].get("unit_own_mask") is not None:
                b["unit_own_mask"] = s["batch"]["unit_own_mask"].to(device)
            feats, valid, role_ids, own = self._unit_ctx(b)
            fmap, _, h, tokens = self.encode(
                b["spatial"], b["scalars"], feats, valid, h,
                unit_role_ids=role_ids, unit_own_mask=own)
            t_idx = s["action"]["type"].to(device)
            dist_t = self.dist_type(h, b["type_mask"])
            dist_u = self.dist_unit(h, t_idx, feats, own, role_ids)
            u_idx = s["action"]["unit_slot"].to(device)
            dist_c = self.dist_cell(
                fmap, t_idx, b["cell_mask"], h,
                tokens, feats, valid, u_idx,
                role_ids=role_ids, unit_own_mask=own)
            has_items = b["item_mask"].any(dim=-1)
            safe_item = self._item_cat_mask(b, t_idx).clone()
            safe_item[~has_items] = True
            dist_i = self.dist_item(h, t_idx, b["item_indices"], safe_item)

            enemy_legal = self._enemy_legal(valid, own)
            e_raw = s["action"].get("enemy_slot")
            if e_raw is None:
                e_idx = torch.zeros_like(t_idx)
            else:
                e_idx = e_raw.to(device).long().clamp(0, MAX_ENEMIES - 1)
            dist_e = self.dist_enemy(h, t_idx, tokens, u_idx, enemy_legal)
            has_ene = enemy_legal.any(dim=-1)

            use_u, use_c, use_i, use_e = _heads_used(t_idx, t_idx.device)
            zero = torch.zeros_like(dist_t.log_prob(t_idx))
            c_idx = s["action"]["cell_flat"].to(device)
            i_idx = s["action"]["item_slot"].to(device)
            had_item = bool(s["action"]["had_item"])
            lp = (dist_t.log_prob(t_idx)
                  + torch.where(use_u, dist_u.log_prob(u_idx), zero)
                  + torch.where(use_c, dist_c.log_prob(c_idx), zero)
                  + torch.where(use_i & has_items & had_item,
                                dist_i.log_prob(i_idx.clamp(
                                    min=0, max=safe_item.size(1) - 1)), zero)
                  + torch.where(use_e & has_ene, dist_e.log_prob(e_idx), zero))

            ht = dist_t.entropy()
            hu = dist_u.entropy()
            hc = dist_c.entropy() * 0.25
            hi = torch.where(has_items, dist_i.entropy(), torch.zeros_like(ht)) * 0.5
            he = torch.where(has_ene, dist_e.entropy(), torch.zeros_like(ht))

            # Entropía ENMASCARADA por cabeza activa: no inyectar gradientes de
            # exploración en cabezas que NO participaron en la acción del paso
            # (p.ej. no_op/train no deben regularizar cell_head/unit_scorer).
            zero = torch.zeros_like(ht)
            use_u_f = use_u.float()
            use_c_f = use_c.float() * 0.25
            use_i_f = (use_i & has_items).float() * 0.5
            use_e_f = (use_e & has_ene).float()
            entropy = ((ht
                        + torch.where(use_u, hu, zero)
                        + torch.where(use_c, hc, zero)
                        + torch.where(use_i & has_items, hi, zero)
                        + torch.where(use_e & has_ene, he, zero))
                       / (1.0 + use_u_f + use_c_f + use_i_f + use_e_f))
            value = self.value_head(h).squeeze(-1)
            lp_list.append(lp)
            ent_list.append(entropy)
            val_list.append(value)
        return (torch.cat(lp_list), torch.cat(ent_list), torch.cat(val_list))


def adapt_capa2_state_dict(net: AlphaLiteNet, raw: dict) -> dict:
    """Net2Net: pesos 922 (cell_head 224→1) → Capa 2 (scatter+unidad extra).

    Copia fmap/tipo/GRU a las mismas rebanadas; scatter y unit_cond quedan 0
    (igual que scatter_proj / unit_cond_proj al init). GRU/U-Net/type-head
    cargan 1:1. Keys nuevas (transformer) las pone load_state_dict missing.
    """
    out = dict(raw)
    key_w = "cell_head.weight"
    if key_w not in out:
        return out
    old_w = out[key_w]
    new_w = net.cell_head.weight.detach().clone()
    if old_w.shape == new_w.shape:
        return out
    if old_w.dim() != 4 or old_w.shape[1] != CELL_HEAD_OLD_IN:
        return out
    new_w.zero_()
    # new cat: fmap(96) + scatter(8) + tipo(64) + hidden(64) + unit(64)
    new_w[:, 0:SPATIAL_CH] = old_w[:, 0:SPATIAL_CH]
    t0 = SPATIAL_CH + SCATTER_CH
    new_w[:, t0:t0 + 64] = old_w[:, SPATIAL_CH:SPATIAL_CH + 64]
    h0 = t0 + 64
    new_w[:, h0:h0 + 64] = old_w[:, SPATIAL_CH + 64:CELL_HEAD_OLD_IN]
    out[key_w] = new_w
    # bias [1] mismo shape
    return out


def _expand_feat_in(old_w: torch.Tensor, old_feat: int, extra: int) -> torch.Tensor:
    """Insert `extra` zero cols after the first `old_feat` (HP..facing)."""
    out_f, tot = old_w.shape
    rest = tot - old_feat
    new_w = old_w.new_zeros(out_f, old_feat + extra + rest)
    new_w[:, :old_feat] = old_w[:, :old_feat]
    if rest:
        new_w[:, old_feat + extra:] = old_w[:, old_feat:]
    return new_w


def adapt_capa2c_state_dict(net: AlphaLiteNet, raw: dict) -> dict:
    """Net2Net 2c-A (feats 10) → 2c-B (11 + role_emb 8 = 19).

    Copia cols 0:10; team+role nacen 0. `role_emb` no está en el ckpt A
    (missing keys). `unit_xf_scale` / GRU / U-Net 1:1 — no resetear.
    """
    out = dict(raw)
    extra = int(UNIT_MLP_IN - 10)
    if extra <= 0:
        return out
    target = net.state_dict()
    for key, old_feat in (
        ("unit_mlp.0.weight", 10),
        ("scatter_proj.weight", 10),
        ("unit_scorer.0.weight", 10),
    ):
        if key not in out or key not in target:
            continue
        old_w = out[key]
        want = target[key]
        if old_w.shape == want.shape:
            continue
        if old_w.dim() != 2 or old_w.shape[0] != want.shape[0]:
            continue
        if old_w.shape[1] < old_feat:
            continue
        padded = _expand_feat_in(old_w, old_feat, extra)
        if padded.shape == want.shape:
            out[key] = padded
    return out
