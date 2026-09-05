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

El log_prob total es la suma SOLO de las cabezas que el tipo elegido usa
(cadena p(tipo)·[p(unidad|tipo)]·[p(celda|tipo)]·[p(ítem|tipo)]) — auditoría
2026-08-24 (F3): antes se sumaban unidad+celda SIEMPRE y ~6000 logits de
celda metían ruido en acciones que ni los miraban.
"""

import numpy as np
import math
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.obs_encoding import MAX_TOKENS, MAX_UNITS, SCALAR_DIM, UNIT_FEAT_DIM, ready_place_items
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
# Capa 2 (doc 12) + 2c-A set 96 + 2c-B role/team + 32 enemigos (tokens ≤128).
# Residual/zero-init para Net2Net desde 922 (GRU y U-Net se conservan).
# 2c-C (attack-actor) se revirtió: smoke wr20→0, pointer casi no se usó.
ROLE_EMB_DIM = 8
UNIT_MLP_IN = UNIT_FEAT_DIM + ROLE_EMB_DIM  # 11 + 8 = 19
XF_DIM = 64
XF_HEADS = 4
XF_LAYERS = 2
XF_FF = 128
SCATTER_CH = 8
UNIT_COND_DIM = 64
QSA_DIM = 32
QSA_BLOCK = 8
QSA_TOPK = 8
SPATIAL_CH = 96
CELL_HEAD_OLD_IN = SPATIAL_CH + 64 + 64  # fmap + tipo + hidden (pre Capa 2)

# Qué cabeza usa cada tipo de acción (FUENTE ÚNICA para log_prob condicional;
# action_adapter debe ser coherente con estos conjuntos). Auditoría 2026-08-24.
TYPES_USE_UNIT = {"move", "attack_move", "attack", "stop", "set_stance",
                  "harvest", "deploy"}
TYPES_USE_CELL = {"move", "attack_move", "attack", "place_building",
                  "army_attack_move"}
TYPES_USE_ITEM = {"train", "build", "place_building", "cancel_production"}

# Tablas [N_ACTION_TYPES] — indexar t_idx en GPU, sin .item() por step
# (el loop Python + sync CUDA era parte de los ~210s de update).
_USE_U = torch.tensor([n in TYPES_USE_UNIT for n in ACTION_TYPES])
_USE_C = torch.tensor([n in TYPES_USE_CELL for n in ACTION_TYPES])
_USE_I = torch.tensor([n in TYPES_USE_ITEM for n in ACTION_TYPES])

# fp16 max ≈ 65504. masked_fill(-1e9) bajo AMP crashea (Half overflow).
_ILLEGAL_FP16 = -1.0e4
_ILLEGAL_FP32 = -1.0e9


def _mask_illegal(logits: torch.Tensor, illegal) -> torch.Tensor:
    """Tapa logits ilegales sin overflow en Half."""
    if illegal.dtype != torch.bool:
        illegal = illegal.bool()
    fill = (_ILLEGAL_FP16
            if logits.dtype in (torch.float16, torch.bfloat16)
            else _ILLEGAL_FP32)
    return logits.masked_fill(illegal, fill)


def _heads_used(t_idx: torch.Tensor, device=None) -> tuple:
    """Máscaras [B] bool: usa_unidad, usa_celda, usa_item para cada tipo."""
    t = t_idx.long().reshape(-1)
    dev = t.device if device is None else device
    if t.device != dev:
        t = t.to(dev, non_blocking=True)
    return (_USE_U.to(dev, non_blocking=True)[t],
            _USE_C.to(dev, non_blocking=True)[t],
            _USE_I.to(dev, non_blocking=True)[t])


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
        # Arch v1.1: GRU unit_vec = proj(own_mean || own_max || ene_mean).
        # Mean solo aplastaba emergencias; max + enemy pool las hacen visibles.
        self.unit_pool_proj = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        # 0 = dense softmax (legacy). >0 = top-k sparse attn on entity XF
        # (Run46 inductive bias; state_dict unchanged — same MHA weights).
        self.xf_topk = 0
        # Map QSA (Run47): block indexer → top-k cells only. 0 = dense cell_head.
        self.qsa_topk = 0
        self.qsa_block = QSA_BLOCK
        # Learned block indexer (fmap keys + query from type/unit/hidden).
        # Small init; block ranking also uses dense logits (distilled prior).
        self.qsa_key = nn.Conv2d(ch, QSA_DIM, 1)
        nn.init.normal_(self.qsa_key.weight, std=0.01)
        nn.init.zeros_(self.qsa_key.bias)
        self.qsa_query = nn.Linear(HIDDEN_DIM + 64 + UNIT_COND_DIM, QSA_DIM)
        nn.init.normal_(self.qsa_query.weight, std=0.01)
        nn.init.zeros_(self.qsa_query.bias)
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
        # Capa 2: fmap + scatter + tipo + GRU + unidad elegida (o pool).
        self.scatter_proj = nn.Linear(UNIT_MLP_IN, SCATTER_CH)
        nn.init.zeros_(self.scatter_proj.weight)
        nn.init.zeros_(self.scatter_proj.bias)
        self.unit_cond_proj = nn.Linear(128, UNIT_COND_DIM)
        nn.init.zeros_(self.unit_cond_proj.weight)
        nn.init.zeros_(self.unit_cond_proj.bias)
        # Arch v1.1: cell head no lineal (AND accion x unidad x geometria local).
        # 1x1 -> SiLU -> 3x3; ~19k params extra vs Conv(296,1,1).
        _cell_in = ch + SCATTER_CH + 64 + 64 + UNIT_COND_DIM
        self.cell_head = nn.Sequential(
            nn.Conv2d(_cell_in, 64, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )
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
        """MLP tokens + transformer residual (scale=0 → identidad 922).

        Si xf_topk>0, cada capa del unit_xf usa atención top-k (filtro de
        ruido entre ~128 tokens) reutilizando los pesos MHA existentes.
        """
        pad = ~unit_valid
        all_pad = pad.all(dim=-1)
        pad = pad.clone()
        pad[all_pad, 0] = False
        x = self.unit_xf_in(u)
        topk = int(getattr(self, "xf_topk", 0) or 0)
        if topk > 0:
            for layer in self.unit_xf.layers:
                x = self._xf_layer_topk(layer, x, pad, topk)
        else:
            x = self.unit_xf(x, src_key_padding_mask=pad)
        x = x.masked_fill(all_pad[:, None, None], 0.0)
        return u + self.unit_xf_scale * self.unit_xf_out(x)

    def _xf_layer_topk(self, layer, x, pad, topk: int):
        """Una TransformerEncoderLayer (norm_first) con self-attn top-k."""
        y = layer.norm1(x)
        y = self._topk_mha(layer.self_attn, y, pad, topk)
        x = x + y
        y = layer.norm2(x)
        y = layer.linear2(layer.activation(layer.linear1(y)))
        x = x + y
        return x

    def _topk_mha(self, mha: nn.MultiheadAttention, x, pad, topk: int):
        """Self-attn batch_first con máscara top-k (resto → -inf).

        Gradiente solo fluye por las k claves elegidas (sesgo inductivo).
        Compatible con AMP: fill -1e4.
        """
        B, N, E = x.shape
        H = mha.num_heads
        Dh = E // H
        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        def _split(t):
            return t.view(B, N, H, Dh).transpose(1, 2)  # [B,H,N,Dh]
        q, k, v = _split(q), _split(k), _split(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        # pad True = ignore key
        if pad is not None:
            scores = scores.masked_fill(pad[:, None, None, :], -1e4)
        k_use = min(int(topk), N)
        if k_use < N:
            vals, idx = torch.topk(scores, k=k_use, dim=-1)
            sparse = scores.new_full(scores.shape, -1e4)
            sparse.scatter_(-1, idx, vals)
            scores = sparse
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B,H,N,Dh]
        out = out.transpose(1, 2).contiguous().view(B, N, E)
        return F.linear(out, mha.out_proj.weight, mha.out_proj.bias)

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


    def _unit_vec_for_gru(self, tokens, unit_valid, unit_own_mask=None):
        """Resumen estrategico para la GRU: mean+max propias + mean enemigas.

        tokens [B,U,128] post-XF. Enemigos = valid & ~own. Sin propias/enemigas
        el max/mean correspondiente es 0 (no -1e9).
        """
        own = unit_own_mask if unit_own_mask is not None else unit_valid
        own_b = own.bool()
        valid_b = unit_valid.bool()
        ene_b = valid_b & ~own_b
        own_f = own_b.float().unsqueeze(-1)
        ene_f = ene_b.float().unsqueeze(-1)

        mean_own = (tokens * own_f).sum(1) / own_f.sum(1).clamp(min=1.0)

        neg = tokens.new_full(tokens.shape, -1e9)
        tok_own = torch.where(own_b.unsqueeze(-1), tokens, neg)
        max_own = tok_own.max(dim=1).values
        has_own = own_b.any(dim=-1)
        max_own = torch.where(has_own.unsqueeze(-1), max_own, torch.zeros_like(max_own))

        mean_ene = (tokens * ene_f).sum(1) / ene_f.sum(1).clamp(min=1.0)
        has_ene = ene_b.any(dim=-1)
        mean_ene = torch.where(has_ene.unsqueeze(-1), mean_ene, torch.zeros_like(mean_ene))

        return self.unit_pool_proj(torch.cat([mean_own, max_own, mean_ene], dim=-1))

    def encode(self, spatial, scalars, unit_feats, unit_valid, hidden,
               unit_role_ids=None, unit_own_mask=None):
        """spatial [B,9,H,W], scalars [B,S], units [B,U,F], valid [B,U] bool.

        Devuelve (fmap, feat_map_flat, new_hidden, tokens). tokens [B,U,128]
        alimentan dist_cell (Capa 2). feat_map_flat se conserva por firma.
        GRU unit_vec: own_mean||own_max||ene_mean -> proj 128. El xf ve own++ene.
        """
        fmap = self._enc_spatial(self._coord_conv(spatial))
        feat_map_flat = fmap.flatten(1)
        spatial_vec = F.adaptive_avg_pool2d(fmap, 1).flatten(1)

        u = self.unit_mlp(self._unit_in(unit_feats, unit_role_ids))
        tokens = self._entity_tokens(u, unit_valid)
        unit_vec = self._unit_vec_for_gru(tokens, unit_valid, unit_own_mask)

        s = self.scalar_mlp(scalars)
        fused = torch.cat([spatial_vec, s, unit_vec], dim=-1)
        new_hidden = self.core(fused, hidden)
        return fmap, feat_map_flat, new_hidden, tokens

    @staticmethod
    def _categorical(logits):
        """Categorical that does not crash on NaN/all-masked rows (PPO 947).

        fp32 siempre: AMP deja -1e9 → -inf en fp16 y softmax se va a NaN.
        """
        logits = logits.float()
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        dead = (logits <= -1e4).all(dim=-1)
        if dead.any():
            logits = logits.clone()
            logits[dead, 0] = 0.0
        return torch.distributions.Categorical(logits=logits)

    def dist_type(self, hidden, type_mask):
        logits = self._logits_type(hidden, type_mask)
        return self._categorical(logits)

    def _logits_type(self, hidden, type_mask):
        return _mask_illegal(self.head_type(hidden), ~type_mask)

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
        return _mask_illegal(scores, ~unit_legal)

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
        Si qsa_topk>0: QSA de mapa — ranking de bloques BxB, top-k, el resto
        a -inf (softmax solo sobre regiones relevantes; doc 13 / Run47).
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
        # unit_cond vector (pre-broadcast) for QSA query
        slot = unit_slot.clamp(0, max(tokens.size(1) - 1, 0))
        chosen = tokens[torch.arange(b, device=tokens.device), slot]
        pool_m = unit_own_mask if unit_own_mask is not None else unit_valid
        valid_f = pool_m.float().unsqueeze(-1)
        pooled = (tokens * valid_f).sum(1) / valid_f.sum(1).clamp(min=1.0)
        use_u = _heads_used(chosen_type, tokens.device)[0]
        unit_cond = torch.where(use_u.unsqueeze(-1), chosen, pooled)
        unit_cond = F.relu(self.unit_cond_proj(unit_cond))

        logits_map = self.cell_head(
            torch.cat([fmap, scatter, emb_map, hd, unit_map], dim=1)).squeeze(1)
        topk = int(getattr(self, "qsa_topk", 0) or 0)
        if topk > 0:
            logits_map = self._apply_map_qsa(
                logits_map, fmap, hidden, emb, unit_cond, topk)
        logits_map = _mask_illegal(logits_map, ~cell_mask.view(b, H, W))
        return logits_map.reshape(b, -1)

    def _apply_map_qsa(self, logits_map, fmap, hidden, type_emb, unit_cond,
                       topk: int):
        """Block top-k mask over cell logits (Query Sparse Attention lite).

        Block score = mean dense logit in block + learned (query·key).
        Cells outside the top-k blocks get -1e4 (AMP-safe).
        """
        b, H, W = logits_map.shape
        block = max(1, int(getattr(self, "qsa_block", QSA_BLOCK) or QSA_BLOCK))
        # Pad to multiple of block
        pad_h = (block - H % block) % block
        pad_w = (block - W % block) % block
        if pad_h or pad_w:
            logits_pad = F.pad(logits_map, (0, pad_w, 0, pad_h), value=-1e4)
            fmap_pad = F.pad(fmap, (0, pad_w, 0, pad_h))
        else:
            logits_pad = logits_map
            fmap_pad = fmap
        Hp, Wp = logits_pad.shape[-2:]
        gh, gw = Hp // block, Wp // block
        n_blocks = gh * gw
        # Dense prior: mean logit per block
        prior = F.avg_pool2d(
            logits_pad.unsqueeze(1), block, block).view(b, n_blocks)
        # Learned indexer (independent of cell_head — no IndexShare)
        keys = self.qsa_key(fmap_pad)
        keys = F.avg_pool2d(keys, block, block)  # [B,D,gh,gw]
        keys = keys.view(b, QSA_DIM, n_blocks).transpose(1, 2)  # [B,N,D]
        q = self.qsa_query(
            torch.cat([hidden, type_emb, unit_cond], dim=-1))  # [B,D]
        learned = torch.einsum("bd,bnd->bn", q, keys) / math.sqrt(QSA_DIM)
        scores = prior + learned
        k_use = min(int(topk), n_blocks)
        _, idx = torch.topk(scores, k=k_use, dim=-1)  # [B,k]
        # Build boolean mask of selected cells
        flat = torch.zeros(b, n_blocks, dtype=torch.bool, device=logits_map.device)
        flat.scatter_(1, idx, True)
        block_mask = flat.view(b, gh, gw)
        cell_keep = block_mask.repeat_interleave(block, dim=1).repeat_interleave(
            block, dim=2)[:, :H, :W]
        out = logits_map.masked_fill(~cell_keep, -1e4)
        # STE: hard mask in forward; let qsa_query/key get grad on kept cells
        learned_map = learned.view(b, gh, gw)
        learned_map = learned_map.repeat_interleave(block, dim=1).repeat_interleave(
            block, dim=2)[:, :H, :W]
        keep_f = cell_keep.float()
        out = out + (learned_map - learned_map.detach()) * keep_f
        return out

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
        return _mask_illegal(scores, ~item_mask)

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

    def _eval_step(self, batch, h, actions):
        """Un paso BPTT, B>=1. actions: type/unit/cell/item/had_item [B].

        Encode puede ir en AMP (fp16). Cabezas siempre fp32: masked_fill(-1e9)
        no entra a Half (crash 1158).
        Devuelve (lp, entropy, value, h_new) todos [B].
        """
        feats, valid, role_ids, own = self._unit_ctx(batch)
        fmap, _, h_new, tokens = self.encode(
            batch["spatial"], batch["scalars"], feats, valid, h,
            unit_role_ids=role_ids, unit_own_mask=own)
        t_idx = actions["type"].reshape(-1)
        try:
            heads_ctx = torch.amp.autocast("cuda", enabled=False)
        except (TypeError, AttributeError):
            heads_ctx = torch.cuda.amp.autocast(enabled=False)
        with heads_ctx:
            fmap_f = fmap.float()
            h_f = h_new.float()
            tok_f = tokens.float()
            feats_f = feats.float()
            dist_t = self.dist_type(h_f, batch["type_mask"])
            dist_u = self.dist_unit(h_f, t_idx, feats_f, own, role_ids)
            u_idx = actions["unit_slot"].reshape(-1)
            dist_c = self.dist_cell(
                fmap_f, t_idx, batch["cell_mask"], h_f,
                tok_f, feats_f, valid, u_idx,
                role_ids=role_ids, unit_own_mask=own)
            has_items = batch["item_mask"].any(dim=-1)
            safe_item = self._item_cat_mask(batch, t_idx).clone()
            safe_item[~has_items] = True
            dist_i = self.dist_item(h_f, t_idx, batch["item_indices"], safe_item)

            use_u, use_c, use_i = _heads_used(t_idx)
            zero = torch.zeros_like(dist_t.log_prob(t_idx))
            c_idx = actions["cell_flat"].reshape(-1)
            i_idx = actions["item_slot"].reshape(-1)
            had_item = actions["had_item"]
            if not torch.is_tensor(had_item):
                had_item = torch.as_tensor(had_item, device=t_idx.device)
            had_item = had_item.reshape(-1).bool()
            lp = (dist_t.log_prob(t_idx)
                  + torch.where(use_u, dist_u.log_prob(u_idx), zero)
                  + torch.where(use_c, dist_c.log_prob(c_idx), zero)
                  + torch.where(use_i & has_items & had_item,
                                dist_i.log_prob(i_idx.clamp(
                                    min=0, max=safe_item.size(1) - 1)), zero))

            ht = dist_t.entropy()
            hu = dist_u.entropy()
            hc = dist_c.entropy() * 0.25
            hi = torch.where(has_items, dist_i.entropy(),
                             torch.zeros_like(ht)) * 0.5
            zero_h = torch.zeros_like(ht)
            use_u_f = use_u.float()
            use_c_f = use_c.float() * 0.25
            use_i_f = (use_i & has_items).float() * 0.5
            entropy = ((ht
                        + torch.where(use_u, hu, zero_h)
                        + torch.where(use_c, hc, zero_h)
                        + torch.where(use_i & has_items, hi, zero_h))
                       / (1.0 + use_u_f + use_c_f + use_i_f))
            value = self.value_head(h_f).squeeze(-1)
        return lp, entropy, value, h_new

    def evaluate_actions_seq(self, seg, device):
        """BPTT truncado sobre UN segmento. Wrapper de evaluate_actions_seq_batch."""
        lp, ent, val, valid = self.evaluate_actions_seq_batch([seg], device)
        m = valid[0]
        return lp[0][m], ent[0][m], val[0][m]

    def evaluate_actions_seq_batch(self, segs, device):
        """BPTT sobre B segmentos en paralelo (pad al final).

        El h_in del primer step de cada seg entra como constante; el GRU
        propaga sin detach. Pasos `_burn` (burn-in) avanzan h pero no
        entran al loss. Padding no entra al loss ni al hidden siguiente.

        Devuelve lp, entropy, value, valid — todos [B, T].
        """
        if not segs:
            z = torch.zeros(0, 0, device=device)
            return z, z, z, z.bool()
        B = len(segs)
        T = max(len(s) for s in segs)
        h0 = []
        for seg in segs:
            h = seg[0]["h_in"]
            if not torch.is_tensor(h):
                raise TypeError("h_in debe ser tensor")
            h = h.to(device, non_blocking=True)
            if h.dim() == 1:
                h = h.unsqueeze(0)
            elif h.dim() > 2:
                h = h.reshape(1, -1)
            h0.append(h)
        h = torch.cat(h0, dim=0)
        if h.dim() > 2:
            h = h.reshape(B, -1)

        # stack, no inplace: un buffer new_zeros no requiere grad y
        # lp_out[:,t]=lp cortaría el grafo de BPTT.
        lp_t, ent_t, val_t, valid_t = [], [], [], []
        for t in range(T):
            steps = []
            active = []
            burn = []
            for seg in segs:
                if t < len(seg):
                    steps.append(seg[t])
                    active.append(True)
                    burn.append(bool(seg[t].get("_burn")))
                else:
                    steps.append(seg[-1])
                    active.append(False)
                    burn.append(False)
            # Pad steps stay inactive. Burn-in steps advance h but skip loss.
            row_active = torch.tensor(active, dtype=torch.bool, device=device)
            row_burn = torch.tensor(burn, dtype=torch.bool, device=device)
            row_valid = row_active & ~row_burn
            batch, actions = _stack_steps(steps, device)
            lp, ent, val, h_new = self._eval_step(batch, h, actions)
            h = torch.where(row_active.unsqueeze(-1), h_new, h)
            lp_t.append(lp)
            ent_t.append(ent)
            val_t.append(val)
            valid_t.append(row_valid)
        return (torch.stack(lp_t, dim=1), torch.stack(ent_t, dim=1),
                torch.stack(val_t, dim=1), torch.stack(valid_t, dim=1))


def _cat_field(steps, getter, device):
    xs = []
    for s in steps:
        v = getter(s)
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        if v.dim() == 0:
            v = v.unsqueeze(0)
        xs.append(v.to(device, non_blocking=True))
    return torch.cat(xs, dim=0)


def _stack_steps(steps, device):
    """Lista de samples (cada uno B=1) → batch [B] + actions [B]."""
    def bget(k):
        return lambda s, kk=k: s["batch"][kk]

    keys = ("spatial", "scalars", "unit_feats", "unit_valid",
            "type_mask", "cell_mask", "item_indices", "item_mask")
    batch = {k: _cat_field(steps, bget(k), device) for k in keys}
    for k in ("train_slot_mask", "build_slot_mask",
              "unit_role_ids", "unit_own_mask"):
        if all(s["batch"].get(k) is not None for s in steps):
            batch[k] = _cat_field(steps, bget(k), device)
        elif k in ("train_slot_mask", "build_slot_mask"):
            batch[k] = batch["item_mask"]
    actions = {}
    for k in ("type", "unit_slot", "cell_flat", "item_slot"):
        actions[k] = _cat_field(
            steps, lambda s, kk=k: s["action"][kk], device)

    had = []
    for s in steps:
        v = s["action"]["had_item"]
        if torch.is_tensor(v):
            had.append(v.reshape(-1).bool().to(device, non_blocking=True))
        else:
            had.append(torch.tensor([bool(v)], dtype=torch.bool, device=device))
    actions["had_item"] = torch.cat(had, dim=0)
    return batch, actions



def cell_head_weight_shape(cell_head) -> tuple:
    """Shape del weight de salida del cell_head (Conv2d o Sequential)."""
    if isinstance(cell_head, nn.Conv2d):
        return tuple(cell_head.weight.shape)
    if isinstance(cell_head, nn.Sequential):
        last = None
        for m in cell_head.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is not None:
            return tuple(last.weight.shape)
    return ()


def adapt_capa2_state_dict(net: AlphaLiteNet, raw: dict) -> dict:
    """Net2Net: pesos 922 (cell_head 224->1) -> Capa 2 (scatter+unidad extra).

    Copia fmap/tipo/GRU a las mismas rebanadas; scatter y unit_cond quedan 0
    (igual que scatter_proj / unit_cond_proj al init). GRU/U-Net/type-head
    cargan 1:1. Keys nuevas (transformer) las pone load_state_dict missing.

    Arch v1.1: cell_head es Sequential — no hay Net2Net 1:1 desde Conv unico;
    se dropean cell_head.weight/bias legacy (las capas nuevas nacen random).
    """
    out = dict(raw)
    key_w = "cell_head.weight"
    if key_w not in out:
        return out
    if isinstance(net.cell_head, nn.Sequential):
        out.pop("cell_head.weight", None)
        out.pop("cell_head.bias", None)
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

def adapt_scalar_state_dict(net: AlphaLiteNet, raw: dict) -> dict:
    """Net2Net: pad scalar_mlp.0.weight when SCALAR_DIM grows (zero new cols).

    Ckpts viejos (in=21) cargan en redes nuevas (in=25); las features AOA
    nacen en 0 y el tronco economico/militar se conserva 1:1.
    """
    out = dict(raw)
    key = "scalar_mlp.0.weight"
    if key not in out:
        return out
    old_w = out[key]
    want = net.state_dict()[key]
    if old_w.shape == want.shape:
        return out
    if old_w.dim() != 2 or old_w.shape[0] != want.shape[0]:
        return out
    if old_w.shape[1] >= want.shape[1]:
        # shrink: truncate (should not happen often)
        out[key] = old_w[:, : want.shape[1]].contiguous()
        return out
    padded = old_w.new_zeros(want.shape)
    padded[:, : old_w.shape[1]] = old_w
    out[key] = padded
    return out

