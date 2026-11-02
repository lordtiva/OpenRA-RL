# Commander v5 — Especificación Técnica del Agente y Contrato gRPC

En `v5`, el agente de aprendizaje por refuerzo evoluciona de un controlador reactivo de frentes de batalla a un **General Autónomo** capaz de coordinar microtácticas especializadas, ejecutar habilidades de unidades de Tier 3, gestionar el mantenimiento de su base y adaptarse dinámicamente a mapas abiertos y cerrados.

---

## 1. Contrato gRPC y Motor C# (`rl_bridge.proto`)

### 1.1 Nuevos Tipos de Decisión (`DecisionKind`)

| `kind` | Cuándo se emite | Opciones / Parámetros | Cooldown C# |
|---|---|---|---|
| `DECIDE_BUILD` | Cola de construcción libre (`Building`, `Defense`) | `choice` en opciones legales | N/A |
| `DECIDE_TRAIN` | Cola de reclutamiento libre (`Infantry`, `Vehicle`, `Aircraft`, `Ship`) | `choice` en opciones legales | N/A |
| `DECIDE_ARMY` | Cada 250 ticks para `MAIN`, `HARASS`, `AIR` | `choice="attack"`, `choice="all_in"` o `target_index=-1` | N/A |
| `DECIDE_SUPPORT` | Cuando un superpoder está listo (`sp.Ready`) | `target_index=k` (`attack`) o `-1` (`hold`) | 150 ticks |
| `DECIDE_ABILITY` | Cuando unidades con habilidades activas están listas (`ctnk`, `e6`, `spy`, `dtrk`, etc.) | `target_index=0..15` (candidato objetivo) o `target_index=16` (`hold`) | 150 ticks por actor |
| `DECIDE_MACRO_OPS` | Cada 250 ticks o cuando un edificio sufre daños graves ($<50\%$ salud) | `choice="repair"`, `choice="sell"` o `choice="hold"` | 100 ticks |

### 1.2 Destacamento Táctico `SPECOPS` en C#
En `SquadManagerBotModule.cs`, las unidades de infiltración y sabotaje ya no se asignan al escuadrón `MAIN`:
- **Unidades asignadas:** Ingenieros (`e6`), Espías (`spy`) y Ladrones (`thf`).
- **Comportamiento:** Permanecen agrupadas cerca de la base aliada en postura evasiva hasta que el agente neuronal emite una orden de `DECIDE_ABILITY` con objetivo específico (edificio enemigo o tecnológico capturable/infiltrable).
- **Protección de transporte:** Si un APC (`apc`) o vehículo de transporte está disponible, el destacamento `SPECOPS` prioriza el embarque para aproximaciones seguras.

### 1.3 Candidatos Tácticos Enriquecidos (`TargetCandidate`)
Cada candidato en `req.targets` (hasta 16) incluye metadatos enriquecidos en `CAND_DIM = 56`:
- `health_percent`: Porcentaje de salud actual ($0..100$).
- `is_capturable`: `true` si la estructura enemiga o neutral puede ser capturada por ingenieros.
- `is_infiltratable`: `true` si la estructura enemiga puede ser saboteada por espías/ladrones (radar, refinería, tech center).
- `is_ally`: `true` si el actor es aliado (usado para reparaciones macro y saltos defensivos).
- `actor_id`: Identificador único de actor para seguimiento temporal y anti-flood.
- `dx, dy, dist`: Desplazamientos y distancias euclidianas normalizadas respecto a la base y al centro de masa del ejército.

---

## 2. Espacio de Estados y Observación (`ra_rl/features.py`)

### 2.1 Vector Global (`GLOBAL_DIM = 305`, `FEATURE_VERSION = 4`)
- **Economía y Poder (0..35):** Créditos, ingresos por tick, capacidad de almacenamiento, energía producida, energía consumida, estado de *brownout*.
- **Unidades y Estructuras Propias (36..150):** Conteo y porcentaje de salud de edificios, defensas, vehículos terrestres, cosechadoras y aeronaves.
- **Memoria de Amenazas y Estructuras Enemigas (151..275):** Ubicaciones observadas (mediante niebla de guerra legítima) de bases enemigas, concentraciones de tropas, defensas antiaéreas y cosechadoras.
- **Relaciones de Alianzas y Macro (276..304):** Añade la nueva característica `ally_building` (presencia y conteo de estructuras aliadas en peligro), balances relativos de fuerza y métricas de expansión territorial.

### 2.2 Vector de Privilegio para el Crítico Asimétrico (`PRIV_DIM = 21`, Total Crítico = 326)
Durante el entrenamiento (CTDE - *Centralized Training with Decentralized Execution*), el crítico asimétrico $V(g, priv)$ accede a:
- Recursos y reservas reales del enemigo.
- Unidades totales y valor militar enemigo sin niebla de guerra.
- Identificador del oponente (`bot_idx` / oponente de liga).

---

## 3. Arquitectura Neuronal `CommanderNet v5` (`ra_rl/model.py`)

```
                          [ Vector Global g (305) ]
                                     │
                             Linear Trunk (256)
                                     │
    [ 16 Candidatos Espaciales (56) ] │
                   │                 │
           Linear Proj (128)         │
                   │                 │
                   └────────┬────────┘
                            ▼
              Concat: Token [CLS] (global) + 16 Tokens Candidatos (17, 256)
                            │
               TransformerEncoderLayer (d_model=256, nhead=4, norm_first=True)
                            │
              Contextualized Representation (17, 256)
             ┌──────────────┴──────────────┬─────────────────────────┐
             ▼                             ▼                         ▼
     [CLS] Global Context            Candidate Tokens          Actor [CLS] + Cand
             │                             │                         │
     prod_head (80 logits)        army_head (34 logits)    ability_head (17 logits)
    (Building/Train queues)       (Micro/All-in/Hold/Keep)  (Hold + Target 0..15)
                                           │
                                  macro_ops_head (3 logits)
                                     (Repair / Sell / Hold)
```

### 3.1 Bloque Espacial Cross-Attention / Transformer
- Combina el vector global como token `[CLS]` en la posición 0 junto con los 16 candidatos espaciales.
- El módulo `nn.TransformerEncoderLayer(d_model=256, nhead=4, norm_first=True)` permite que los candidatos se atiendan entre sí (ej. detectar si una refinería enemiga está protegida por una Tesla Coil) y atiendan a la situación global de la base.

### 3.2 Cabezas de Acción Especializadas
1. **`prod_head` (80 clases):** Puntuación de ítems de construcción y entrenamiento con máscara de legalidad económica dinámica.
2. **`army_head` (34 clases):** 16 objetivos para ataque normal + 16 objetivos para asalto `all_in` + 1 `HOLD` + 1 `KEEP`.
3. **`ability_head` (17 clases):** Puntero directo sobre el candidato $0..15$ para ejecutar la habilidad especial, o clase 16 para posponer (`HOLD`).
4. **`macro_ops_head` (3 clases):** `repair` (0), `sell` (1), o `hold` (2) sobre el edificio objetivo reportado en la solicitud macro.

---

## 4. Guardias Dinámicas de Viabilidad Económica

En lugar de la lista negra estática `FORBIDDEN_TRAIN_ITEMS` utilizada en v4, la v5 implementa en `ra_rl/features.py` y `ra_rl/vocab.py` **guardias dinámicas condicionales**:

$$\text{is\_viable}(u) = \begin{cases}
\text{True} & \text{si } u \in \text{Tier 1-2} \\
\text{True} & \text{si } u \in \text{Tier 3} \land \text{Créditos} > \$1.200 \land \text{Cosechadoras} \ge 2 \land \text{Energía} \ge 0 \\
\text{False} & \text{en caso contrario}
\end{cases}$$

Esto permite que PPO y los *Discovery Exploiters* exploren libremente unidades como `ctnk`, `qtnk`, `mslo` o `stek` cuando la base cuenta con excedente económico, evitando suicidios macroeconómicos tempranos sin restringir la creatividad tardía del agente.
