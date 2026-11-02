# Comparativa Arquitectónica y Metodológica: OpenRA-RL v4 vs v5

Documento de referencia técnica que sintetiza la evolución entre la versión **v4 (Commander + Self-Play Híbrido)** y la versión **v5 (Autonomous General & Creative Meta)**.

---

## 1. Matriz de Diferencias Principales

| Dimensión | OpenRA-RL v4 | OpenRA-RL v5 | Justificación Técnica del Cambio |
| :--- | :--- | :--- | :--- |
| **Arquitectura de Red** | MLP estándar con Linear Embeddings (`CommanderNet v4`). | **Transformer Spatial Cross-Attention** (17 tokens con `norm_first=True`). | Permite que los 16 candidatos espaciales se atiendan entre sí y al token global `[CLS]`, identificando sinergias y amenazas contiguas. |
| **Vector Global (`GLOBAL_DIM`)** | 304 dimensiones (`FEATURE_VERSION = 3`). | **305 dimensiones** (`FEATURE_VERSION = 4`). | Incorpora `ally_building` para soporte macro y métricas relativas de expansión territorial. |
| **Metadatos de Candidatos (`CAND_DIM`)** | 51 dimensiones. | **56 dimensiones**. | Añade `health_percent`, `is_capturable`, `is_infiltratable`, `is_ally`, `actor_id` y offsets euclidianos normalizados. |
| **Cabezas de Decisión** | 2 cabezas: `prod_head` (75) y `army_head` (34). | **4 cabezas**: `prod_head` (80), `army_head` (34), **`ability_head` (17)** y **`macro_ops_head` (3)**. | Permite controlar habilidades activas de unidades Tier 3 y reparaciones/ventas de edificios sin saturar las colas generales. |
| **Destacamentos Tácticos C#** | `MAIN`, `HARASS`, `AIR`. | `MAIN`, `HARASS`, `AIR` + **`SPECOPS`**. | Evita que ingenieros (`e6`) y espías (`spy`) mueran en el *blob* general de combate, manteniéndolos a salvo para misiones de captura. |
| **Selección de Unidades** | Lista negra estática `FORBIDDEN_TRAIN_ITEMS` (14 unidades bloqueadas). | **Guardias Dinámicas de Viabilidad Económica**. | Elimina el bloqueo estático: unidades Tier 3 (`ctnk`, `qtnk`, `mslo`) se habilitan cuando hay solvencia ($>\$1.200$, $\ge 2$ harvs). |
| **Entorno de Mapas** | Exclusivo `singles.oramap` (ratonera de combate rápido). | **Pool Dual 50/50: `singles.oramap` + `doubles.oramap`**. | Garantiza invarianza espacial y previene sobreajuste convolucional a distancias cortas, probando macro-expansión. |
| **Ancla de Regularización** | KL Inversa continua ($D_{KL}(\pi_\theta \parallel \pi_{\text{ref}})$) hasta el final. | **Clipped Reverse KL + Slack Temporal ($t > 2.500$)**. | Evita explosiones de gradiente cuando $\pi_{\text{ref}}(a) \approx 0$ y libera la creatividad del agente en el juego medio y tardío. |
| **Warm-Start Supervisado** | Entrenamiento Cross-Entropy clásico en `bc_v4`. | **Label Smoothing ($\epsilon = 0.04$) sobre acciones legales**. | Impide que unidades no vistas en el maestro supervisado caigan en pozos de logits de $-12.0$, permitiendo su exploración en PPO. |
| **Composición de Liga** | 30% Mirror, 20% Snapshots, 35% Bots, 15% Ancla BC. | 30% Mirror, 20% Snapshots, 25% Bots, 15% Ancla, **10% Discovery Exploiters ($\pi_{\text{disc}}$)**. | Los *Exploiters* descubren usos viables de nuevas tecnologías con bonus intrínseco; el agente principal aprende a utilizarlas por suma cero. |
| **Telemetría y Resumen** | Parcial (solo consola simple en fases intermedias). | **Monitor de Progreso en Vivo + TensorBoard Exhaustivo + Safe Stop & Resume**. | Barra interactiva con tiempo transcurrido, tiempo restante (ETA) y hora proyectada de finalización en todas las fases. |

---

## 2. Transferencia de Pesos: v4 a v5

Para aprovechar las ~4.5 horas de entrenamiento de la v4 sin reiniciar el aprendizaje desde cero, la v5 implementa `warmstart_v4_to_v5()`:
- **27 tensores de pesos coincidentes** (tronco principal, capas ocultas, cabezas de producción y ejército) se copian directamente sin alteración.
- La dimensión adicional de entrada en `trunk.0.weight` (`304 -> 305`) y en `critic.0.weight` (`325 -> 326`) se inicializa con **zero-padding**, garantizando matemáticamente que:
  $$\pi_{\theta_{v5}}(a \mid s) = \pi_{\theta_{v4}}(a \mid s) \quad \forall s \in \mathcal{S}_{v4}$$
- Esto preserva el conocimiento táctico de 1418.4 Elo al comenzar el entrenamiento de la v5.

---

## 3. Benchmarks Esperados

| Métrica | Resultado Oficial v4 | Objetivo de Desempeño v5 |
| :--- | :--- | :--- |
| **Elo en Liga de Self-Play** | 1418.4 Elo | **$\ge 1550.0$ Elo** |
| **Composite Score (Harmonic)** | 0.817 | **$\ge 0.880$** |
| **H2H Directo vs v4-champion** | N/A (modelo base) | **$\ge 75.0\%$ Win Rate** |
| **Win Rate vs Bot Hard** | 100.0% (`singles`) | **$\ge 98.0\%$ (pool dual: `singles` + `doubles`)** |
| **Win Rate vs Bot Brutal** | 88.3% (`singles`) | **$\ge 90.0\%$ (pool dual: `singles` + `doubles`)** |
| **Win Rate vs Bot Turtle** | 98.3% (`singles`) | **$\ge 98.0\%$ (pool dual: `singles` + `doubles`)** |
| **Despliegue Tecnológico Tier 3** | 0.0% (bloqueado por lista negra) | **$\ge 15.0\%$ de partidas con uso de `ctnk`, `qtnk` o `e6`** |
| **Captura y Sabotaje (`SPECOPS`)** | 0.0% | **$\ge 5.0\%$ de partidas con captura exitosa de estructuras** |
