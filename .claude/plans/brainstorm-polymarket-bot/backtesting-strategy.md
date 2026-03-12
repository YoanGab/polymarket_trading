# Polymarket Bot — Backtesting Strategy

> Synthèse de 5 agents de recherche — 12 mars 2026
> Objectif : prouver (ou invalider) l'edge avant d'investir un centime

---

## Table des matières

1. [Pourquoi le backtest est la priorité #1](#1-pourquoi-le-backtest-est-la-priorité-1)
2. [Architecture de replay sans fuite](#2-architecture-de-replay-sans-fuite)
3. [Sources de données historiques](#3-sources-de-données-historiques)
4. [Setup Grok en mode replay](#4-setup-grok-en-mode-replay)
5. [Stratégies à tester](#5-stratégies-à-tester)
6. [Métriques et seuils de validation](#6-métriques-et-seuils-de-validation)
7. [Validation gratuite sur Metaculus](#7-validation-gratuite-sur-metaculus)
8. [Coûts du backtest](#8-coûts-du-backtest)
9. [Plan d'exécution](#9-plan-dexécution)
10. [Frameworks et outils existants](#10-frameworks-et-outils-existants)

---

## 1. Pourquoi le backtest est la priorité #1

### Le problème
- 70% des traders Polymarket perdent de l'argent
- L'edge "Grok + X/Twitter" est plausible mais non démontré
- Sans mesure instrumentée, on ne sait pas si on gagne grâce à l'AI ou malgré elle

### Ce que le backtest doit prouver
1. **Grok produit des probabilités mieux calibrées que le marché** (Brier score < marché)
2. **L'edge survit aux coûts** (fees, slippage, impact)
3. **Les signaux sont actionnables** (liquide, exécutable, sortie propre)
4. **Le sizing fonctionne** (Kelly fractionnel ne ruine pas le bankroll)

### Ce que le backtest NE peut PAS prouver
- Performance en conditions adverses non observées
- Résilience aux changements de régime
- Edge dans un environnement où d'autres bots similaires existent

---

## 2. Architecture de replay sans fuite

### Principe fondamental
> Le modèle ne doit JAMAIS voir d'information postérieure au timestamp simulé T.

```
                    +----------------------------------+
                    | Snapshot Store (SQLite)           |
                    | markets, rules, books, news      |
                    +----------------+-----------------+
                                     |
                              as_of <= replay clock T
                                     |
                    +----------------v-----------------+
                    | Replay Context Builder            |
                    | queries bornées temporellement    |
                    +----------------+-----------------+
                                     |
        +----------------------------v----------------------------+
        | Grok API                                                |
        | - x_search avec from_date/to_date bornés <= T           |
        | - PAS de web_search (pas de filtre temporel)            |
        | - Client-side tools custom pour web historique           |
        | - System prompt : "simulated date = T"                  |
        | - Modèle versionné (pas -latest)                        |
        +----------------------------+----------------------------+
                                     |
                    +----------------v-----------------+
                    | Strategy Engine                   |
                    | carry, news, research, arb       |
                    | Kelly / edge thresholds / stops   |
                    +----------------+-----------------+
                                     |
                    +----------------v-----------------+
                    | Market Simulator                  |
                    | orderbook walk, impact, fees      |
                    | passive fill queue model          |
                    +----------------+-----------------+
                                     |
                    +----------------v-----------------+
                    | Metrics + Reports                 |
                    | Brier, markout, calibration       |
                    | Sharpe, edge decay, bootstrap CI  |
                    +---------------------------------+
```

### Prévention des fuites — 3 couches

| Couche | Mécanisme | Risque résiduel |
|--------|-----------|-----------------|
| **Données** | Toutes les queries SQL ont `WHERE ts <= as_of` | Aucun si le code est correct |
| **Grok tools** | `x_search` borné par `to_date <= T`, `web_search` remplacé par outil custom | Le modèle peut savoir des choses via son training (cutoff nov. 2024) |
| **Modèle** | Seuls les marchés **post-novembre 2024** sont valides pour mesurer l'alpha prédictif | Marchés pré-cutoff = test de pipeline uniquement |

### Règle critique
> **Ne backtester la qualité de forecast que sur des marchés résolus APRÈS novembre 2024.**
> Les marchés avant cette date testent le pipeline, PAS l'edge.

---

## 3. Sources de données historiques

### Stack Polymarket (gratuit)

| Donnée | Source | Granularité | Coût |
|--------|--------|-------------|------|
| **Prix historiques** | CLOB `/prices-history` | Seconde | Gratuit (1000 req/10s) |
| **Trades tick-level** | Goldsky subgraph `orderFilledEvents` | Tick | Gratuit |
| **Marchés résolus** | Gamma `/events?closed=true&active=false` | — | Gratuit (500 req/10s) |
| **Règles/clarifications** | Gamma `/markets` + `/comments` | — | Gratuit |
| **Orderbook historique** | **pmxt archive** (archive.pmxt.dev) | Horaire (Parquet) | Gratuit |
| **On-chain lifecycle** | Goldsky subgraphs (splits, merges, redemptions) | Tick | Gratuit |
| **Volume/OI courant** | Data API `/oi`, `/live-volume` | Snapshot | Gratuit |

### Datasets pré-collectés

| Dataset | Taille | Contenu |
|---------|--------|---------|
| **Jon-Becker/prediction-market-analysis** (2.2k stars) | ~36 GiB compressé | Polymarket + Kalshi historique complet |
| **Kaggle (ismetsemedov)** | 43,840 events, 100,795 marchés | Metadata snapshot déc. 2025 |
| **warproxxx/poly_data** | Variable | Pipeline + S3 snapshot trades |

### Stack News/Events

| Source | Couverture | Date filter | Coût | Usage |
|--------|-----------|------------|------|-------|
| **X/Twitter via `x_search` Grok** | Temps réel | `from_date`/`to_date` natif | $5/1k calls | Signal principal |
| **GDELT bulk files** | 2015+ (15 min) | Par fichier | Gratuit | Découverte news large |
| **Wayback Machine** | 2008+ | `timestamp` | Gratuit | Figer pages web à T |
| **Wikipedia pageviews** | Juil. 2015+ | Journalier | Gratuit | Attention publique |
| **Congress.gov API** | 1973+ | Par date | Gratuit (5k/hr) | Événements législatifs |
| **FEC/OpenFEC** | 2007+ | `min_date`/`max_date` | Gratuit (1k/hr) | Finance de campagne |
| **CourtListener** | 9M+ décisions | Par date | Gratuit (5k/hr) | Legal/judiciaire |
| **Arctic Shift (Reddit)** | 2005+ | Par date | Gratuit | Discussions politiques |

### Sources payantes (Phase 2 seulement)

| Source | Coût | Valeur ajoutée |
|--------|------|---------------|
| **Event Registry** | $90/mo | News structurées depuis 2014, entités, événements |
| **X API v2 Full Archive** | $0.005/tweet | Archive complète depuis 2006 |
| **GNews** | €49.99/mo | 80k sources depuis 2020 |

### Architecture de corpus

```
Deux couches :
1. DISCOVERY (index mutable)     → GDELT, news APIs, x_search, Wikipedia
2. FROZEN CONTENT (immuable)     → Wayback snapshots, raw HTML/PDF, Parquet

Chaque document stocke :
- published_at          (quand l'info est publique)
- first_seen_at         (quand notre collecteur l'a vue)
- ingested_at           (quand elle est entrée dans la DB)
- source, source_id, canonical_url
- content_hash          (SHA-256 du contenu)
```

---

## 4. Setup Grok en mode replay

### Modèles

| Modèle | Usage backtest | Pourquoi |
|--------|---------------|---------|
| `grok-4.20-beta-0309-reasoning` | Forecasting | Version datée, reproductible |
| `grok-4-fast-non-reasoning` | Monitoring/triage | Rapide, cheap ($0.20/$0.50 par 1M) |

**Règle** : toujours utiliser un modèle versionné daté (pas `-latest`). Si aucun daté disponible, logger `system_fingerprint` à chaque call.

### Tools autorisés en replay

```python
# AUTORISÉ — x_search avec dates bornées
tools = [
    {
        "type": "x_search",
        "from_date": "2025-03-01",   # >= date de début du marché
        "to_date": "2025-03-05",     # <= T (timestamp simulé)
    }
]

# AUTORISÉ — client-side tools custom
tools += [
    {
        "type": "function",
        "name": "historical_web_search",    # NOTRE outil, corpus figé
        "parameters": {"query": "str", "as_of_date": "str"}
    },
    {
        "type": "function",
        "name": "get_market_snapshot",      # NOTRE outil, DB locale
        "parameters": {"market_id": "str", "as_of_date": "str"}
    },
]

# INTERDIT
# web_search  ← pas de filtre temporel, fuiterait
```

### System prompt temporel

```
You are running inside a historical replay.
Current replay time: {T}
You must behave as if nothing after this timestamp exists.

Hard rules:
1. Use only the supplied replay context + x_search results within date bounds.
2. Never reference future prices, news, or final outcomes.
3. If evidence is insufficient, say so explicitly.
4. Return: probability_yes, confidence, thesis, reasoning, evidence, key_unknowns.
5. Output strict JSON only.
```

### Paramètres déterministes
- `temperature = 0`
- `store = false` (pas de logging côté xAI)
- `parallel_tool_calls = false`
- Logger : `model`, `system_fingerprint`, `usage`, `prompt_hash`, `context_hash`

### Coût par évaluation
- ~5k tokens input + 1k output = **$0.0015**
- + 1-3 `x_search` calls = **$0.005-$0.015**
- **Total : ~$0.01-$0.02 par marché évalué**

---

## 5. Stratégies à tester

### Grille d'expérimentation

| Dimension | Valeurs à tester |
|-----------|-----------------|
| **Famille** | carry_only, news_driven, deep_research, cross_market_arb |
| **Kelly fraction** | 0.05, 0.10, 0.15, 0.25 |
| **Edge threshold** | 25, 50, 100, 200 bps |
| **Exécution** | aggressive (taker), passive (maker) |
| **Holding** | 5min, 30min, 240min, hold-to-resolution |
| **Stops** | aucun, thesis_stop, time_stop, les deux |

### Stratégies par famille

**Carry only** (acheter 95-99c, hold to resolution)
- Edge minimum : 25 bps
- Kelly : 0.05
- Pas de time stop
- Marchés : résolution claire, < 30 jours, non-ambigu

**News-driven** (réagir aux breaking events)
- Edge minimum : 50 bps
- Kelly : 0.10-0.15
- Time stop : 240 min
- Thesis stop : si P(yes) baisse de 8+ points
- Marchés : événements catalyseurs identifiables

**Deep research** (mispricings structurels)
- Edge minimum : 100 bps
- Kelly : 0.05-0.10
- Holding : jours-semaines
- Marchés : mid-liquidity, pas les headlines

**Cross-market arb** (incohérences logiques entre marchés liés)
- Scanner sémantique, pas l'alpha principal
- Vérification déterministe des payoffs

### Configuration MVP recommandée (par ChatGPT corrigé)

Au lieu de 4 stratégies en parallèle avec $1k :
- **Phase 1** : carry_only SEUL — prouver que la sélection de marchés + sizing fonctionne
- **Phase 2** : ajouter news_driven SI carry profitable
- **Phase 3** : ajouter deep_research SI news profitable

---

## 6. Métriques et seuils de validation

### Métriques obligatoires

| Métrique | Formule | Seuil go/no-go |
|----------|---------|---------------|
| **Brier score** | `mean((p_yes - outcome)²)` | < Brier du marché (= prix) |
| **Markout 1/5/30/240 min** | `future_mid - fill_price` | Positif en moyenne |
| **Adverse selection** | `% trades où markout < 0` | < 55% |
| **Fill ratio** | `filled_qty / requested_qty` | > 60% |
| **PnL pré-résolution** | Realized PnL des exits avant résolution | Positif |
| **PnL hold-to-resolution** | PnL si on gardait jusqu'à résolution | Positif |
| **Calibration curve** | Forecast par décile vs fréquence réalisée | Proche de la diagonale |
| **Sharpe-like** | Daily return Sharpe sur equity | > 0.5 |
| **Edge decay** | `realized_markout / initial_edge` par horizon | > 0.3 à 30min |

### Sample sizes nécessaires

| Objectif | Minimum |
|----------|---------|
| Signal directionnel | 200-300 trades, 50 marchés résolus |
| Ranking de stratégies | 1000+ trades, 200+ marchés résolus |
| Calibration | 100+ outcomes par décile |

### Tests statistiques

- **Bootstrap CI** sur le PnL moyen — l'intervalle doit être strictement > 0
- **Paired bootstrap** pour comparer deux stratégies sur les mêmes marchés
- **Clustered standard errors** par marché/événement (pas naive t-test)
- **Diebold-Mariano** pour comparer les pertes de forecast

### Critères go/no-go pour passer au live

| Critère | Seuil |
|---------|-------|
| Brier score agent < Brier score marché | Sur 50+ marchés résolus |
| Bootstrap CI(mean PnL) > 0 | Sur 200+ trades |
| Edge stable across domains | Pas juste un burst sur un marché |
| Markout et hold-to-resolution PnL dans la même direction | Cohérence |
| Edge decay < latence d'exécution | L'info est encore exploitable |

---

## 7. Validation gratuite sur Metaculus

### Pourquoi Metaculus d'abord ?

- **Gratuit** (pas de capital, pas de frais API Polymarket)
- **Questions curées** avec critères de résolution explicites
- **Scoring rigoureux** (Brier, log score, calibration)
- **Benchmark** : Metaculus Pros score 36.00, Community 22.38, Claude Opus 4.5 19.54
- **Grok 4.20 score 67.9 sur ForecastBench** (superforecasters: 70.6)

### Plan de validation

1. Faire tourner Grok sur ~200 questions Metaculus ouvertes
2. Mesurer calibration et Brier score
3. Comparer aux forecasts de la communauté
4. Si Grok bat la communauté → confiance pour passer à Polymarket
5. Si non → affiner les prompts et la méthodologie avant de dépenser

### API Metaculus
- Questions avec historique de forecasts
- Exports zip téléchargeables
- Résolution explicite

### Ordre de validation recommandé
```
Metaculus (forecast quality, gratuit)
  → Manifold Markets (market interaction, play money)
    → Polymarket paper trading (exécution simulée, gratuit)
      → Polymarket micro-live ($100-250, réel)
```

---

## 8. Coûts du backtest

### Phase "Collecte de données" (~$0/mo)

| Poste | Coût |
|-------|------|
| Polymarket APIs (Gamma, CLOB, Goldsky) | Gratuit |
| Jon-Becker dataset (36 GiB) | Gratuit |
| pmxt archive (orderbook Parquet) | Gratuit |
| GDELT bulk files | Gratuit |
| Wikipedia pageviews | Gratuit |
| Congress.gov / FEC / CourtListener | Gratuit |
| Stockage local | ~0 (SSD existant) |

### Phase "Replay" (~$20-50 par run complet)

| Poste | Calcul | Coût |
|-------|--------|------|
| Grok API (200 marchés × 5 timestamps × $0.015/eval) | 1000 evals | ~$15 |
| x_search calls (~3 par eval × $0.005) | 3000 calls | ~$15 |
| Compute (local Mac) | — | $0 |
| **Total par run complet** | | **~$30** |

### Phase "Grid search" (~$150-300 total)

- 5-10 runs avec différentes configs de stratégie
- Total : **$150-300** pour explorer la grille complète

### Comparaison avec le coût d'un échec
- Perdre $500 sur du live sans backtest = bien plus cher
- $150-300 de backtest = assurance contre les erreurs coûteuses

---

## 9. Plan d'exécution

### Étape 1 — Data Pipeline (semaine 1)
- [ ] Télécharger le dataset Jon-Becker (36 GiB)
- [ ] Télécharger les archives pmxt (orderbook Parquet)
- [ ] Fetcher tous les marchés résolus post-nov 2024 via Gamma API
- [ ] Fetcher les price histories CLOB pour ces marchés
- [ ] Fetcher les trades Goldsky pour ces marchés
- [ ] Constituer le corpus news : GDELT + Wikipedia pageviews
- [ ] Tout stocker dans SQLite avec le schéma existant

### Étape 2 — Validation Metaculus (semaine 1-2, en parallèle)
- [ ] Setup Grok API avec clé xAI
- [ ] Écrire un script qui soumet des forecasts Metaculus
- [ ] Faire tourner sur ~200 questions ouvertes
- [ ] Mesurer Brier score et calibration
- [ ] Go/no-go : Grok bat-il la communauté ?

### Étape 3 — Premier replay (semaine 2-3)
- [ ] Identifier 50-100 marchés politiques résolus post-nov 2024
- [ ] Lancer le replay engine avec `carry_only` (config la plus simple)
- [ ] x_search borné par dates pour chaque évaluation
- [ ] Analyser : Brier, markout, PnL, calibration
- [ ] Go/no-go : edge > 0 après fees ?

### Étape 4 — Grid search (semaine 3-4)
- [ ] Tester les variantes : Kelly fractions, edge thresholds, aggressive vs passive
- [ ] Ajouter `news_driven` au mix
- [ ] Comparer carry vs news vs combiné
- [ ] Identifier la meilleure config
- [ ] Bootstrap CI : l'edge est-il statistiquement significatif ?

### Étape 5 — Paper trading live (semaine 5-6)
- [ ] Connecter au WebSocket Polymarket en temps réel
- [ ] Exécuter les signaux sans argent réel
- [ ] Mesurer : latence signal→décision, fill simulation, markout réel
- [ ] Comparer aux résultats du backtest

### Étape 6 — Go/no-go pour le live
- [ ] Tous les critères de validation remplis ?
- [ ] Brier agent < Brier marché sur 50+ marchés ?
- [ ] PnL simulé positif après fees sur 200+ trades ?
- [ ] Aucune anomalie de calibration ?
- → Si oui : micro-live avec $100-250

---

## 10. Frameworks et outils existants

### À utiliser

| Outil | Pourquoi |
|-------|---------|
| **py-clob-client** | SDK officiel Polymarket, plumbing |
| **scoringrules** (pip) | Brier, CRPS, log score — meilleur package scoring |
| **agent-next/polymarket-paper-trader** | Scaffold simulateur Polymarket-natif |
| **Jon-Becker/prediction-market-analysis** | Dataset 36 GiB pré-collecté |
| **pmxt archive** | Snapshots orderbook horaires |
| **Goldsky subgraphs** | Données on-chain tick-level |

### Comme référence (pas à forker)

| Outil | Pourquoi |
|-------|---------|
| **PredictionMarketBench** | Design patterns, mais Kalshi-spécifique |
| **ent0n29/polybot** | Stack ops complète, mais Java microservices |
| **Polymarket/agents** | Agent scaffolding, mais pas de backtest |

### Code déjà écrit (dans ce repo)

Le framework de backtesting existe déjà dans `src/polymarket_backtest/` :
- `schema.sql` — 13 tables SQLite
- `db.py` — queries bornées temporellement + seed data
- `grok_replay.py` — client Grok en mode replay, leakage prevention
- `market_simulator.py` — fills agressifs/passifs, fees Polymarket, impact model
- `strategies.py` — carry, news-driven, deep research, grille paramétrable
- `replay_engine.py` — boucle principale, portfolio, résolution
- `metrics.py` — Brier, markout, calibration, Sharpe, edge decay, bootstrap CI
- `report.py` — rapports markdown
- `demo.py` — démo end-to-end

### Ce qu'il reste à construire

1. **Data pipeline** : downloader Jon-Becker + pmxt + Gamma + CLOB + Goldsky → SQLite
2. **News corpus builder** : GDELT + Wikipedia + Wayback → news_documents table
3. **Grok transport réel** : connecter `XAIResponsesTransport` avec x_search borné
4. **Rapport de comparaison** : stratégie A vs B vs C avec bootstrap CI
5. **Metaculus validator** : script de validation forecast

---

## Références

### Polymarket
- [Polymarket API docs](https://docs.polymarket.com/)
- [pmxt archive](https://archive.pmxt.dev/)
- [Goldsky subgraphs](https://docs.polymarket.com/market-data/subgraph)
- [Jon-Becker/prediction-market-analysis](https://github.com/Jon-Becker/prediction-market-analysis)
- [agent-next/polymarket-paper-trader](https://github.com/agent-next/polymarket-paper-trader)

### Grok/xAI
- [xAI Models](https://docs.x.ai/developers/models)
- [x_search tool](https://docs.x.ai/developers/tools/x-search)
- [Function calling](https://docs.x.ai/developers/tools/function-calling)

### Benchmarks forecasting
- [ForecastBench](https://www.forecastbench.org/) — Grok 4.20: 67.9, superforecasters: 70.6
- [Metaculus FutureEval](https://www.metaculus.com/futureeval/)
- [Prophet Arena](https://www.prophetarena.co/research/welcome)

### Papers
- [The Anatomy of Polymarket](https://arxiv.org/html/2603.03136v1)
- [Pitfalls in Evaluating LLM Forecasters](https://arxiv.org/html/2506.00723v1)
- [Evaluating LLMs vs Expert Forecasters](https://arxiv.org/html/2507.04562v3)
- [PredictionMarketBench](https://github.com/Oddpool/PredictionMarketBench)

### News/Data
- [GDELT](https://www.gdeltproject.org/data.html)
- [Congress.gov API](https://api.congress.gov/)
- [OpenFEC](https://api.open.fec.gov/developers/)
- [CourtListener](https://www.courtlistener.com/help/api/)
- [Wikipedia Pageviews](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html)
