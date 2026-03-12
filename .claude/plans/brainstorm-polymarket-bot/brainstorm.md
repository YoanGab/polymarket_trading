# Polymarket AI Trading Bot — Complete Strategy

> Brainstorm complet — 22 agents de recherche, Mars 2026
> Contexte : bot multi-agents Grok, marchés de prédiction politiques/sociétaux, basé en Irlande

---

## Table des matières

1. [Vision & Thesis](#1-vision--thesis)
2. [Marchés cibles](#2-marchés-cibles)
3. [Architecture des agents Grok](#3-architecture-des-agents-grok)
4. [Stratégies de trading](#4-stratégies-de-trading)
5. [Trade lifecycle](#5-trade-lifecycle)
6. [Risk management](#6-risk-management)
7. [Sources de données](#7-sources-de-données)
8. [Infrastructure technique](#8-infrastructure-technique)
9. [Coûts et projections financières](#9-coûts-et-projections-financières)
10. [Légal & compliance (Irlande)](#10-légal--compliance-irlande)
11. [Wallet & on-chain](#11-wallet--on-chain)
12. [Roadmap de déploiement](#12-roadmap-de-déploiement)
13. [Risques et mitigations](#13-risques-et-mitigations)
14. [Références clés](#14-références-clés)

---

## 1. Vision & Thesis

### Le concept
Un bot de trading autonome sur Polymarket utilisant des agents Grok (xAI) spécialisés qui analysent en temps réel les données X/Twitter, le web, et les sources officielles pour détecter des mispricings sur les marchés de prédiction politiques et sociétaux.

### Pourquoi ça peut marcher
- **Grok + X/Twitter = edge unique** : confirmé par l'expérience Defend Intelligence (Grok seul AI profitable sur 6 testées, grâce à l'accès X)
- **Marchés politiques = prédictibles** : les résultats dépendent de données analysables (sondages, déclarations, historique), pas du hasard
- **Mispricings réels** : Michael B. Jordan Best Actor à 55.8% sur Polymarket vs 81.1% chez Gold Derby (gap de 25 points détecté en live)
- **Coûts faibles** : MVP à ~$25/mo, viable avec $500-$1000 de capital
- **Irlande = juridiction favorable** : pas géobloqué, gains de paris potentiellement non imposables

### Pourquoi ça peut échouer
- **70% des traders Polymarket perdent de l'argent** (médiane : -$0.90)
- **<0.04% des adresses capturent >70% des profits**
- L'edge théorique peut ne pas survivre aux frais, slippage, et traders plus rapides
- Les LLMs sont encore derrière les superforecasters humains sur les benchmarks de calibration

### Position honnête
> Avec <$1k, c'est un **investissement en apprentissage**, pas un revenu. Si les résultats suivent après 2-3 mois de MVP, le scaling est possible.

---

## 2. Marchés cibles

### Catégories prioritaires (décisions humaines, pas hasard)

| Priorité | Catégorie | Liquidité totale | AI Edge | Exemples |
|----------|-----------|-----------------|---------|----------|
| 1 | **Politique US** | $109M | Très élevé | Elections 2026/2028, nominations, policy |
| 2 | **Géopolitique / Moyen-Orient** | $22.7M | Élevé | Ceasefires, régimes, conflits |
| 3 | **Fed / Politique monétaire** | $23.4M | Moyen | Décisions FOMC, nominations |
| 4 | **Entertainment (awards)** | $25.2M | Très élevé | Oscars, Eurovision, comités |
| 5 | **Science & Tech** | $2.4M | Élevé | Releases AI, lancements, rankings |
| 6 | **Legal / Judicial** | Faible | Élevé | SCOTUS, régulations FDA |
| 7 | **Europe / Monde** | Variable | Élevé | Elections, Brexit, EU policy |

### Catégories exclues
- **Sports** : trop de hasard, dominé par les bookmakers
- **Crypto prix** : dominé par les bots HFT, taker fees 1.56%
- **Climate / météo** : exogène, peu prédictible par l'AI
- **Compteurs (cas COVID, etc.)** : randomness domine

### Caractéristiques des marchés politiques (données live)
- ~5,981 marchés politiques actifs, ~60 nouveaux contrats/jour
- Spread médian marchés liquides : **0.2c** (très serré)
- Durée médiane d'un marché : ~243 jours, résolution médiane dans ~112 jours
- Les marchés les plus tradables : **binaires, résolution claire, narrative chaude, 1-30 jours**

### Exemples de marchés actifs (11 mars 2026)

| Marché | Odds | Volume 24h | Liquidité | Spread |
|--------|------|-----------|-----------|--------|
| Fed no change March | 99.2% | $2.83M | $1.26M | 0.1c |
| Dems control House 2026 | 84.5% | $3k | $213k | 0.8c |
| Iran regime falls June | 22.5% | $365k | $682k | 0.8c |
| Peter Magyar next Hungarian PM | 61.5% | $40k | $155k | 1.0c |
| One Battle After Another Best Picture | 75.5% | $2.07M | $130.6k | 0.8c |
| Anthropic best AI model March | 84.2% | $772k | $29.8k | 0.4c |

---

## 3. Architecture des agents Grok

### Principe fondamental
> **Grok raisonne et explique. Le code déterministe décide.** Risk et execution ne sont JAMAIS LLM-only.

### Modèles Grok utilisés

| Modèle | Prix (in/out 1M) | Contexte | Usage |
|--------|-----------------|----------|-------|
| `grok-4-fast-non-reasoning` | $0.20 / $0.50 | 2M | Monitoring, triage, détection rapide |
| `grok-4-fast-reasoning` | $0.20 / $0.50 | 2M | Analyse, estimation probabilités, recherche |
| `grok-4.20-beta` (Enterprise) | $2.00 / $6.00 | 2M | Deep analysis haute valeur (optionnel) |

### Tools (function calling) — le LLM NE calcule PAS
```
calculate_kelly(p_model, p_market, bankroll) → taille de position
calculate_ev(p_model, p_market, fees, slippage) → expected value
calculate_annualized_carry(price, days_to_resolution) → rendement carry
check_correlation(new_position, portfolio) → risque de cluster
get_market_price(market_id) → prix, spread, volume, depth
get_order_book(market_id, depth) → book complet
get_related_markets(market_id) → marchés liés (arb detection)
get_positions() → positions ouvertes, PnL, exposure
check_risk_limits(proposed_trade) → pass/fail + raison
check_liquidity(market_id, size) → can_exit + slippage estimé
place_order(market_id, side, size, price, type) → order ID
cancel_order(order_id) → confirmation
```

### Roster des agents

#### Phase 1 — MVP (4 agents)

| Agent | Modèle | Refresh | Rôle |
|-------|--------|---------|------|
| **News/Event Detector** | `grok-4-fast-non-reasoning` | Continu (15-30s) | Détecte breaking news via `x_search` + `web_search`, mappe aux marchés impactés |
| **Politics Analyst** | `grok-4-fast-reasoning` | 30min + on trigger | Estime P(YES) depuis institutions, sondages, déclarations, analyse de fond |
| **Forecaster + Risk** | `grok-4-fast-reasoning` + déterministe | 5min + on signal | Agrège signaux, calibre probabilités, vérifie limites risk |
| **Execution** | Déterministe (reasoning optionnel) | On trade intent | Choisit order type, taille, prix, timing |

#### Phase 2 (ajouter)
- **Market Scanner** : découvre nouveaux marchés intéressants
- **Market Structure Agent** : analyse orderbook, liquidité, whale activity
- **Resolution Monitor** : surveille marchés proches de résolution, détecte ambiguïté

#### Phase 3 (ajouter)
- **Sentiment Aggregator** : sentiment cross-plateforme (X, Reddit, YouTube)
- **EU/World Politics Analyst** : spécialiste géopolitique séparé
- Calibration en ligne, graphe de dépendances cross-marchés

### Prompt structure (superforecasting)
Chaque agent suit cette structure pour maximiser la calibration :
1. **Outside view** : référence classes, base rates historiques
2. **Décomposer** en 3-5 drivers clés
3. **Probabilité initiale** (avant de voir le prix marché — évite l'anchoring)
4. **Update** avec les preuves actuelles
5. **Meilleur argument CONTRE** sa propre prédiction
6. **Output JSON structuré** : `p_base`, `p_final`, `evidence_for`, `evidence_against`, `key_unknowns`

### Agrégation des forecasts
- Combiner en **log-odds** (pas simple moyenne — ça compresse trop vers 50%)
- Pondérer par **Brier score historique** de chaque agent
- **Extremiser légèrement** après backtesting
- Traiter le **prix marché comme prior**, l'ensemble des agents comme adjustment
- Ne trader que quand `edge post-calibration > fees + slippage + erreur de calibration`

### Inter-agent communication
- **Event bus** : Redis Streams (MVP), NATS JetStream (full)
- **Shared state** : PostgreSQL (source of truth), Redis (hot cache)
- Les agents ne s'appellent PAS entre eux — l'orchestrateur coordonne

### Signal format inter-agents
```json
{
  "signal_id": "uuid",
  "agent": "us_politics",
  "market_id": "pm_123",
  "kind": "probability_update",
  "p_yes": 0.58,
  "confidence": 0.76,
  "evidence": [{"url": "...", "tier": "official", "ts": "..."}],
  "risk_flags": ["ambiguous_rules"],
  "decay_half_life_min": 180
}
```

### Workflow concret — "Un candidat se retire"
```
T+0-20s   → News Detector voit un cluster sur X (candidat + 2 reporters crédibles)
T+20-60s  → web_search confirme sur le site de campagne + Reuters/AP
T+60-90s  → Orchestrateur mappe les marchés impactés (nominee, primaire, élection générale)
T+90-180s → Politics Analyst met à jour fair values (nominee 21% → 1%, rival 34% → 58%)
T+180-230s → Forecaster agrège, Sentiment vérifie consolidation des élites
T+230-235s → Risk Manager clips la taille si exposure corrélée existante
T+235-260s → Execution poste des limit orders laddered, repricing toutes les 2-3s
= Premier trade en < 3 minutes, recalibration complète en 5-10 minutes
```

---

## 4. Stratégies de trading

### Allocation recommandée (~$1000)

| Stratégie | Allocation | ROI annualisé | Risque | AI Edge |
|-----------|-----------|---------------|--------|---------|
| **Resolution carry (95-99c)** | 40% ($400) | 10-30% | 3/10 | Moyen |
| **Deep research politique** | 30% ($300) | 10-50% | 5/10 | Élevé |
| **Event-driven (breaking news)** | 20% ($200) | 20-100%+ (lumpy) | 8/10 | Très élevé |
| **Cash réserve** | 10% ($100) | 0% | 0/10 | — |

### Où l'AI ajoute du VRAI alpha
- Parsing de documents/discours/filings en temps réel
- Mapping sémantique entre marchés liés (cross-market logical arb)
- Screening du risque de résolution ambiguë
- Détection de relations leader-follower entre marchés
- Détection de cascades d'information cross-plateformes

### Où c'est du théâtre (pas d'edge réel)
- Demander à Grok "quels sont les odds" sur des gros marchés
- Sentiment Twitter générique sans filtre de qualité
- Market making piloté par LLM (trop de capital nécessaire, trop lent)

### Arbitrage en background
- Scanner cross-market logique : Grok détecte les relations sémantiques, le code vérifie les payoffs
- Pas la stratégie principale à <$5k de capital, mais un scanner complémentaire
- Alpha historique : $39.6M extraits en arb sur Polymarket en 1 an (politique = catégorie la plus lucrative)

---

## 5. Trade lifecycle

### Entrée
- **Edge minimum requis** : 3c (marchés liquides), 5c (normaux), 8c+ (thin books)
- **Scaling** : 30% starter → 40% confirmation → 30% completion
- **Limit orders par défaut** (GTC/GTD post-only)
- **FOK** seulement pour les news urgentes quand le book est stale et la profondeur suffisante
- **Pre-trade checks obligatoires** :
  - Profondeur bid à ±5c ≥ 10x la position
  - Slippage exit estimé < 1c
  - ≥20 trades en 24h
  - Dernier trade < 6h
  - Spread < 5c

### Gestion de position
- **Management actif > hold-to-resolution** pour les marchés politiques
- **Thesis stop** (pas price stop) : sortir quand l'EV post-coûts devient négative
- **Time stop** : couper si rien ne bouge après 7-14 jours (medium-term) ou 30 jours (long-term)
- Recompute fair value à chaque nouvelle info significative

### Sortie
- **Sortir AVANT résolution** dans la plupart des cas (quand 60-80% du move est fait)
- Resolution carry (95-99c) réservé aux marchés **proches, clairs, non-ambigus**
- Exits passifs (laddered GTC/GTD), actifs (FAK) seulement en urgence
- Annualized carry formula : `((1 - p_net) / p_net) * (365 / days_to_resolution)`

### Position sizing
- **Formule Kelly** : `f* = max(0, (q - p_exec) / (1 - p_exec))` pour YES à résolution
- **Toujours 0.25x Kelly** par défaut (probas incertaines, marchés corrélés)
- **Confidence adjustment** : `f = 0.25 * confidence * f*`
- **$500** : 3-4 positions core + 1 tactique max
- **$1000** : 4-6 positions core + 1-2 tactiques max
- **Cap cluster corrélé** : 35-40% max du bankroll (tous marchés liés au même événement)

### Horizons temporels
- **Court terme (minutes-heures)** : 20% du book — news trades, petit sizing, time stop serré
- **Moyen terme (jours-semaines)** : 45% — cœur du book, recherche + catalystes programmés
- **Long terme (semaines-mois)** : 15% — fondamental pur, plus petite taille
- **Cash** : 20% — toujours en réserve

### Rebalancing
- Rebalancer sur **changement d'edge**, pas sur calendrier fixe
- Fermer la position avec le **plus faible edge en premier** (pas le plus gros perdant)
- Rotation vers de meilleures opportunités quand return annualisé < hurdle rate

---

## 6. Risk management

### Limites

| Contrôle | Seuil | Action |
|----------|-------|--------|
| Per-trade max loss | 0.5% equity (hard cap 1%) | Rejeter le trade |
| Daily loss | 1.5% | Stop new trades |
| Daily loss | 2% | **Kill switch** |
| Weekly drawdown | 4% (high-water mark) | Réduire agressivité |
| Monthly drawdown | 8% (high-water mark) | Revue complète |
| Per-market exposure | 3% equity (1-2% si illiquide) | Limiter la taille |
| Total directionnel | 15% equity | Stop new risk |
| Total gross | 30% equity | Stop new risk |

### Kill switch
1. Set `TRADING_DISABLED=true` (flag externe au bot)
2. Stop tous les nouveaux ordres
3. Call Polymarket `cancel-all`
4. Disconnect execution workers
5. Page humain (Telegram/SMS)
6. Réconciliation complète `/orders` + `/activity` + `/positions` avant de réactiver

### Triggers de kill switch automatique
- Breach daily loss limit
- Order state inconnu > 30s
- WebSocket stale > 15s ou market data stale > 5s
- Mismatch réconciliation > 0.25% equity
- HTTP 503 (cancel-only) de Polymarket
- Marché détenu entre en clarification/dispute
- Slippage réalisé > 2x le modèle

### State machine du bot
```
NORMAL → CAUTIOUS → HALT → KILL_SWITCH → RECONCILE → RESUME
```

### Source of truth
- **L'exchange est TOUJOURS la source of truth**, pas l'état local
- Réconciliation sur chaque fill, chaque reconnexion, et snapshot complet toutes les 30-60s

---

## 7. Sources de données

### Stack MVP
| Source | Accès | Signal | Coût |
|--------|-------|--------|------|
| **X/Twitter** (via Grok `x_search`) | Natif Grok | Breaking news, sentiment élite | $5/1k calls |
| **Web** (via Grok `web_search`) | Natif Grok | Confirmation, recherche | $5/1k calls |
| **Polymarket WebSocket/CLOB** | Gratuit | Prix, orderbook, trades temps réel | $0 |
| **Polymarket Gamma API** | Gratuit | Métadonnées marchés, événements | $0 |
| **Google Trends API** | Alpha (gratuit) | Attention publique large | $0 |
| **Wikipedia pageviews** | Gratuit | Spikes d'intérêt candidats/sujets | $0 |

### Stack étendu (Phase 2+)
| Source | Signal | Coût |
|--------|--------|------|
| **YouTube Data API** | Uploads, vues, commentaires politiques | Gratuit (10k quota/jour) |
| **FEC/OpenFEC** | Fundraising, dépenses campagne | Gratuit |
| **CourtListener** | Dockets, filings, décisions judiciaires | Gratuit |
| **Telegram Bot API** | Channels politiques/activistes | Gratuit |
| **GDELT** | News large spectre (gratuit, 15min lag) | Gratuit |
| **CME FedWatch** | Probabilités FOMC | $25/mo |
| **The Odds API** | Cotes bookmakers (arb sports si besoin) | $30/mo |
| **TikTok via Apify** | Signal jeunesse/attention (optionnel) | $1.70/1k résultats |

### Détection de cascades cross-plateformes
```
X/Telegram (breaking, 1er) → TikTok/Threads (amplification, 2ème)
  → YouTube (persistance, 3ème) → Google Trends/Wikipedia (attention large, 4ème)
  → TV/mainstream (confirmation, 5ème)
```

Le vrai alpha : **accélération + confirmation cross-plateforme + persistance**, pas le sentiment brut.

---

## 8. Infrastructure technique

### Stack technique

| Composant | MVP (Phase 1) | Production (Phase 2) |
|-----------|--------------|---------------------|
| **Runtime** | asyncio (single process) | asyncio + event bus |
| **Cloud** | GCP e2-micro (free tier) | AWS ECS Fargate eu-west-1 |
| **Database** | SQLite | PostgreSQL (RDS) |
| **Event bus** | In-process | Redis Streams → NATS |
| **Monitoring** | Structured JSON logs | Prometheus + Grafana |
| **IaC** | Docker local | Terraform |
| **CI/CD** | Manual deploy | GitHub Actions + approval gate |

### Architecture cible
```
Polymarket WS + Polygon RPC
  → Feed service (always-on, 1 container)
  → Event bus (in-process ou Redis)
  → Agents Grok (workers logiques dans le même process asyncio)
  → Risk gate (déterministe)
  → Execution (déterministe + CLOB)
  → PostgreSQL/SQLite (source of truth)
  → Monitoring/Alerts (Telegram)
```

### Points clés
- Les agents sont des **workers logiques dans un seul process**, PAS des services séparés
- WebSocket : ping toutes les 10s, reconnexion rapide, réconciliation après chaque drop
- Tuesday 7AM ET : matching engine restart (~90s), HTTP 425, ordres persistent
- Deployments : active/passive avec leader election via DB advisory lock

### API Polymarket utilisées
- **Gamma API** (`gamma-api.polymarket.com`) : métadonnées marchés
- **CLOB API** (`clob.polymarket.com`) : trading, orderbook
- **Data API** (`data-api.polymarket.com`) : historique, positions
- **WebSocket** : market channel (public), user channel (auth)
- **Bridge API** : dépôts/retraits USDC

### SDK Python
- `py-clob-client` : auth, signing, ordres
- `xai-sdk` ou OpenAI-compatible : agents Grok
- Custom : retry, throttling, reconnexion (py-clob-client n'a rien de built-in)

---

## 9. Coûts et projections financières

### Phase 1 — MVP (~$25/mo)

| Poste | Coût mensuel |
|-------|-------------|
| Grok API (1M in + 0.15M out/jour, 100 search/jour) | $23.25 |
| GCP e2-micro | $0 (free tier) |
| SQLite | $0 |
| Polygon RPC (Alchemy free) | $0 |
| Gas Polygon (~250 tx/mo) | $0.87 |
| Monitoring | $0 |
| **Total** | **~$25/mo** |

### Phase 2 — Full (~$143/mo)

| Poste | Coût mensuel |
|-------|-------------|
| Grok API (3M in + 0.45M out/jour, 300 search/jour) | $69.75 |
| AWS EC2 t4g.medium (eu-west-1) | $26.86 |
| RDS PostgreSQL db.t4g.small | $27.85 |
| Polygon RPC (Alchemy PAYG) | $9.00 |
| Gas Polygon (~1000 tx/mo) | $3.49 |
| CloudWatch | $3.00 |
| **Total** | **~$143/mo** |

### Projections de revenus (honnêtes)

| Scénario | Return mensuel | Sur $500 | Sur $1000 |
|----------|---------------|---------|-----------|
| **A : Mauvais (bottom 70%)** | -10%/mo | -$50 | -$100 |
| **B : Moyen (break-even)** | 0%/mo | $0 | $0 |
| **C : Bon (top 10%)** | +5%/mo | +$25 | +$50 |
| **D : Excellent (top 1%)** | +15%/mo | +$75 | +$150 |

### Break-even

| Phase | Capital nécessaire à 5%/mo | Capital nécessaire à 15%/mo |
|-------|--------------------------|---------------------------|
| MVP ($25/mo) | ~$505 | ~$168 |
| Full ($143/mo) | ~$2,850 | ~$948 |

### Règle d'or
> Ne passer en Phase 2 qu'après que le MVP soit profitable ET que le capital dépasse $3k-$5k.

### Optimisation des coûts Grok
- Déclencher `x_search`/`web_search` **seulement sur événements** (pas à chaque refresh)
- Cacher les résultats de recherche et les partager entre agents
- Utiliser `grok-4-fast-non-reasoning` pour le monitoring, reasoning seulement pour l'analyse
- Batch API de xAI = 50% de réduction sur les recherches non-urgentes

---

## 10. Légal & compliance (Irlande)

### Statut
- **Irlande PAS géobloquée** par Polymarket
- Pas de déclaration réglementaire irlandaise spécifique contre Polymarket
- Le GRAI (Gambling Regulatory Authority of Ireland) est nouveau et commence à délivrer des licences

### Fiscalité — potentiellement très favorable
- **Gains de paris exempts de CGT** (Capital Gains Tax) en Irlande
- **Gains de paris exempts de CAT** (Capital Acquisitions Tax)
- La taxe sur les paris (2% remote betting duty) est **côté opérateur**, pas côté joueur
- Si classé comme "betting winnings" → **potentiellement non imposable**
- Recommandation : consulter un tax advisor irlandais pour confirmer

### Risques réglementaires
- Le GRAI pourrait exiger des licences pour les opérateurs servant l'Irlande
- Si Polymarket ajoute KYC → les positions pourraient être affectées
- MiCA (EU) s'applique aux crypto-assets mais ne couvre pas explicitement les prediction markets

### Bot legality
- Polymarket **ne bannit PAS les bots** — API dédiée au trading programmatique
- Interdit : spoofing, wash trading, front-running, manipulation

---

## 11. Wallet & on-chain

### Setup initial (MetaMask EOA)
- MetaMask fonctionne directement avec Polymarket
- `signatureType=0`, wallet = funder
- Besoin de USDC.e + POL sur Polygon
- API creds dérivées automatiquement via `ClobClient.create_or_derive_api_creds()`

### On-ramp depuis l'Irlande
- **MoonPay** : SEPA, carte, Apple Pay, Google Pay — enregistré CASP Ireland
- Frais : SEPA ~0.99%, carte ~3.5-4%
- Bridging : ~$0.06 de frais pour $100 USDC vers Polygon

### Architecture wallet recommandée (long terme)
- **Hot wallet** (Gnosis Safe) : capital de trading, gasless via relayer
- **Treasury** (MetaMask multi-sig) : sweep profits, gros dépôts
- **Sécurité** : GCP KMS pour signing (support secp256k1), spending limits via Safe modules

---

## 12. Roadmap de déploiement

### Phase 0 — Setup (semaine 1-2)
- [ ] Créer wallet MetaMask dédié
- [ ] Déposer $100 USDC test via MoonPay SEPA
- [ ] Setup GCP e2-micro + projet Python (uv, ruff, ty)
- [ ] Implémenter connexion Polymarket (py-clob-client + Gamma API)
- [ ] Implémenter connexion Grok API (xai-sdk)
- [ ] Premier agent : News Detector avec `x_search`

### Phase 1 — Paper Trading (semaine 3-6)
- [ ] Construire paper trading engine (simule trades sans argent réel)
- [ ] Implémenter les 4 agents MVP
- [ ] Système de scoring/calibration (Brier score par agent)
- [ ] Backtester sur données historiques Polymarket
- [ ] Dashboard monitoring (PnL paper, signaux, décisions)
- [ ] Objectif : 200+ signal opportunities, calibration correcte

### Phase 2 — Micro-live (semaine 7-10)
- [ ] Déposer $100-$250 USDC réel
- [ ] Un marché à la fois, une position à la fois
- [ ] Kill switch + circuit breakers actifs
- [ ] Réconciliation exchange toutes les 30s
- [ ] Objectif : 30-50 trades réels, slippage réel vs modèle, aucun bug critique

### Phase 3 — Small-live (semaine 11-16)
- [ ] Augmenter à $500-$1000 si Phase 2 profitable
- [ ] Multi-marchés, multi-positions
- [ ] Ajouter agents Phase 2 (Scanner, Market Structure, Resolution Monitor)
- [ ] Arbitrage scanner en background
- [ ] Objectif : ROI positif après coûts sur 4+ semaines

### Phase 4 — Scaling (mois 4+)
- [ ] Augmenter capital de 25-50% par mois si résultats stables
- [ ] Migration AWS ECS Fargate
- [ ] PostgreSQL, monitoring Prometheus/Grafana
- [ ] Agents Phase 3 (Sentiment, EU/World Analyst)
- [ ] Wallet Gnosis Safe + GCP KMS

### Critères de go/no-go entre phases
- Positive expectancy après fees
- Max drawdown < 10-15%
- Slippage réalisé ≈ modèle
- Aucun bug d'exécution non-résolu
- Fill ratio stable
- Pas de cluster corrélé > 20% du bankroll

---

## 13. Risques et mitigations

| Risque | Probabilité | Impact | Mitigation |
|--------|------------|--------|------------|
| Bug d'exécution (mauvais côté, taille incorrecte) | Moyen | Très élevé | Tests extensifs, paper trading d'abord, kill switch |
| Perte de capital (mauvaises prédictions) | Élevé | Élevé | Fractional Kelly, drawdown limits, diversification |
| Polymarket shutdown/restriction | Faible | Très élevé | Non-custodial (USDC safe), monitoring réglementaire |
| Grok API down/changement prix | Faible | Moyen | Fallback mode (pas de nouveaux trades), multi-provider ready |
| Liquidity trap (impossible de sortir) | Moyen | Élevé | Pre-trade liquidity check, position ≤ 10% depth |
| Résolution ambiguë/disputée | Moyen | Moyen | Resolution Monitor agent, éviter marchés ambigus |
| Régulation irlandaise change | Faible | Élevé | Monitoring GRAI, consultation légale |
| Slippage supérieur au modèle | Moyen | Moyen | Limit orders, position sizing conservateur |
| Concurrent plus rapide | Élevé | Moyen | Focus niches, pas les marchés les plus liquides/efficients |
| Knight Capital scenario (bug catastrophique) | Très faible | Catastrophique | Kill switch automatique, max 2% daily loss |

---

## 14. Références clés

### Papers académiques
- [The Anatomy of Polymarket (2026)](https://arxiv.org/html/2603.03136v1) — microstructure et arb
- [Political Shocks and Price Discovery (2026)](https://arxiv.org/html/2603.03152v1) — réaction aux chocs politiques
- [Unravelling the Probabilistic Forest (2025)](https://arxiv.org/html/2508.03474v1) — $39.6M d'arb, méthodes
- [Application of Kelly Criterion to Prediction Markets](https://arxiv.org/abs/2412.14144)
- [LLM Forecasting (Science Advances)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11800985/) — LLMs vs humains
- [Metaculus FutureEval Benchmark](https://arxiv.org/abs/2507.04562v1)
- [Extremizing in Aggregation (Baron/Ungar)](https://pubsonline.informs.org/doi/10.1287/deca.2014.0293)

### Documentation officielle
- [Polymarket API](https://docs.polymarket.com/)
- [xAI/Grok API](https://docs.x.ai/developers/models)
- [py-clob-client](https://github.com/Polymarket/py-clob-client)
- [Polymarket/agents (référence)](https://github.com/Polymarket/agents)

### Repos utiles
- [poly-maker](https://github.com/warproxxx/poly-maker) — market maker (927 stars)
- [PredictionMarketBench](https://github.com/Oddpool/PredictionMarketBench) — backtesting
- [TradingAgents](https://github.com/TauricResearch/TradingAgents) — multi-agent framework

### Expérience Defend Intelligence
- [Vidéo "J'ai codé des bots de trading"](https://www.youtube.com/watch?v=3tlhWmNTXKs) — Grok seul AI profitable (accès X)
- 6 LLMs testés avec ~$1,100 chacun sur le marché boursier US
- Validation clé : Grok + X/Twitter = edge réel confirmé en conditions live
