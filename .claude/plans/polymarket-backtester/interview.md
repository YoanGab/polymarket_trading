# Interview Summary — Polymarket Backtester

## Mode & Interaction
- **Execution mode**: Normal (interactif)
- **Interaction**: Terminal (AskUserQuestion)

## Scope
- **In scope**: Backtest historique + validation Metaculus + CLI
- **Out of scope**: Live trading, paper trading temps réel, UI web

## Data
- **Primary**: warproxxx/poly_data (2 GiB, trades pré-traités)
- **Fallback**: Goldsky subgraph (1.3M fills depuis 2022)
- **Metadata**: Gamma API (marchés résolus, règles, résolutions)
- **Signal**: Grok x_search avec dates bornées (quand clé API disponible)

## API Keys
- **xAI/Grok**: pas encore de clé — commencer avec transport déterministe
- **Metaculus**: à créer pour validation

## Capital
- **Backtest simulé**: $1,000 (reflète le capital réel initial prévu)
- **Capital réel prévu si validé**: $1,000-5,000

## Marchés
- **Pas de restriction de domaine** — tester toutes les catégories (sauf sports/crypto)
- Tester différentes stratégies sur différents domaines

## Code
- **Partir de l'existant** dans `src/polymarket_backtest/`
- Ajouter : data pipeline, CLI, transport Grok réel, Metaculus validation

## Interface
- **CLI simple** avec output markdown dans le terminal
- `uv run backtest --strategy carry --markets politics`
