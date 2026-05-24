# CBB Quant Model

Sports betting prediction model using machine learning. Covers college basketball (men's + women's spreads and game-winner markets) and MLB moneyline.

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Copy env template (optional, for Kalshi integration)
cp .env.example .env
```

## Usage

### Generate Predictions (Men's default)
```bash
uv run python predict.py
```
Predictions now include both spread picks and Kalshi game-market picks when available.

### Run Dashboard
```bash
uv run streamlit run app.py
```
Use the in-app league selector to switch between Men's CBB, Women's CBB, and MLB.
Use the in-app **Jump to section** control to quickly navigate directly to:
- Spread Value Bets
- Kalshi Game Markets
- Spread Slate
- Betting Log
- Performance History
- System

### Run Dashboard + Telegram Bot Together
```bash
./scripts/run-stack.sh
```
Starts both services as background process groups (Streamlit logs to `streamlit.log`, the bot logs to `telegram_bot.log`). Cleanup runs on Ctrl-C, SIGTERM, or normal exit, so neither service is left orphaned. If Streamlit crashes immediately, the last 20 lines of `streamlit.log` are printed before the script aborts.

### Multi-League Commands

All major scripts support `--league` with `mens` (default), `womens`, or `mlb`.

```bash
# Update data + run features/backtest for women's CBB
uv run python main.py --league womens

# Train women's spread model
uv run python model.py --league womens

# Train women's game-winner P(win) model bundle
uv run python model_win.py --league womens

# Generate women's predictions
uv run python predict.py --league womens

# Backtest women's model
uv run python backtest.py --league womens

# Grade yesterday's women's predictions
uv run python grade_predictions.py --league womens
```

### MLB Predictions

```bash
# Generate MLB moneyline predictions (with Kalshi + DK edge)
uv run python mlb/predict.py

# Backtest MLB model
uv run python backtest.py --league mlb
```

The MLB model is a GBM classifier (28 features) predicting P(home win) for moneyline bets. Features include starting pitcher rolling stats, team Pythagorean win%, weather, ballpark factors, and bullpen ERA differentials. Kalshi edges are capped at 15% with CBB-style rating gates (model prob + price range).

### Data Migration

After pulling, run once to move any existing generated files (betting history, predictions, etc.) to their new locations:

```bash
uv run python migrate_data.py
```

### Model Artifacts

- Men's canonical model: `models/cbb_model_v2.pkl`
- Women's canonical model: `models/womens_cbb_spread_model_v2.pkl`
- Men's game-winner model bundle: `models/cbb_win_model_v1.pkl`
- Women's game-winner model bundle: `models/womens_cbb_win_model_v1.pkl`
- MLB moneyline model: `models/mlb_win_model_v1.pkl`

### Neutral-Site and Venue Features

The spread model includes neutral-site detection and travel distance monitoring:
- `is_neutral` flag from ESPN's `neutralSite` field (in FEATURES, used by model)
- `distance_advantage` computed from geocoded team/venue locations (stored in CSV for monitoring, not yet in FEATURES)
- `venue.py` handles geocoding with Nominatim + state-centroid fallback
- `venue_geocode.json` ships 357 pre-geocoded locations for offline use

### Kalshi GAME Markets (P(win))

`predict.py` now evaluates Kalshi GAME contracts with the P(win) model bundle:
- Uses `model_win.py` bundle (`no_line` / `with_line` variants) to estimate `P(home wins)`.
- Finds matching Kalshi `GAME` markets and chooses the better side (`YES` or `NO`) by edge.
- Writes combined output to the daily predictions CSV with a `Bet_Type` column:
  - `spread` for spread picks
  - `game` for Kalshi game-market picks

### Telegram Bot

The Telegram bot is a local bet logger and settlement assistant. It can:
- log bets from screenshots
- log bets from shorthand text messages
- parse DraftKings and Kalshi share links
- settle pending bets against scoreboard data and imported Kalshi settlements
- show today's model picks, pending bets, and record/ROI

#### Prerequisites

- Run `uv sync` first so the Telegram and OCR dependencies are installed.
- The current bot process should be run on macOS because it imports `ocrmac` at startup.
- Screenshot parsing uses macOS native OCR via `ocrmac`.
- Text shorthand entry and share-link logging are supported workflows, but they still use the same bot process.
- `/today` reads `data/daily_predictions.csv` (CBB), so run `uv run python predict.py` first if you want live model picks in Telegram. MLB predictions are generated separately via `uv run python mlb/predict.py`.

#### Environment Configuration

Copy the template if you have not already:

```bash
cp .env.example .env
```

Required and relevant bot settings:

```dotenv
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_ALLOWED_USERS=123456789
BET_PARSE_AUDIT_FILE=./telegram_parse_audit.jsonl
```

- `TELEGRAM_BOT_TOKEN` is required. Create the bot with `@BotFather`.
- `TELEGRAM_ALLOWED_USERS` is strongly recommended. This is a comma-separated list of Telegram user IDs allowed to use the bot. If left empty, anyone who can reach the bot can interact with it.
- `BET_PARSE_AUDIT_FILE` is optional. If unset, the bot writes parse-audit events to `telegram_parse_audit.jsonl` in the repo root.

If you use Kalshi settlement import through `/settle`, also configure the Kalshi credentials already shown in `.env.example`:

```dotenv
KALSHI_API_KEY=your_api_key_here
KALSHI_PRIVATE_KEY_PATH=./kalshi_private_key.pem
```

#### Create the Telegram Bot

1. Open Telegram and search for `@BotFather`.
2. Send `/newbot` and follow the prompts.
3. Copy the bot token BotFather returns.
4. Add that token to `.env` as `TELEGRAM_BOT_TOKEN=...`.
5. Get your personal Telegram user ID from a helper bot such as `@userinfobot`, then add it to `TELEGRAM_ALLOWED_USERS`.

#### Run the Bot

```bash
uv run python telegram_bot.py
```

The bot runs in polling mode. A successful startup prints:

```text
Bot started. Send /start in Telegram to begin.
```

The process also creates a lock file at `.telegram_bot.lock` so you do not accidentally run multiple bot instances at once.

#### Commands

| Command | Description |
|---------|-------------|
| `/start` | Show the quick-start help text |
| `/help` | Alias for `/start` |
| `/today` | Show today's value bets from `data/daily_predictions.csv` |
| `/pending` | List all unsettled bets in `data/betting_history.csv` |
| `/settle` | Settle pending bets and import recent Kalshi settlements |
| `/record` | Show all-time record, profit, ROI, last 7 days, and pending count |
| `/delete N` | Delete the `N`th pending bet |

#### Supported Input Workflows

**1. Screenshot logging**

Send a photo of a bet slip. The bot attempts OCR-based parsing for books such as FanDuel, DraftKings, Kalshi, BetMGM, and Caesars. If a screenshot contains multiple bets, the bot responds with a numbered list and you can reply with `all` or selected numbers like `1 3`.

If OCR cannot fully parse a bet, the bot will either show the raw OCR text or ask for a manual follow-up.

**2. Text shorthand logging**

Send a plain-text message in this format:

```text
PLATFORM TEAM SPREAD ODDS WAGER
```

Example:

```text
FD Providence +15.5 -110 1.25
```

Common platform aliases include:
- `FD` for FanDuel
- `DK` for DraftKings
- `K` or `KAL` for Kalshi

**3. Share-link logging**

Paste a supported share URL directly into chat:
- DraftKings social bet-share links
- Kalshi market links

The bot will parse the URL, extract the bet, and log or settle it when possible.

#### What the Bot Reads and Writes

The bot works out of the repository root and uses these local files:

- `data/betting_history.csv`: primary ledger of logged bets
- `data/daily_predictions.csv`: source for `/today`
- `data/performance_log.csv`: performance data used elsewhere in the project
- `telegram_bot.log`: rotating bot log output
- `telegram_parse_audit.jsonl`: parse audit trail unless overridden by `BET_PARSE_AUDIT_FILE`
- `screenshots/`: stored screenshot artifacts when parsing needs them

#### Typical Usage

1. Start the bot with `uv run python telegram_bot.py`.
2. Open Telegram and send `/start`.
3. Log bets by sending a screenshot, shorthand text, or a supported share URL.
4. Check `/pending` to review open bets.
5. Run `/settle` after games finish to grade pending bets and pull Kalshi settlements.
6. Run `/record` to review results and ROI.
