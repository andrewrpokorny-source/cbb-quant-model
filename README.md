# CBB Quant Model

College basketball spread prediction model using machine learning.

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

### Generate Predictions
```bash
uv run python predict.py
```

### Run Dashboard
```bash
uv run streamlit run app.py
```

### Telegram Bot

The Telegram bot lets you log bets by sending screenshots or shorthand text, settle bets against ESPN scores, and view your record -- all from Telegram.

#### Creating the Bot

1. Open Telegram and search for **@BotFather**.
2. Send `/newbot` and follow the prompts to choose a name and username.
3. BotFather will reply with a token (e.g. `123456:ABC-DEF...`). Copy it.
4. Add the token to your `.env` file:
   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   ```

#### Running the Bot

```bash
uv run python telegram_bot.py
```

The bot runs in polling mode and will print `Bot started. Send /start in Telegram to begin.` when ready.

**Note:** Screenshot parsing uses macOS native OCR (`ocrmac`), so the bot must run on a Mac for image-based bet logging. Text-based logging works on any platform.

#### Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Show available commands |
| `/help` | Same as /start |
| `/today` | Show today's model picks (STRONG/GOOD rated) |
| `/pending` | List all unsettled bets |
| `/settle` | Settle pending bets against ESPN final scores |
| `/record` | Show W-L record, ROI, and last 7 days |

#### Logging Bets

**Via screenshot:** Send a photo of a bet slip from DraftKings, FanDuel, Kalshi, BetMGM, or Caesars. The bot will OCR-parse and log it automatically.

**Via text shorthand:** Send a message in the format:
```
PLATFORM TEAM SPREAD ODDS WAGER
```
For example:
```
FD Providence +15.5 -110 1.25
```
Platform aliases: `FD` (FanDuel), `DK` (DraftKings), `K` or `KAL` (Kalshi).
