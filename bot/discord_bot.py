import discord
from discord.ext import tasks
import json
import asyncio
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from .env
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
FREE_CHANNEL_ID = int(os.getenv('FREE_CHANNEL_ID'))
PREMIUM_CHANNEL_ID = int(os.getenv('PREMIUM_CHANNEL_ID'))
SIGNALS_FILE = Path(os.getenv('SIGNALS_FILE', 'logs/signal_history.jsonl'))

# Discord client setup
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Track last signal posted
last_signal_id = None

@tasks.loop(minutes=5)
async def check_for_signals():
    """Check for new signals every 5 minutes"""
    global last_signal_id
    
    if not SIGNALS_FILE.exists():
        print("⚠️  No signals file found at:", SIGNALS_FILE)
        return
    
    try:
        # Read latest signal
        with open(SIGNALS_FILE, 'r') as f:
            lines = f.readlines()
            if not lines:
                print("📭 No signals yet")
                return
            signal = json.loads(lines[-1])
    except Exception as e:
        print(f"❌ Error reading signals: {e}")
        return
    
    # Skip if already posted
    if signal.get("id") == last_signal_id:
        return
    
    last_signal_id = signal["id"]
    print(f"\n🆕 New signal detected: {signal['symbol']}")
    
    # Format message
    direction_emoji = "🟢" if signal["direction"] == "long" else "🔴"
    confidence_pct = signal['confidence'] * 100
    
    msg = f"""{direction_emoji} **{signal['symbol']} SIGNAL**

**Direction:** {signal['direction'].upper()}
**Confidence:** {confidence_pct:.0f}%
**Entry Price:** ${signal['price']:,.2f}
**Timestamp:** {signal['ts']}

━━━━━━━━━━━━━━━━━
*MoonWire ML Signal Detection*
"""
    
    # Post to premium immediately
    try:
        premium_channel = client.get_channel(PREMIUM_CHANNEL_ID)
        if premium_channel:
            await premium_channel.send(msg)
            print(f"✅ Posted to #premium-signals")
        else:
            print(f"⚠️  Premium channel not found (ID: {PREMIUM_CHANNEL_ID})")
    except Exception as e:
        print(f"❌ Error posting to premium: {e}")
    
    # Wait 15 minutes, then post to free
    print("⏳ Waiting 15 minutes before posting to free...")
    await asyncio.sleep(900)  # 900 seconds = 15 minutes
    
    try:
        free_channel = client.get_channel(FREE_CHANNEL_ID)
        if free_channel:
            free_msg = msg + "\n⏰ *Delayed 15 minutes • [Upgrade for real-time](https://moonwire.app/pricing)*"
            await free_channel.send(free_msg)
            print(f"✅ Posted to #free-signals (delayed)")
        else:
            print(f"⚠️  Free channel not found (ID: {FREE_CHANNEL_ID})")
    except Exception as e:
        print(f"❌ Error posting to free: {e}")

@client.event
async def on_ready():
    """Bot startup"""
    print(f'\n{"="*50}')
    print(f'🤖 MoonWire Bot Online!')
    print(f'📡 Logged in as: {client.user}')
    print(f'🔍 Monitoring signals every 5 minutes...')
    print(f'📂 Signals file: {SIGNALS_FILE.absolute()}')
    print(f'{"="*50}\n')
    check_for_signals.start()

@client.event
async def on_error(event, *args, **kwargs):
    """Error handler"""
    print(f"❌ Error in {event}")

if __name__ == "__main__":
    print("🚀 Starting MoonWire Discord Bot...")
    if not TOKEN:
        print("❌ ERROR: DISCORD_BOT_TOKEN not found in .env file!")
        exit(1)
    try:
        client.run(TOKEN)
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
