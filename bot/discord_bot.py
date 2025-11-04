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
STANDARD_CHANNEL_ID = int(os.getenv('STANDARD_CHANNEL_ID'))
ELITE_CHANNEL_ID = int(os.getenv('ELITE_CHANNEL_ID'))

# Signal files for each tier
STANDARD_SIGNALS_FILE = Path('models/standard/signals.jsonl')
ELITE_SIGNALS_FILE = Path('models/elite/signals.jsonl')

# Discord client setup
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Track last signals posted
last_standard_signal_id = None
last_elite_signal_id = None

async def post_signal(channel_id, signal, tier_name, tier_color):
    """Post a signal to a specific channel"""
    try:
        channel = client.get_channel(channel_id)
        if not channel:
            print(f"⚠️  {tier_name} channel not found (ID: {channel_id})")
            return False
        
        # Format message
        direction_emoji = "🟢" if signal["direction"] == "long" else "🔴"
        confidence_pct = signal.get('confidence', 0) * 100
        
        # Create embed for prettier display
        embed = discord.Embed(
            title=f"{direction_emoji} {signal['symbol']} {signal['direction'].upper()} SIGNAL",
            color=tier_color,
            timestamp=datetime.fromisoformat(signal['ts'].replace('Z', '+00:00'))
        )
        
        embed.add_field(name="Direction", value=signal['direction'].upper(), inline=True)
        embed.add_field(name="Confidence", value=f"{confidence_pct:.0f}%", inline=True)
        embed.add_field(name="Entry Price", value=f"${signal['price']:,.2f}", inline=True)
        
        embed.set_footer(text=f"MoonWire {tier_name} • ML-Validated Signal")
        
        await channel.send(embed=embed)
        print(f"✅ Posted {signal['symbol']} to #{tier_name.lower()}-signals")
        return True
        
    except Exception as e:
        print(f"❌ Error posting to {tier_name}: {e}")
        return False

@tasks.loop(minutes=5)
async def check_for_signals():
    """Check for new signals every 5 minutes"""
    global last_standard_signal_id, last_elite_signal_id
    
    # Check Standard tier signals (270d model)
    if STANDARD_SIGNALS_FILE.exists():
        try:
            with open(STANDARD_SIGNALS_FILE, 'r') as f:
                lines = f.readlines()
                if lines:
                    signal = json.loads(lines[-1])
                    if signal.get("id") != last_standard_signal_id:
                        last_standard_signal_id = signal["id"]
                        print(f"\n🆕 New STANDARD signal: {signal['symbol']}")
                        await post_signal(STANDARD_CHANNEL_ID, signal, "Standard", 0x10b981)  # Green
        except Exception as e:
            print(f"❌ Error reading standard signals: {e}")
    
    # Check Elite tier signals (365d model)
    if ELITE_SIGNALS_FILE.exists():
        try:
            with open(ELITE_SIGNALS_FILE, 'r') as f:
                lines = f.readlines()
                if lines:
                    signal = json.loads(lines[-1])
                    if signal.get("id") != last_elite_signal_id:
                        last_elite_signal_id = signal["id"]
                        print(f"\n🆕 New ELITE signal: {signal['symbol']}")
                        await post_signal(ELITE_CHANNEL_ID, signal, "Elite", 0x3b82f6)  # Blue
        except Exception as e:
            print(f"❌ Error reading elite signals: {e}")

@client.event
async def on_ready():
    """Bot startup"""
    print(f'\n{"="*60}')
    print(f'🤖 MoonWire Bot Online!')
    print(f'📡 Logged in as: {client.user}')
    print(f'🔍 Monitoring signals every 5 minutes...')
    print(f'📂 Standard signals: {STANDARD_SIGNALS_FILE.absolute()}')
    print(f'📂 Elite signals: {ELITE_SIGNALS_FILE.absolute()}')
    print(f'{"="*60}\n')
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
