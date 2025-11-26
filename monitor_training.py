#!/usr/bin/env python3
"""Monitor training progress by checking Tinker service and local files."""
import asyncio
import tinker
from dotenv import load_dotenv
from datetime import datetime, timezone
import time
import sys

load_dotenv()

async def monitor():
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    
    print("Monitoring training status... (Ctrl+C to stop)")
    print("=" * 60)
    
    while True:
        try:
            runs_response = await rest_client.list_training_runs_async()
            
            if not runs_response.training_runs:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No training runs found")
            else:
                for run in runs_response.training_runs:
                    now = datetime.now(timezone.utc)
                    time_diff = (now - run.last_request_time).total_seconds()
                    status = "🟢 ACTIVE" if time_diff < 60 else "🔴 IDLE"
                    
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {status} | "
                          f"Last request: {time_diff:.0f}s ago | "
                          f"Checkpoints: {run.last_checkpoint or 'None'}", 
                          end="", flush=True)
            
            await asyncio.sleep(5)
            
        except KeyboardInterrupt:
            print("\n\nStopping monitor...")
            break
        except Exception as e:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(monitor())
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        sys.exit(0)
