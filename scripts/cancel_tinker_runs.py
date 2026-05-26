#!/usr/bin/env python3
"""
Script to list and cancel active Tinker training runs.

Usage:
    # List all training runs
    python scripts/cancel_tinker_runs.py --list
    
    # Cancel all running sessions
    python scripts/cancel_tinker_runs.py --cancel-all
    
    # Cancel specific run by ID
    python scripts/cancel_tinker_runs.py --cancel-run <run_id>
"""

import argparse
import asyncio
from typing import List, Optional
import tinker


async def list_training_runs() -> List:
    """List all training runs."""
    print("\n=== Fetching Training Runs ===")
    
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    
    # List all training runs
    response = await rest_client.list_training_runs_async()
    
    # Extract training_runs from response
    runs = response.training_runs if hasattr(response, 'training_runs') else []
    
    if not runs:
        print("No training runs found.")
        return []
    
    print(f"\nFound {len(runs)} training run(s):")
    print(f"Total in system: {response.cursor.total_count if hasattr(response, 'cursor') else 'unknown'}")
    print("=" * 100)
    
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    
    for i, run in enumerate(runs, 1):
        # Calculate time since last request
        time_diff = None
        status = "UNKNOWN"
        if run.last_request_time:
            time_diff = now - run.last_request_time
            hours_ago = time_diff.total_seconds() / 3600
            
            if hours_ago < 0.5:  # Less than 30 minutes
                status = "🟢 ACTIVE (< 30 min ago)"
            elif hours_ago < 24:
                status = f"🟡 RECENT (~{int(hours_ago)}h ago)"
            else:
                days_ago = int(hours_ago / 24)
                status = f"🔵 IDLE ({days_ago}d ago)"
        
        print(f"\n{i}. Training Run [{status}]")
        print(f"   ID: {run.training_run_id}")
        print(f"   Base Model: {run.base_model}")
        print(f"   LoRA: {run.is_lora} (rank={run.lora_rank if run.is_lora else 'N/A'})")
        print(f"   Last Request: {run.last_request_time}")
        print(f"   Corrupted: {run.corrupted}")
        
        if run.last_checkpoint:
            print(f"   Last Checkpoint: {run.last_checkpoint.checkpoint_id}")
            print(f"   Checkpoint Path: {run.last_checkpoint.tinker_path}")
        else:
            print(f"   Last Checkpoint: None")
        
        if run.last_sampler_checkpoint:
            print(f"   Last Sampler: {run.last_sampler_checkpoint.checkpoint_id}")
        else:
            print(f"   Last Sampler: None")
    
    print("=" * 100)
    
    # Count active runs
    active_runs = [r for r in runs if r.last_request_time and (now - r.last_request_time).total_seconds() < 1800]  # 30 min
    if active_runs:
        print(f"\n⚠️  {len(active_runs)} ACTIVE run(s) detected (last request < 30 min ago)")
        print("   To stop training, press Ctrl+C in the training script terminal")
    
    return runs


async def list_sessions() -> List:
    """List all active sessions (which may have associated training runs)."""
    print("\n=== Fetching Active Sessions ===")
    
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    
    # List all sessions
    sessions = await rest_client.list_sessions_async()
    
    if not sessions:
        print("No active sessions found.")
        return []
    
    print(f"\nFound {len(sessions)} active session(s):")
    print("-" * 80)
    
    for i, session in enumerate(sessions, 1):
        print(f"\n{i}. Session:")
        print(f"   ID: {session.id if hasattr(session, 'id') else 'N/A'}")
        # Print all available attributes
        for attr in dir(session):
            if not attr.startswith('_') and not callable(getattr(session, attr)):
                try:
                    value = getattr(session, attr)
                    print(f"   {attr}: {value}")
                except Exception:
                    pass
    
    print("-" * 80)
    return sessions


async def get_run_details(run_id: str):
    """Get detailed information about a specific training run."""
    print(f"\n=== Fetching Details for Run: {run_id} ===")
    
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    
    try:
        run = await rest_client.get_training_run_async(run_id)
        print(f"\nTraining Run Details:")
        print(f"   ID: {run.id if hasattr(run, 'id') else 'N/A'}")
        # Print all attributes
        for attr in dir(run):
            if not attr.startswith('_') and not callable(getattr(run, attr)):
                try:
                    value = getattr(run, attr)
                    print(f"   {attr}: {value}")
                except Exception:
                    pass
        return run
    except Exception as e:
        print(f"Error fetching run details: {e}")
        return None


async def cancel_session(session_id: str):
    """
    Cancel a session. Note: The RestClient API doesn't have a direct
    session cancellation method exposed, but we can try to delete
    associated checkpoints to clean up resources.
    """
    print(f"\n=== Attempting to Clean Up Session: {session_id} ===")
    
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    
    try:
        # Get session details first
        session = await rest_client.get_session_async(session_id)
        print(f"Session found: {session}")
        
        # Try to list and delete associated checkpoints
        checkpoints = await rest_client.list_checkpoints_async(session_id)
        print(f"Found {len(checkpoints)} checkpoint(s) for this session")
        
        for checkpoint in checkpoints:
            print(f"  - Checkpoint: {checkpoint}")
            # Note: Be careful with deletion - this is destructive!
            # Uncomment the line below only if you're sure you want to delete checkpoints
            # await rest_client.delete_checkpoint_async(checkpoint.id)
        
        print("\n⚠️  WARNING: To fully stop training, you may need to:")
        print("   1. Stop the training script (Ctrl+C)")
        print("   2. Delete checkpoints manually if needed")
        print("   3. Contact Tinker support for session termination")
        
    except Exception as e:
        print(f"Error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Manage Tinker training runs")
    parser.add_argument("--list", action="store_true", help="List all training runs")
    parser.add_argument("--list-sessions", action="store_true", help="List all active sessions")
    parser.add_argument("--run-details", type=str, metavar="RUN_ID", help="Get details for specific run")
    parser.add_argument("--cancel-session", type=str, metavar="SESSION_ID", help="Cancel a specific session")
    parser.add_argument("--cancel-all", action="store_true", help="Show info about all active runs/sessions")
    
    args = parser.parse_args()
    
    if args.list or args.cancel_all or (not any(vars(args).values())):
        # Default: list all training runs
        await list_training_runs()
    
    if args.list_sessions or args.cancel_all:
        await list_sessions()
    
    if args.run_details:
        await get_run_details(args.run_details)
    
    if args.cancel_session:
        await cancel_session(args.cancel_session)
    
    if args.cancel_all:
        print("\n" + "=" * 80)
        print("MANUAL CANCELLATION REQUIRED")
        print("=" * 80)
        print("To stop active training:")
        print("1. Press Ctrl+C in the terminal where training is running")
        print("2. The Tinker API will handle cleanup automatically")
        print("3. If needed, contact Tinker support for session termination")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
