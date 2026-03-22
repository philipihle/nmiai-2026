"""
Auto-improvement script: reads recent Cloud Run logs, sends to Gemini,
gets suggested system prompt improvements, optionally deploys.
Usage: python improve.py [--minutes 30] [--auto-deploy]
"""
import subprocess
import sys
import os
import argparse
from google import genai
from google.genai import types

PROJECT = "ai-nm26osl-1847"
SERVICE = "tripletex-agent"
SYSTEM_PROMPT_PATH = "app/prompts/system_prompt.py"
GEMINI_API_KEY = "REMOVED"


def fetch_logs(minutes: int) -> str:
    filter_str = f'resource.type=cloud_run_revision AND resource.labels.service_name={SERVICE}'
    gcloud = "gcloud.cmd" if sys.platform == "win32" else "gcloud"
    cmd = [
        gcloud, "logging", "read", filter_str,
        f"--project={PROJECT}",
        "--limit=1000",
        "--format=value(textPayload)",
        f"--freshness={minutes}m",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.stdout or result.stderr


def filter_logs(logs: str) -> str:
    """Keep only error lines and key tool calls to reduce payload size."""
    keywords = ["422", "404", "500", "error", "Error", "ERROR", "Received task",
                "create_voucher sending", "create_voucher response", "systemgenererte",
                "validationMessages", "tool_use", "tool_result", "FAIL", "fail",
                "Traceback", "Exception", "task_complete"]
    lines = logs.splitlines()
    filtered = [l for l in lines if any(kw in l for kw in keywords)]
    result = "\n".join(filtered)
    # Cap at 8000 chars
    if len(result) > 8000:
        result = result[:8000]
    return result


def read_current_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH) as f:
        return f.read()


def analyze_with_gemini(logs: str, current_prompt: str) -> str:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY))

    analysis_prompt = f"""You are analyzing logs from a Tripletex accounting AI agent to improve its system prompt.

CURRENT SYSTEM PROMPT:
{current_prompt}

RECENT LOGS (errors, tool calls, results):
{logs[:8000]}

Tasks:
1. Identify all errors, failures, wrong field names, wrong endpoints, wrong account numbers
2. Find patterns  what types of tasks keep failing
3. Suggest SPECIFIC changes to the system prompt that would fix these issues
4. Return the COMPLETE updated system_prompt.py file content (the entire Python file with SYSTEM_PROMPT = \"\"\"...\"\"\")

Focus on:
- Wrong API endpoint names (e.g. /:pay vs /:payment)
- Wrong field names (e.g. category vs costCategory)
- Wrong account numbers used in vouchers
- Missing required fields
- Tasks where the agent gave up or timed out

Return ONLY the complete updated Python file content, nothing else."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=analysis_prompt)])],
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return response.text


def deploy():
    cmd = [
        "gcloud", "run", "deploy", SERVICE,
        "--source", ".",
        "--region", "europe-west1",
        f"--project={PROJECT}",
        "--allow-unauthenticated",
        "--memory=2Gi",
        "--cpu=2",
        "--timeout=3600",
        "--concurrency=10",
        f"--set-env-vars=GEMINI_API_KEY={os.environ.get('GEMINI_API_KEY', GEMINI_API_KEY)}",
    ]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=int, default=30, help="How many minutes of logs to analyze")
    parser.add_argument("--auto-deploy", action="store_true", help="Deploy automatically without confirmation")
    args = parser.parse_args()

    print(f" Fetching logs from last {args.minutes} minutes...")
    logs = fetch_logs(args.minutes)
    if not logs.strip():
        print("No logs found. Try increasing --minutes.")
        sys.exit(1)

    error_count = logs.count("422") + logs.count("404") + logs.count("500")
    print(f"   Found {len(logs)} chars of logs, ~{error_count} errors")

    filtered = filter_logs(logs)
    print(f"   Filtered to {len(filtered)} chars for analysis")
    # If filtering removed everything, use raw logs
    analysis_logs = filtered if len(filtered) > 50 else logs[:8000]

    print(" Analyzing with Gemini...")
    current_prompt = read_current_prompt()
    new_content = analyze_with_gemini(analysis_logs, current_prompt)

    # Clean up response if wrapped in markdown
    if new_content.startswith("```"):
        lines = new_content.split("\n")
        new_content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    print("\n" + "="*60)
    print("SUGGESTED CHANGES:")
    print("="*60)
    # Show diff summary
    old_lines = set(current_prompt.split("\n"))
    new_lines = set(new_content.split("\n"))
    added = [l for l in new_lines - old_lines if l.strip()]
    removed = [l for l in old_lines - new_lines if l.strip()]
    print(f"Lines added: {len(added)}, Lines removed: {len(removed)}")
    for l in added[:10]:
        print(("  + " + l[:100]).encode("ascii", "replace").decode())
    for l in removed[:5]:
        print(("  - " + l[:100]).encode("ascii", "replace").decode())
    print("="*60)

    if args.auto_deploy:
        confirm = "y"
    else:
        confirm = input("\nApply changes and deploy [y/N] ").strip().lower()

    if confirm == "y":
        print(" Writing new system prompt...")
        with open(SYSTEM_PROMPT_PATH, "w") as f:
            f.write(new_content)
        print(" Deploying...")
        deploy()
        print(" Done!")
    else:
        # Save to file for review
        with open("suggested_prompt.py", "w") as f:
            f.write(new_content)
        print(" Saved to suggested_prompt.py for review")


if __name__ == "__main__":
    main()
