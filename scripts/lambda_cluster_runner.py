#!/usr/bin/env python3
"""
Lambda Labs Cluster Runner for Parallel Scenario Execution

Automatically launches Lambda Cloud instances, runs one scenario per GPU,
and aggregates results when complete.

Setup:
    1. Copy .env.example to .env and fill in your credentials:
       cp .env.example .env

    2. Edit .env with your keys:
       - LAMBDA_API_KEY from https://cloud.lambdalabs.com/api-keys
       - LAMBDA_SSH_KEY_NAME from https://cloud.lambdalabs.com/ssh-keys
       - OPENAI_API_KEY from https://platform.openai.com/api-keys
       - REPO_URL (your git repo)

Usage:
    # Run all scenarios in parallel (one per instance)
    python scripts/lambda_cluster_runner.py --scenarios all

    # Run specific scenarios
    python scripts/lambda_cluster_runner.py --scenarios ultimatum_bluff,alliance_betrayal

    # Dry run (show what would happen)
    python scripts/lambda_cluster_runner.py --scenarios all --dry-run

    # Use specific GPU type
    python scripts/lambda_cluster_runner.py --scenarios all --gpu-type gpu_1x_h100_pcie
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load .env file from project root
try:
    from dotenv import load_dotenv
    # Look for .env in project root (parent of scripts/)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
    else:
        # Also try current directory
        load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using system environment variables.")
    print("Install with: pip install python-dotenv\n")

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)

try:
    import paramiko
except ImportError:
    paramiko = None


# =============================================================================
# CONFIGURATION
# =============================================================================

LAMBDA_API_BASE = "https://cloud.lambdalabs.com/api/v1"

# All available scenarios
ALL_SCENARIOS = [
    "ultimatum_bluff",
    "alliance_betrayal",
    "capability_bluff",
    "info_withholding",
    "promise_break",
    "hidden_value",
]

# GPU types in order of preference (for Gemma 27B, need 80GB)
GPU_PREFERENCES = [
    "gpu_1x_h100_pcie",      # $2.49/hr - 80GB, best value
    "gpu_1x_h100_sxm",       # $3.29/hr - 80GB, faster
    "gpu_1x_gh200",          # $1.49/hr - 96GB, great if available
    "gpu_1x_a100_sxm",       # $1.79/hr - 80GB (8x node, but 1x might exist)
]

# Instance setup script (runs on each instance)
SETUP_SCRIPT = """#!/bin/bash
set -e

echo "=== Setting up instance ==="

# Clone repo if not exists
if [ ! -d "multiagent-emergent-deception" ]; then
    git clone {repo_url} multiagent-emergent-deception
fi

cd multiagent-emergent-deception

# Pull latest
git pull origin main

# Install dependencies
pip install -e . --quiet

# Set API keys
export OPENAI_API_KEY="{openai_key}"

echo "=== Setup complete ==="
"""

# Run script template
RUN_SCRIPT = """#!/bin/bash
set -e

cd ~/multiagent-emergent-deception

export OPENAI_API_KEY="{openai_key}"

echo "=== Running scenario: {scenario} ==="
echo "Started at: $(date)"

python interpretability/run_deception_experiment.py \
    --scenario-name {scenario} \
    --model {model} \
    --trials {trials} \
    --max-rounds {max_rounds} \
    --device cuda \
    --dtype bfloat16 \
    --output-dir ./results/{scenario}_{timestamp}

echo "=== Scenario complete ==="
echo "Finished at: $(date)"
"""


# =============================================================================
# LAMBDA API CLIENT
# =============================================================================

@dataclass
class LambdaInstance:
    """Represents a Lambda Cloud instance."""
    id: str
    name: str
    ip: Optional[str]
    status: str
    gpu_type: str
    scenario: Optional[str] = None


class LambdaClient:
    """Simple Lambda Cloud API client."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make API request."""
        url = f"{LAMBDA_API_BASE}/{endpoint}"
        response = requests.request(
            method, url, headers=self.headers, json=data, timeout=30
        )
        if response.status_code >= 400:
            raise Exception(f"API error {response.status_code}: {response.text}")
        return response.json()

    def list_instance_types(self) -> Dict[str, Any]:
        """List available instance types and their availability."""
        result = self._request("GET", "instance-types")
        return result.get("data", {})

    def get_available_gpus(self) -> List[str]:
        """Get list of currently available GPU types."""
        types = self.list_instance_types()
        available = []
        for gpu_type, info in types.items():
            regions = info.get("regions_with_capacity_available", [])
            if regions:
                available.append(gpu_type)
        return available

    def find_best_gpu(self, preferences: List[str] = None) -> Optional[str]:
        """Find best available GPU from preference list."""
        available = self.get_available_gpus()
        prefs = preferences or GPU_PREFERENCES

        for gpu in prefs:
            if gpu in available:
                return gpu

        # Return any available 80GB+ GPU
        for gpu in available:
            if "h100" in gpu or "a100" in gpu or "gh200" in gpu:
                return gpu

        return available[0] if available else None

    def launch_instance(
        self,
        gpu_type: str,
        ssh_key_name: str,
        name: str = None,
        region: str = None,
    ) -> LambdaInstance:
        """Launch a new instance."""
        # Get available region for this GPU type
        types = self.list_instance_types()
        gpu_info = types.get(gpu_type, {})
        regions = gpu_info.get("regions_with_capacity_available", [])

        if not regions:
            raise Exception(f"No capacity available for {gpu_type}")

        target_region = region or regions[0].get("name", regions[0])
        if isinstance(target_region, dict):
            target_region = target_region.get("name")

        data = {
            "instance_type_name": gpu_type,
            "region_name": target_region,
            "ssh_key_names": [ssh_key_name],
            "quantity": 1,
        }
        if name:
            data["name"] = name

        result = self._request("POST", "instance-operations/launch", data)
        instance_ids = result.get("data", {}).get("instance_ids", [])

        if not instance_ids:
            raise Exception(f"Failed to launch instance: {result}")

        return LambdaInstance(
            id=instance_ids[0],
            name=name or instance_ids[0],
            ip=None,
            status="launching",
            gpu_type=gpu_type,
        )

    def get_instance(self, instance_id: str) -> LambdaInstance:
        """Get instance details."""
        result = self._request("GET", f"instances/{instance_id}")
        data = result.get("data", {})
        return LambdaInstance(
            id=data.get("id"),
            name=data.get("name"),
            ip=data.get("ip"),
            status=data.get("status"),
            gpu_type=data.get("instance_type", {}).get("name"),
        )

    def wait_for_instance(
        self, instance_id: str, timeout: int = 300, poll_interval: int = 10
    ) -> LambdaInstance:
        """Wait for instance to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            instance = self.get_instance(instance_id)
            if instance.status == "active" and instance.ip:
                return instance
            if instance.status in ["terminated", "error"]:
                raise Exception(f"Instance {instance_id} failed: {instance.status}")
            print(f"  Waiting for {instance.name}... ({instance.status})")
            time.sleep(poll_interval)
        raise Exception(f"Timeout waiting for instance {instance_id}")

    def terminate_instance(self, instance_id: str) -> None:
        """Terminate an instance."""
        self._request(
            "POST",
            "instance-operations/terminate",
            {"instance_ids": [instance_id]},
        )

    def list_instances(self) -> List[LambdaInstance]:
        """List all instances."""
        result = self._request("GET", "instances")
        instances = []
        for data in result.get("data", []):
            instances.append(LambdaInstance(
                id=data.get("id"),
                name=data.get("name"),
                ip=data.get("ip"),
                status=data.get("status"),
                gpu_type=data.get("instance_type", {}).get("name"),
            ))
        return instances


# =============================================================================
# SSH EXECUTION
# =============================================================================

def run_ssh_command(ip: str, command: str, timeout: int = 3600) -> tuple:
    """Run command on remote instance via SSH."""
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=30",
        f"ubuntu@{ip}",
        command,
    ]

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def wait_for_ssh(ip: str, timeout: int = 300) -> bool:
    """Wait for SSH to become available."""
    start = time.time()
    while time.time() - start < timeout:
        code, _, _ = run_ssh_command(ip, "echo 'SSH ready'", timeout=10)
        if code == 0:
            return True
        time.sleep(10)
    return False


def upload_file(ip: str, local_path: str, remote_path: str) -> bool:
    """Upload file to remote instance."""
    scp_cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        local_path,
        f"ubuntu@{ip}:{remote_path}",
    ]
    result = subprocess.run(scp_cmd, capture_output=True, timeout=120)
    return result.returncode == 0


def download_file(ip: str, remote_path: str, local_path: str) -> bool:
    """Download file from remote instance."""
    scp_cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-r",  # Recursive for directories
        f"ubuntu@{ip}:{remote_path}",
        local_path,
    ]
    result = subprocess.run(scp_cmd, capture_output=True, timeout=300)
    return result.returncode == 0


# =============================================================================
# CLUSTER RUNNER
# =============================================================================

@dataclass
class ClusterConfig:
    """Configuration for cluster run."""
    scenarios: List[str]
    gpu_type: Optional[str] = None
    model: str = "google/gemma-2-27b-it"
    trials: int = 50
    max_rounds: int = 5
    repo_url: str = field(default_factory=lambda: os.environ.get(
        "REPO_URL", "https://github.com/your-username/multiagent-emergent-deception.git"
    ))
    output_dir: str = "./cluster_results"
    dry_run: bool = False


class ClusterRunner:
    """Orchestrates parallel scenario execution across Lambda instances."""

    def __init__(self, config: ClusterConfig):
        self.config = config
        self.api_key = os.environ.get("LAMBDA_API_KEY")
        self.ssh_key_name = os.environ.get("LAMBDA_SSH_KEY_NAME")
        self.openai_key = os.environ.get("OPENAI_API_KEY")

        # Only validate credentials if not dry run
        if not config.dry_run:
            if not self.api_key:
                raise ValueError("LAMBDA_API_KEY not set. Add it to your .env file.")
            if not self.ssh_key_name:
                raise ValueError("LAMBDA_SSH_KEY_NAME not set. Add it to your .env file.")
            if not self.openai_key:
                raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")
            self.client = LambdaClient(self.api_key)
        else:
            self.client = None

        self.instances: Dict[str, LambdaInstance] = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def check_availability(self) -> str:
        """Check GPU availability and return best option."""
        print("\n=== Checking GPU Availability ===")

        if self.config.dry_run:
            gpu = self.config.gpu_type or "gpu_1x_h100_pcie"
            print(f"[DRY RUN] Would use GPU: {gpu}")
            return gpu

        available = self.client.get_available_gpus()

        if not available:
            raise Exception("No GPUs currently available on Lambda Labs")

        print(f"Available GPU types: {available}")

        if self.config.gpu_type:
            if self.config.gpu_type in available:
                return self.config.gpu_type
            else:
                print(f"Requested {self.config.gpu_type} not available")

        best = self.client.find_best_gpu()
        print(f"Selected GPU: {best}")
        return best

    def launch_instances(self, gpu_type: str) -> None:
        """Launch one instance per scenario."""
        print(f"\n=== Launching {len(self.config.scenarios)} Instances ===")

        for scenario in self.config.scenarios:
            name = f"deception-{scenario}-{self.timestamp[:8]}"
            print(f"Launching {name} for scenario '{scenario}'...")

            if self.config.dry_run:
                print(f"  [DRY RUN] Would launch {gpu_type} as {name}")
                self.instances[scenario] = LambdaInstance(
                    id=f"dry-run-{scenario}",
                    name=name,
                    ip="0.0.0.0",
                    status="dry-run",
                    gpu_type=gpu_type,
                    scenario=scenario,
                )
                continue

            try:
                instance = self.client.launch_instance(
                    gpu_type=gpu_type,
                    ssh_key_name=self.ssh_key_name,
                    name=name,
                )
                instance.scenario = scenario
                self.instances[scenario] = instance
                print(f"  Launched: {instance.id}")
            except Exception as e:
                print(f"  Failed to launch: {e}")

    def wait_for_all_instances(self) -> None:
        """Wait for all instances to be ready."""
        if self.config.dry_run:
            return

        print("\n=== Waiting for Instances ===")

        for scenario, instance in self.instances.items():
            print(f"Waiting for {instance.name}...")
            try:
                updated = self.client.wait_for_instance(instance.id)
                self.instances[scenario] = updated
                self.instances[scenario].scenario = scenario
                print(f"  Ready: {updated.ip}")

                # Wait for SSH
                print(f"  Waiting for SSH...")
                if not wait_for_ssh(updated.ip):
                    print(f"  Warning: SSH not ready for {updated.ip}")
            except Exception as e:
                print(f"  Error: {e}")

    def setup_instance(self, instance: LambdaInstance) -> bool:
        """Setup a single instance."""
        print(f"Setting up {instance.name} ({instance.ip})...")

        setup_script = SETUP_SCRIPT.format(
            repo_url=self.config.repo_url,
            openai_key=self.openai_key,
        )

        code, stdout, stderr = run_ssh_command(
            instance.ip,
            setup_script,
            timeout=600,
        )

        if code != 0:
            print(f"  Setup failed: {stderr}")
            return False

        print(f"  Setup complete")
        return True

    def run_scenario(self, instance: LambdaInstance) -> Dict[str, Any]:
        """Run scenario on a single instance."""
        scenario = instance.scenario
        print(f"\n=== Running {scenario} on {instance.name} ===")

        run_script = RUN_SCRIPT.format(
            scenario=scenario,
            model=self.config.model,
            trials=self.config.trials,
            max_rounds=self.config.max_rounds,
            openai_key=self.openai_key,
            timestamp=self.timestamp,
        )

        start_time = time.time()
        code, stdout, stderr = run_ssh_command(
            instance.ip,
            run_script,
            timeout=14400,  # 4 hour timeout
        )
        elapsed = time.time() - start_time

        result = {
            "scenario": scenario,
            "instance_id": instance.id,
            "instance_ip": instance.ip,
            "exit_code": code,
            "elapsed_seconds": elapsed,
            "success": code == 0,
        }

        if code == 0:
            print(f"  Completed in {elapsed/60:.1f} minutes")
        else:
            print(f"  Failed: {stderr[:500]}")
            result["error"] = stderr[:1000]

        return result

    def collect_results(self) -> None:
        """Download results from all instances."""
        print("\n=== Collecting Results ===")

        output_dir = Path(self.config.output_dir) / self.timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        for scenario, instance in self.instances.items():
            if instance.status == "dry-run":
                continue

            print(f"Downloading results from {instance.name}...")
            remote_path = f"~/multiagent-emergent-deception/results/{scenario}_{self.timestamp}"
            local_path = output_dir / scenario

            if download_file(instance.ip, remote_path, str(local_path)):
                print(f"  Saved to {local_path}")
            else:
                print(f"  Failed to download results")

    def terminate_all(self) -> None:
        """Terminate all instances."""
        print("\n=== Terminating Instances ===")

        for scenario, instance in self.instances.items():
            if instance.status == "dry-run":
                continue

            print(f"Terminating {instance.name}...")
            try:
                self.client.terminate_instance(instance.id)
                print(f"  Terminated")
            except Exception as e:
                print(f"  Error: {e}")

    def run(self) -> Dict[str, Any]:
        """Run the full cluster execution."""
        print("=" * 60)
        print("LAMBDA LABS PARALLEL CLUSTER RUNNER")
        print("=" * 60)
        print(f"Scenarios: {self.config.scenarios}")
        print(f"Model: {self.config.model}")
        print(f"Trials per scenario: {self.config.trials}")
        print(f"Dry run: {self.config.dry_run}")

        results = {
            "timestamp": self.timestamp,
            "config": {
                "scenarios": self.config.scenarios,
                "model": self.config.model,
                "trials": self.config.trials,
            },
            "scenario_results": {},
        }

        try:
            # Check availability
            gpu_type = self.check_availability()
            results["gpu_type"] = gpu_type

            # Launch instances
            self.launch_instances(gpu_type)

            if self.config.dry_run:
                print("\n[DRY RUN] Would execute scenarios in parallel")
                return results

            # Wait for instances
            self.wait_for_all_instances()

            # Setup instances in parallel
            print("\n=== Setting Up Instances ===")
            with ThreadPoolExecutor(max_workers=len(self.instances)) as executor:
                futures = {
                    executor.submit(self.setup_instance, inst): scenario
                    for scenario, inst in self.instances.items()
                }
                for future in as_completed(futures):
                    scenario = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Setup failed for {scenario}: {e}")

            # Run scenarios in parallel
            print("\n=== Running Scenarios in Parallel ===")
            with ThreadPoolExecutor(max_workers=len(self.instances)) as executor:
                futures = {
                    executor.submit(self.run_scenario, inst): scenario
                    for scenario, inst in self.instances.items()
                }
                for future in as_completed(futures):
                    scenario = futures[future]
                    try:
                        result = future.result()
                        results["scenario_results"][scenario] = result
                    except Exception as e:
                        print(f"Run failed for {scenario}: {e}")
                        results["scenario_results"][scenario] = {"error": str(e)}

            # Collect results
            self.collect_results()

        finally:
            # Always terminate instances
            self.terminate_all()

        # Save summary
        output_dir = Path(self.config.output_dir) / self.timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "cluster_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary saved to {summary_path}")

        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run deception experiment scenarios in parallel on Lambda Labs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all scenarios in parallel
  python scripts/lambda_cluster_runner.py --scenarios all

  # Run specific scenarios
  python scripts/lambda_cluster_runner.py --scenarios ultimatum_bluff,alliance_betrayal

  # Dry run to see what would happen
  python scripts/lambda_cluster_runner.py --scenarios all --dry-run

  # Use specific GPU type
  python scripts/lambda_cluster_runner.py --scenarios all --gpu-type gpu_1x_h100_pcie

Environment variables (set in .env file):
  LAMBDA_API_KEY      - Your Lambda Labs API key
  LAMBDA_SSH_KEY_NAME - Name of SSH key in Lambda dashboard
  OPENAI_API_KEY      - OpenAI API key for DeepEval
  REPO_URL            - Git repository URL to clone

Setup:
  cp .env.example .env
  # Edit .env with your credentials
        """,
    )

    parser.add_argument(
        "--scenarios",
        type=str,
        required=True,
        help="Scenarios to run: 'all' or comma-separated list",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default=None,
        help="Specific GPU type (default: auto-select best available)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-27b-it",
        help="Model to use (default: google/gemma-2-27b-it)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Trials per scenario (default: 50)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Max negotiation rounds (default: 5)",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        default=os.environ.get("REPO_URL", "https://github.com/your-username/multiagent-emergent-deception.git"),
        help="Git repo URL to clone on instances (or set REPO_URL in .env)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./cluster_results",
        help="Local directory for results (default: ./cluster_results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without launching instances",
    )
    parser.add_argument(
        "--check-availability",
        action="store_true",
        help="Just check GPU availability and exit",
    )

    args = parser.parse_args()

    # Check availability only
    if args.check_availability:
        api_key = os.environ.get("LAMBDA_API_KEY")
        if not api_key:
            print("Error: LAMBDA_API_KEY not set")
            sys.exit(1)
        client = LambdaClient(api_key)
        print("\n=== Lambda Labs GPU Availability ===")
        types = client.list_instance_types()
        for gpu_type, info in sorted(types.items()):
            regions = info.get("regions_with_capacity_available", [])
            price = info.get("instance_type", {}).get("price_cents_per_hour", 0) / 100
            status = "AVAILABLE" if regions else "sold out"
            print(f"  {gpu_type}: ${price:.2f}/hr - {status}")
        return

    # Parse scenarios
    if args.scenarios.lower() == "all":
        scenarios = ALL_SCENARIOS
    else:
        scenarios = [s.strip() for s in args.scenarios.split(",")]
        for s in scenarios:
            if s not in ALL_SCENARIOS:
                print(f"Error: Unknown scenario '{s}'")
                print(f"Available: {ALL_SCENARIOS}")
                sys.exit(1)

    # Create config
    config = ClusterConfig(
        scenarios=scenarios,
        gpu_type=args.gpu_type,
        model=args.model,
        trials=args.trials,
        max_rounds=args.max_rounds,
        repo_url=args.repo_url,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    # Run
    runner = ClusterRunner(config)
    results = runner.run()

    # Print summary
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    for scenario, result in results.get("scenario_results", {}).items():
        status = "SUCCESS" if result.get("success") else "FAILED"
        elapsed = result.get("elapsed_seconds", 0) / 60
        print(f"  {scenario}: {status} ({elapsed:.1f} min)")


if __name__ == "__main__":
    main()
