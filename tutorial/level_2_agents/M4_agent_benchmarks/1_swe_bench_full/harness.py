"""Container-based SWE-Bench evaluation harness.

Manages Docker/Podman containers for applying patches and running
repository test suites — the same pipeline used by the real
SWE-Bench leaderboard.
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

CONTAINER_RUNTIME = os.environ.get("SWE_BENCH_RUNTIME", "podman")
IMAGE_NAME = "swe-bench-eval"
CONTAINER_TIMEOUT = 600
TEST_TIMEOUT = 120
MAX_P2P_TESTS = 10


@dataclass
class EvalResult:
    instance_id: str
    repo: str
    status: str  # resolved, applied, patch_failed, setup_failed, error
    resolved: bool = False
    f2p_passed: int = 0
    f2p_total: int = 0
    p2p_passed: int = 0
    p2p_total: int = 0
    agent_patch: str = ""
    test_output: str = ""
    latency_s: float = 0.0
    error: str = ""


def ensure_image_built() -> None:
    """Build the base evaluation image if it doesn't exist."""
    result = subprocess.run(
        [CONTAINER_RUNTIME, "image", "inspect", IMAGE_NAME],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        print(f"  Image {IMAGE_NAME} already exists.")
        return

    print(f"  Building {IMAGE_NAME} image (one-time) ...")
    dockerfile_dir = str(Path(__file__).parent)
    result = subprocess.run(
        [CONTAINER_RUNTIME, "build", "-t", IMAGE_NAME, dockerfile_dir],
        capture_output=True,
        text=True,
        timeout=CONTAINER_TIMEOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Image build failed:\n{result.stderr}")
    print(f"  Image {IMAGE_NAME} ready.")


def _sanitize_name(instance_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", instance_id).lower()


def start_container(instance_id: str) -> str:
    """Start a container for evaluating one SWE-Bench instance."""
    name = f"swe-eval-{_sanitize_name(instance_id)}"
    # Remove stale container with same name
    subprocess.run(
        [CONTAINER_RUNTIME, "rm", "-f", name],
        capture_output=True,
        check=False,
    )
    result = subprocess.run(
        [CONTAINER_RUNTIME, "run", "-d", "--name", name, IMAGE_NAME, "sleep", "infinity"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Container start failed:\n{result.stderr}")
    return name


def exec_in_container(
    container_id: str, cmd: str, timeout: int = CONTAINER_TIMEOUT
) -> tuple[int, str, str]:
    """Execute a command inside a container. Returns (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            [CONTAINER_RUNTIME, "exec", container_id, "bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Timeout after {timeout}s"


def setup_repo(container_id: str, repo: str, base_commit: str) -> bool:
    """Clone the repo at the given commit and install dependencies."""
    print(f"    Cloning {repo} at {base_commit[:8]} ...")
    rc, out, err = exec_in_container(
        container_id,
        f"git clone --quiet https://github.com/{repo}.git /workspace/repo 2>&1",
    )
    if rc != 0:
        print(f"    Clone failed: {err or out}")
        return False

    rc, out, err = exec_in_container(
        container_id,
        f"cd /workspace/repo && git checkout -q {base_commit}",
    )
    if rc != 0:
        print(f"    Checkout failed: {err}")
        return False

    print("    Installing dependencies ...")
    rc, out, err = exec_in_container(
        container_id,
        "cd /workspace/repo && pip install -e . -q 2>&1 | tail -3",
        timeout=CONTAINER_TIMEOUT,
    )
    if rc != 0:
        print(f"    Install failed: {err or out}")
        return False

    print("    Repo setup complete.")
    return True


def apply_patch(
    container_id: str, patch_text: str, patch_name: str = "fix.patch"
) -> bool:
    """Apply a patch inside the container."""
    if not patch_text.strip():
        return False

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False
    ) as f:
        f.write(patch_text)
        host_path = f.name

    try:
        subprocess.run(
            [CONTAINER_RUNTIME, "cp", host_path, f"{container_id}:/workspace/{patch_name}"],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"    Failed to copy patch into container: {exc}")
        return False
    finally:
        os.unlink(host_path)

    rc, _out, _err = exec_in_container(
        container_id,
        f"cd /workspace/repo && git apply /workspace/{patch_name} 2>&1",
    )
    if rc == 0:
        return True

    rc2, _out2, _err2 = exec_in_container(
        container_id,
        f"cd /workspace/repo && git apply --reject /workspace/{patch_name} 2>&1",
    )
    return rc2 == 0


def run_tests(
    container_id: str, test_names: list[str], repo: str
) -> tuple[int, int, str]:
    """Run specific tests inside the container. Returns (passed, failed, raw_output)."""
    if not test_names:
        return 0, 0, "No tests specified"

    tests_str = " ".join(test_names)
    _rc, out, err = exec_in_container(
        container_id,
        f"cd /workspace/repo && python -m pytest --no-header -rN --tb=short -q {tests_str} 2>&1",
        timeout=TEST_TIMEOUT,
    )

    raw = out + err
    passed, failed = _parse_pytest_summary(raw)
    return passed, failed, raw


def _parse_pytest_summary(output: str) -> tuple[int, int]:
    """Parse pytest summary line to extract passed/failed counts."""
    passed = 0
    failed = 0

    match = re.search(r"(\d+) passed", output)
    if match:
        passed = int(match.group(1))

    match = re.search(r"(\d+) failed", output)
    if match:
        failed = int(match.group(1))

    match = re.search(r"(\d+) error", output)
    if match:
        failed += int(match.group(1))

    return passed, failed


def cleanup_container(container_id: str) -> None:
    """Stop and remove a container."""
    subprocess.run(
        [CONTAINER_RUNTIME, "stop", "-t", "5", container_id],
        capture_output=True,
        check=False,
    )
    subprocess.run(
        [CONTAINER_RUNTIME, "rm", "-f", container_id],
        capture_output=True,
        check=False,
    )
