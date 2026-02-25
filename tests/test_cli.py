import json
import subprocess
import sys


def test_cli_json_output():
    cmd = [
        sys.executable,
        "-m",
        "example_repo",
        "--n-samples",
        "150",
        "--n-features",
        "4",
        "--noise",
        "0.5",
        "--seed",
        "1",
        "--json",
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    assert "train_accuracy" in data
    assert "test_accuracy" in data
    assert 0.0 <= data["train_accuracy"] <= 1.0
    assert 0.0 <= data["test_accuracy"] <= 1.0