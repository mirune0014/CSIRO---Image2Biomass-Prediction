"""Experiment logging utilities.

This module implements a lightweight, dependency-free experiment logger to
support the hypothesis-driven workflow described in AGENTS.md. It records
experiment metadata, parameters, metrics (CV/Holdout), and Git information
to JSON files under `project/results/<experiment_id>/` and maintains a
CSV index for quick comparison.

Usage example:
    from project.utils.experiment_logger import ExperimentLogger

    logger = ExperimentLogger(
        experiment_id="exp002",
        hypothesis_id="H-02",
        change_type="feature",
        hypothesis="Add NDVI feature to improve AUC",
        expected_effect="Higher correlation with target, better CV AUC",
        metrics=["auc_cv", "auc_holdout"],
    )
    logger.log_params({"feature": "ndvi", "window": 7})
    logger.log_metrics({"auc_cv": 0.829})
    logger.log_metrics({"auc_holdout": 0.821})
    logger.finalize()

All outputs are written atomically where possible, ensuring reproducibility.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Any


def _now_iso() -> str:
    """Return current UTC time in ISO8601 (without microseconds)."""
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _git_cmd(args: List[str]) -> Optional[str]:
    """Run a git command and return stdout stripped, or None on failure.

    Avoids introducing external dependencies. Failures are swallowed so that
    logging still works even outside of a Git repo.
    """
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _atomic_write_text(path: str, text: str) -> None:
    """Write text to `path` atomically by using a temporary file + replace."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


@dataclasses.dataclass
class ExperimentMeta:
    """Metadata captured at experiment start."""

    experiment_id: str
    hypothesis_id: str
    change_type: str  # preprocessing | feature | model | hyperparam
    hypothesis: str
    expected_effect: str
    metrics: List[str]
    started_at: str
    git_commit: Optional[str]
    git_branch: Optional[str]


class ExperimentLogger:
    """Logger for hypothesis-driven experiments.

    - Creates a result directory: `project/results/<experiment_id>/`
    - Writes a run JSON file: `run-<timestamp>.json`
    - Appends/updates a CSV index: `project/results/experiments.csv`
    """

    def __init__(
        self,
        experiment_id: str,
        hypothesis_id: str,
        change_type: str,
        hypothesis: str,
        expected_effect: str,
        metrics: Optional[List[str]] = None,
        results_root: str = os.path.join("project", "results"),
    ) -> None:
        """Initialize the logger and capture metadata.

        Args:
            experiment_id: Short identifier (e.g., "exp002").
            hypothesis_id: Hypothesis identifier (e.g., "H-02").
            change_type: One of {"preprocessing", "feature", "model", "hyperparam"}.
            hypothesis: One-sentence hypothesis.
            expected_effect: Brief rationale/expected impact.
            metrics: List of metric names to track.
            results_root: Root directory for results output.
        """
        self.results_root = results_root
        self.experiment_id = experiment_id
        self.metrics = list(metrics or [])
        self.run_started_at = _now_iso()
        self.run_file_path = os.path.join(
            self.results_root,
            self.experiment_id,
            f"run-{self.run_started_at.replace(':', '').replace('-', '')}.json",
        )

        git_commit = _git_cmd(["rev-parse", "HEAD"])  # may be None
        git_branch = _git_cmd(["rev-parse", "--abbrev-ref", "HEAD"])  # may be None

        self.meta = ExperimentMeta(
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
            change_type=change_type,
            hypothesis=hypothesis,
            expected_effect=expected_effect,
            metrics=self.metrics,
            started_at=self.run_started_at,
            git_commit=git_commit,
            git_branch=git_branch,
        )

        self.params: Dict[str, Any] = {}
        self.logged_metrics: Dict[str, Any] = {}
        self.notes: List[str] = []

        # Write initial skeleton
        self._sync()

    # ---------------------------- public API ---------------------------------
    def log_params(self, params: Dict[str, Any]) -> None:
        """Merge experiment parameters into the log."""
        self.params.update(params)
        self._sync()

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Merge metric values into the log (e.g., {"auc_cv": 0.829})."""
        self.logged_metrics.update(metrics)
        self._sync()

    def add_note(self, text: str) -> None:
        """Append a free-text note to the log."""
        self.notes.append(text)
        self._sync()

    def finalize(self) -> None:
        """Mark the run as completed and update the CSV index."""
        data = self._current_payload()
        data["finished_at"] = _now_iso()
        _atomic_write_text(self.run_file_path, json.dumps(data, ensure_ascii=False, indent=2))
        self._update_index_csv(data)

    # --------------------------- internal helpers ----------------------------
    def _current_payload(self) -> Dict[str, Any]:
        return {
            "meta": dataclasses.asdict(self.meta),
            "params": self.params,
            "metrics": self.logged_metrics,
            "notes": self.notes,
        }

    def _sync(self) -> None:
        """Write the current JSON log state to disk."""
        payload = self._current_payload()
        _atomic_write_text(self.run_file_path, json.dumps(payload, ensure_ascii=False, indent=2))

    def _update_index_csv(self, data: Dict[str, Any]) -> None:
        """Append/update an experiments CSV index for quick comparison.

        The CSV is stored at `project/results/experiments.csv` with columns:
        experiment_id, hypothesis_id, change_type, started_at, finished_at,
        git_commit, git_branch, metrics(json), params(json)
        """
        index_path = os.path.join(self.results_root, "experiments.csv")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        row = {
            "experiment_id": data["meta"]["experiment_id"],
            "hypothesis_id": data["meta"]["hypothesis_id"],
            "change_type": data["meta"]["change_type"],
            "started_at": data["meta"]["started_at"],
            "finished_at": data.get("finished_at", ""),
            "git_commit": data["meta"].get("git_commit", ""),
            "git_branch": data["meta"].get("git_branch", ""),
            "metrics": json.dumps(data.get("metrics", {}), ensure_ascii=False),
            "params": json.dumps(data.get("params", {}), ensure_ascii=False),
        }

        write_header = not os.path.exists(index_path) or os.path.getsize(index_path) == 0
        with open(index_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "experiment_id",
                    "hypothesis_id",
                    "change_type",
                    "started_at",
                    "finished_at",
                    "git_commit",
                    "git_branch",
                    "metrics",
                    "params",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)


__all__ = ["ExperimentLogger", "ExperimentMeta"]

