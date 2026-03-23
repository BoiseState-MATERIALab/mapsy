from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SlurmTemplate:
    """Starter utility for rendering and optionally submitting Slurm job scripts."""

    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    time: str | None = None
    nodes: int | None = 1
    ntasks: int | None = 1
    cpus_per_task: int | None = None
    mem: str | None = None
    job_name: str = "mapsy-qe"
    output: str = "slurm-%j.out"
    error: str | None = None
    modules: list[str] = field(default_factory=list)
    setup_commands: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    custom_directives: list[str] = field(default_factory=list)
    script_name: str = "submit.slurm"
    submit_command: str = "sbatch"
    shebang: str = "#!/bin/bash"
    job_id_pattern: re.Pattern[str] = field(
        init=False,
        default=re.compile(r"Submitted batch job\s+(\d+)", re.IGNORECASE),
    )

    def workflow_description(self) -> dict[str, Any]:
        """Return scheduler metadata suitable for a CalculationWorkflow description."""
        return {
            "scheduler": "slurm",
            "partition": self.partition,
            "account": self.account,
            "qos": self.qos,
            "time": self.time,
            "nodes": self.nodes,
            "ntasks": self.ntasks,
            "cpus_per_task": self.cpus_per_task,
            "mem": self.mem,
            "job_name": self.job_name,
            "output": self.output,
            "error": self.error,
            "modules": list(self.modules),
            "setup_commands": list(self.setup_commands),
            "environment": dict(self.environment),
            "custom_directives": list(self.custom_directives),
            "script_name": self.script_name,
            "submit_command": self.submit_command,
        }

    def render(
        self,
        run_command: str,
        *,
        job_name: str | None = None,
        workdir: str | Path | None = None,
    ) -> str:
        """Render a Slurm submission script."""
        lines = [self.shebang]
        directives: list[tuple[str, Any]] = [
            ("job-name", job_name or self.job_name),
            ("partition", self.partition),
            ("account", self.account),
            ("qos", self.qos),
            ("time", self.time),
            ("nodes", self.nodes),
            ("ntasks", self.ntasks),
            ("cpus-per-task", self.cpus_per_task),
            ("mem", self.mem),
            ("output", self.output),
            ("error", self.error),
        ]
        for key, value in directives:
            if value is not None:
                lines.append(f"#SBATCH --{key}={value}")
        for directive in self.custom_directives:
            lines.append(directive if directive.startswith("#SBATCH") else f"#SBATCH {directive}")
        lines.append("")
        if workdir is not None:
            lines.append(f"cd {shlex.quote(str(Path(workdir).expanduser()))}")
        for module in self.modules:
            lines.append(f"module load {module}")
        for key, value in self.environment.items():
            lines.append(f"export {key}={shlex.quote(str(value))}")
        lines.extend(self.setup_commands)
        if lines[-1] != "":
            lines.append("")
        lines.append(run_command)
        return "\n".join(lines) + "\n"

    def write(
        self,
        directory: str | Path,
        run_command: str,
        *,
        job_name: str | None = None,
        filename: str | None = None,
        workdir: str | Path | None = None,
    ) -> Path:
        """Write a Slurm submission script into a job directory."""
        directory_path = Path(directory).expanduser()
        directory_path.mkdir(parents=True, exist_ok=True)
        script_path = directory_path / (filename or self.script_name)
        script = self.render(
            run_command,
            job_name=job_name,
            workdir=directory_path if workdir is None else workdir,
        )
        script_path.write_text(script)
        script_path.chmod(0o755)
        return script_path

    def submit(
        self,
        script_path: str | Path,
        *,
        subprocess_run: Callable[..., Any] | None = None,
        check: bool = True,
    ) -> dict[str, Any]:
        """Submit a Slurm script and return submission metadata."""
        if subprocess_run is None:
            subprocess_run = subprocess.run

        completed = subprocess_run(
            [self.submit_command, str(Path(script_path).expanduser())],
            check=check,
            capture_output=True,
            text=True,
        )
        stdout = getattr(completed, "stdout", "") or ""
        stderr = getattr(completed, "stderr", "") or ""
        match = self.job_id_pattern.search(stdout)
        metadata: dict[str, Any] = {
            "scheduler": "slurm",
            "submit_script": str(Path(script_path).expanduser()),
            "scheduler_submit_output": stdout.strip(),
        }
        if stderr.strip():
            metadata["scheduler_submit_error"] = stderr.strip()
        if match:
            metadata["scheduler_job_id"] = int(match.group(1))
        return metadata


__all__ = ["SlurmTemplate"]
