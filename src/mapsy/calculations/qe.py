from __future__ import annotations

import re
import shlex
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian

from .cluster import SlurmTemplate

QE_RUNNER_RESULT = dict[str, Any]
BOHR_TO_ANG = 0.529177210903
ATTEMPT_PATTERN = re.compile(r"^(?P<stem>.+)\.r(?P<index>\d{3})(?P<suffix>\.[^.]+)$")


def _with_attempt_suffix(filename: str, attempt_index: int) -> str:
    if attempt_index <= 0:
        return filename
    path = Path(filename)
    return f"{path.stem}.r{attempt_index:03d}{path.suffix}"


def _attempt_index_from_name(filename: str, base_filename: str) -> int | None:
    if filename == base_filename:
        return 0
    match = ATTEMPT_PATTERN.match(filename)
    if match is None:
        return None
    base = Path(base_filename)
    if match.group("stem") != base.stem or match.group("suffix") != base.suffix:
        return None
    return int(match.group("index"))


@dataclass
class QuantumEspressoSetup:
    """Reusable helper for configuring and running Quantum ESPRESSO jobs through ASE."""

    runprefix: str = ""
    qepath: str = ""
    pseudodir: str = ""
    pseudopotentials: dict[str, str] = field(default_factory=dict)
    input_data: dict[str, Any] = field(default_factory=dict)
    kpts: tuple[int, int, int] = (1, 1, 1)
    koffset: tuple[int, int, int] = (0, 0, 0)
    executable: str = "pw.x"
    input_filename: str = "espresso.pwi"
    output_filename: str = "espresso.pwo"
    outdir_root: str | None = None
    outdir_template: str | None = None
    tstress: bool | None = None
    tprnfor: bool | None = None
    calculator_kwargs: dict[str, Any] = field(default_factory=dict)

    def command(self) -> str:
        """Return the full QE command used by ASE."""
        executable_path = str(Path(self.qepath).expanduser() / self.executable)
        return f"{self.runprefix}{executable_path}"

    def workflow_description(
        self,
        *,
        scheduler_template: SlurmTemplate | None = None,
    ) -> dict[str, Any]:
        """Return calculation metadata suitable for a CalculationWorkflow description."""
        metadata = {
            "engine": "quantum_espresso",
            "command": self.command(),
            "run_command": self.run_command(),
            "pseudo_dir": str(Path(self.pseudodir).expanduser()),
            "pseudopotentials": dict(self.pseudopotentials),
            "input_data": deepcopy(self.input_data),
            "kpts": tuple(self.kpts),
            "koffset": tuple(self.koffset),
            "input_filename": self.input_filename,
            "output_filename": self.output_filename,
            "outdir_root": self.outdir_root,
            "outdir_template": self.outdir_template,
            "tstress": self.tstress,
            "tprnfor": self.tprnfor,
            "calculator_kwargs": deepcopy(self.calculator_kwargs),
        }
        if scheduler_template is not None:
            metadata["scheduler"] = scheduler_template.workflow_description()
        return metadata

    def run_command(
        self,
        *,
        input_filename: str | None = None,
        output_filename: str | None = None,
    ) -> str:
        """Return a shell command suitable for running QE in a batch script."""
        input_name = input_filename or self.input_filename
        output_name = output_filename or self.output_filename
        return f"{self.command()} -in {shlex.quote(input_name)}" f" > {shlex.quote(output_name)}"

    def make_profile(self, profile_cls: type[Any] | None = None) -> Any:
        """Create an EspressoProfile using the configured command and pseudo directory."""
        profile_factory: type[Any]
        if profile_cls is None:
            from ase.calculators.espresso import EspressoProfile

            profile_factory = EspressoProfile
        else:
            profile_factory = profile_cls

        return profile_factory(
            command=self.command(),
            pseudo_dir=str(Path(self.pseudodir).expanduser()),
        )

    def make_calculator(
        self,
        directory: str | Path,
        *,
        qe_outdir: str | Path | None = None,
        profile: Any | None = None,
        espresso_cls: type[Any] | None = None,
        profile_cls: type[Any] | None = None,
        calculator_overrides: dict[str, Any] | None = None,
    ) -> Any:
        """Create an ASE Espresso calculator for a specific work directory."""
        calculator_factory: type[Any]
        if espresso_cls is None:
            from ase.calculators.espresso import Espresso

            calculator_factory = Espresso
        else:
            calculator_factory = espresso_cls

        local_input_data = self.build_input_data(qe_outdir=qe_outdir)
        kwargs = self.build_calculator_kwargs(calculator_overrides=calculator_overrides)

        return calculator_factory(
            profile=self.make_profile(profile_cls=profile_cls) if profile is None else profile,
            pseudopotentials=dict(self.pseudopotentials),
            input_data=local_input_data,
            directory=str(Path(directory).expanduser()),
            kpts=tuple(self.kpts),
            koffset=tuple(self.koffset),
            **kwargs,
        )

    def build_input_data(
        self,
        *,
        qe_outdir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Build QE input_data with pseudo_dir and optional outdir resolved."""
        local_input_data = deepcopy(self.input_data)
        control = deepcopy(local_input_data.get("control", {}))
        control.setdefault("pseudo_dir", str(Path(self.pseudodir).expanduser()))
        if qe_outdir is not None:
            control["outdir"] = str(Path(qe_outdir).expanduser())
        local_input_data["control"] = control
        return local_input_data

    def build_calculator_kwargs(
        self,
        *,
        calculator_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build calculator/write kwargs derived from the QE setup."""
        kwargs = deepcopy(self.calculator_kwargs)
        if self.tstress is not None:
            kwargs.setdefault("tstress", self.tstress)
        if self.tprnfor is not None:
            kwargs.setdefault("tprnfor", self.tprnfor)
        if calculator_overrides:
            kwargs.update(calculator_overrides)
        return kwargs

    def resolve_workdir(
        self,
        point_index: int,
        *,
        directory_template: str = "qe_job_{point_index}",
        workdir_root: str | Path | None = None,
    ) -> Path:
        """Resolve the per-job working directory."""
        folder_name = directory_template.format(point_index=point_index)
        if workdir_root is None:
            return Path(folder_name)
        return Path(workdir_root).expanduser() / folder_name

    def resolve_outdir(
        self,
        point_index: int,
        *,
        directory_template: str = "qe_job_{point_index}",
        outdir_root: str | Path | None = None,
        outdir_template: str | None = None,
    ) -> Path | None:
        """Resolve the per-job QE outdir, if configured."""
        root = self.outdir_root if outdir_root is None else outdir_root
        if root is None:
            return None

        template = self.outdir_template if outdir_template is None else outdir_template
        name_template = directory_template if template is None else template
        folder_name = name_template.format(point_index=point_index)
        return Path(root).expanduser() / folder_name

    def attempt_filename(self, filename: str, attempt_index: int) -> str:
        """Return the filename for a specific retry attempt."""
        return _with_attempt_suffix(filename, attempt_index)

    def attempt_path(self, directory: str | Path, filename: str, attempt_index: int) -> Path:
        """Return the path for a specific retry attempt file."""
        return Path(directory).expanduser() / self.attempt_filename(filename, attempt_index)

    def attempt_outdir(self, qe_outdir: str | Path | None, attempt_index: int) -> Path | None:
        """Return the attempt-specific QE outdir."""
        if qe_outdir is None:
            return None
        path = Path(qe_outdir).expanduser()
        if attempt_index <= 0:
            return path
        return path.with_name(_with_attempt_suffix(path.name, attempt_index))

    def discover_attempt_indexes(
        self,
        directory: str | Path,
        *,
        output_filename: str | None = None,
    ) -> list[int]:
        """Return discovered attempt indexes for one job directory."""
        directory_path = Path(directory).expanduser()
        base_output = output_filename or self.output_filename
        indexes: list[int] = []
        for path in directory_path.iterdir() if directory_path.exists() else []:
            if not path.is_file():
                continue
            index = _attempt_index_from_name(path.name, base_output)
            if index is not None:
                indexes.append(index)
        return sorted(set(indexes))

    def next_attempt_index(
        self,
        directory: str | Path,
        *,
        output_filename: str | None = None,
    ) -> int:
        """Return the next retry attempt index for one job directory."""
        indexes = self.discover_attempt_indexes(directory, output_filename=output_filename)
        if not indexes:
            return 1
        return max(indexes) + 1

    def add_single_atom_adsorbate(
        self,
        substrate: Atoms,
        symbol: str,
        position: Sequence[float],
        *,
        freeze_substrate: bool = True,
        adsorbate_mask: Sequence[bool] | None = (True, True, False),
    ) -> Atoms:
        """Return a copy of the substrate with one adsorbate atom and optional constraints."""
        atoms = cast(Atoms, substrate.copy())
        n_substrate = len(atoms)
        atoms += Atoms(symbol, positions=[list(position)])

        constraints: list[Any] = []
        if freeze_substrate:
            constraints.append(FixAtoms(indices=range(n_substrate)))
        if adsorbate_mask is not None:
            constraints.append(FixCartesian(n_substrate, list(adsorbate_mask)))
        if constraints:
            atoms.set_constraint(constraints)

        return atoms

    def run_single_calculation(
        self,
        point_index: int,
        atoms: Atoms,
        *,
        directory_template: str = "qe_job_{point_index}",
        workdir_root: str | Path | None = None,
        outdir_root: str | Path | None = None,
        outdir_template: str | None = None,
        calculator_overrides: dict[str, Any] | None = None,
        espresso_cls: type[Any] | None = None,
        profile_cls: type[Any] | None = None,
        energy_key: str = "runner_energy",
    ) -> QE_RUNNER_RESULT:
        """Run a single QE job in its own directory and return workflow metadata."""
        workdir = self.resolve_workdir(
            point_index,
            directory_template=directory_template,
            workdir_root=workdir_root,
        )
        workdir.mkdir(parents=True, exist_ok=True)
        qe_outdir = self.resolve_outdir(
            point_index,
            directory_template=directory_template,
            outdir_root=outdir_root,
            outdir_template=outdir_template,
        )
        if qe_outdir is not None:
            qe_outdir.mkdir(parents=True, exist_ok=True)

        atoms_to_run = atoms.copy()
        atoms_to_run.calc = self.make_calculator(
            directory=workdir,
            qe_outdir=qe_outdir,
            espresso_cls=espresso_cls,
            profile_cls=profile_cls,
            calculator_overrides=calculator_overrides,
        )
        energy = atoms_to_run.get_potential_energy()

        return {
            "label_file": str(workdir),
            "qe_outdir": str(qe_outdir) if qe_outdir is not None else None,
            energy_key: energy,
        }

    def write_inputs(
        self,
        atoms: Atoms,
        calculator: Any,
        *,
        qe_outdir: str | Path | None = None,
        directory: str | Path | None = None,
        input_writer: Callable[..., Any] | None = None,
    ) -> None:
        """Write QE input files for a prepared job directory."""
        if input_writer is not None:
            input_writer(atoms=atoms, calculator=calculator, setup=self)
        else:
            target_directory = directory
            if target_directory is None:
                target_directory = getattr(calculator, "directory", None)
            if target_directory is None:
                raise AttributeError(
                    "Missing target directory for QE input writing and no input_writer was provided."
                )
            self.write_espresso_input(
                atoms,
                directory=target_directory,
                qe_outdir=qe_outdir,
            )

        if qe_outdir is None or directory is None:
            return

        self.ensure_input_outdir(
            Path(directory).expanduser() / self.input_filename,
            qe_outdir,
        )

    def write_espresso_input(
        self,
        atoms: Atoms,
        *,
        directory: str | Path,
        qe_outdir: str | Path | None = None,
        write_func: Callable[..., Any] | None = None,
    ) -> Path:
        """Write a QE input file using setup-owned serialization defaults."""
        writer: Callable[..., Any]
        if write_func is None:
            from ase.io import write as ase_write

            writer = ase_write
        else:
            writer = write_func

        directory_path = Path(directory).expanduser()
        directory_path.mkdir(parents=True, exist_ok=True)
        input_path = directory_path / self.input_filename
        writer(
            input_path,
            atoms,
            format="espresso-in",
            input_data=self.build_input_data(qe_outdir=qe_outdir),
            pseudopotentials=dict(self.pseudopotentials),
            kpts=tuple(self.kpts),
            koffset=tuple(self.koffset),
            **self.build_calculator_kwargs(),
        )
        return input_path

    def ensure_input_outdir(
        self,
        input_path: str | Path,
        qe_outdir: str | Path,
    ) -> None:
        """Ensure the QE input file contains the requested outdir in the CONTROL block."""
        path = Path(input_path).expanduser()
        if not path.exists():
            return

        outdir_value = str(Path(qe_outdir).expanduser())
        lines = path.read_text().splitlines()
        output: list[str] = []
        in_control = False
        control_found = False
        outdir_found = False
        inserted = False

        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()
            if lower.startswith("&control"):
                in_control = True
                control_found = True
                output.append(line)
                continue
            if in_control and lower.startswith("outdir"):
                indent = line[: len(line) - len(line.lstrip())]
                output.append(f"{indent}outdir = '{outdir_value}'")
                outdir_found = True
                continue
            if in_control and stripped == "/":
                if not outdir_found:
                    output.append(f"   outdir = '{outdir_value}'")
                    outdir_found = True
                output.append(line)
                in_control = False
                inserted = True
                continue
            output.append(line)

        if control_found:
            if not inserted and not outdir_found:
                output.append(f"   outdir = '{outdir_value}'")
        else:
            output = ["&CONTROL", f"   outdir = '{outdir_value}'", "/", *output]

        path.write_text("\n".join(output) + "\n")

    def extract_last_atomic_positions_block(
        self,
        output_path: str | Path,
    ) -> tuple[str, list[str]] | None:
        """Extract the last ATOMIC_POSITIONS block from a QE output file."""
        lines = Path(output_path).expanduser().read_text(errors="replace").splitlines()
        atomic_re = re.compile(
            r"^\s*ATOMIC_POSITIONS(?:\s*[\(\{]?\s*([A-Za-z_]+)\s*[\)\}]?)?",
            re.IGNORECASE,
        )
        atom_re = re.compile(r"^\s*([A-Za-z]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)")
        blocks: list[tuple[str, list[str]]] = []
        index = 0
        while index < len(lines):
            match = atomic_re.match(lines[index])
            if match is None:
                index += 1
                continue
            unit = (match.group(1) or "").strip().lower()
            block_lines = [lines[index].strip()]
            index += 1
            while index < len(lines):
                if atom_re.match(lines[index]) is None:
                    break
                block_lines.append(lines[index].strip())
                index += 1
            if len(block_lines) > 1:
                blocks.append((unit, block_lines))
        if not blocks:
            return None
        return blocks[-1]

    def replace_atomic_positions_block(
        self,
        input_path: str | Path,
        block_lines: Sequence[str],
    ) -> None:
        """Replace the first ATOMIC_POSITIONS block in a QE input file."""
        path = Path(input_path).expanduser()
        lines = path.read_text(errors="replace").splitlines()
        atomic_re = re.compile(
            r"^\s*ATOMIC_POSITIONS(?:\s*[\(\{]?\s*([A-Za-z_]+)\s*[\)\}]?)?",
            re.IGNORECASE,
        )
        atom_re = re.compile(r"^\s*([A-Za-z]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)")
        output: list[str] = []
        index = 0
        replaced = False
        while index < len(lines):
            if not replaced and atomic_re.match(lines[index]) is not None:
                output.extend(block_lines)
                replaced = True
                index += 1
                while index < len(lines) and atom_re.match(lines[index]) is not None:
                    index += 1
                continue
            output.append(lines[index])
            index += 1
        if not replaced:
            output.extend(["", *block_lines])
        path.write_text("\n".join(output) + "\n")

    def prepare_retry_from_previous_attempt(
        self,
        point_index: int,
        *,
        scheduler_template: SlurmTemplate,
        directory_template: str = "qe_job_{point_index}",
        job_name_template: str = "qe-retry-{point_index}",
        workdir_root: str | Path | None = None,
        outdir_root: str | Path | None = None,
        outdir_template: str | None = None,
        submit: bool = False,
        subprocess_run: Callable[..., Any] | None = None,
        update_positions_from_output: bool = False,
    ) -> QE_RUNNER_RESULT:
        """Prepare a retry attempt from the latest attempt files in a job directory."""
        workdir = self.resolve_workdir(
            point_index,
            directory_template=directory_template,
            workdir_root=workdir_root,
        )
        workdir.mkdir(parents=True, exist_ok=True)
        latest_indexes = self.discover_attempt_indexes(
            workdir, output_filename=self.output_filename
        )
        if not latest_indexes:
            raise FileNotFoundError(f"No previous QE output files found in {workdir}.")
        latest_index = max(latest_indexes)
        next_index = latest_index + 1

        source_input = self.attempt_path(workdir, self.input_filename, latest_index)
        source_output = self.attempt_path(workdir, self.output_filename, latest_index)
        if not source_input.exists():
            raise FileNotFoundError(f"Missing source QE input file {source_input}.")
        if not source_output.exists():
            raise FileNotFoundError(f"Missing source QE output file {source_output}.")

        retry_input = self.attempt_path(workdir, self.input_filename, next_index)
        retry_output = self.attempt_path(workdir, self.output_filename, next_index)
        retry_script = self.attempt_path(workdir, scheduler_template.script_name, next_index)

        retry_input.write_text(source_input.read_text(errors="replace"))

        if update_positions_from_output:
            block = self.extract_last_atomic_positions_block(source_output)
            if block is None:
                raise ValueError(f"No ATOMIC_POSITIONS block found in {source_output}.")
            _, block_lines = block
            self.replace_atomic_positions_block(retry_input, block_lines)

        base_outdir = self.resolve_outdir(
            point_index,
            directory_template=directory_template,
            outdir_root=outdir_root,
            outdir_template=outdir_template,
        )
        retry_outdir = self.attempt_outdir(base_outdir, next_index)
        if retry_outdir is not None:
            retry_outdir.mkdir(parents=True, exist_ok=True)
            self.ensure_input_outdir(retry_input, retry_outdir)

        scheduler_for_job = scheduler_template
        if retry_outdir is not None:
            mkdir_outdir = f"mkdir -p {shlex.quote(str(retry_outdir))}"
            if mkdir_outdir not in scheduler_template.setup_commands:
                scheduler_for_job = replace(
                    scheduler_template,
                    setup_commands=[*scheduler_template.setup_commands, mkdir_outdir],
                )
        script_path = scheduler_for_job.write(
            workdir,
            self.run_command(
                input_filename=retry_input.name,
                output_filename=retry_output.name,
            ),
            job_name=job_name_template.format(point_index=point_index, attempt=next_index),
            filename=retry_script.name,
        )

        metadata: QE_RUNNER_RESULT = {
            "label_file": str(workdir),
            "qe_outdir": str(retry_outdir) if retry_outdir is not None else None,
            "submit_script": str(script_path),
            "latest_output_file": str(retry_output),
            "retry_attempt_index": next_index,
            "label_status": "unlabeled",
            "label_error": None,
            "scheduler": "slurm",
        }
        if submit:
            metadata.update(
                scheduler_for_job.submit(
                    script_path,
                    subprocess_run=subprocess_run,
                )
            )
        return metadata

    def prepare_single_calculation(
        self,
        point_index: int,
        atoms: Atoms,
        *,
        scheduler_template: SlurmTemplate,
        directory_template: str = "qe_job_{point_index}",
        job_name_template: str = "qe-{point_index}",
        workdir_root: str | Path | None = None,
        outdir_root: str | Path | None = None,
        outdir_template: str | None = None,
        calculator_overrides: dict[str, Any] | None = None,
        espresso_cls: type[Any] | None = None,
        profile_cls: type[Any] | None = None,
        input_writer: Callable[..., Any] | None = None,
        submit: bool = False,
        subprocess_run: Callable[..., Any] | None = None,
    ) -> QE_RUNNER_RESULT:
        """Prepare a QE job directory and Slurm script, optionally submitting it."""
        workdir = self.resolve_workdir(
            point_index,
            directory_template=directory_template,
            workdir_root=workdir_root,
        )
        workdir.mkdir(parents=True, exist_ok=True)
        qe_outdir = self.resolve_outdir(
            point_index,
            directory_template=directory_template,
            outdir_root=outdir_root,
            outdir_template=outdir_template,
        )
        if qe_outdir is not None:
            qe_outdir.mkdir(parents=True, exist_ok=True)

        atoms_to_write = atoms.copy()
        calculator = self.make_calculator(
            directory=workdir,
            qe_outdir=qe_outdir,
            espresso_cls=espresso_cls,
            profile_cls=profile_cls,
            calculator_overrides=calculator_overrides,
        )
        atoms_to_write.calc = calculator
        self.write_inputs(
            atoms_to_write,
            calculator,
            qe_outdir=qe_outdir,
            directory=workdir,
            input_writer=input_writer,
        )
        scheduler_for_job = scheduler_template
        if qe_outdir is not None:
            mkdir_outdir = f"mkdir -p {shlex.quote(str(qe_outdir))}"
            if mkdir_outdir not in scheduler_template.setup_commands:
                scheduler_for_job = replace(
                    scheduler_template,
                    setup_commands=[*scheduler_template.setup_commands, mkdir_outdir],
                )

        script_path = scheduler_for_job.write(
            workdir,
            self.run_command(),
            job_name=job_name_template.format(point_index=point_index),
        )

        metadata: QE_RUNNER_RESULT = {
            "label_file": str(workdir),
            "qe_outdir": str(qe_outdir) if qe_outdir is not None else None,
            "submit_script": str(script_path),
            "scheduler": "slurm",
        }
        if submit:
            metadata.update(
                scheduler_for_job.submit(
                    script_path,
                    subprocess_run=subprocess_run,
                )
            )
        return metadata

    def build_single_adsorbate_runner(
        self,
        substrate: Atoms,
        adsorbate_symbol: str,
        *,
        position_columns: tuple[str, str, str] = ("x", "y", "z"),
        freeze_substrate: bool = True,
        adsorbate_mask: Sequence[bool] | None = (True, True, False),
        directory_template: str = "qe_job_{point_index}",
        workdir_root: str | Path | None = None,
        outdir_root: str | Path | None = None,
        outdir_template: str | None = None,
        calculator_overrides: dict[str, Any] | None = None,
        espresso_cls: type[Any] | None = None,
        profile_cls: type[Any] | None = None,
        energy_key: str = "runner_energy",
    ) -> Callable[..., QE_RUNNER_RESULT]:
        """Build a CalculationWorkflow runner that creates one adsorbate from special-point coordinates."""

        def runner(*, maps: Any, workflow: Any, special_point: Any, **_: Any) -> QE_RUNNER_RESULT:
            point_index = int(special_point["point_index"])
            position = [float(special_point[column]) for column in position_columns]
            atoms = self.add_single_atom_adsorbate(
                substrate,
                adsorbate_symbol,
                position,
                freeze_substrate=freeze_substrate,
                adsorbate_mask=adsorbate_mask,
            )
            return self.run_single_calculation(
                point_index,
                atoms,
                directory_template=directory_template,
                workdir_root=workdir_root,
                outdir_root=outdir_root,
                outdir_template=outdir_template,
                calculator_overrides=calculator_overrides,
                espresso_cls=espresso_cls,
                profile_cls=profile_cls,
                energy_key=energy_key,
            )

        return runner

    def build_single_adsorbate_prepare_runner(
        self,
        substrate: Atoms,
        adsorbate_symbol: str,
        *,
        scheduler_template: SlurmTemplate,
        position_columns: tuple[str, str, str] = ("x", "y", "z"),
        freeze_substrate: bool = True,
        adsorbate_mask: Sequence[bool] | None = (True, True, False),
        directory_template: str = "qe_job_{point_index}",
        job_name_template: str = "qe-{point_index}",
        workdir_root: str | Path | None = None,
        outdir_root: str | Path | None = None,
        outdir_template: str | None = None,
        calculator_overrides: dict[str, Any] | None = None,
        espresso_cls: type[Any] | None = None,
        profile_cls: type[Any] | None = None,
        input_writer: Callable[..., Any] | None = None,
        submit: bool = False,
        subprocess_run: Callable[..., Any] | None = None,
    ) -> Callable[..., QE_RUNNER_RESULT]:
        """Build a CalculationWorkflow runner that prepares and optionally submits a cluster QE job."""

        def runner(*, maps: Any, workflow: Any, special_point: Any, **_: Any) -> QE_RUNNER_RESULT:
            point_index = int(special_point["point_index"])
            position = [float(special_point[column]) for column in position_columns]
            atoms = self.add_single_atom_adsorbate(
                substrate,
                adsorbate_symbol,
                position,
                freeze_substrate=freeze_substrate,
                adsorbate_mask=adsorbate_mask,
            )
            return self.prepare_single_calculation(
                point_index,
                atoms,
                scheduler_template=scheduler_template,
                directory_template=directory_template,
                job_name_template=job_name_template,
                workdir_root=workdir_root,
                outdir_root=outdir_root,
                outdir_template=outdir_template,
                calculator_overrides=calculator_overrides,
                espresso_cls=espresso_cls,
                profile_cls=profile_cls,
                input_writer=input_writer,
                submit=submit,
                subprocess_run=subprocess_run,
            )

        return runner

    def build_retry_runner(
        self,
        *,
        scheduler_template: SlurmTemplate,
        directory_template: str = "qe_job_{point_index}",
        job_name_template: str = "qe-retry-{point_index}-r{attempt:03d}",
        workdir_root: str | Path | None = None,
        outdir_root: str | Path | None = None,
        outdir_template: str | None = None,
        submit: bool = False,
        subprocess_run: Callable[..., Any] | None = None,
        update_positions_from_output: bool = False,
    ) -> Callable[..., QE_RUNNER_RESULT]:
        """Build a CalculationWorkflow runner that prepares a retry from previous attempt files."""

        def runner(*, maps: Any, workflow: Any, special_point: Any, **_: Any) -> QE_RUNNER_RESULT:
            point_index = int(special_point["point_index"])
            return self.prepare_retry_from_previous_attempt(
                point_index,
                scheduler_template=scheduler_template,
                directory_template=directory_template,
                job_name_template=job_name_template,
                workdir_root=workdir_root,
                outdir_root=outdir_root,
                outdir_template=outdir_template,
                submit=submit,
                subprocess_run=subprocess_run,
                update_positions_from_output=update_positions_from_output,
            )

        return runner

    def build_relax_retry_runner(
        self,
        *,
        scheduler_template: SlurmTemplate,
        directory_template: str = "qe_job_{point_index}",
        job_name_template: str = "qe-relax-retry-{point_index}-r{attempt:03d}",
        workdir_root: str | Path | None = None,
        outdir_root: str | Path | None = None,
        outdir_template: str | None = None,
        submit: bool = False,
        subprocess_run: Callable[..., Any] | None = None,
    ) -> Callable[..., QE_RUNNER_RESULT]:
        """Build a retry runner that seeds the next input from the last relax geometry."""
        return self.build_retry_runner(
            scheduler_template=scheduler_template,
            directory_template=directory_template,
            job_name_template=job_name_template,
            workdir_root=workdir_root,
            outdir_root=outdir_root,
            outdir_template=outdir_template,
            submit=submit,
            subprocess_run=subprocess_run,
            update_positions_from_output=True,
        )


@dataclass
class _QuantumEspressoParserBase(ABC):
    """Shared QE parsing helpers for workflow-facing parser classes."""

    iteration: int = 1
    pwo_name: str = "espresso.pwo"
    pwi_name: str = "espresso.pwi"
    adsorbate_label: str = "last_atom"

    re_iter: re.Pattern[str] = field(
        init=False,
        default=re.compile(r"^\s*iteration\s*#\s*(\d+)\b", re.IGNORECASE),
    )
    re_energy: re.Pattern[str] = field(
        init=False,
        default=re.compile(r"^\s*total energy\s*=\s*([-\d.]+)\s*Ry\b", re.IGNORECASE),
    )
    re_accuracy: re.Pattern[str] = field(
        init=False,
        default=re.compile(
            r"^\s*estimated scf accuracy\s*<\s*([-\d.Ee+]+)\s*Ry\b",
            re.IGNORECASE,
        ),
    )
    re_alat_au: re.Pattern[str] = field(
        init=False,
        default=re.compile(
            r"lattice parameter\s*\(alat\)\s*=\s*([-\d.Ee+]+)\s*a\.u\.",
            re.IGNORECASE,
        ),
    )
    re_atomic: re.Pattern[str] = field(
        init=False,
        default=re.compile(
            r"^\s*ATOMIC_POSITIONS(?:\s*[\(\{]?\s*([A-Za-z_]+)\s*[\)\}]?)?",
            re.IGNORECASE,
        ),
    )
    re_atom: re.Pattern[str] = field(
        init=False,
        default=re.compile(r"^\s*([A-Za-z]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)"),
    )
    re_energy_converged: re.Pattern[str] = field(
        init=False,
        default=re.compile(r"!\s+total energy\s*=\s*([-\d.]+)\s*Ry\b", re.IGNORECASE),
    )
    re_total_force: re.Pattern[str] = field(
        init=False,
        default=re.compile(r"total force\s*=\s*([-\d.Ee+]+)", re.IGNORECASE),
    )
    re_force_header: re.Pattern[str] = field(
        init=False,
        default=re.compile(r"^\s*forces acting on atoms", re.IGNORECASE),
    )
    re_force_atom: re.Pattern[str] = field(
        init=False,
        default=re.compile(
            r"^\s*atom\s+\d+\s+type\s+\d+\s+force\s*=\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)",
            re.IGNORECASE,
        ),
    )
    re_celldm1: re.Pattern[str] = field(
        init=False,
        default=re.compile(r"celldm\s*\(\s*1\s*\)\s*=\s*([-\d.Ee+]+)", re.IGNORECASE),
    )
    re_cellpar: re.Pattern[str] = field(
        init=False,
        default=re.compile(
            r"^\s*CELL_PARAMETERS(?:\s*[\(\{]?\s*([A-Za-z_]+)\s*[\)\}]?)?",
            re.IGNORECASE,
        ),
    )
    re_relax_converged: re.Pattern[str] = field(
        init=False,
        default=re.compile(
            r"(bfgs converged|end of bfgs geometry optimization|optimization achieved convergence)",
            re.IGNORECASE,
        ),
    )

    @abstractmethod
    def parse(self, output_file: str | Path) -> dict[str, Any]:
        """Parse one QE output directory/file into workflow metadata."""

    def __call__(self, output_file: str | Path, **_: Any) -> dict[str, Any]:
        return self.parse(output_file)

    def collect(
        self,
        root: str | Path = ".",
        *,
        pattern: str = "qe_job_*",
    ) -> pd.DataFrame:
        """Collect parsed QE metrics from a folder of job directories."""
        rows = []
        root_path = Path(root).expanduser()
        for folder in sorted(root_path.glob(pattern)):
            parsed = self.parse(folder)
            try:
                job_id: int | str = int(folder.name.split("_")[-1])
            except ValueError:
                job_id = folder.name
            rows.append({"job_id": job_id, **parsed})

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("job_id").reset_index(drop=True)

    def scf_metrics_at_iteration(
        self,
        pwo_path: Path,
        *,
        iteration: int,
    ) -> tuple[float, float]:
        if not pwo_path.exists():
            return (np.nan, np.nan)

        wanted = int(iteration)
        in_target_iteration = False
        energy = np.nan
        accuracy = np.nan

        with pwo_path.open("r", errors="replace") as handle:
            for line in handle:
                match_iteration = self.re_iter.match(line)
                if match_iteration:
                    if in_target_iteration:
                        break
                    in_target_iteration = int(match_iteration.group(1)) == wanted
                    continue
                if not in_target_iteration:
                    continue

                match_energy = self.re_energy.match(line)
                if match_energy:
                    energy = float(match_energy.group(1))
                    continue

                match_accuracy = self.re_accuracy.match(line)
                if match_accuracy:
                    accuracy = float(match_accuracy.group(1))
                    continue

                if not np.isnan(energy) and not np.isnan(accuracy):
                    return (energy, accuracy)

        return (energy, accuracy)

    def scf_metrics_first_below_accuracy(
        self,
        pwo_path: Path,
        *,
        accuracy_threshold: float,
    ) -> tuple[int | None, float, float]:
        if not pwo_path.exists():
            return (None, np.nan, np.nan)

        current_iteration: int | None = None
        current_energy = np.nan
        current_accuracy = np.nan

        with pwo_path.open("r", errors="replace") as handle:
            for line in handle:
                match_iteration = self.re_iter.match(line)
                if match_iteration:
                    current_iteration = int(match_iteration.group(1))
                    current_energy = np.nan
                    current_accuracy = np.nan
                    continue

                if current_iteration is None:
                    continue

                match_energy = self.re_energy.match(line)
                if match_energy:
                    current_energy = float(match_energy.group(1))
                    continue

                match_accuracy = self.re_accuracy.match(line)
                if match_accuracy:
                    current_accuracy = float(match_accuracy.group(1))
                    if current_accuracy < accuracy_threshold:
                        return (current_iteration, current_energy, current_accuracy)

        return (None, np.nan, np.nan)

    def converged_energies_first_last(self, pwo_path: Path) -> tuple[float, float]:
        if not pwo_path.exists():
            return (np.nan, np.nan)

        energy_first = np.nan
        energy_last = np.nan
        with pwo_path.open("r", errors="replace") as handle:
            for line in handle:
                match = self.re_energy_converged.search(line)
                if not match:
                    continue
                value = float(match.group(1))
                if np.isnan(energy_first):
                    energy_first = value
                energy_last = value
        return (energy_first, energy_last)

    def converged_energy_history(self, pwo_path: Path) -> np.ndarray:
        if not pwo_path.exists():
            return np.zeros(0, dtype=float)

        energies: list[float] = []
        with pwo_path.open("r", errors="replace") as handle:
            for line in handle:
                match = self.re_energy_converged.search(line)
                if match:
                    energies.append(float(match.group(1)))
        return np.array(energies, dtype=float)

    def scf_iteration_count(self, pwo_path: Path) -> int:
        if not pwo_path.exists():
            return 0
        count = 0
        with pwo_path.open("r", errors="replace") as handle:
            for line in handle:
                if self.re_iter.match(line):
                    count += 1
        return count

    def total_forces_first_last(self, pwo_path: Path) -> tuple[float, float]:
        if not pwo_path.exists():
            return (np.nan, np.nan)
        forces = self._parse_total_forces(self._read_lines(pwo_path))
        if not forces:
            return (np.nan, np.nan)
        return (forces[0], forces[-1])

    def last_atom_input_coordinates_from_pwi(
        self,
        pwi_path: Path,
        *,
        fallback_alat_angstrom: float = np.nan,
    ) -> np.ndarray:
        if not pwi_path.exists():
            return np.full(3, np.nan, dtype=float)

        lines = self._read_lines(pwi_path)
        alat_angstrom = np.nan
        for line in lines:
            match = self.re_celldm1.search(line)
            if match:
                alat_angstrom = float(match.group(1)) * BOHR_TO_ANG
                break
        if not np.isfinite(alat_angstrom):
            alat_angstrom = fallback_alat_angstrom

        cell, cell_unit = self._parse_cell_parameters_from_lines(lines)
        cell_angstrom = None
        if cell is not None:
            if "ang" in cell_unit:
                cell_angstrom = cell
            elif "bohr" in cell_unit:
                cell_angstrom = cell * BOHR_TO_ANG
            elif "alat" in cell_unit:
                cell_angstrom = cell * alat_angstrom if np.isfinite(alat_angstrom) else None

        block = self._parse_atomic_positions_first_block(lines)
        return self._last_atom_coordinates_from_block_angstrom(
            block,
            alat_angstrom=alat_angstrom,
            cell_angstrom=cell_angstrom,
        )

    def last_atom_final_coordinates_from_pwo(self, pwo_path: Path) -> np.ndarray:
        if not pwo_path.exists():
            return np.full(3, np.nan, dtype=float)

        lines = self._read_lines(pwo_path)
        alat_angstrom = self._extract_alat_angstrom_from_pwo(lines)
        blocks = self._parse_atomic_positions_blocks(lines)

        if not blocks:
            return np.full(3, np.nan, dtype=float)
        return self._last_atom_coordinates_from_block_angstrom(
            blocks[-1],
            alat_angstrom=alat_angstrom,
        )

    def last_atom_coordinates_history_from_pwo(self, pwo_path: Path) -> np.ndarray:
        if not pwo_path.exists():
            return np.zeros((0, 3), dtype=float)

        lines = self._read_lines(pwo_path)
        alat_angstrom = self._extract_alat_angstrom_from_pwo(lines)
        blocks = self._parse_atomic_positions_blocks(lines)
        if not blocks:
            return np.zeros((0, 3), dtype=float)

        coordinates = [
            self._last_atom_coordinates_from_block_angstrom(
                block,
                alat_angstrom=alat_angstrom,
            )
            for block in blocks
        ]
        return np.vstack(coordinates).astype(float)

    def ionic_steps_from_pwo(self, pwo_path: Path) -> int:
        if not pwo_path.exists():
            return 0
        return len(self._parse_atomic_positions_blocks(self._read_lines(pwo_path)))

    def adsorbate_forces_first_last(
        self,
        pwo_path: Path,
        pwi_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not pwo_path.exists():
            nan_force = np.full(3, np.nan, dtype=float)
            return (nan_force, nan_force.copy())

        index = self.adsorbate_atom_index_from_pwi(pwi_path)
        blocks = self._parse_force_blocks(self._read_lines(pwo_path))
        if not blocks:
            nan_force = np.full(3, np.nan, dtype=float)
            return (nan_force, nan_force.copy())

        first = self._adsorbate_force_from_block(blocks[0], index=index)
        last = self._adsorbate_force_from_block(blocks[-1], index=index)
        return (first, last)

    def relaxation_converged(self, pwo_path: Path) -> bool:
        if not pwo_path.exists():
            return False
        with pwo_path.open("r", errors="replace") as handle:
            return any(self.re_relax_converged.search(line) for line in handle)

    def adsorbate_atom_index_from_pwi(self, pwi_path: Path) -> int:
        if not pwi_path.exists():
            return -1
        block = self._parse_atomic_positions_first_block(self._read_lines(pwi_path))
        if block is None:
            return -1
        symbols = list(block["symbols"])
        matches = [index for index, symbol in enumerate(symbols) if symbol == self.adsorbate_label]
        if matches:
            return matches[-1]
        return len(symbols) - 1

    def _resolve_paths(self, output_file: str | Path) -> tuple[Path, Path, Path]:
        path = Path(output_file).expanduser()
        if path.is_dir():
            folder = path
            return folder, folder / self.pwi_name, folder / self.pwo_name
        folder = path.parent
        pwo_path = path if path.name == self.pwo_name else folder / self.pwo_name
        return folder, folder / self.pwi_name, pwo_path

    @staticmethod
    def _read_lines(path: Path) -> list[str]:
        if not path.exists():
            return []
        return path.read_text(errors="replace").splitlines()

    def _extract_alat_angstrom_from_pwo(self, lines: list[str]) -> float:
        for line in lines:
            match = self.re_alat_au.search(line)
            if match:
                return float(match.group(1)) * BOHR_TO_ANG
        return float("nan")

    def _parse_cell_parameters_from_lines(
        self,
        lines: list[str],
    ) -> tuple[np.ndarray | None, str]:
        for index, line in enumerate(lines):
            match = self.re_cellpar.match(line)
            if not match:
                continue
            unit = (match.group(1) or "").strip().lower()
            rows = []
            for offset in range(1, 4):
                if index + offset >= len(lines):
                    break
                parts = lines[index + offset].split()
                if len(parts) < 3:
                    break
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if len(rows) == 3:
                return np.array(rows, dtype=float), unit
        return None, ""

    def _parse_atomic_positions_first_block(self, lines: list[str]) -> dict[str, Any] | None:
        for index, line in enumerate(lines):
            match = self.re_atomic.match(line)
            if not match:
                continue
            unit = (match.group(1) or "").strip().lower()
            symbols = []
            positions = []
            cursor = index + 1
            while cursor < len(lines):
                atom_match = self.re_atom.match(lines[cursor])
                if not atom_match:
                    break
                symbols.append(atom_match.group(1))
                positions.append(
                    [
                        float(atom_match.group(2)),
                        float(atom_match.group(3)),
                        float(atom_match.group(4)),
                    ]
                )
                cursor += 1
            if positions:
                return {"unit": unit, "symbols": symbols, "pos": np.array(positions, dtype=float)}
            return None
        return None

    def _parse_atomic_positions_blocks(self, lines: list[str]) -> list[dict[str, Any]]:
        blocks = []
        index = 0
        while index < len(lines):
            match = self.re_atomic.match(lines[index])
            if not match:
                index += 1
                continue

            unit = (match.group(1) or "").strip().lower()
            index += 1
            symbols: list[str] = []
            positions: list[list[float]] = []
            while index < len(lines):
                atom_match = self.re_atom.match(lines[index])
                if not atom_match:
                    break
                symbols.append(atom_match.group(1))
                positions.append(
                    [
                        float(atom_match.group(2)),
                        float(atom_match.group(3)),
                        float(atom_match.group(4)),
                    ]
                )
                index += 1
            if positions:
                blocks.append(
                    {
                        "unit": unit,
                        "symbols": symbols,
                        "pos": np.array(positions, dtype=float),
                    }
                )
        return blocks

    def _parse_total_forces(self, lines: list[str]) -> list[float]:
        forces = []
        for line in lines:
            match = self.re_total_force.search(line)
            if match:
                forces.append(float(match.group(1)))
        return forces

    def _parse_force_blocks(self, lines: list[str]) -> list[np.ndarray]:
        blocks: list[np.ndarray] = []
        index = 0
        while index < len(lines):
            if not self.re_force_header.match(lines[index]):
                index += 1
                continue

            index += 1
            forces: list[list[float]] = []
            while index < len(lines):
                match = self.re_force_atom.match(lines[index])
                if match:
                    forces.append(
                        [
                            float(match.group(1)),
                            float(match.group(2)),
                            float(match.group(3)),
                        ]
                    )
                    index += 1
                    continue
                if forces:
                    break
                index += 1
            if forces:
                blocks.append(np.array(forces, dtype=float))
        return blocks

    @staticmethod
    def _adsorbate_force_from_block(block: np.ndarray, *, index: int) -> np.ndarray:
        if block.size == 0:
            return np.full(3, np.nan, dtype=float)
        resolved_index = index
        if resolved_index < 0 or resolved_index >= block.shape[0]:
            resolved_index = block.shape[0] - 1
        return np.array(block[resolved_index], dtype=float)

    @staticmethod
    def _last_atom_coordinates_from_block_angstrom(
        block: dict[str, Any] | None,
        *,
        alat_angstrom: float = np.nan,
        cell_angstrom: np.ndarray | None = None,
    ) -> np.ndarray:
        if block is None:
            return np.full(3, np.nan, dtype=float)

        unit = str(block["unit"])
        coords = np.array(block["pos"][-1, :], dtype=float)
        if "ang" in unit:
            return coords
        if "bohr" in unit:
            return coords * BOHR_TO_ANG
        if "alat" in unit:
            return (
                coords * alat_angstrom
                if np.isfinite(alat_angstrom)
                else np.full(3, np.nan, dtype=float)
            )
        if "crystal" in unit:
            if cell_angstrom is None:
                return np.full(3, np.nan, dtype=float)
            return cast(np.ndarray, coords @ cell_angstrom)
        return np.full(3, np.nan, dtype=float)


@dataclass
class QuantumEspressoScfParser(_QuantumEspressoParserBase):
    """Parser for QE SCF-oriented outputs without geometry bookkeeping."""

    accuracy_threshold: float | None = None

    def parse(self, output_file: str | Path) -> dict[str, Any]:
        folder, _, pwo_path = self._resolve_paths(output_file)
        energy_first, energy_last = self.converged_energies_first_last(pwo_path)
        parsed = {
            "label_file": str(folder),
            "E_converged_first_Ry": energy_first,
            "E_converged_last_Ry": energy_last,
            "n_scf_iterations": self.scf_iteration_count(pwo_path),
        }
        if self.accuracy_threshold is None:
            energy_iter, accuracy_iter = self.scf_metrics_at_iteration(
                pwo_path, iteration=self.iteration
            )
            parsed[f"E_iter{self.iteration}_Ry"] = energy_iter
            parsed[f"acc_iter{self.iteration}_Ry"] = accuracy_iter
        else:
            match_iteration, match_energy, match_accuracy = self.scf_metrics_first_below_accuracy(
                pwo_path,
                accuracy_threshold=self.accuracy_threshold,
            )
            parsed["scf_accuracy_threshold_Ry"] = self.accuracy_threshold
            parsed["first_iteration_below_accuracy_threshold"] = match_iteration
            parsed["E_first_below_accuracy_threshold_Ry"] = match_energy
            parsed["acc_first_below_accuracy_threshold_Ry"] = match_accuracy
        return parsed


@dataclass
class QuantumEspressoRelaxParser(_QuantumEspressoParserBase):
    """Parser for QE relax outputs focused on ionic-step energies, forces, and geometry."""

    mark_unconverged_as_failed: bool = True

    def parse(self, output_file: str | Path) -> dict[str, Any]:
        folder, pwi_path, pwo_path = self._resolve_paths(output_file)
        pwo_lines = self._read_lines(pwo_path)
        alat_angstrom = self._extract_alat_angstrom_from_pwo(pwo_lines)
        input_coordinates = self.last_atom_input_coordinates_from_pwi(
            pwi_path,
            fallback_alat_angstrom=alat_angstrom,
        )
        final_coordinates = self.last_atom_final_coordinates_from_pwo(pwo_path)
        coordinates_history = self.last_atom_coordinates_history_from_pwo(pwo_path)
        energy_first, energy_last = self.converged_energies_first_last(pwo_path)
        energy_history = self.converged_energy_history(pwo_path)
        total_force_first, total_force_last = self.total_forces_first_last(pwo_path)
        adsorbate_force_first, adsorbate_force_last = self.adsorbate_forces_first_last(
            pwo_path,
            pwi_path,
        )
        relax_converged = self.relaxation_converged(pwo_path)
        label = self.adsorbate_label
        parsed = {
            "label_file": str(folder),
            "E_bfgs_initial_Ry": energy_first,
            "E_bfgs_final_Ry": energy_last,
            "E_bfgs_steps_Ry": energy_history,
            "F_bfgs_initial_total_Ry_au": total_force_first,
            "F_bfgs_final_total_Ry_au": total_force_last,
            f"fx_{label}_initial_Ry_au": adsorbate_force_first[0],
            f"fy_{label}_initial_Ry_au": adsorbate_force_first[1],
            f"fz_{label}_initial_Ry_au": adsorbate_force_first[2],
            f"f_{label}_initial_abs_Ry_au": float(np.linalg.norm(adsorbate_force_first)),
            f"fx_{label}_final_Ry_au": adsorbate_force_last[0],
            f"fy_{label}_final_Ry_au": adsorbate_force_last[1],
            f"fz_{label}_final_Ry_au": adsorbate_force_last[2],
            f"f_{label}_final_abs_Ry_au": float(np.linalg.norm(adsorbate_force_last)),
            f"x_{label}_input_A": input_coordinates[0],
            f"y_{label}_input_A": input_coordinates[1],
            f"z_{label}_input_A": input_coordinates[2],
            f"x_{label}_final_A": final_coordinates[0],
            f"y_{label}_final_A": final_coordinates[1],
            f"z_{label}_final_A": final_coordinates[2],
            f"{label}_positions_bfgs_A": coordinates_history,
            f"x_{label}_bfgs_steps_A": (
                coordinates_history[:, 0] if coordinates_history.size else np.zeros(0, dtype=float)
            ),
            f"y_{label}_bfgs_steps_A": (
                coordinates_history[:, 1] if coordinates_history.size else np.zeros(0, dtype=float)
            ),
            f"z_{label}_bfgs_steps_A": (
                coordinates_history[:, 2] if coordinates_history.size else np.zeros(0, dtype=float)
            ),
            "n_ionic_steps": self.ionic_steps_from_pwo(pwo_path),
            "relax_converged": relax_converged,
        }
        if self.mark_unconverged_as_failed and not relax_converged:
            parsed["label_status"] = "failed"
            parsed["label_error"] = (
                "Relax calculation did not converge: missing 'End of BFGS Geometry Optimization' "
                f"marker in {pwo_path.name}."
            )
        return parsed


@dataclass
class QuantumEspressoMultiRelaxParser(_QuantumEspressoParserBase):
    """Retry-aware parser for relax jobs with one or more attempt files in one directory."""

    mark_unconverged_as_failed: bool = True
    include_attempt_summaries: bool = True

    def parse(self, output_file: str | Path) -> dict[str, Any]:
        folder, _, _ = self._resolve_paths(output_file)
        attempt_indexes = self._attempt_indexes_from_folder(folder)
        if not attempt_indexes:
            attempt_indexes = [0]

        attempt_results: list[tuple[int, dict[str, Any]]] = []
        for attempt_index in attempt_indexes:
            parser = QuantumEspressoRelaxParser(
                iteration=self.iteration,
                pwo_name=_with_attempt_suffix(self.pwo_name, attempt_index),
                pwi_name=_with_attempt_suffix(self.pwi_name, attempt_index),
                adsorbate_label=self.adsorbate_label,
                mark_unconverged_as_failed=False,
            )
            attempt_results.append((attempt_index, parser.parse(folder)))

        latest_attempt_index, latest_attempt = attempt_results[-1]
        converged_attempts = [
            (attempt_index, parsed)
            for attempt_index, parsed in attempt_results
            if bool(parsed.get("relax_converged"))
        ]
        if converged_attempts:
            selected_attempt_index, selected_attempt = converged_attempts[-1]
        else:
            selected_attempt_index, selected_attempt = latest_attempt_index, latest_attempt

        parsed = dict(selected_attempt)
        parsed["label_file"] = str(folder)
        parsed["n_relax_attempts"] = len(attempt_results)
        parsed["latest_attempt_index"] = latest_attempt_index
        parsed["selected_attempt_index"] = selected_attempt_index
        parsed["selected_output_file"] = str(
            folder / _with_attempt_suffix(self.pwo_name, selected_attempt_index)
        )
        parsed["selected_input_file"] = str(
            folder / _with_attempt_suffix(self.pwi_name, selected_attempt_index)
        )

        if self.include_attempt_summaries:
            for attempt_index, attempt_parsed in attempt_results:
                prefix = f"attempt_{attempt_index:03d}"
                parsed[f"{prefix}_output_file"] = str(
                    folder / _with_attempt_suffix(self.pwo_name, attempt_index)
                )
                parsed[f"{prefix}_input_file"] = str(
                    folder / _with_attempt_suffix(self.pwi_name, attempt_index)
                )
                parsed[f"{prefix}_relax_converged"] = bool(
                    attempt_parsed.get("relax_converged", False)
                )
                parsed[f"{prefix}_n_ionic_steps"] = attempt_parsed.get("n_ionic_steps")
                parsed[f"{prefix}_E_bfgs_final_Ry"] = attempt_parsed.get("E_bfgs_final_Ry")
                if "label_error" in attempt_parsed:
                    parsed[f"{prefix}_label_error"] = attempt_parsed["label_error"]

        if self.mark_unconverged_as_failed and not converged_attempts:
            latest_output_name = _with_attempt_suffix(self.pwo_name, latest_attempt_index)
            parsed["label_status"] = "failed"
            parsed["label_error"] = (
                "Relax calculation did not converge in any attempt: missing "
                f"'End of BFGS Geometry Optimization' marker through {latest_output_name}."
            )
        else:
            parsed.pop("label_status", None)
            parsed.pop("label_error", None)

        return parsed

    def _attempt_indexes_from_folder(self, folder: Path) -> list[int]:
        indexes: list[int] = []
        if not folder.exists():
            return indexes
        for path in folder.iterdir():
            if not path.is_file():
                continue
            attempt_index = _attempt_index_from_name(path.name, self.pwo_name)
            if attempt_index is not None:
                indexes.append(attempt_index)
        return sorted(set(indexes))
