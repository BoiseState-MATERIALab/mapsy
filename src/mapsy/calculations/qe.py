from __future__ import annotations

import re
import shlex
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

        local_input_data = deepcopy(self.input_data)
        control = deepcopy(local_input_data.get("control", {}))
        control.setdefault("pseudo_dir", str(Path(self.pseudodir).expanduser()))
        if qe_outdir is not None:
            control["outdir"] = str(Path(qe_outdir).expanduser())
        local_input_data["control"] = control

        kwargs = deepcopy(self.calculator_kwargs)
        if self.tstress is not None:
            kwargs.setdefault("tstress", self.tstress)
        if self.tprnfor is not None:
            kwargs.setdefault("tprnfor", self.tprnfor)
        if calculator_overrides:
            kwargs.update(calculator_overrides)

        return calculator_factory(
            profile=self.make_profile(profile_cls=profile_cls) if profile is None else profile,
            pseudopotentials=dict(self.pseudopotentials),
            input_data=local_input_data,
            directory=str(Path(directory).expanduser()),
            kpts=tuple(self.kpts),
            koffset=tuple(self.koffset),
            **kwargs,
        )

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
        input_writer: Callable[..., Any] | None = None,
    ) -> None:
        """Write QE input files for a prepared job directory."""
        if input_writer is not None:
            input_writer(atoms=atoms, calculator=calculator, setup=self)
            return
        write_input = getattr(calculator, "write_input", None)
        if callable(write_input):
            write_input(atoms)
            return
        raise AttributeError(
            "Calculator does not expose write_input and no input_writer was provided."
        )

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
        self.write_inputs(atoms_to_write, calculator, input_writer=input_writer)

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


@dataclass
class QuantumEspressoOutputParser:
    """Parser for common QE output metrics, usable as a CalculationWorkflow parser."""

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

    def __call__(self, output_file: str | Path, **_: Any) -> dict[str, Any]:
        return self.parse(output_file)

    def parse(self, output_file: str | Path) -> dict[str, Any]:
        """Parse one QE work directory or one QE output file into structured metrics."""
        folder, pwi_path, pwo_path = self._resolve_paths(output_file)

        energy_iter, accuracy_iter = self.scf_metrics_at_iteration(
            pwo_path, iteration=self.iteration
        )
        energy_first, energy_last = self.converged_energies_first_last(pwo_path)
        pwo_lines = self._read_lines(pwo_path)
        alat_angstrom = self._extract_alat_angstrom_from_pwo(pwo_lines)

        input_coordinates = self.last_atom_input_coordinates_from_pwi(
            pwi_path,
            fallback_alat_angstrom=alat_angstrom,
        )
        final_coordinates = self.last_atom_final_coordinates_from_pwo(pwo_path)
        ionic_steps = self.ionic_steps_from_pwo(pwo_path)
        relax_converged = self.relaxation_converged(pwo_path)

        label = self.adsorbate_label
        return {
            "label_file": str(folder),
            f"E_iter{self.iteration}_Ry": energy_iter,
            f"acc_iter{self.iteration}_Ry": accuracy_iter,
            "E_after_first_SCF_Ry": energy_first,
            "E_end_BFGS_Ry": energy_last,
            f"x_{label}_input_A": input_coordinates[0],
            f"y_{label}_input_A": input_coordinates[1],
            f"z_{label}_input_A": input_coordinates[2],
            f"x_{label}_last_A": final_coordinates[0],
            f"y_{label}_last_A": final_coordinates[1],
            f"z_{label}_last_A": final_coordinates[2],
            "n_ionic_steps": ionic_steps,
            "relax_converged": relax_converged,
        }

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

    def ionic_steps_from_pwo(self, pwo_path: Path) -> int:
        if not pwo_path.exists():
            return 0
        return len(self._parse_atomic_positions_blocks(self._read_lines(pwo_path)))

    def relaxation_converged(self, pwo_path: Path) -> bool:
        if not pwo_path.exists():
            return False
        with pwo_path.open("r", errors="replace") as handle:
            return any(self.re_relax_converged.search(line) for line in handle)

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
