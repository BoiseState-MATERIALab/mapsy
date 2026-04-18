import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian

from mapsy import Maps, QuantumEspressoSetup, SlurmTemplate
from mapsy.data import Grid, System


def _build_maps(frame: pd.DataFrame) -> Maps:
    cell = np.diag([10.0, 10.0, 10.0])
    grid = Grid(scalars=[2, 2, 2], cell=cell)
    atoms = Atoms("Co2", positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], cell=cell, pbc=True)
    maps = Maps(System(grid=grid, atoms=atoms), [])
    maps.data = frame.copy()
    maps.features = [column for column in frame.columns if column not in {"x", "y", "z"}]
    return maps


class FakeProfile:
    def __init__(self, command: str, pseudo_dir: str) -> None:
        self.command = command
        self.pseudo_dir = pseudo_dir


class FakeEspresso:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeCalc:
    def get_potential_energy(self, atoms=None, force_consistent=False):
        return -1.234


class FakePreparedCalc:
    def __init__(self, input_path: str | None = None) -> None:
        self.written = False
        self.input_path = input_path

    def write_input(self, atoms) -> None:
        self.written = True
        if self.input_path is not None:
            with open(self.input_path, "w", encoding="utf-8") as handle:
                handle.write("&CONTROL\n")
                handle.write("   calculation = 'relax'\n")
                handle.write("/\n")


def test_qe_setup_generates_metadata_and_profile() -> None:
    setup = QuantumEspressoSetup(
        runprefix="mpirun -np 1 ",
        qepath="/opt/qe/bin",
        pseudodir="/opt/qe/pseudo",
        pseudopotentials={"H": "H.UPF"},
        input_data={"control": {"calculation": "relax", "prefix": "CoP"}},
        tstress=True,
        tprnfor=True,
    )

    metadata = setup.workflow_description()
    assert metadata["engine"] == "quantum_espresso"
    assert metadata["command"] == "mpirun -np 1 /opt/qe/bin/pw.x"
    assert (
        metadata["run_command"] == "mpirun -np 1 /opt/qe/bin/pw.x -in espresso.pwi > espresso.pwo"
    )
    assert metadata["pseudo_dir"] == "/opt/qe/pseudo"
    assert metadata["input_data"]["control"]["calculation"] == "relax"

    profile = setup.make_profile(profile_cls=FakeProfile)
    assert profile.command == "mpirun -np 1 /opt/qe/bin/pw.x"
    assert profile.pseudo_dir == "/opt/qe/pseudo"


def test_qe_setup_adds_single_adsorbate_with_constraints() -> None:
    substrate = Atoms("Co2", positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    setup = QuantumEspressoSetup()

    atoms = setup.add_single_atom_adsorbate(substrate, "H", [2.0, 3.0, 4.0])

    assert len(atoms) == 3
    assert atoms[-1].symbol == "H"
    assert atoms.positions[-1].tolist() == [2.0, 3.0, 4.0]
    assert len(atoms.constraints) == 2
    assert isinstance(atoms.constraints[0], FixAtoms)
    assert isinstance(atoms.constraints[1], FixCartesian)


def test_qe_setup_runs_single_fake_calculation_and_builds_workflow_runner(tmp_path) -> None:
    setup = QuantumEspressoSetup(
        runprefix="mpirun -np 1 ",
        qepath="/opt/qe/bin",
        pseudodir="/opt/qe/pseudo",
        pseudopotentials={"Co": "Co.UPF", "H": "H.UPF"},
        input_data={"control": {"calculation": "relax", "prefix": "CoP"}},
    )

    created: dict[str, object] = {}

    def fake_make_calculator(**kwargs):
        created.update(kwargs)
        return FakeCalc()

    setup.make_calculator = fake_make_calculator  # type: ignore[method-assign]

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    result = setup.run_single_calculation(5, atoms, workdir_root=tmp_path)

    assert result["label_file"] == str(tmp_path / "qe_job_5")
    assert result["runner_energy"] == -1.234
    assert created["directory"] == tmp_path / "qe_job_5"

    frame = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})
    maps = _build_maps(frame)
    maps.add_special_points([0], kind="centroid")

    runner = setup.build_single_adsorbate_runner(
        maps.system.atoms,
        "H",
        workdir_root=tmp_path,
    )
    special_point = maps.get_special_points(kind="centroid").iloc[0]
    metadata = runner(maps=maps, workflow=None, special_point=special_point)

    assert metadata["label_file"] == str(tmp_path / "qe_job_0")


def test_qe_setup_prepares_slurm_job_and_cluster_runner(tmp_path) -> None:
    setup = QuantumEspressoSetup(
        runprefix="srun ",
        qepath="/opt/qe/bin",
        pseudodir="/opt/qe/pseudo",
        pseudopotentials={"Co": "Co.UPF", "H": "H.UPF"},
        input_data={"control": {"calculation": "relax", "prefix": "CoP"}},
    )
    scheduler = SlurmTemplate(
        partition="debug",
        time="00:10:00",
        modules=["qe/7.2"],
        setup_commands=["echo preparing"],
    )

    created: dict[str, object] = {}

    def fake_make_calculator(**kwargs):
        created.update(kwargs)
        return FakePreparedCalc(input_path=str(tmp_path / "qe_job_7" / "espresso.pwi"))

    class FakeCompleted:
        stdout = "Submitted batch job 321\n"
        stderr = ""

    def fake_subprocess_run(*args, **kwargs):
        return FakeCompleted()

    setup.make_calculator = fake_make_calculator  # type: ignore[method-assign]

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    metadata = setup.prepare_single_calculation(
        7,
        atoms,
        scheduler_template=scheduler,
        workdir_root=tmp_path,
        submit=True,
        subprocess_run=fake_subprocess_run,
    )

    script_path = tmp_path / "qe_job_7" / "submit.slurm"
    script_text = script_path.read_text()

    input_text = (tmp_path / "qe_job_7" / "espresso.pwi").read_text()

    assert created["directory"] == tmp_path / "qe_job_7"
    assert metadata["label_file"] == str(tmp_path / "qe_job_7")
    assert metadata["submit_script"] == str(script_path)
    assert metadata["scheduler_job_id"] == 321
    assert "#SBATCH --partition=debug" in script_text
    assert "module load qe/7.2" in script_text
    assert "srun /opt/qe/bin/pw.x -in espresso.pwi > espresso.pwo" in script_text
    assert "outdir =" not in input_text

    frame = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})
    maps = _build_maps(frame)
    maps.add_special_points([0], kind="centroid")

    runner = setup.build_single_adsorbate_prepare_runner(
        maps.system.atoms,
        "H",
        scheduler_template=scheduler,
        workdir_root=tmp_path,
        submit=False,
    )
    special_point = maps.get_special_points(kind="centroid").iloc[0]
    prepared = runner(maps=maps, workflow=None, special_point=special_point)

    assert prepared["label_file"] == str(tmp_path / "qe_job_0")
    assert prepared["submit_script"] == str(tmp_path / "qe_job_0" / "submit.slurm")


def test_qe_setup_generates_dynamic_scratch_outdir(tmp_path) -> None:
    scratch_root = tmp_path / "scratch"
    setup = QuantumEspressoSetup(
        runprefix="srun ",
        qepath="/opt/qe/bin",
        pseudodir="/opt/qe/pseudo",
        pseudopotentials={"Co": "Co.UPF", "H": "H.UPF"},
        input_data={"control": {"calculation": "relax", "prefix": "CoP"}},
        outdir_root=str(scratch_root),
    )
    scheduler = SlurmTemplate(
        partition="debug",
        time="00:10:00",
        modules=["qe/7.2"],
    )

    created: dict[str, object] = {}

    def fake_make_calculator(**kwargs):
        created.update(kwargs)
        return FakePreparedCalc(input_path=str(tmp_path / "qe_job_7" / "espresso.pwi"))

    setup.make_calculator = fake_make_calculator  # type: ignore[method-assign]

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    metadata = setup.prepare_single_calculation(
        7,
        atoms,
        scheduler_template=scheduler,
        workdir_root=tmp_path,
    )

    expected_outdir = scratch_root / "qe_job_7"
    script_text = (tmp_path / "qe_job_7" / "submit.slurm").read_text()
    input_text = (tmp_path / "qe_job_7" / "espresso.pwi").read_text()

    assert created["qe_outdir"] == expected_outdir
    assert expected_outdir.exists()
    assert metadata["qe_outdir"] == str(expected_outdir)
    assert f"mkdir -p {expected_outdir}" in script_text
    assert f"outdir = '{expected_outdir}'" in input_text


def test_qe_setup_prepares_relax_retry_from_previous_attempt(tmp_path) -> None:
    scratch_root = tmp_path / "scratch"
    workdir = tmp_path / "qe_job_7"
    workdir.mkdir()
    setup = QuantumEspressoSetup(
        runprefix="srun ",
        qepath="/opt/qe/bin",
        pseudodir="/opt/qe/pseudo",
        pseudopotentials={"Co": "Co.UPF", "H": "H.UPF"},
        input_data={"control": {"calculation": "relax", "prefix": "CoP"}},
        outdir_root=str(scratch_root),
    )
    scheduler = SlurmTemplate(partition="debug", time="00:10:00", modules=["qe/7.2"])

    (workdir / "espresso.pwi").write_text(
        "\n".join(
            [
                "&CONTROL",
                "   calculation = 'relax'",
                "/",
                "ATOMIC_POSITIONS (angstrom)",
                "Co 0.0 0.0 1.0",
                "H 0.1 0.2 2.0",
            ]
        )
        + "\n"
    )
    (workdir / "espresso.pwo").write_text(
        "\n".join(
            [
                "!    total energy              =   -123.4000 Ry",
                "ATOMIC_POSITIONS (angstrom)",
                "Co 0.0 0.0 1.1",
                "H 0.3 0.4 2.2",
            ]
        )
        + "\n"
    )

    metadata = setup.prepare_retry_from_previous_attempt(
        7,
        scheduler_template=scheduler,
        workdir_root=tmp_path,
        update_positions_from_output=True,
    )

    retry_input = workdir / "espresso.r001.pwi"
    retry_script = workdir / "submit.r001.slurm"
    retry_input_text = retry_input.read_text()
    retry_script_text = retry_script.read_text()

    assert metadata["label_file"] == str(workdir)
    assert metadata["retry_attempt_index"] == 1
    assert metadata["submit_script"] == str(retry_script)
    assert "H 0.3 0.4 2.2" in retry_input_text
    assert "espresso.r001.pwi > espresso.r001.pwo" in retry_script_text
    assert "mkdir -p" in retry_script_text
    assert "outdir = '" in retry_input_text
