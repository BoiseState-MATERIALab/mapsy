import threading

import numpy as np
import pandas as pd
from ase import Atoms

from mapsy import CalculationWorkflow, Maps
from mapsy.data import Grid, System


def _build_maps(frame: pd.DataFrame) -> Maps:
    cell = np.diag([10.0, 10.0, 10.0])
    grid = Grid(scalars=[2, 2, 2], cell=cell)
    atoms = Atoms("H", positions=[[5.0, 5.0, 5.0]], cell=cell, pbc=True)
    maps = Maps(System(grid=grid, atoms=atoms), [])
    maps.data = frame.copy()
    maps.features = [column for column in frame.columns if column not in {"x", "y", "z"}]
    return maps


def test_calculation_workflow_run_and_collect_updates_special_points() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "predicted_label": [-0.2, -0.1],
        }
    )
    maps = _build_maps(frame)
    maps.add_special_points([1], kind="adaptive", iteration=1, label_status="unlabeled")

    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock", "method": "relax"},
        runner=lambda maps, workflow, special_point: f"calc_{int(special_point['point_index'])}.out",
        parser=lambda output_file, maps, workflow, special_point: {
            "observed_energy_initial": -2.0,
            "observed_energy_relaxed": -1.5 * int(special_point["point_index"]),
            "observed_n_scf_steps": 7,
            "parsed_from": output_file,
        },
    )

    workflow.run(maps, kind="adaptive")
    after_run = maps.get_special_points(kind="adaptive")
    assert after_run["label_status"].tolist() == ["unlabeled"]
    assert after_run["calculation_name"].tolist() == ["qe_relax"]
    assert after_run["label_name"].tolist() == ["qe_relax"]
    assert after_run["label_file"].tolist() == ["calc_1.out"]

    workflow.collect(maps, kind="adaptive")
    collected = maps.get_special_points(kind="adaptive")
    assert collected["label_status"].tolist() == ["completed"]
    assert collected["observed_energy_initial"].tolist() == [-2.0]
    assert collected["observed_energy_relaxed"].tolist() == [-1.5]
    assert collected["observed_n_scf_steps"].tolist() == [7]
    assert collected["parsed_from"].tolist() == ["calc_1.out"]


def test_calculation_workflow_record_outputs_supports_external_runner_and_failures() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "predicted_label": [-0.2],
        }
    )
    maps = _build_maps(frame)
    maps.add_special_points([0], kind="manual", iteration=0, label_status="unlabeled")

    workflow = CalculationWorkflow(
        calculation_name="relaxation",
        calculation_description="mock relaxation workflow",
        parser=lambda output_file, maps, workflow, special_point: (_ for _ in ()).throw(
            RuntimeError(f"Could not parse {output_file}")
        ),
    )

    workflow.record_outputs(maps, [0], ["failed.out"], kind="manual")
    attached = maps.get_special_points(kind="manual")
    assert attached["label_file"].tolist() == ["failed.out"]
    assert attached["calculation_name"].tolist() == ["relaxation"]

    workflow.collect(maps, kind="manual")
    failed = maps.get_special_points(kind="manual")
    assert failed["label_status"].tolist() == ["failed"]
    assert "Could not parse failed.out" in failed["label_error"].tolist()[0]


def test_calculation_workflow_run_supports_parallel_runner_execution() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }
    )
    maps = _build_maps(frame)
    maps.add_special_points([0, 1], kind="adaptive", iteration=1, label_status="unlabeled")

    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock", "method": "relax"},
        runner=lambda maps, workflow, special_point: {
            "label_file": f"calc_{int(special_point['point_index'])}.out",
            "runner_thread": threading.current_thread().name,
        },
    )

    workflow.run(maps, kind="adaptive", parallel=True, max_workers=2)
    after_run = maps.get_special_points(kind="adaptive").sort_values("point_index")

    assert after_run["label_file"].tolist() == ["calc_0.out", "calc_1.out"]
    assert all(name != "MainThread" for name in after_run["runner_thread"].tolist())
