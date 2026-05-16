import threading

import numpy as np
import pandas as pd
from ase import Atoms

from mapsy import (
    CalculationWorkflow,
    Maps,
    MultiMaps,
    build_adsorption_energy_parser,
    build_che_adsorption_energy_parser,
    build_relax_adsorption_parser,
)
from mapsy.data import Grid, System


def _build_maps(frame: pd.DataFrame) -> Maps:
    cell = np.diag([10.0, 10.0, 10.0])
    grid = Grid(scalars=[2, 2, 2], cell=cell)
    atoms = Atoms("H", positions=[[5.0, 5.0, 5.0]], cell=cell, pbc=True)
    maps = Maps(System(grid=grid, atoms=atoms), [])
    maps.data = frame.copy()
    maps.features = [column for column in frame.columns if column not in {"x", "y", "z"}]
    return maps


def _build_multimaps(frames: list[pd.DataFrame], names: list[str] | None = None) -> MultiMaps:
    maps_list = [_build_maps(frame) for frame in frames]
    return MultiMaps(maps_list, names=names)


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
    assert after_run["calculation_description"].tolist() == [{"code": "mock", "method": "relax"}]
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
    assert after_run["label_status"].tolist() == ["unlabeled", "unlabeled"]
    assert all(name != "MainThread" for name in after_run["runner_thread"].tolist())


def test_calculation_workflow_retry_failed_targets_only_failed_points() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }
    )
    maps = _build_maps(frame)
    maps.add_special_points([0], kind="adaptive", iteration=1, label_status="failed")
    maps.add_special_points([1], kind="adaptive", iteration=1, label_status="completed")

    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock", "method": "relax"},
        runner=lambda maps, workflow, special_point: {
            "label_file": f"retry_{int(special_point['point_index'])}.out",
        },
    )

    workflow.retry_failed(maps, kind="adaptive")
    special = maps.get_special_points(kind="adaptive").sort_values("point_index")

    assert special.iloc[0]["label_file"] == "retry_0.out"
    assert pd.isna(special.iloc[1]["label_file"])
    assert special["label_status"].tolist() == ["unlabeled", "completed"]


def test_calculation_workflow_record_outputs_resets_points_for_recollection() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
        }
    )
    maps = _build_maps(frame)
    maps.add_special_points([0], kind="centroid", iteration=0, label_status="completed")

    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock", "method": "relax"},
        parser=lambda output_file, maps, workflow, special_point: {
            "parsed_from": output_file,
        },
    )

    workflow.record_outputs(maps, [0], ["qe_job_0"], kind="centroid")
    after_record = maps.get_special_points(kind="centroid")

    assert after_record["label_status"].tolist() == ["unlabeled"]
    assert after_record["label_file"].tolist() == ["qe_job_0"]

    collected = workflow.collect(maps, kind="centroid")
    assert collected["label_status"].tolist() == ["completed"]
    assert collected["parsed_from"].tolist() == ["qe_job_0"]


def test_calculation_workflow_collect_discovered_finds_existing_job_folders(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }
    )
    maps = _build_maps(frame)
    maps.add_special_points([0, 1], kind="centroid", iteration=0, label_status="completed")

    (tmp_path / "qe_job_0").mkdir()
    (tmp_path / "qe_job_1").mkdir()

    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock", "method": "relax"},
        parser=lambda output_file, maps, workflow, special_point: {
            "parsed_from": output_file,
        },
    )

    collected = workflow.collect(
        maps,
        root=tmp_path,
        kind="centroid",
    ).sort_values("point_index")

    assert collected["point_index"].tolist() == [0, 1]
    assert collected["label_status"].tolist() == ["completed", "completed"]
    assert collected["parsed_from"].tolist() == [
        str(tmp_path / "qe_job_0"),
        str(tmp_path / "qe_job_1"),
    ]


def test_calculation_workflow_collect_discovers_missing_label_files_for_unlabeled_points(
    tmp_path,
) -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
        }
    )
    maps = _build_maps(frame)
    maps.add_special_points([0], kind="centroid", iteration=0, label_status="unlabeled")

    (tmp_path / "qe_job_0").mkdir()

    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock", "method": "relax"},
        parser=lambda output_file, maps, workflow, special_point: {
            "parsed_from": output_file,
        },
    )

    collected = workflow.collect(
        maps,
        root=tmp_path,
        kind="centroid",
    )

    assert collected["label_status"].tolist() == ["completed"]
    assert collected["parsed_from"].tolist() == [str(tmp_path / "qe_job_0")]


def test_calculation_workflow_collect_accepts_array_metadata_for_single_point() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
        }
    )
    maps = _build_maps(frame)
    maps.add_special_points([0], kind="centroid", iteration=0, label_status="unlabeled")

    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock", "method": "relax"},
        parser=lambda output_file, maps, workflow, special_point: {
            "E_bfgs_steps_Ry": np.array([-1.0, -2.0, -3.0]),
            "H_positions_bfgs_A": np.array([[0.0, 0.0, 1.0], [0.1, 0.2, 1.1]]),
        },
    )

    workflow.record_outputs(maps, [0], ["qe_job_0"], kind="centroid")
    collected = workflow.collect(maps, kind="centroid")

    assert collected["label_status"].tolist() == ["completed"]
    np.testing.assert_allclose(collected.iloc[0]["E_bfgs_steps_Ry"], np.array([-1.0, -2.0, -3.0]))
    np.testing.assert_allclose(
        collected.iloc[0]["H_positions_bfgs_A"],
        np.array([[0.0, 0.0, 1.0], [0.1, 0.2, 1.1]]),
    )


def test_calculation_workflow_can_run_and_collect_global_reference() -> None:
    maps = _build_maps(pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]}))
    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock"},
    )
    workflow.add_reference(
        "H2",
        scope="global",
        runner=lambda maps, workflow, reference, subject, **kwargs: "H2.out",
        parser=lambda output_file, maps, workflow, reference, subject, **kwargs: {
            "reference_energy": -6.8,
            "parsed_from": output_file,
        },
    )

    workflow.run_references(maps)
    workflow.collect_references(maps)
    record = workflow.get_reference("H2")

    assert record["label_status"] == "completed"
    assert record["reference_energy"] == -6.8
    assert record["parsed_from"] == "H2.out"


def test_calculation_workflow_can_collect_per_map_references_on_multimaps() -> None:
    multimaps = _build_multimaps(
        [
            pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]}),
            pd.DataFrame({"x": [1.0], "y": [0.0], "z": [0.0]}),
        ],
        names=["slab_a", "slab_b"],
    )
    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock"},
    )
    workflow.add_reference(
        "clean_slab",
        scope="per_map",
        runner=lambda maps, workflow, reference, subject, map_index, system_name, **kwargs: {
            "reference_file": f"{system_name}_{map_index}.out",
        },
        parser=lambda output_file, maps, workflow, reference, subject, map_index, system_name, **kwargs: {
            "reference_energy": -10.0 - float(map_index),
            "parsed_from": output_file,
            "system_name_seen": system_name,
        },
    )

    run_frame = workflow.run_references(multimaps).sort_values("map_index")
    collected = workflow.collect_references(multimaps).sort_values("map_index")

    assert run_frame["reference_file"].tolist() == ["slab_a_0.out", "slab_b_1.out"]
    assert collected["reference_energy"].tolist() == [-10.0, -11.0]
    assert workflow.get_reference("clean_slab", map_index=0)["system_name_seen"] == "slab_a"
    assert workflow.get_reference("clean_slab", map_index=1)["system_name_seen"] == "slab_b"


def test_calculation_workflow_reference_postprocess_supports_derived_metadata() -> None:
    maps = _build_maps(pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]}))
    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock"},
    )
    workflow.add_reference(
        "H2",
        scope="global",
        runner=lambda maps, workflow, reference, subject, **kwargs: "H2.out",
        parser=lambda output_file, maps, workflow, reference, subject, **kwargs: {
            "reference_energy": -6.8,
            "zpe": 0.27,
            "entropy_term": -0.41,
        },
        postprocess=lambda parsed, **kwargs: {
            "mu_half_h2": 0.5
            * (parsed["reference_energy"] + parsed["zpe"] + parsed["entropy_term"])
        },
    )

    workflow.run_references(maps)
    workflow.collect_references(maps)
    record = workflow.get_reference("H2")

    assert np.isclose(record["mu_half_h2"], 0.5 * (-6.8 + 0.27 - 0.41))


def test_collect_references_discovers_existing_global_reference_folder(tmp_path) -> None:
    maps = _build_maps(pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]}))
    (tmp_path / "global" / "H2").mkdir(parents=True)

    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock"},
    )
    workflow.add_reference(
        "H2",
        scope="global",
        parser=lambda output_file, maps, workflow, reference, subject, **kwargs: {
            "reference_energy": -6.8,
            "parsed_from": output_file,
        },
    )

    collected = workflow.collect_references(maps, names=["H2"], root=tmp_path)

    assert collected["label_status"].tolist() == ["completed"]
    assert collected["reference_energy"].tolist() == [-6.8]
    assert collected["parsed_from"].tolist() == [str(tmp_path / "global" / "H2")]


def test_collect_references_discovers_existing_per_map_reference_folders(tmp_path) -> None:
    multimaps = _build_multimaps(
        [
            pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]}),
            pd.DataFrame({"x": [1.0], "y": [0.0], "z": [0.0]}),
        ],
        names=["slab_a", "slab_b"],
    )
    (tmp_path / "map_0" / "clean_slab").mkdir(parents=True)
    (tmp_path / "map_1" / "clean_slab").mkdir(parents=True)

    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock"},
    )
    workflow.add_reference(
        "clean_slab",
        scope="per_map",
        parser=lambda output_file, maps, workflow, reference, subject, map_index, system_name, **kwargs: {
            "reference_energy": -10.0 - float(map_index),
            "parsed_from": output_file,
            "system_name_seen": system_name,
        },
    )

    collected = workflow.collect_references(
        multimaps,
        names=["clean_slab"],
        root=tmp_path,
    ).sort_values("map_index")

    assert collected["reference_energy"].tolist() == [-10.0, -11.0]
    assert collected["parsed_from"].tolist() == [
        str(tmp_path / "map_0" / "clean_slab"),
        str(tmp_path / "map_1" / "clean_slab"),
    ]
    assert collected["system_name_seen"].tolist() == ["slab_a", "slab_b"]


def test_adsorption_energy_parser_combines_point_and_reference_energies() -> None:
    maps = _build_maps(pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]}))
    maps.add_special_points([0], kind="centroid", iteration=0, label_status="unlabeled")

    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock"},
        parser=build_adsorption_energy_parser(
            lambda output_file, maps, workflow, special_point, **kwargs: {
                "E_bfgs_final_Ry": -15.0,
                "parsed_from": output_file,
            }
        ),
    )
    workflow.add_reference(
        "clean_slab",
        scope="per_map",
        runner=lambda **kwargs: None,
        parser=lambda **kwargs: None,
    )
    workflow.add_reference(
        "H2",
        scope="global",
        runner=lambda **kwargs: None,
        parser=lambda **kwargs: None,
    )
    workflow.reference_records[("clean_slab", 0)] = {
        "name": "clean_slab",
        "scope": "per_map",
        "map_index": 0,
        "label_status": "completed",
        "reference_energy_Ry": -10.0,
    }
    workflow.reference_records[("H2", None)] = {
        "name": "H2",
        "scope": "global",
        "map_index": None,
        "label_status": "completed",
        "reference_energy_Ry": -6.0,
    }

    workflow.record_outputs(maps, [0], ["qe_job_0"], kind="centroid")
    collected = workflow.collect(maps, kind="centroid")

    assert np.isclose(collected.iloc[0]["E_ads_Ry"], -2.0)
    assert np.isclose(collected.iloc[0]["clean_slab_reference_energy_Ry"], -10.0)
    assert np.isclose(collected.iloc[0]["H2_reference_energy_Ry"], -6.0)
    assert np.isclose(collected.iloc[0]["H2_multiplier"], 0.5)


def test_adsorption_energy_parser_can_use_derived_gas_reference_column() -> None:
    maps = _build_maps(pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]}))
    maps.add_special_points([0], kind="centroid", iteration=0, label_status="unlabeled")

    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock"},
        parser=build_adsorption_energy_parser(
            lambda output_file, maps, workflow, special_point, **kwargs: {
                "E_bfgs_final_Ry": -15.0,
            },
            gas_reference_column="mu_half_h2",
            gas_multiplier=1.0,
        ),
    )
    workflow.add_reference(
        "clean_slab",
        scope="per_map",
        runner=lambda **kwargs: None,
        parser=lambda **kwargs: None,
    )
    workflow.add_reference(
        "H2",
        scope="global",
        runner=lambda **kwargs: None,
        parser=lambda **kwargs: None,
    )
    workflow.reference_records[("clean_slab", 0)] = {
        "name": "clean_slab",
        "scope": "per_map",
        "map_index": 0,
        "label_status": "completed",
        "reference_energy_Ry": -10.0,
    }
    workflow.reference_records[("H2", None)] = {
        "name": "H2",
        "scope": "global",
        "map_index": None,
        "label_status": "completed",
        "mu_half_h2": -3.2,
    }

    workflow.record_outputs(maps, [0], ["qe_job_0"], kind="centroid")
    collected = workflow.collect(maps, kind="centroid")

    assert np.isclose(collected.iloc[0]["E_ads_Ry"], -1.8)
    assert np.isclose(collected.iloc[0]["H2_mu_half_h2"], -3.2)


def test_che_adsorption_energy_parser_adds_standard_shift() -> None:
    maps = _build_maps(pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]}))
    maps.add_special_points([0], kind="centroid", iteration=0, label_status="unlabeled")

    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock"},
        parser=build_che_adsorption_energy_parser(
            lambda output_file, maps, workflow, special_point, **kwargs: {
                "E_bfgs_final_Ry": -15.0,
            },
            gas_reference_column="mu_half_h2",
            gas_multiplier=1.0,
            free_energy_shift_eV=0.24,
        ),
    )
    workflow.add_reference(
        "clean_slab",
        scope="per_map",
        runner=lambda **kwargs: None,
        parser=lambda **kwargs: None,
    )
    workflow.add_reference(
        "H2",
        scope="global",
        runner=lambda **kwargs: None,
        parser=lambda **kwargs: None,
    )
    workflow.reference_records[("clean_slab", 0)] = {
        "name": "clean_slab",
        "scope": "per_map",
        "map_index": 0,
        "label_status": "completed",
        "reference_energy_Ry": -10.0,
    }
    workflow.reference_records[("H2", None)] = {
        "name": "H2",
        "scope": "global",
        "map_index": None,
        "label_status": "completed",
        "mu_half_h2": -3.2,
    }

    workflow.record_outputs(maps, [0], ["qe_job_0"], kind="centroid")
    collected = workflow.collect(maps, kind="centroid")

    expected_eads_ry = -15.0 - (-10.0) - (-3.2)
    expected_g_ry = expected_eads_ry + 0.24 / 13.605693122994
    assert np.isclose(collected.iloc[0]["E_ads_Ry"], expected_eads_ry)
    assert np.isclose(collected.iloc[0]["G_ads_CHE_Ry"], expected_g_ry)
    assert np.isclose(collected.iloc[0]["G_ads_CHE_eV"], expected_g_ry * 13.605693122994)
    assert np.isclose(collected.iloc[0]["G_ads_CHE_shift_eV"], 0.24)


def test_build_relax_adsorption_parser_simplifies_adsorption_setup(tmp_path) -> None:
    jobdir = tmp_path / "qe_job_0"
    jobdir.mkdir()
    (jobdir / "espresso.pwi").write_text(
        "\n".join(
            [
                "ATOMIC_POSITIONS (angstrom)",
                "Co 0.0 0.0 1.00",
                "H  0.1 0.2 2.00",
            ]
        )
        + "\n"
    )
    (jobdir / "espresso.pwo").write_text(
        "\n".join(
            [
                "!    total energy              =   -15.000000 Ry",
                "ATOMIC_POSITIONS (angstrom)",
                "Co 0.0 0.0 1.10",
                "H  0.3 0.4 2.20",
                "End of BFGS Geometry Optimization",
            ]
        )
        + "\n"
    )

    maps = _build_maps(pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]}))
    maps.add_special_points([0], kind="centroid", iteration=0, label_status="unlabeled")

    workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock"},
        parser=build_relax_adsorption_parser(
            adsorbate_label="H",
            gas_reference_column="mu_half_h2",
            gas_multiplier=1.0,
            che_shift_eV=0.24,
        ),
    )
    workflow.add_reference(
        "clean_slab",
        scope="per_map",
        runner=lambda **kwargs: None,
        parser=lambda **kwargs: None,
    )
    workflow.add_reference(
        "H2",
        scope="global",
        runner=lambda **kwargs: None,
        parser=lambda **kwargs: None,
    )
    workflow.reference_records[("clean_slab", 0)] = {
        "name": "clean_slab",
        "scope": "per_map",
        "map_index": 0,
        "label_status": "completed",
        "reference_energy_Ry": -10.0,
    }
    workflow.reference_records[("H2", None)] = {
        "name": "H2",
        "scope": "global",
        "map_index": None,
        "label_status": "completed",
        "mu_half_h2": -3.2,
    }

    workflow.record_outputs(maps, [0], [str(jobdir)], kind="centroid")
    collected = workflow.collect(maps, kind="centroid")

    expected_eads_ry = -15.0 - (-10.0) - (-3.2)
    expected_g_ry = expected_eads_ry + 0.24 / 13.605693122994
    assert np.isclose(collected.iloc[0]["E_bfgs_final_Ry"], -15.0)
    assert np.isclose(collected.iloc[0]["E_ads_Ry"], expected_eads_ry)
    assert np.isclose(collected.iloc[0]["G_ads_CHE_Ry"], expected_g_ry)
