import numpy as np

from mapsy import (
    QuantumEspressoMultiRelaxParser,
    QuantumEspressoRelaxParser,
    QuantumEspressoScfParser,
)


def test_qe_relax_parser_extracts_common_metrics(tmp_path) -> None:
    jobdir = tmp_path / "qe_job_7"
    jobdir.mkdir()

    (jobdir / "espresso.pwi").write_text(
        "\n".join(
            [
                "&system",
                "  celldm(1) = 10.0",
                "/",
                "CELL_PARAMETERS (angstrom)",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 20.0",
                "ATOMIC_POSITIONS (alat)",
                "Co 0.0 0.0 0.10",
                "H  0.0 0.0 0.40",
            ]
        )
        + "\n"
    )

    (jobdir / "espresso.pwo").write_text(
        "\n".join(
            [
                " lattice parameter (alat)  =      10.0000  a.u.",
                "     iteration #  1",
                "     total energy              =   -123.4567 Ry",
                "     estimated scf accuracy <  1.2E-09 Ry",
                "!    total energy              =   -123.4000 Ry",
                "     Total force =     0.0500     Total SCF correction =     0.0001",
                "     Forces acting on atoms (Ry/au):",
                "     atom    1 type  1   force =     0.00000000    0.00000000    0.01000000",
                "     atom    2 type  1   force =     0.00100000   -0.00200000    0.03000000",
                "ATOMIC_POSITIONS (angstrom)",
                "Co 0.0 0.0 1.00",
                "H  0.1 0.2 2.00",
                "!    total energy              =   -124.0000 Ry",
                "     Total force =     0.0200     Total SCF correction =     0.0001",
                "     Forces acting on atoms (Ry/au):",
                "     atom    1 type  1   force =     0.00000000    0.00000000    0.00500000",
                "     atom    2 type  1   force =     0.00300000   -0.00400000    0.00600000",
                "ATOMIC_POSITIONS (angstrom)",
                "Co 0.0 0.0 1.10",
                "H  0.3 0.4 2.20",
                "End of BFGS Geometry Optimization",
            ]
        )
        + "\n"
    )

    parser = QuantumEspressoRelaxParser(iteration=1, adsorbate_label="H")
    parsed = parser(jobdir)

    assert parsed["label_file"] == str(jobdir)
    assert np.isclose(parsed["E_bfgs_initial_Ry"], -123.4)
    assert np.isclose(parsed["E_bfgs_final_Ry"], -124.0)
    assert np.isclose(parsed["x_H_input_A"], 0.0)
    assert np.isclose(parsed["y_H_input_A"], 0.0)
    assert np.isclose(parsed["z_H_input_A"], 0.40 * 10.0 * 0.529177210903)
    assert np.isclose(parsed["x_H_final_A"], 0.3)
    assert np.isclose(parsed["y_H_final_A"], 0.4)
    assert np.isclose(parsed["z_H_final_A"], 2.20)
    np.testing.assert_allclose(parsed["E_bfgs_steps_Ry"], np.array([-123.4, -124.0]))
    np.testing.assert_allclose(
        parsed["H_positions_bfgs_A"],
        np.array([[0.1, 0.2, 2.0], [0.3, 0.4, 2.2]]),
    )
    np.testing.assert_allclose(parsed["x_H_bfgs_steps_A"], np.array([0.1, 0.3]))
    np.testing.assert_allclose(parsed["y_H_bfgs_steps_A"], np.array([0.2, 0.4]))
    np.testing.assert_allclose(parsed["z_H_bfgs_steps_A"], np.array([2.0, 2.2]))
    assert parsed["n_ionic_steps"] == 2
    assert parsed["relax_converged"] is True


def test_qe_scf_parser_keeps_iteration_metrics_only(tmp_path) -> None:
    jobdir = tmp_path / "qe_job_8"
    jobdir.mkdir()
    (jobdir / "espresso.pwi").write_text("ATOMIC_POSITIONS (angstrom)\nH 0.0 0.0 1.0\n")
    (jobdir / "espresso.pwo").write_text(
        "\n".join(
            [
                "     iteration #  1",
                "     total energy              =   -10.0000 Ry",
                "     estimated scf accuracy <  5.0E-06 Ry",
                "     iteration #  2",
                "     total energy              =   -10.5000 Ry",
                "     estimated scf accuracy <  1.0E-08 Ry",
                "!    total energy              =   -10.6000 Ry",
            ]
        )
        + "\n"
    )

    parsed = QuantumEspressoScfParser(iteration=2, adsorbate_label="H").parse(jobdir)

    assert np.isclose(parsed["E_iter2_Ry"], -10.5)
    assert np.isclose(parsed["acc_iter2_Ry"], 1.0e-08)
    assert np.isclose(parsed["E_converged_first_Ry"], -10.6)
    assert np.isclose(parsed["E_converged_last_Ry"], -10.6)
    assert parsed["n_scf_iterations"] == 2
    assert "n_ionic_steps" not in parsed


def test_qe_scf_parser_can_target_accuracy_threshold(tmp_path) -> None:
    jobdir = tmp_path / "qe_job_10"
    jobdir.mkdir()
    (jobdir / "espresso.pwi").write_text("ATOMIC_POSITIONS (angstrom)\nH 0.0 0.0 1.0\n")
    (jobdir / "espresso.pwo").write_text(
        "\n".join(
            [
                "     iteration #  1",
                "     total energy              =   -10.0000 Ry",
                "     estimated scf accuracy <  5.0E-04 Ry",
                "     iteration #  2",
                "     total energy              =   -10.5000 Ry",
                "     estimated scf accuracy <  5.0E-06 Ry",
                "     iteration #  3",
                "     total energy              =   -10.7500 Ry",
                "     estimated scf accuracy <  1.0E-08 Ry",
                "!    total energy              =   -10.8000 Ry",
            ]
        )
        + "\n"
    )

    parsed = QuantumEspressoScfParser(
        accuracy_threshold=1.0e-05,
        adsorbate_label="H",
    ).parse(jobdir)

    assert parsed["first_iteration_below_accuracy_threshold"] == 2
    assert np.isclose(parsed["E_first_below_accuracy_threshold_Ry"], -10.5)
    assert np.isclose(parsed["acc_first_below_accuracy_threshold_Ry"], 5.0e-06)
    assert np.isclose(parsed["scf_accuracy_threshold_Ry"], 1.0e-05)
    assert "E_iter1_Ry" not in parsed


def test_qe_relax_parser_extracts_force_and_geometry_metrics(tmp_path) -> None:
    jobdir = tmp_path / "qe_job_9"
    jobdir.mkdir()
    (jobdir / "espresso.pwi").write_text(
        "\n".join(
            [
                "&system",
                "  celldm(1) = 10.0",
                "/",
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
                "!    total energy              =   -123.4000 Ry",
                "     Total force =     0.0500     Total SCF correction =     0.0001",
                "     Forces acting on atoms (Ry/au):",
                "     atom    1 type  1   force =     0.00000000    0.00000000    0.01000000",
                "     atom    2 type  1   force =     0.00100000   -0.00200000    0.03000000",
                "ATOMIC_POSITIONS (angstrom)",
                "Co 0.0 0.0 1.00",
                "H  0.1 0.2 2.00",
                "!    total energy              =   -124.0000 Ry",
                "     Total force =     0.0200     Total SCF correction =     0.0001",
                "     Forces acting on atoms (Ry/au):",
                "     atom    1 type  1   force =     0.00000000    0.00000000    0.00500000",
                "     atom    2 type  1   force =     0.00300000   -0.00400000    0.00600000",
                "ATOMIC_POSITIONS (angstrom)",
                "Co 0.0 0.0 1.10",
                "H  0.3 0.4 2.20",
                "End of BFGS Geometry Optimization",
            ]
        )
        + "\n"
    )

    parsed = QuantumEspressoRelaxParser(adsorbate_label="H").parse(jobdir)

    assert np.isclose(parsed["E_bfgs_initial_Ry"], -123.4)
    assert np.isclose(parsed["E_bfgs_final_Ry"], -124.0)
    assert np.isclose(parsed["F_bfgs_initial_total_Ry_au"], 0.05)
    assert np.isclose(parsed["F_bfgs_final_total_Ry_au"], 0.02)
    assert np.isclose(parsed["fx_H_initial_Ry_au"], 0.001)
    assert np.isclose(parsed["fy_H_initial_Ry_au"], -0.002)
    assert np.isclose(parsed["fz_H_initial_Ry_au"], 0.03)
    assert np.isclose(parsed["fx_H_final_Ry_au"], 0.003)
    assert np.isclose(parsed["fy_H_final_Ry_au"], -0.004)
    assert np.isclose(parsed["fz_H_final_Ry_au"], 0.006)
    assert np.isclose(parsed["x_H_final_A"], 0.3)
    assert np.isclose(parsed["y_H_final_A"], 0.4)
    assert np.isclose(parsed["z_H_final_A"], 2.2)
    np.testing.assert_allclose(parsed["E_bfgs_steps_Ry"], np.array([-123.4, -124.0]))
    np.testing.assert_allclose(
        parsed["H_positions_bfgs_A"],
        np.array([[0.1, 0.2, 2.0], [0.3, 0.4, 2.2]]),
    )
    assert parsed["n_ionic_steps"] == 2
    assert parsed["relax_converged"] is True


def test_qe_relax_parser_marks_unconverged_runs_as_failed(tmp_path) -> None:
    jobdir = tmp_path / "qe_job_11"
    jobdir.mkdir()
    (jobdir / "espresso.pwi").write_text(
        "ATOMIC_POSITIONS (angstrom)\nCo 0.0 0.0 1.0\nH 0.1 0.2 2.0\n"
    )
    (jobdir / "espresso.pwo").write_text(
        "\n".join(
            [
                "!    total energy              =   -123.4000 Ry",
                "     Total force =     0.0500     Total SCF correction =     0.0001",
                "     Forces acting on atoms (Ry/au):",
                "     atom    1 type  1   force =     0.00000000    0.00000000    0.01000000",
                "     atom    2 type  1   force =     0.00100000   -0.00200000    0.03000000",
            ]
        )
        + "\n"
    )

    parsed = QuantumEspressoRelaxParser(adsorbate_label="H").parse(jobdir)

    assert parsed["relax_converged"] is False
    assert parsed["label_status"] == "failed"
    assert "End of BFGS Geometry Optimization" in parsed["label_error"]


def test_qe_parsers_collect_job_directories(tmp_path) -> None:
    scf_parser = QuantumEspressoScfParser()
    relax_parser = QuantumEspressoRelaxParser()
    for index in [0, 1]:
        jobdir = tmp_path / f"qe_job_{index}"
        jobdir.mkdir()
        (jobdir / "espresso.pwi").write_text("ATOMIC_POSITIONS (angstrom)\nH 0.0 0.0 1.0\n")
        (jobdir / "espresso.pwo").write_text("!    total energy              =   -1.0000 Ry\n")

    scf_frame = scf_parser.collect(tmp_path)
    relax_frame = relax_parser.collect(tmp_path)

    assert scf_frame["job_id"].tolist() == [0, 1]
    assert relax_frame["job_id"].tolist() == [0, 1]
    assert "E_converged_last_Ry" in scf_frame.columns
    assert "E_bfgs_final_Ry" in relax_frame.columns
    assert "n_ionic_steps" in relax_frame.columns


def test_qe_multi_relax_parser_prefers_latest_converged_attempt(tmp_path) -> None:
    jobdir = tmp_path / "qe_job_12"
    jobdir.mkdir()

    (jobdir / "espresso.pwi").write_text(
        "ATOMIC_POSITIONS (angstrom)\nCo 0.0 0.0 1.0\nH 0.1 0.2 2.0\n"
    )
    (jobdir / "espresso.pwo").write_text(
        "\n".join(
            [
                "!    total energy              =   -123.4000 Ry",
                "     Total force =     0.0500     Total SCF correction =     0.0001",
                "ATOMIC_POSITIONS (angstrom)",
                "Co 0.0 0.0 1.0",
                "H 0.2 0.3 2.1",
            ]
        )
        + "\n"
    )
    (jobdir / "espresso.r001.pwi").write_text(
        "ATOMIC_POSITIONS (angstrom)\nCo 0.0 0.0 1.0\nH 0.2 0.3 2.1\n"
    )
    (jobdir / "espresso.r001.pwo").write_text(
        "\n".join(
            [
                "!    total energy              =   -124.0000 Ry",
                "     Total force =     0.0200     Total SCF correction =     0.0001",
                "     Forces acting on atoms (Ry/au):",
                "     atom    1 type  1   force =     0.00000000    0.00000000    0.00500000",
                "     atom    2 type  1   force =     0.00300000   -0.00400000    0.00600000",
                "ATOMIC_POSITIONS (angstrom)",
                "Co 0.0 0.0 1.10",
                "H  0.3 0.4 2.20",
                "End of BFGS Geometry Optimization",
            ]
        )
        + "\n"
    )

    parsed = QuantumEspressoMultiRelaxParser(adsorbate_label="H").parse(jobdir)

    assert parsed["label_file"] == str(jobdir)
    assert parsed["n_relax_attempts"] == 2
    assert parsed["latest_attempt_index"] == 1
    assert parsed["selected_attempt_index"] == 1
    assert parsed["attempt_000_relax_converged"] is False
    assert parsed["attempt_001_relax_converged"] is True
    assert np.isclose(parsed["E_bfgs_final_Ry"], -124.0)
    assert np.isclose(parsed["x_H_final_A"], 0.3)
    assert np.isclose(parsed["y_H_final_A"], 0.4)
    assert np.isclose(parsed["z_H_final_A"], 2.2)
    assert "label_status" not in parsed


def test_qe_multi_relax_parser_marks_all_failed_attempts(tmp_path) -> None:
    jobdir = tmp_path / "qe_job_13"
    jobdir.mkdir()
    (jobdir / "espresso.pwi").write_text(
        "ATOMIC_POSITIONS (angstrom)\nCo 0.0 0.0 1.0\nH 0.1 0.2 2.0\n"
    )
    (jobdir / "espresso.pwo").write_text(
        "!    total energy              =   -123.4000 Ry\nATOMIC_POSITIONS (angstrom)\nCo 0.0 0.0 1.0\nH 0.2 0.3 2.1\n"
    )

    parsed = QuantumEspressoMultiRelaxParser(adsorbate_label="H").parse(jobdir)

    assert parsed["selected_attempt_index"] == 0
    assert parsed["label_status"] == "failed"
    assert "missing 'End of BFGS Geometry Optimization'" in parsed["label_error"]
