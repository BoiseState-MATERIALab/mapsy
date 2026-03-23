import numpy as np

from mapsy import QuantumEspressoOutputParser


def test_qe_output_parser_extracts_common_metrics(tmp_path) -> None:
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
                "ATOMIC_POSITIONS (angstrom)",
                "Co 0.0 0.0 1.00",
                "H  0.1 0.2 2.00",
                "!    total energy              =   -124.0000 Ry",
                "ATOMIC_POSITIONS (angstrom)",
                "Co 0.0 0.0 1.10",
                "H  0.3 0.4 2.20",
                "End of BFGS Geometry Optimization",
            ]
        )
        + "\n"
    )

    parser = QuantumEspressoOutputParser(iteration=1, adsorbate_label="H")
    parsed = parser(jobdir)

    assert parsed["label_file"] == str(jobdir)
    assert np.isclose(parsed["E_iter1_Ry"], -123.4567)
    assert np.isclose(parsed["acc_iter1_Ry"], 1.2e-09)
    assert np.isclose(parsed["E_after_first_SCF_Ry"], -123.4)
    assert np.isclose(parsed["E_end_BFGS_Ry"], -124.0)
    assert np.isclose(parsed["x_H_input_A"], 0.0)
    assert np.isclose(parsed["y_H_input_A"], 0.0)
    assert np.isclose(parsed["z_H_input_A"], 0.40 * 10.0 * 0.529177210903)
    assert np.isclose(parsed["x_H_last_A"], 0.3)
    assert np.isclose(parsed["y_H_last_A"], 0.4)
    assert np.isclose(parsed["z_H_last_A"], 2.20)
    assert parsed["n_ionic_steps"] == 2
    assert parsed["relax_converged"] is True


def test_qe_output_parser_collects_job_directories(tmp_path) -> None:
    parser = QuantumEspressoOutputParser()
    for index in [0, 1]:
        jobdir = tmp_path / f"qe_job_{index}"
        jobdir.mkdir()
        (jobdir / "espresso.pwi").write_text("ATOMIC_POSITIONS (angstrom)\nH 0.0 0.0 1.0\n")
        (jobdir / "espresso.pwo").write_text("!    total energy              =   -1.0000 Ry\n")

    frame = parser.collect(tmp_path)

    assert frame["job_id"].tolist() == [0, 1]
    assert "E_end_BFGS_Ry" in frame.columns
    assert "n_ionic_steps" in frame.columns
