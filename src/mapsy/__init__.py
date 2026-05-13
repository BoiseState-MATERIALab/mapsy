"""A Python Tool to Compute Local Symmetry Maps"""

from contextlib import suppress
from typing import Any

__author__ = "MATERIALab"
__contact__ = "olivieroandreuss@boisestate.edu"
__license__ = "MIT"
__version__ = "0.0.1"  # fallback if package metadata isn't available
__date__ = "2024-05-24"

try:
    import importlib.metadata as _stdlib_metadata

    _md: Any = _stdlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata as _backport_metadata

    _md = _backport_metadata

# Optional: define a default so __version__ always exists
# __version__ = "0.0.1"

with suppress(_md.PackageNotFoundError):
    __version__ = _md.version("mapsy")

__all__ = [
    "__version__",
    "Maps",
    "MapsFromFile",
    "MultiMaps",
    "MultiMapsFromFile",
    "PCAAnalysisResult",
    "PCAResult",
    "ArchetypeSelectionResult",
    "ArchetypePropagationResult",
    "ClusterResult",
    "ClusterScreeningResult",
    "GraphResult",
    "CalculationWorkflow",
    "WorkflowReferenceSpec",
    "AdaptiveWorkflow",
    "ModelSuite",
    "ModelTrainingSpec",
    "PointPropertyDatasetBuilder",
    "AdsorptionEnergyParser",
    "CHEAdsorptionEnergyParser",
    "build_che_adsorption_energy_parser",
    "build_relax_adsorption_parser",
    "QuantumEspressoEnergyParser",
    "QuantumEspressoMultiRelaxParser",
    "QuantumEspressoRelaxParser",
    "QuantumEspressoScfParser",
    "QuantumEspressoSetup",
    "X2RelaxFrequencyParser",
    "build_adsorption_energy_parser",
    "RelaxStepDatasetBuilder",
    "RobustGaussianProcessSurrogate",
    "SlurmTemplate",
    "SupervisedDataset",
    "WarmStartProfile",
    "plot_cluster_screening",
    "plot_pca_scree",
]


def __getattr__(name: str) -> Any:
    if name in {"Maps", "MapsFromFile"}:
        from .maps import Maps, MapsFromFile

        exports = {"Maps": Maps, "MapsFromFile": MapsFromFile}
        return exports[name]
    if name in {"MultiMaps", "MultiMapsFromFile"}:
        from .multimaps import MultiMaps, MultiMapsFromFile

        exports = {"MultiMaps": MultiMaps, "MultiMapsFromFile": MultiMapsFromFile}
        return exports[name]
    if name in {
        "PCAAnalysisResult",
        "PCAResult",
        "ArchetypeSelectionResult",
        "ArchetypePropagationResult",
        "ClusterResult",
        "ClusterScreeningResult",
        "GraphResult",
    }:
        from .results import (
            ArchetypePropagationResult,
            ArchetypeSelectionResult,
            ClusterResult,
            ClusterScreeningResult,
            GraphResult,
            PCAAnalysisResult,
            PCAResult,
        )

        exports = {
            "PCAAnalysisResult": PCAAnalysisResult,
            "PCAResult": PCAResult,
            "ArchetypeSelectionResult": ArchetypeSelectionResult,
            "ArchetypePropagationResult": ArchetypePropagationResult,
            "ClusterResult": ClusterResult,
            "ClusterScreeningResult": ClusterScreeningResult,
            "GraphResult": GraphResult,
        }
        return exports[name]
    if name in {"CalculationWorkflow", "WorkflowReferenceSpec"}:
        from .workflows import CalculationWorkflow, WorkflowReferenceSpec

        exports = {
            "CalculationWorkflow": CalculationWorkflow,
            "WorkflowReferenceSpec": WorkflowReferenceSpec,
        }
        return exports[name]
    if name in {
        "AdaptiveWorkflow",
        "ModelTrainingSpec",
        "SupervisedDataset",
        "PointPropertyDatasetBuilder",
        "RelaxStepDatasetBuilder",
        "WarmStartProfile",
        "RobustGaussianProcessSurrogate",
        "ModelSuite",
    }:
        from .learning import (
            AdaptiveWorkflow,
            ModelSuite,
            ModelTrainingSpec,
            PointPropertyDatasetBuilder,
            RelaxStepDatasetBuilder,
            RobustGaussianProcessSurrogate,
            SupervisedDataset,
            WarmStartProfile,
        )

        exports = {
            "AdaptiveWorkflow": AdaptiveWorkflow,
            "ModelTrainingSpec": ModelTrainingSpec,
            "SupervisedDataset": SupervisedDataset,
            "PointPropertyDatasetBuilder": PointPropertyDatasetBuilder,
            "RelaxStepDatasetBuilder": RelaxStepDatasetBuilder,
            "WarmStartProfile": WarmStartProfile,
            "RobustGaussianProcessSurrogate": RobustGaussianProcessSurrogate,
            "ModelSuite": ModelSuite,
        }
        return exports[name]
    if name in {
        "AdsorptionEnergyParser",
        "CHEAdsorptionEnergyParser",
        "QuantumEspressoSetup",
        "QuantumEspressoEnergyParser",
        "X2RelaxFrequencyParser",
    }:
        from .calculations import (
            AdsorptionEnergyParser,
            CHEAdsorptionEnergyParser,
            QuantumEspressoEnergyParser,
            QuantumEspressoSetup,
            X2RelaxFrequencyParser,
        )

        exports = {
            "AdsorptionEnergyParser": AdsorptionEnergyParser,
            "CHEAdsorptionEnergyParser": CHEAdsorptionEnergyParser,
            "QuantumEspressoSetup": QuantumEspressoSetup,
            "QuantumEspressoEnergyParser": QuantumEspressoEnergyParser,
            "X2RelaxFrequencyParser": X2RelaxFrequencyParser,
        }
        return exports[name]
    if name == "build_adsorption_energy_parser":
        from .calculations import build_adsorption_energy_parser

        return build_adsorption_energy_parser
    if name == "build_che_adsorption_energy_parser":
        from .calculations import build_che_adsorption_energy_parser

        return build_che_adsorption_energy_parser
    if name == "build_relax_adsorption_parser":
        from .calculations import build_relax_adsorption_parser

        return build_relax_adsorption_parser
    if name in {"QuantumEspressoMultiRelaxParser"}:
        from .calculations import QuantumEspressoMultiRelaxParser

        return QuantumEspressoMultiRelaxParser
    if name in {"QuantumEspressoRelaxParser"}:
        from .calculations import QuantumEspressoRelaxParser

        return QuantumEspressoRelaxParser
    if name in {"QuantumEspressoScfParser"}:
        from .calculations import QuantumEspressoScfParser

        return QuantumEspressoScfParser
    if name in {"SlurmTemplate"}:
        from .calculations import SlurmTemplate

        return SlurmTemplate
    if name == "plot_cluster_screening":
        from .plotting import plot_cluster_screening, plot_pca_scree

        return plot_cluster_screening
    if name == "plot_pca_scree":
        from .plotting import plot_cluster_screening, plot_pca_scree

        return plot_pca_scree
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
