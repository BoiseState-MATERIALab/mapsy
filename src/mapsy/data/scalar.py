# Refactored from Stephen Weitzner cube_vizkit
from typing import Optional
import numpy.typing as npt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ase import Atoms
from mapsy.data import Grid, VolumetricField


class ScalarField(VolumetricField):
    """ """

    def __new__(
        cls,
        grid: Grid,
        rank: Optional[int] = 1,
        name: Optional[str] = None,
        label: Optional[str] = None,
        data: Optional[npt.NDArray] = None,
    ):

        if label is None:
            label = "SCA"
        if name is None:
            name = "scalar"

        obj = super().__new__(cls, grid, rank, name, label, data)

        obj._integral = None
        return obj

    def __array_finalize__(self, obj) -> None:
        # Restore attributes when we are taking a slice
        super().__array_finalize__(obj)
        if obj is None:
            return
        if isinstance(obj, ScalarField):
            obj._integral = getattr(obj, "_integral", None)

    @property
    def integral(self) -> np.float64:
        """"""
        self._calc_integral()
        return self._integral

    def _calc_integral(self) -> None:
        self._integral = np.sum(self) * self.grid.volume / self.grid.ndata

    def tocube(
        self,
        filename: str,
        atoms: Atoms,
    ) -> None:
        with open(filename,"w") as cubefile:       
            cubefile.write("Cubefile generated by MapSy\n Loops along x, y, z \n")
            nat : int = len(atoms)
            cubefile.write(f"{nat:5d}{self.grid.origin[0]:12.6f}{self.grid.origin[1]:12.6f}{self.grid.origin[2]:12.6f}\n")
            N1, N2, N3 = self.grid.scalars
            cubefile.write(f"{N1:5d}{self.grid.basis[0,0]:12.6f}{self.grid.basis[0,1]:12.6f}{self.grid.basis[0,2]:12.6f}\n")
            cubefile.write(f"{N2:5d}{self.grid.basis[1,0]:12.6f}{self.grid.basis[1,1]:12.6f}{self.grid.basis[1,2]:12.6f}\n")
            cubefile.write(f"{N3:5d}{self.grid.basis[2,0]:12.6f}{self.grid.basis[2,1]:12.6f}{self.grid.basis[2,2]:12.6f}\n")
            for mass, charge, pos in zip(atoms.get_atomic_numbers(),atoms.get_initial_charges(),atoms.positions):
                cubefile.write(f"{mass:5d}{charge:12.6f}{pos[0]:12.6f}{pos[1]:12.6f}{pos[2]:12.6f}\n")
            count: int = 0
            for i in range(N1):
                for j in range(N2):
                    for k in range(N3):
                        if (i or j or k) and count%6==0:
                            cubefile.write("\n")
                        cubefile.write(" {0: .5E}".format(self.data[i,j,k]))
                        count+=1



    def toline(
        self,
        center: npt.NDArray[np.float64],
        iaxis: int,
        planaraverage: bool = False,
    ):
        icenter = np.array(
            [np.rint(center[i] / self.grid.basis[i, i]) for i in range(3)], dtype="int"
        )
        icenter = icenter - (
            self.grid.scalars * np.trunc(icenter // self.grid.scalars)
        ).astype("int")
        if iaxis == 0:
            axis = self.grid.coordinates[0, :, icenter[1], icenter[2]]  # type: ignore
            if planaraverage:
                value = np.mean(self, (1, 2))
            else:
                value = self[:, icenter[1], icenter[2]]
        elif iaxis == 1:
            axis = self.grid.coordinates[1, icenter[0], :, icenter[2]]  # type: ignore
            if planaraverage:
                value = np.mean(self, (0, 2))
            else:
                value = self[icenter[0], :, icenter[2]]
        elif iaxis == 2:
            axis = self.grid.coordinates[2, icenter[0], icenter[1], :]  # type: ignore
            if planaraverage:
                value = np.mean(self, (0, 1))
            else:
                value = self[icenter[0], icenter[1], :]
        else:
            raise ValueError("Axis out of range")
        return axis, value

    def tocontour(self, center, axis):
        icenter = np.array(
            [np.rint(center[i] / self.grid.basis[i, i]) for i in range(3)], dtype="int"
        )
        icenter = icenter - (
            self.grid.scalars * np.trunc(icenter // self.grid.scalars)
        ).astype("int")
        if axis == 0:
            ax1 = self.grid.coordinates[1, icenter[0], :, :]
            ax2 = self.grid.coordinates[2, icenter[0], :, :]
            value = self[icenter[0], :, :]
        elif axis == 1:
            ax1 = self.grid.coordinates[0, :, icenter[1], :]
            ax2 = self.grid.coordinates[2, :, icenter[1], :]
            value = self[:, icenter[1], :]
        elif axis == 2:
            ax1 = self.grid.coordinates[0, :, :, icenter[2]]
            ax2 = self.grid.coordinates[1, :, :, icenter[2]]
            value = self[:, :, icenter[2]]
        else:
            raise ValueError("Axis out of range")
        return ax1, ax2, value

    def plotprojections(
        self,
        center: npt.NDArray[np.float64],
        colormap: str = "plasma",
        centermap: bool = False,
    ):
        cmap = mpl.colormaps[colormap]
        axis1_yz, axis2_yz, values_yz = self.tocontour(center, 0)
        axis1_xz, axis2_xz, values_xz = self.tocontour(center, 1)
        axis1_xy, axis2_xy, values_xy = self.tocontour(center, 2)
        width_x = self.grid.cell[0, 0]  # NEED TO FIX FOR NON ORTHOROMBIC CELLS
        width_y = self.grid.cell[1, 1]  # NEED TO FIX FOR NON ORTHOROMBIC CELLS
        width_z = self.grid.cell[2, 2]  # NEED TO FIX FOR NON ORTHOROMBIC CELLS
        vmin = np.min(self)
        vmax = np.max(self)
        # The following is an option for centering the colorbar on zero
        if centermap:
            vmax = -np.max([abs(np.min(self)), abs(np.max(self))]) * 0.6
            vmin = -vmax
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=(width_x, width_z),
            height_ratios=(width_z, width_y),
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.05,
            hspace=0.05,
        )

        ax1 = fig.add_subplot(gs[0, 0])
        cont1 = ax1.contourf(
            axis1_xz, axis2_xz, values_xz, levels=100, cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax1.scatter(center[0], center[2])
        ax1.tick_params(labelbottom=False)
        ax1.set_ylabel("Z (a.u.)")

        ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
        cont3 = ax3.contourf(
            axis1_xy, axis2_xy, values_xy, levels=100, cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax3.scatter(center[0], center[1])
        ax3.set_xlabel("X (a.u.)")
        ax3.set_ylabel("Y (a.u.)")

        ax4 = fig.add_subplot(gs[1, 1], sharey=ax3)
        cont4 = ax4.contourf(
            axis2_yz, axis1_yz, values_yz, levels=100, cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax4.scatter(center[2], center[1])
        ax4.tick_params(labelleft=False)
        ax4.set_xlabel("Z (a.u.)")

        # Colorbar
        ax2 = fig.add_subplot(gs[0, 1])
        ax2_pos = ax2.get_position().bounds
        ax2.set_position(
            [
                ax2_pos[0] + ax2_pos[2] * 0.35,
                ax2_pos[1] + ax2_pos[3] * 0.05,
                ax2_pos[2] * 0.1,
                ax2_pos[3] * 0.9,
            ]
        )
        fig.colorbar(cont4, cax=ax2)
        plt.show()
