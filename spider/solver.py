"""Solver

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from configparser import ConfigParser, SectionProxy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from spider.mesh import StaggeredMesh
from spider.phase import (
    PhaseEvaluator,
    PhaseStateBasic,
    PhaseStateStaggered,
    phase_from_configuration,
)
from spider.scalings import Constants, Scalings, scalings_from_configuration

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class State:
    """Stores the state at temperature and pressure.

    Args:
        _phase_evaluator: A PhaseEvaluator
        _mesh: A StaggeredMesh

    Attributes:
        phase_basic: Phase properties at the basic nodes
        phase_staggered: Phase properties at the staggered nodes
        conductive_heat_flux: Conductive heat flux at the basic nodes
        convective_heat_flux: Convective heat flux at the basic nodes
        critical_reynolds_number: Critical Reynolds number
        dTdr: Temperature gradient with respect to radius at the basic nodes
        eddy_diffusivity: Eddy diffusivity at the basic nodes
        heat_flux: Heat flux at the basic nodes
        inviscid_regime: Array with True if the flow is inviscid and otherwise False
        inviscid_velocity: Inviscid velocity
        is_convective: Array with True if the flow is convecting and otherwise False
        reynolds_number: Reynolds number
        super_adiabatic_temperature_gradient: Super adiabatic temperature gradient
        viscous_regime: Array with True if the flow is viscous and otherwise False
        viscous_velocity: Viscous velocity
    """

    _phase_evaluator: PhaseEvaluator
    _mesh: StaggeredMesh
    phase_basic: PhaseStateBasic = field(init=False)
    phase_staggered: PhaseStateStaggered = field(init=False)
    _dTdr: np.ndarray = field(init=False)
    _eddy_diffusivity: np.ndarray = field(init=False)
    _is_convective: np.ndarray = field(init=False)
    _reynolds_number: np.ndarray = field(init=False)
    _super_adiabatic_temperature_gradient: np.ndarray = field(init=False)
    _viscous_velocity: np.ndarray = field(init=False)
    _inviscid_velocity: np.ndarray = field(init=False)

    def __post_init__(self):
        self.phase_basic = PhaseStateBasic(self._phase_evaluator)
        self.phase_staggered = PhaseStateStaggered(self._phase_evaluator)

    @property
    def conductive_heat_flux(self) -> np.ndarray:
        """Conductive heat flux is only accessed once so therefore it is a property."""
        conductive_heat_flux: np.ndarray = -self.phase_basic.thermal_conductivity * self._dTdr
        return conductive_heat_flux

    @property
    def convective_heat_flux(self) -> np.ndarray:
        """Convective heat flux is only accessed once so therefore it is a property."""
        convective_heat_flux: np.ndarray = (
            -self.phase_basic.density
            * self.phase_basic.heat_capacity
            * self._eddy_diffusivity
            * self._super_adiabatic_temperature_gradient
        )
        return convective_heat_flux

    @property
    def critical_reynolds_number(self) -> float:
        """Critical Reynolds number from Abe (1993)"""
        return 9 / 8

    @property
    def dTdr(self) -> np.ndarray:
        return self._dTdr

    @property
    def eddy_diffusivity(self) -> np.ndarray:
        return self._eddy_diffusivity

    @property
    def heat_flux(self) -> np.ndarray:
        return self.conductive_heat_flux + self.convective_heat_flux

    @property
    def inviscid_regime(self) -> np.ndarray:
        return self._reynolds_number > self.critical_reynolds_number

    @property
    def inviscid_velocity(self) -> np.ndarray:
        return self._inviscid_velocity

    @property
    def is_convective(self) -> np.ndarray:
        return self._is_convective

    @property
    def reynolds_number(self) -> np.ndarray:
        return self._reynolds_number

    @property
    def super_adiabatic_temperature_gradient(self) -> np.ndarray:
        return self._super_adiabatic_temperature_gradient

    @property
    def viscous_regime(self) -> np.ndarray:
        return self._reynolds_number <= self.critical_reynolds_number

    @property
    def viscous_velocity(self) -> np.ndarray:
        return self._viscous_velocity

    def update(self, temperature: np.ndarray, pressure: np.ndarray) -> None:
        """Updates the state.

        The evaluation order matters because we want to minimise the number of evaluations.

        Args:
            temperature: Temperature at the staggered nodes.
            pressure: Pressure at the staggered nodes.
        """
        logger.info("Updating the state")
        self.phase_staggered.update(temperature, pressure)
        temperature_basic: np.ndarray = self._mesh.quantity_at_basic_nodes(temperature)
        pressure_basic: np.ndarray = self._mesh.quantity_at_basic_nodes(pressure)
        self.phase_basic.update(temperature_basic, pressure_basic)
        self._dTdr = self._mesh.d_dr_at_basic_nodes(temperature)
        self._super_adiabatic_temperature_gradient = self._dTdr - self.phase_basic.dTdrs
        self._is_convective = self._super_adiabatic_temperature_gradient < 0
        velocity_prefactor: np.ndarray = (
            -self.phase_basic.gravitational_acceleration
            * self.phase_basic.thermal_expansivity
            * self._super_adiabatic_temperature_gradient
        )
        # Viscous velocity
        self._viscous_velocity = (velocity_prefactor * self._mesh.basic.mixing_length_cubed) / (
            18 * self.phase_basic.kinematic_viscosity
        )
        self._viscous_velocity[~self.is_convective] = 0  # Must be super-adiabatic
        # Inviscid velocity
        self._inviscid_velocity = (
            velocity_prefactor * self._mesh.basic.mixing_length_squared
        ) / 16
        self._inviscid_velocity[~self.is_convective] = 0  # Must be super-adiabatic
        self._inviscid_velocity[self._is_convective] = np.sqrt(
            self._inviscid_velocity[self._is_convective]
        )
        # Reynolds number
        self._reynolds_number = (
            self._viscous_velocity
            * self._mesh.basic.mixing_length
            / self.phase_basic.kinematic_viscosity
        )
        # Eddy diffusivity
        self._eddy_diffusivity = np.where(
            self.viscous_regime, self._viscous_velocity, self._inviscid_velocity
        )
        self._eddy_diffusivity *= self._mesh.basic.mixing_length


@dataclass
class SpiderSolver:
    filename: Union[str, Path]
    root_path: Union[str, Path] = ""
    root: Path = field(init=False)
    scalings: Scalings = field(init=False)
    constants: Constants = field(init=False)
    mesh: StaggeredMesh = field(init=False)
    phase_liquid_evaluator: PhaseEvaluator = field(init=False)
    phase_solid_evaluator: PhaseEvaluator = field(init=False)
    # Phase for calculations, could be a composite phase.
    phase_evaluator: PhaseEvaluator = field(init=False)
    state: State = field(init=False)
    initial_temperature: np.ndarray = field(init=False)
    initial_time: float = field(init=False, default=0)
    end_time: float = field(init=False, default=0)
    _solution: OptimizeResult = field(init=False, default_factory=OptimizeResult)

    def __post_init__(self):
        logger.info("Creating a SPIDER model")
        self.config: ConfigParser = MyConfigParser(self.filename)
        self.root = Path(self.root_path)
        self.scalings = scalings_from_configuration(self.config["scalings"])
        self.constants = Constants(self.scalings)
        self.mesh = StaggeredMesh.uniform_radii(self.scalings, **self.config["mesh"])
        self.phase_liquid_evaluator = phase_from_configuration(
            self.config["phase_liquid_evaluator"], self.scalings
        )
        self.phase_solid_evaluator = phase_from_configuration(
            self.config["phase_solid_evaluator"], self.scalings
        )
        # FIXME: For time being just set phase to liquid phase.
        self.phase_evaluator = self.phase_liquid_evaluator
        self.state = State(self.phase_evaluator, self.mesh)
        # Set the time stepping.
        self.initial_time = self.config.getfloat("timestepping", "start_time_years")
        self.initial_time *= self.constants.YEAR_IN_SECONDS
        self.end_time = self.config.getfloat("timestepping", "end_time_years")
        self.end_time *= self.constants.YEAR_IN_SECONDS
        # Set the initial condition.
        initial: SectionProxy = self.config["initial_condition"]
        self.initial_temperature = np.linspace(
            initial.getfloat("basal_temperature"),
            initial.getfloat("surface_temperature"),
            self.mesh.staggered.number,
        )
        self.initial_temperature /= self.scalings.temperature

    @property
    def solution(self) -> OptimizeResult:
        """The solution."""
        return self._solution

    def dTdt(
        self,
        time: float,
        temperature: np.ndarray,
        pressure: np.ndarray,
    ) -> np.ndarray:
        """dT/dt at the staggered nodes.

        Args:
            time: Time.
            temperature: Temperature at the staggered nodes.
            mesh: Mesh.
            pressure: Pressure at the staggered nodes.

        Returns:
            dT/dt at the staggered nodes.
        """
        self.state.update(temperature, pressure)

        # energy: SectionProxy = self.config["energy"]
        # heat_flux: np.ndarray = total_heat_flux(
        #     energy,
        #     self.mesh,
        #     self.state,
        #     temperature,
        #     pressure,
        # )

        heat_flux: np.ndarray = self.state.heat_flux
        # TODO: Clean up boundary conditions.
        # No heat flux from the core.
        heat_flux[0] = 0
        # Blackbody cooling.
        equilibrium_temperature: float = self.config.getfloat(
            "boundary_conditions", "equilibrium_temperature"
        )
        equilibrium_temperature /= self.scalings.temperature
        heat_flux[-1] = (
            self.config.getfloat("boundary_conditions", "emissivity")
            * self.constants.STEFAN_BOLTZMANN_CONSTANT
            * (
                self.mesh.quantity_at_basic_nodes(temperature)[-1] ** 4
                - equilibrium_temperature**4
            )
        )

        energy_flux: np.ndarray = heat_flux * self.mesh.basic.area
        logger.info("energy_flux = %s", energy_flux)

        delta_energy_flux: np.ndarray = np.diff(energy_flux)
        logger.info("delta_energy_flux = %s", delta_energy_flux)
        capacitance: np.ndarray = self.state.phase_staggered.capacitance * self.mesh.basic.volume

        dTdt: np.ndarray = -delta_energy_flux / capacitance

        # FIXME: Need to non-dimensionalise heating
        # dTdt += (
        #     self.phase.density
        #     * total_heating(self.config, time)
        #     * self.mesh.basic.volume
        #     / capacitance
        # )
        # logger.info("dTdt = %s", dTdt)

        return dTdt

    def plot(self, num_lines: int = 11) -> None:
        """Plots the solution with labelled lines according to time.

        Args:
            num_lines: Number of lines to plot. Defaults to 11.
        """
        assert self.solution is not None
        radii: np.ndarray = self.mesh.basic.radii
        y_basic: np.ndarray = self.mesh.quantity_at_basic_nodes(self.solution.y)
        times: np.ndarray = self.solution.t

        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111)

        # Ensure there are at least 2 lines to plot (first and last).
        num_lines = max(2, num_lines)

        # Calculate the time range.
        time_range: float = times[-1] - times[0]

        # Calculate the time step based on the total number of lines.
        time_step: float = time_range / (num_lines - 1)

        # Plot the first line.
        label_first: str = f"{times[0]:.2f}"
        ax.plot(y_basic[:, 0], radii, label=label_first)

        # Loop through the selected lines and plot each with a label.
        for i in range(1, num_lines - 1):
            desired_time: float = times[0] + i * time_step
            # Find the closest available time step.
            closest_time_index: int = np.argmin(np.abs(times - desired_time))
            time: float = times[closest_time_index]
            label: str = f"{time:.2f}"  # Create a label based on the time.
            plt.plot(y_basic[:, closest_time_index], radii, label=label)

        # Plot the last line.
        label_last: str = f"{times[-1]:.2f}"
        ax.plot(y_basic[:, -1], radii, label=label_last)

        # Shrink current axis by 20%.
        box = ax.get_position()
        ax.set_position((box.x0, box.y0, box.width * 0.8, box.height))

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Radii (m)")
        ax.set_title("Magma ocean thermal profile")
        ax.grid(True)
        legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        legend.set_title("Time (yr)")
        plt.show()

    def solve(self, atol: float = 1.0e-6, rtol: float = 1.0e-6) -> None:
        """Solves the system of ODEs to determine the interior temperature profile.

        Args:
            atol: Absolute tolerance. Defaults to 1.0e-6.
            rtol: Relative tolerance. Defaults to 1.0e-6.
        """
        self._solution = solve_ivp(
            self.dTdt,
            (self.initial_time, self.end_time),
            self.initial_temperature,
            method="BDF",
            vectorized=False,  # TODO: True could speed up BDF according to the documentation.
            args=(self.initial_temperature,),  # FIXME: Should be pressure.
            atol=atol,
            rtol=rtol,
        )

        logger.info(self.solution)


class MyConfigParser(ConfigParser):
    """A configuration parser with some default options

    Args:
        *filenames: Filenames of one or several configuration files
    """

    getpath: Callable[..., Path]  # For typing.

    def __init__(self, *filenames):
        kwargs: dict = {"comment_prefixes": ("#",), "converters": {"path": Path}}
        super().__init__(**kwargs)
        self.read(filenames)
