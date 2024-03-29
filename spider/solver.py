#
# Copyright 2024 Dan J. Bower
#
# This file is part of Spider.
#
# Spider is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Spider is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Spider. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Solver"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

from spider.core import SpiderData
from spider.parser import Parameters
from spider.phase import PhaseStateBasic, PhaseStateStaggered

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class State:
    """Stores the state at temperature and pressure.

    Args:
        _data: SpiderData
        conduction: Include conduction flux
        convection: Include convection flux
        gravitational_separation: Include gravitational separation flux
        mixing: Include mixing flux
        radionuclides: Include radionuclides
        tidal: Include tidal heating

    Attributes:
        conduction: Include conduction flux
        convection: Include convection flux
        gravitational_separation: Include gravitational separation flux
        mixing: Include mixing flux
        radionuclides: Include radionuclides
        tidal: Include tidal heating
        phase_basic: Phase properties at the basic nodes
        phase_staggered: Phase properties at the staggered nodes
        conductive_heat_flux: Conductive heat flux at the basic nodes
        convective_heat_flux: Convective heat flux at the basic nodes
        critical_reynolds_number: Critical Reynolds number
        dTdr: Temperature gradient with respect to radius at the basic nodes
        eddy_diffusivity: Eddy diffusivity at the basic nodes
        gravitational_separation: Gravitational separation at the basic nodes
        heating: Heat generation at the staggered nodes
        heat_flux: Heat flux at the basic nodes
        inviscid_regime: Array with True if the flow is inviscid and otherwise False
        inviscid_velocity: Inviscid velocity
        is_convective: Array with True if the flow is convecting and otherwise False
        mixing: Mixing heat flux at the basic nodes
        reynolds_number: Reynolds number
        super_adiabatic_temperature_gradient: Super adiabatic temperature gradient
        temperature_basic: Temperature at the basic nodes
        bottom_temperature: Temperature at the bottom basic node
        top_temperature: Temperature at the top basic node
        viscous_regime: Array with True if the flow is viscous and otherwise False
        viscous_velocity: Viscous velocity
    """

    data: SpiderData
    phase_basic: PhaseStateBasic = field(init=False)
    phase_staggered: PhaseStateStaggered = field(init=False)
    _dTdr: np.ndarray = field(init=False)
    _eddy_diffusivity: np.ndarray = field(init=False)
    _heat_flux: np.ndarray = field(init=False)
    _heating: np.ndarray = field(init=False)
    _is_convective: np.ndarray = field(init=False)
    _reynolds_number: np.ndarray = field(init=False)
    _super_adiabatic_temperature_gradient: np.ndarray = field(init=False)
    _temperature_basic: np.ndarray = field(init=False)
    _viscous_velocity: np.ndarray = field(init=False)
    _inviscid_velocity: np.ndarray = field(init=False)

    def __post_init__(self):
        self.phase_basic = PhaseStateBasic(self.data.phase)
        self.phase_staggered = PhaseStateStaggered(self.data.phase)

    def conductive_heat_flux(self) -> np.ndarray:
        """Conductive heat flux"""
        logger.debug(
            "thermal_conductivity.shape = %s", self.phase_basic.thermal_conductivity.shape
        )
        conductive_heat_flux: np.ndarray = -self.phase_basic.thermal_conductivity * self._dTdr
        return conductive_heat_flux

    def convective_heat_flux(self) -> np.ndarray:
        """Convective heat flux"""
        convective_heat_flux: np.ndarray = (
            -self.phase_basic.density
            * self.phase_basic.heat_capacity
            * self._eddy_diffusivity
            * self._super_adiabatic_temperature_gradient
        )
        return convective_heat_flux

    def radiogenic_heating(self, time: np.ndarray | float) -> np.ndarray | float:
        """Radiogenic heating

        Args:
            time: Time

        Returns:
            Radiogenic heating as a single column (in a 2-D array) if time is a float, otherwise a
                2-D array with each column associated with a single time in the time array.
        """
        radiogenic_heating_float: np.ndarray | float = 0
        for radionuclide in self.data.radionuclides:
            radiogenic_heating_float += radionuclide.get_heating(time)

        radiogenic_heating: np.ndarray | float = radiogenic_heating_float * (
            self.phase_staggered.density / self.phase_staggered.capacitance
        )

        return radiogenic_heating

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

    def gravitational_separation_flux(self) -> np.ndarray:
        """Gravitational separation"""
        raise NotImplementedError

    @property
    def heating(self) -> np.ndarray:
        """The total heating rate according to the heat sources specified in the configuration."""
        return self._heating

    @property
    def heat_flux(self) -> np.ndarray:
        """The total heat flux according to the fluxes specified in the configuration."""
        return self._heat_flux

    # TODO: Check this again
    @heat_flux.setter
    def heat_flux(self, value):
        """Setter for applying boundary conditions"""
        self._heat_flux = value

    @property
    def inviscid_regime(self) -> np.ndarray:
        return self._reynolds_number > self.critical_reynolds_number

    @property
    def inviscid_velocity(self) -> np.ndarray:
        return self._inviscid_velocity

    @property
    def is_convective(self) -> np.ndarray:
        return self._is_convective

    def mixing_flux(self) -> np.ndarray:
        """Mixing heat flux"""
        raise NotImplementedError

    @property
    def reynolds_number(self) -> np.ndarray:
        return self._reynolds_number

    @property
    def super_adiabatic_temperature_gradient(self) -> np.ndarray:
        return self._super_adiabatic_temperature_gradient

    @property
    def temperature_basic(self) -> np.ndarray:
        return self._temperature_basic

    @property
    def top_temperature(self) -> np.ndarray:
        return self._temperature_basic[-1, :]

    @property
    def bottom_temperature(self) -> np.ndarray:
        return self._temperature_basic[0, :]

    @property
    def viscous_regime(self) -> np.ndarray:
        return self._reynolds_number <= self.critical_reynolds_number

    @property
    def viscous_velocity(self) -> np.ndarray:
        return self._viscous_velocity

    def _set_temperature(self, temperature: np.ndarray) -> None:
        """Sets the temperature at the basic nodes

        This also ensures that the temperature profile adheres to any imposed thermal boundary
        conditions at the top or bottom surfaces.

        Args:
            temperature: Temperature at the staggered nodes
        """
        logger.debug("Setting the temperature profile")
        self._temperature_basic = self.data.mesh.quantity_at_basic_nodes(temperature)
        self._dTdr = self.data.mesh.d_dr_at_basic_nodes(temperature)
        self.data.boundary_conditions.conform_temperature_boundary_conditions(
            temperature, self._temperature_basic, self._dTdr
        )
        logger.debug("_temperature_basic = %s", self._temperature_basic)
        logger.debug("_dTdr = %s", self._dTdr)

    def update(self, temperature: np.ndarray, time: np.ndarray | float) -> None:
        """Updates the state.

        The evaluation order matters because we want to minimise the number of evaluations.

        Args:
            temperature: Temperature at the staggered nodes
            pressure: Pressure at the staggered nodes
            time: Time
        """
        logger.debug("Updating the state")
        self._set_temperature(temperature)
        self.phase_staggered.update(temperature, self.data.mesh.staggered.eos.pressure)
        self.phase_basic.update(self._temperature_basic, self.data.mesh.basic.eos.pressure)
        self._super_adiabatic_temperature_gradient = self._dTdr - self.phase_basic.dTdrs
        self._is_convective = self._super_adiabatic_temperature_gradient < 0
        velocity_prefactor: np.ndarray = (
            -self.phase_basic.gravitational_acceleration
            * self.phase_basic.thermal_expansivity
            * self._super_adiabatic_temperature_gradient
        )
        # Viscous velocity
        self._viscous_velocity = (
            velocity_prefactor * self.data.mesh.basic.mixing_length_cubed
        ) / (18 * self.phase_basic.kinematic_viscosity)
        self._viscous_velocity[~self.is_convective] = 0  # Must be super-adiabatic
        # Inviscid velocity
        self._inviscid_velocity = (
            velocity_prefactor * self.data.mesh.basic.mixing_length_squared
        ) / 16
        self._inviscid_velocity[~self.is_convective] = 0  # Must be super-adiabatic
        self._inviscid_velocity[self._is_convective] = np.sqrt(
            self._inviscid_velocity[self._is_convective]
        )
        # Reynolds number
        self._reynolds_number = (
            self._viscous_velocity
            * self.data.mesh.basic.mixing_length
            / self.phase_basic.kinematic_viscosity
        )
        # Eddy diffusivity
        self._eddy_diffusivity = np.where(
            self.viscous_regime, self._viscous_velocity, self._inviscid_velocity
        )
        self._eddy_diffusivity *= self.data.mesh.basic.mixing_length
        logger.debug("Before evaluating heat flux")
        # Heat flux
        # FIXME: try and remove zeros_like
        self._heat_flux = 0  # np.zeros_like(self.temperature_basic)
        if self.data.parameters.energy.conduction:
            self._heat_flux += self.conductive_heat_flux()
            logger.debug("heat_flux.shape = %s", self._heat_flux.shape)
        if self.data.parameters.energy.convection:
            self._heat_flux += self.convective_heat_flux()
            logger.debug("heat_flux.shape = %s", self._heat_flux.shape)
        if self.data.parameters.energy.gravitational_separation:
            self._heat_flux += self.gravitational_separation_flux()
        if self.data.parameters.energy.mixing:
            self._heat_flux += self.mixing_flux()
        # Heating
        # FIXME: try and remove zeros_like
        self._heating = 0  # np.zeros_like(temperature)
        if self.data.parameters.energy.radionuclides:
            self._heating += self.radiogenic_heating(time)


class SpiderSolver:
    """Creates the system and solves the interior dynamics

    Args:
        filename: Filename of a file with configuration settings
        root: Root path to the flename

    Attributes:
        filename: Filename of a file with configuration settings
        root: Root path to the filename. Defaults to empty
        config: Configuration data
        data: Model data
        state: Model state
    """

    def __init__(self, filename: str | Path, root: str | Path = Path()):
        logger.info("Creating a SPIDER model")
        self.filename = Path(filename)
        self.root = Path(root)
        self.parameters: Parameters
        self.data: SpiderData
        self.state: State
        self._solution: OptimizeResult
        self.parse_configuration()

    def parse_configuration(self) -> None:
        """Parses a configuration file"""
        configuration_file: Path = self.root / self.filename
        logger.info("Parsing configuration file = %s", configuration_file)
        self.parameters = Parameters.from_file(configuration_file)

    def initialize(self) -> None:
        """Initializes the model using configuration data"""
        logger.info("Initializing %s", self.__class__.__name__)
        self.data = SpiderData(self.parameters)
        self.state = State(self.data)

    def get_temperature(self) -> np.ndarray:
        """Temperature in kelvin

        Returns:
            Temperature in kelvin at the staggered nodes
        """
        temperature: np.ndarray = self.solution.y * self.data.parameters.scalings.temperature
        return temperature

    @property
    def solution(self) -> OptimizeResult:
        """The solution."""
        return self._solution

    def dTdt(
        self,
        time: np.ndarray | float,
        temperature: np.ndarray,
    ) -> np.ndarray:
        """dT/dt at the staggered nodes

        Args:
            time: Time
            temperature: Temperature at the staggered nodes

        Returns:
            dT/dt at the staggered nodes
        """
        logger.debug("temperature passed into dTdt = %s", temperature)
        logger.debug("temperature.shape = %s", temperature.shape)
        self.state.update(temperature, time)
        heat_flux: np.ndarray = self.state.heat_flux
        logger.debug("heat_flux = %s", heat_flux)
        self.data.boundary_conditions.apply(self.state)
        logger.debug("heat_flux = %s", heat_flux)
        logger.debug("mesh.basic.area.shape = %s", self.data.mesh.basic.area.shape)

        energy_flux: np.ndarray = heat_flux * self.data.mesh.basic.area
        logger.debug("energy_flux size = %s", energy_flux.shape)

        delta_energy_flux: np.ndarray = np.diff(energy_flux, axis=0)
        logger.debug("delta_energy_flux size = %s", delta_energy_flux.shape)
        logger.debug("capacitance = %s", self.state.phase_staggered.capacitance.shape)
        capacitance: np.ndarray = (
            self.state.phase_staggered.capacitance * self.data.mesh.basic.volume
        )

        dTdt: np.ndarray = -delta_energy_flux / capacitance
        logger.debug("dTdt (fluxes only) = %s", dTdt)

        dTdt += self.state.heating
        logger.debug("dTdt (with internal heating) = %s", dTdt)

        return dTdt

    def solve(self) -> None:
        """Solves the system of ODEs to determine the interior temperature profile."""

        start_time: float = self.data.parameters.solver.start_time
        logger.debug("start_time = %f", start_time)
        end_time: float = self.data.parameters.solver.end_time
        logger.debug("end_time = %f", end_time)
        atol: float = self.data.parameters.solver.atol
        rtol: float = self.data.parameters.solver.rtol

        self._solution = solve_ivp(
            self.dTdt,
            (start_time, end_time),
            self.data.initial_condition.temperature,
            method="BDF",
            vectorized=True,
            atol=atol,
            rtol=rtol,
        )

        logger.info(self.solution)
