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
"""A phase defines the equation of state (EOS) and transport properties."""

from __future__ import annotations

import copy
import logging
import sys
from dataclasses import KW_ONLY, Field, dataclass, field, fields

import numpy as np
from scipy.interpolate import RectBivariateSpline

from spider.interfaces import (
    PhaseEvaluatorABC,
    PhaseEvaluatorProtocol,
    PropertyProtocol,
)
from spider.parser import _MeshSettings, _PhaseMixedSettings, _PhaseSettings
from spider.utilities import (
    FloatOrArray,
    combine_properties,
    is_file,
    is_number,
    tanh_weight,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class ConstantProperty(PropertyProtocol):
    """A property with a constant value

    Args:
        name: Name of the property
        value: The constant value

    Attributes:
        name: Name of the property
        value: The constant value
        ndim: Number of dimensions, which is equal to zero for a constant property
    """

    name: str
    _: KW_ONLY
    value: float
    ndim: int = field(init=False, default=0)

    def eval(self) -> float:
        return self.value

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> float:
        """Evaluates the property.

        Args:
            temperature: A 2-D array with temperature at constant time in columns
            pressure: A 2-D array with pressure
        """
        del temperature
        del pressure
        return self.eval()


@dataclass
class LookupProperty1D(PropertyProtocol):
    """A property from a 1-D lookup

    Args:
        name: Name of the property
        value: A 2-D array, with x values in the first column and y values in the second column.

    Attributes:
        name: Name of the property
        value: A 2-D array
        ndim: Number of dimensions, which is equal to one for a 1-D lookup
    """

    name: str
    _: KW_ONLY
    value: np.ndarray
    ndim: int = field(init=False, default=1)
    _gradient: np.ndarray = field(init=False)

    def __post_init__(self):
        # Sort the data to ensure x is increasing
        self.value = self.value[self.value[:, 0].argsort()]
        self._gradient = np.gradient(self.value[:, 1], self.value[:, 0])

    def eval(self, pressure: np.ndarray) -> np.ndarray:
        return np.interp(pressure, self.value[:, 0], self.value[:, 1])

    def gradient(self, pressure: np.ndarray) -> np.ndarray:
        """Computes the gradient"""
        return np.interp(pressure, self.value[:, 0], self._gradient)

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        del temperature
        return self.eval(pressure)


@dataclass
class LookupProperty2D(PropertyProtocol):
    """A property from a 2-D lookup

    Args:
        name: Name of the property
        value: The 2-D array

    Attributes:
        name: Name of the property
        value: The 2-D array
        ndim: Number of dimensions, which is equal to two for a 2-D lookup
    """

    name: str
    _: KW_ONLY
    value: np.ndarray
    ndim: int = field(init=False, default=2)
    _lookup: RectBivariateSpline = field(init=False)

    def __post_init__(self):
        # x and y must be increasing otherwise the interpolation might give unexpected behaviour
        x_values = self.value[:, 0].round(decimals=0)
        logger.debug("x_values round = %s", x_values)
        logger.debug("self.value.shape = %s", self.value.shape)
        x_values: np.ndarray = np.unique(self.value[:, 0])
        logger.debug("x_values.shape = %s", x_values.shape)
        y_values: np.ndarray = np.unique(self.value[:, 1])
        logger.debug("y_values.shape = %s", y_values.shape)
        z_values: np.ndarray = self.value[:, 2]
        logger.debug("z_values = %s", z_values)
        z_values = z_values.reshape((x_values.size, y_values.size), order="F")
        self._lookup = RectBivariateSpline(x_values, y_values, z_values, kx=1, ky=1, s=0)

    def eval(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self._lookup(pressure, temperature, grid=False)

    def __call__(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return self.eval(temperature, pressure)


class SinglePhaseEvaluator(PhaseEvaluatorABC):
    """Contains the objects to evaluate the EOS and transport properties of a phase.

    Args:
        settings: phase settings
        mesh: mesh settings
    """

    _density: PropertyProtocol
    _gravitational_acceleration: PropertyProtocol
    _heat_capacity: PropertyProtocol
    _melt_fraction: ConstantProperty
    _thermal_conductivity: PropertyProtocol
    _thermal_expansivity: PropertyProtocol
    _viscosity: PropertyProtocol

    def __init__(self, settings: _PhaseSettings, mesh: _MeshSettings):
        self._settings: _PhaseSettings = settings
        self._mesh: _MeshSettings = mesh
        cls_fields: tuple[Field, ...] = fields(self._settings)
        for field_ in cls_fields:
            name: str = field_.name
            private_name: str = f"_{name}"
            value = getattr(self._settings, field_.name)

            if is_number(value):
                # Numbers have already been scaled by the parser
                setattr(self, private_name, ConstantProperty(name=name, value=value))

            elif is_file(value):
                with open(value, encoding="utf-8") as infile:
                    logger.debug("%s is a file = %s", name, value)
                    header = infile.readline()
                    col_names = header[1:].split()
                value_array: np.ndarray = np.loadtxt(value, ndmin=2)
                logger.debug("before scaling, value_array = %s", value_array)
                # Scale lookup data
                for nn, col_name in enumerate(col_names):
                    logger.info("Scaling %s from %s", col_name, value)
                    value_array[:, nn] /= getattr(self._settings.scalings_, col_name)
                logger.debug("after scaling, value_array = %s", value_array)
                ndim = value_array.shape[1]
                logger.debug("ndim = %d", ndim)
                if ndim == 2:
                    setattr(self, private_name, LookupProperty1D(name=name, value=value_array))
                elif ndim == 3:
                    setattr(self, private_name, LookupProperty2D(name=name, value=value_array))
                else:
                    raise ValueError(f"Lookup data must have 2 or 3 dimensions, not {ndim}")
            else:
                msg: str = f"Cannot interpret value ({value}): not a number or a file"
                raise ValueError(msg)

        self._gravitational_acceleration = ConstantProperty(
            "gravitational_acceleration", value=self._mesh.gravitational_acceleration
        )

    @override
    def density(self) -> FloatOrArray:
        return self._density(self.temperature, self.pressure)

    @override
    def gravitational_acceleration(self) -> FloatOrArray:
        return self._gravitational_acceleration(self.temperature, self.pressure)

    @override
    def heat_capacity(self) -> FloatOrArray:
        return self._heat_capacity(self.temperature, self.pressure)

    @override
    def melt_fraction(self) -> float:
        return self._melt_fraction(self.temperature, self.pressure)

    @override
    def thermal_conductivity(self) -> FloatOrArray:
        return self._thermal_conductivity(self.temperature, self.pressure)

    @override
    def thermal_expansivity(self) -> FloatOrArray:
        return self._thermal_expansivity(self.temperature, self.pressure)

    @override
    def viscosity(self) -> FloatOrArray:
        return self._viscosity(self.temperature, self.pressure)


class MixedPhaseEvaluator(PhaseEvaluatorABC):
    """Evaluates the EOS and transport properties of a mixed phase.

    This only computes quantities within the mixed phase region between the solidus and the
    liquidus. Computing quantities outside of this region will give incorrect results.

    Args:
        settings: Mixed phase settings
        solid: Solid phase evaluator
        liquid: Liquid phase evaluator
    """

    _delta_density: FloatOrArray
    _delta_fusion: np.ndarray
    _density: np.ndarray
    _gravitational_acceleration: FloatOrArray
    _heat_capacity: np.ndarray
    _melt_fraction: np.ndarray
    _melt_fraction_no_clip: np.ndarray
    _porosity: np.ndarray
    _thermal_conductivity: np.ndarray
    _thermal_expansivity: np.ndarray
    _viscosity: np.ndarray

    def __init__(
        self,
        settings: _PhaseMixedSettings,
        solid: PhaseEvaluatorProtocol,
        liquid: PhaseEvaluatorProtocol,
    ):
        self.settings: _PhaseMixedSettings = settings
        self._solid: PhaseEvaluatorProtocol = copy.deepcopy(solid)
        self._liquid: PhaseEvaluatorProtocol = copy.deepcopy(liquid)
        self._solidus: LookupProperty1D = self._get_melting_curve_lookup(
            "solidus", self.settings.solidus
        )
        self._liquidus: LookupProperty1D = self._get_melting_curve_lookup(
            "liquidus", self.settings.liquidus
        )

    def set_pressure(self, pressure: np.ndarray) -> None:
        """Sets pressure and updates quantities that only depend on pressure"""
        super().set_pressure(pressure)
        # Sets the temperature of the solid and liquid phases to the appropriate melting curve in
        # order to evaluate mixed properties
        self._solid.set_temperature(self.solidus())
        self._solid.set_pressure(pressure)
        self._liquid.set_temperature(self.solidus())
        self._liquid.set_pressure(pressure)
        # Precompute below to avoid repeat calculation
        self._delta_density = self._solid.density() - self._liquid.density()
        self._delta_fusion = self.liquidus() - self.solidus()
        # Heat capacity of the mixed phase :cite:p:`{Equation 4,}SOLO07`
        self._heat_capacity = self.settings.latent_heat_of_fusion / self.delta_fusion()

    def update(self):
        """Minimises the number of function evaluations."""

        # Melt fraction without clipping
        # phi<0 for the solid, 0<phi<1 for the mixed phase, and phi>1 for the melt
        self._melt_fraction_no_clip = (self.temperature - self.solidus()) / self.delta_fusion()
        logger.debug("_melt_fraction_no_clip = %s", self.melt_fraction_no_clip())
        self._melt_fraction = np.clip(self._melt_fraction_no_clip, 0, 1)

        # Mixed density by volume additivity
        self._density = combine_properties(
            self.melt_fraction(), 1 / self._liquid.density(), 1 / self._solid.density()
        )
        self._density = 1 / self._density

        # Porosity
        self._porosity = (self._solid.density() - self.density()) / self.delta_density()

        # Thermal conductivity
        self._thermal_conductivity = combine_properties(
            self.melt_fraction(),
            self._liquid.thermal_conductivity(),
            self._solid.thermal_conductivity(),
        )

        # Thermal expansivity :cite:p:`{Equation 3,}SOLO07`
        # The first term in :cite:t:`{Equation 3,}SOLO07` is not included because it is small
        # compared to the latent heat term :cite:p:`{Equation 33,}SS93`.
        self._thermal_expansivity = self.delta_density() / self.delta_fusion() / self.density()

        # Viscosity
        weight: np.ndarray = tanh_weight(
            self.melt_fraction(),
            self.settings.rheological_transition_melt_fraction,
            self.settings.rheological_transition_width,
        )
        log10_viscosity: np.ndarray = combine_properties(
            weight, np.log10(self._liquid.viscosity()), np.log10(self._solid.viscosity())
        )
        self._viscosity = 10**log10_viscosity

    def delta_density(self) -> FloatOrArray:
        return self._delta_density

    def delta_fusion(self) -> np.ndarray:
        return self._delta_fusion

    @override
    def density(self) -> np.ndarray:
        return self._density

    @override
    def gravitational_acceleration(self) -> FloatOrArray:
        return self._solid.gravitational_acceleration()

    @override
    def heat_capacity(self) -> FloatOrArray:
        """Heat capacity of the mixed phase :cite:p:`{Equation 4,}SOLO07`"""
        return self._heat_capacity

    def liquidus(self) -> np.ndarray:
        """Liquidus"""
        liquidus: np.ndarray = self._liquidus.eval(self.pressure)
        logger.debug("liquidus = %s", liquidus)

        return liquidus

    def liquidus_gradient(self) -> np.ndarray:
        """Liquidus gradient with respect to pressure"""
        return self._liquidus.gradient(self.pressure)

    def melt_fraction_no_clip(self) -> np.ndarray:
        """Melt fraction without clipping"""
        return self._melt_fraction_no_clip

    @override
    def melt_fraction(self) -> np.ndarray:
        """Melt fraction of the mixed phase

        The melt fraction is always between zero and one.
        """
        return self._melt_fraction

    def porosity(self) -> np.ndarray:
        """Porosity of the mixed phase, that is the volume fraction occupied by the melt"""
        return self._porosity

    def solidus(self) -> np.ndarray:
        """Solidus"""
        solidus: np.ndarray = self._solidus.eval(self.pressure)
        logger.debug("solidus = %s", solidus)

        return solidus

    def solidus_gradient(self) -> np.ndarray:
        """Solidus gradient with respect to pressure"""
        return self._solidus.gradient(self.pressure)

    @override
    def thermal_conductivity(self) -> np.ndarray:
        return self._thermal_conductivity

    @override
    def thermal_expansivity(self) -> np.ndarray:
        return self._thermal_expansivity

    @override
    def viscosity(self) -> np.ndarray:
        return self._viscosity

    def _get_melting_curve_lookup(self, name: str, value: str) -> LookupProperty1D:
        with open(value, encoding="utf-8") as infile:
            header = infile.readline()
            col_names = header[1:].split()
        value_array: np.ndarray = np.loadtxt(value, ndmin=2)
        logger.debug("before scaling, value_array = %s", value_array)
        for nn, col_name in enumerate(col_names):
            logger.info("Scaling %s from %s", col_name, value)
            value_array[:, nn] /= getattr(self.settings.scalings_, col_name)

        return LookupProperty1D(name=name, value=value_array)


class CompositePhaseEvaluator(PhaseEvaluatorABC):
    """Evaluates the EOS and transport properties of a composite phase.

    This combines the single phase evaluators for the liquid and solid regions with the mixed phase
    evaluator for the mixed phase region. This ensure that the phase properties are computed
    correctly for all temperatures and pressures.

    Args:
        solid: Solid phase evaluator
        liquid: Liquid phase evaluator
        mixed: Mixed phase evaluator
    """

    _density: np.ndarray
    _dTdPs: np.ndarray
    _heat_capacity: np.ndarray
    _melt_fraction: np.ndarray
    _thermal_conductivity: np.ndarray
    _thermal_expansivity: np.ndarray
    _viscosity: np.ndarray
    _blending_factor: np.ndarray
    _liquid_mask: np.ndarray
    _solid_mask: np.ndarray

    def __init__(
        self,
        solid: PhaseEvaluatorProtocol,
        liquid: PhaseEvaluatorProtocol,
        mixed: MixedPhaseEvaluator,
    ):
        self.solid: PhaseEvaluatorProtocol = copy.deepcopy(solid)
        self.liquid: PhaseEvaluatorProtocol = copy.deepcopy(liquid)
        self.mixed: MixedPhaseEvaluator = copy.deepcopy(mixed)

    @override
    def set_temperature(self, temperature: np.ndarray) -> None:
        super().set_temperature(temperature)
        self.solid.set_temperature(temperature)
        self.liquid.set_temperature(temperature)
        self.mixed.set_temperature(temperature)

    @override
    def set_pressure(self, pressure: np.ndarray) -> None:
        """Sets pressure and updates quantities that only depend on pressure"""
        super().set_pressure(pressure)
        self.solid.set_pressure(pressure)
        self.liquid.set_pressure(pressure)
        self.mixed.set_pressure(pressure)

    @override
    def update(self) -> None:
        self.mixed.update()
        self._set_blending_and_masks()
        self._density = self._get_composite("density")
        self._dTdPs = self._get_composite("dTdPs")
        self._heat_capacity = self._get_composite("heat_capacity")
        self._thermal_conductivity = self._get_composite("thermal_conductivity")
        self._thermal_expansivity = self._get_composite("thermal_expansivity")

        name: str = "viscosity"
        log10_mixed_phase: np.ndarray = np.log10(getattr(self.mixed, name)())
        single_phase: np.ndarray = np.empty_like(self._blending_factor)
        try:
            single_phase[self._liquid_mask] = getattr(self.liquid, name)()[self._liquid_mask]
        except (IndexError, TypeError):
            single_phase[self._liquid_mask] = getattr(self.liquid, name)()
        try:
            single_phase[self._solid_mask] = getattr(self.solid, name)()[self._solid_mask]
        except (IndexError, TypeError):
            single_phase[self._solid_mask] = getattr(self.solid, name)()
        log10_single_phase: np.ndarray = np.log10(single_phase)
        self._viscosity = combine_properties(
            self._blending_factor, log10_mixed_phase, log10_single_phase
        )
        self._viscosity = 10**self._viscosity

    @override
    def density(self) -> np.ndarray:
        return self._density

    @override
    def dTdPs(self) -> np.ndarray:
        return self._dTdPs

    @override
    def gravitational_acceleration(self) -> FloatOrArray:
        return self.mixed.gravitational_acceleration()

    @override
    def heat_capacity(self) -> np.ndarray:
        """Heat capacity"""
        return self._heat_capacity

    def liquidus(self) -> np.ndarray:
        return self.mixed.liquidus()

    def liquidus_gradient(self) -> np.ndarray:
        return self.mixed.liquidus_gradient()

    @override
    def melt_fraction(self) -> np.ndarray:
        """Melt fraction"""
        return self.mixed.melt_fraction()

    def solidus(self) -> np.ndarray:
        return self.mixed.solidus()

    def solidus_gradient(self) -> np.ndarray:
        return self.mixed.solidus_gradient()

    @override
    def thermal_conductivity(self) -> np.ndarray:
        """Thermal conductivity"""
        return self._thermal_conductivity

    @override
    def thermal_expansivity(self) -> np.ndarray:
        """Thermal expansivity"""
        return self._thermal_expansivity

    def viscosity(self) -> np.ndarray:
        """Viscosity"""
        return self._viscosity

    def _set_blending_and_masks(self) -> None:
        """Sets blending and masks."""

        phase_transition_width: float = self.mixed.settings.phase_transition_width
        melt_fraction_no_clip: np.ndarray = self.mixed.melt_fraction_no_clip()

        if phase_transition_width == 0.0:
            blending_factor: np.ndarray = np.where(
                ((melt_fraction_no_clip < 0.0) | (melt_fraction_no_clip > 1.0)),
                0,
                1,
            )
        else:
            blending_liquid: np.ndarray = 1.0 - tanh_weight(
                melt_fraction_no_clip, 1.0, phase_transition_width
            )
            blending_solid: np.ndarray = tanh_weight(
                melt_fraction_no_clip, 0.0, phase_transition_width
            )
            blending_factor = np.where(
                melt_fraction_no_clip > 0.5, blending_liquid, blending_solid
            )

        self._blending_factor = blending_factor
        logger.debug("_blending_factor = %s", self._blending_factor)
        self._liquid_mask = melt_fraction_no_clip > 0.5
        logger.debug("_liquid_mask = %s", self._liquid_mask)
        self._solid_mask = ~self._liquid_mask
        logger.debug("_solid_mask = %s", self._solid_mask)

    def _get_composite(self, property_name: str) -> np.ndarray:
        """Evaluates the composite property"""
        mixed_phase: np.ndarray = getattr(self.mixed, property_name)()
        single_phase: np.ndarray = np.empty_like(self._blending_factor)
        logger.debug("single_phase = %s", single_phase)
        logger.debug("_liquid_mask = %s", self._liquid_mask)
        logger.debug("_solid_mask = %s", self._solid_mask)
        test = getattr(self.liquid, property_name)()
        logger.debug("test = %s", test)

        # logger.debug(self.temperature.shape)
        # logger.debug(self.pressure.shape)
        # logger.debug(mixed_phase.shape)
        # logger.debug(single_phase.shape)

        try:
            single_phase[self._liquid_mask] = getattr(self.liquid, property_name)()[
                self._liquid_mask
            ]
        except (IndexError, TypeError):
            single_phase[self._liquid_mask] = getattr(self.liquid, property_name)()
        try:
            single_phase[self._solid_mask] = getattr(self.solid, property_name)()[self._solid_mask]
        except (IndexError, TypeError):
            single_phase[self._solid_mask] = getattr(self.solid, property_name)()

        combined: np.ndarray = combine_properties(self._blending_factor, mixed_phase, single_phase)

        return combined
