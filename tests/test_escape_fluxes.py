"""
Tests for the species number-flux functions in isofate.isofunks
(Phi_1/Phi_2 binary diffusion-limited escape, and the Zahnle et al. 1990
minor-species fluxes).
"""

import numpy as np
import pytest

from isofate import constants as c
from isofate.isofunks import (
    Johnson_reduction,
    Phi_1,
    Phi_2,
    Phi_C_Z90,
    Phi_D_GC23,
    Phi_D_Z90,
    Phi_D_Z90_mod,
    Phi_minor_species,
    Phi_N_Z90,
    Phi_O_Z90,
    Phi_S_Z90,
    get_binary_diffusion_coeff,
    phi_RR,
    phiE_CP,
)

T = 800.0  # [K]
GRAV = 10.0  # [m/s2]


def scale_height(M_molar, T=T, g=GRAV):
    return c.R_gas * T / (M_molar * g)


H_H, H_He, H_D = scale_height(c.M_H), scale_height(c.M_He), scale_height(c.M_D)
H_O, H_C, H_N, H_S = scale_height(c.M_O), scale_height(c.M_C), scale_height(c.M_N), scale_height(c.M_S)


# ---------------------------------------------------------------- binary H/He escape


class TestBinaryEscape:
    m1, m2 = c.mu_H, c.mu_He
    x1, x2 = 0.9, 0.1
    b = c.b_H_He(T)
    mu = x1 * m1 + x2 * m2

    @property
    def phi_c(self):
        return self.b * self.x1 * (self.m2 - self.m1) / H_H

    def fluxes(self, phi):
        args = (phi, self.b, H_H, H_He, self.m1, self.m2, self.x1, self.x2, self.mu)
        return Phi_1(*args), Phi_2(*args)

    def test_critical_flux_is_returned_with_output_1(self):
        _, phi_c = Phi_1(1e-8, self.b, H_H, H_He, self.m1, self.m2, self.x1, self.x2, self.mu, output=1)
        assert phi_c == pytest.approx(self.phi_c)

    def test_subcritical_only_light_species_escapes(self):
        phi = 0.5 * self.phi_c
        Phi1, Phi2 = self.fluxes(phi)
        assert Phi1 == pytest.approx(phi / self.m1)
        assert Phi2 == 0

    @pytest.mark.parametrize("multiple", [1.5, 10.0, 1e3])
    def test_supercritical_both_species_escape(self, multiple):
        Phi1, Phi2 = self.fluxes(multiple * self.phi_c)
        assert Phi1 > 0
        assert Phi2 > 0

    @pytest.mark.parametrize("multiple", [0.1, 0.99, 1.0, 1.01, 5.0, 1e4])
    def test_mass_flux_is_conserved(self, multiple):
        phi = multiple * self.phi_c
        Phi1, Phi2 = self.fluxes(phi)
        assert self.m1 * Phi1 + self.m2 * Phi2 == pytest.approx(phi, rel=1e-10)

    def test_fluxes_are_continuous_at_critical_flux(self):
        Phi1_below, Phi2_below = self.fluxes(self.phi_c * (1 - 1e-9))
        Phi1_above, Phi2_above = self.fluxes(self.phi_c * (1 + 1e-9))
        assert Phi2_below == 0
        assert Phi2_above == pytest.approx(0, abs=1e-6 * Phi1_below)
        assert Phi1_above == pytest.approx(Phi1_below, rel=1e-6)

    def test_heavy_species_is_fractionated_supercritically(self):
        # The heavy species escapes less efficiently than its share of the gas
        Phi1, Phi2 = self.fluxes(3 * self.phi_c)
        assert Phi2 / Phi1 < self.x2 / self.x1

    def test_fractionation_vanishes_at_high_flux(self):
        Phi1, Phi2 = self.fluxes(1e6 * self.phi_c)
        assert Phi2 / Phi1 == pytest.approx(self.x2 / self.x1, rel=1e-3)

    def test_empty_atmosphere_returns_zero(self):
        assert Phi_1(1e-8, self.b, H_H, H_He, self.m1, self.m2, 0, 0, 0) == 0
        assert Phi_1(1e-8, self.b, H_H, H_He, self.m1, self.m2, 0, 0, 0, output=1) == (0, 0)
        assert Phi_2(1e-8, self.b, H_H, H_He, self.m1, self.m2, 0, 0, 0) == 0


# ---------------------------------------------------------------- minor species (Zahnle et al. 1990)

N_H, N_He, N_D = 1e40, 1e39, 2e35
N_O, N_C, N_N, N_S = 5e36, 3e36, 1e36, 2e35
PHI_H, PHI_He = 1e18, 1e16

# (function, scale height, index of the minor species in the [H, He, D, O, C, N, S] list)
HE_D_LOOKUP_BUG = pytest.mark.xfail(
    strict=True,
    reason="get_binary_diffusion_coeff('He', 'D') sorts D before He, misses ('He', 'D') in its table and "
    "silently returns b_H_He instead of b_He_D",
)

Z90_FUNCS = [
    pytest.param(Phi_D_Z90, H_D, 2, id="D"),
    pytest.param(Phi_O_Z90, H_O, 3, id="O"),
    pytest.param(Phi_C_Z90, H_C, 4, id="C"),
    pytest.param(Phi_N_Z90, H_N, 5, id="N"),
    pytest.param(Phi_S_Z90, H_S, 6, id="S"),
]


def call_z90(fn, H_minor, Phi_H=PHI_H, Phi_He=PHI_He, N=None):
    N = N or [N_H, N_He, N_D, N_O, N_C, N_N, N_S]
    return fn(Phi_H, Phi_He, H_H, H_minor, H_He, *N, T)


@pytest.mark.parametrize("fn, H_minor, idx", Z90_FUNCS)
class TestZahnleMinorSpecies:
    def test_positive_for_vigorous_escape(self, fn, H_minor, idx):
        assert call_z90(fn, H_minor) > 0

    def test_zero_without_hydrogen(self, fn, H_minor, idx):
        N = [0, N_He, N_D, N_O, N_C, N_N, N_S]
        assert call_z90(fn, H_minor, N=N) == 0

    def test_zero_for_empty_atmosphere(self, fn, H_minor, idx):
        assert call_z90(fn, H_minor, N=[0] * 7) == 0

    def test_never_negative_for_weak_escape(self, fn, H_minor, idx):
        assert call_z90(fn, H_minor, Phi_H=1.0, Phi_He=0.0) == 0

    def test_zero_when_species_absent(self, fn, H_minor, idx):
        N = [N_H, N_He, N_D, N_O, N_C, N_N, N_S]
        N[idx] = 0
        assert call_z90(fn, H_minor, N=N) == 0

    def test_linear_in_own_abundance(self, fn, H_minor, idx):
        # Doubling a trace species doubles its flux (x_He changes negligibly)
        N = [N_H, N_He, N_D, N_O, N_C, N_N, N_S]
        base = call_z90(fn, H_minor, N=N)
        N[idx] *= 2
        assert call_z90(fn, H_minor, N=N) == pytest.approx(2 * base, rel=1e-3)

    def test_escapes_less_efficiently_than_hydrogen(self, fn, H_minor, idx):
        # Heavier species are fractionated: per-atom loss rate below that of H
        N = [N_H, N_He, N_D, N_O, N_C, N_N, N_S]
        assert call_z90(fn, H_minor) / N[idx] < PHI_H / N_H

    def test_matches_generalised_minor_species_function(self, fn, H_minor, idx, request):
        # Phi_minor_species with H (0) and He (1) dominant should reproduce the
        # species-specific Z90 functions.
        if idx == 2:
            request.applymarker(HE_D_LOOKUP_BUG)
        N = [N_H, N_He, N_D, N_O, N_C, N_N, N_S]
        general = Phi_minor_species(PHI_H, PHI_He, H_H, H_He, H_minor, N, T, idx, 0, 1)
        assert general == pytest.approx(call_z90(fn, H_minor), rel=1e-12)


def test_heavier_minor_species_are_more_fractionated():
    per_atom = {
        name: call_z90(fn, H) / n
        for name, fn, H, n in [("D", Phi_D_Z90, H_D, N_D), ("C", Phi_C_Z90, H_C, N_C), ("O", Phi_O_Z90, H_O, N_O)]
    }
    assert per_atom["D"] > per_atom["C"] > per_atom["O"]


def test_deuterium_flux_reduces_to_closed_form_without_helium():
    # With no He and no other heavies: Phi_D = f_D * (Phi_H - b_HD (1/H_D - 1/H_H))
    got = Phi_D_Z90(PHI_H, 0, H_H, H_D, H_He, N_H, 0, N_D, 0, 0, 0, 0, T)
    expected = (N_D / N_H) * (PHI_H - c.b_H_D(T) * (1 / H_D - 1 / H_H))
    assert got == pytest.approx(expected)


def test_Z90_mod_close_to_Z90_without_helium_for_trace_D():
    full = Phi_D_Z90(PHI_H, 0, H_H, H_D, H_He, N_H, 0, N_D, 0, 0, 0, 0, T)
    mod = Phi_D_Z90_mod(PHI_H, H_D, N_D, N_H, T)
    # Z90_mod drops the 1/H_H diffusion term, so it should be the smaller of the two
    assert mod <= full
    assert mod == pytest.approx(full, rel=0.2)


def test_gu_chen_deuterium_flux_is_non_negative():
    assert Phi_D_GC23(PHI_H, PHI_He, H_H, H_D, H_He, N_H, N_He, N_D, T) > 0
    assert Phi_D_GC23(1.0, 0.0, H_H, H_D, H_He, N_H, N_He, N_D, T) == 0


# ---------------------------------------------------------------- generalised minor-species flux


def test_minor_species_zero_for_empty_atmosphere():
    assert Phi_minor_species(PHI_H, PHI_He, H_H, H_He, H_D, [0] * 7, T, 2, 0, 1) == 0


def test_minor_species_zero_when_light_dominant_absent():
    N = [0, N_He, N_D, N_O, N_C, N_N, N_S]
    assert Phi_minor_species(PHI_H, PHI_He, H_H, H_He, H_D, N, T, 2, 0, 1) == 0


def test_minor_species_with_non_hydrogen_dominant_pair():
    # e.g. a desiccated atmosphere dominated by He (light) and O (heavy), with trace C
    N = [0, 1e39, 0, 1e39, 1e35, 0, 0]
    flux = Phi_minor_species(1e17, 1e15, H_He, H_O, H_C, N, T, 4, 1, 3)
    assert flux >= 0


# ---------------------------------------------------------------- binary diffusion lookup


@pytest.mark.parametrize(
    "pair, fn",
    [
        (("H", "D"), c.b_H_D),
        (("H", "He"), c.b_H_He),
        (("H", "O"), c.b_H_O),
        (("H", "C"), c.b_H_C),
        (("H", "N"), c.b_H_N),
        (("H", "S"), c.b_H_S),
        pytest.param(("He", "D"), c.b_He_D, marks=HE_D_LOOKUP_BUG),
        (("He", "O"), c.b_He_O),
        (("He", "C"), c.b_He_C),
        (("He", "N"), c.b_He_N),
        (("He", "S"), c.b_He_S),
    ],
)
def test_binary_diffusion_lookup_is_order_independent(pair, fn):
    a, b = pair
    assert get_binary_diffusion_coeff(a, b, T) == pytest.approx(fn(T))
    assert get_binary_diffusion_coeff(b, a, T) == pytest.approx(fn(T))


def test_binary_diffusion_lookup_falls_back_to_H_He():
    # Pairs without a tabulated coefficient silently use b_H_He; this pins that behaviour.
    assert get_binary_diffusion_coeff("O", "C", T) == pytest.approx(c.b_H_He(T))


# ---------------------------------------------------------------- other escape prescriptions


def test_radiation_recombination_flux_positive_and_scales_with_sqrt_flux():
    Rp, Mp, Teq, t = 2 * c.Re, 5 * c.Me, 800.0, 1e7 / c.s2yr
    low = phi_RR(Rp, Mp, Teq, t, 1.0, t_pms=0)
    high = phi_RR(Rp, Mp, Teq, t, 4.0, t_pms=0)
    assert low > 0
    assert high / low == pytest.approx(2.0)


def test_core_powered_mass_loss_is_positive_and_bondi_limited():
    Mp, Teq, rho, R_env = 5 * c.Me, 800.0, 1.0, c.Re
    Rp = 2 * c.Re
    phi = phiE_CP(Teq, Mp, rho, 0.1, c.G * Mp / Rp, 4 * np.pi * Rp**2, c.mu_solar, R_env)
    c_s = np.sqrt(c.kb * Teq / c.mu_H)
    R_rcb = c.Re * (Mp / c.Me) ** 0.25 + R_env
    phi_B = c_s * rho * np.exp(-c.G * Mp / (c_s**2 * R_rcb))
    assert 0 < phi <= phi_B


@pytest.mark.parametrize("F_xuv", [1e-3, 1.0, 1e3])
def test_johnson_reduction_is_between_zero_and_one(F_xuv):
    Rp, Mp = 2 * c.Re, 5 * c.Me
    f_r = Johnson_reduction(0.15, F_xuv, Rp, c.G * Mp / Rp)
    assert 0 < f_r <= 1


def test_johnson_reduction_saturates_at_one_for_weak_heating():
    Rp, Mp = 2 * c.Re, 5 * c.Me
    assert Johnson_reduction(0.15, 1e-12, Rp, c.G * Mp / Rp) == 1
