# isofate.mcmc: posterior evolutionary histories from IsoFATE–Atmodeller

`isofate.mcmc` fits the end state of coupled IsoFATE–Atmodeller evolutionary tracks (`isofate_coupler.isocalc`)
to whatever observables exist for one planet, with emcee. Everything is set in one TOML config. The result is a
posterior over the free inputs (initial atmospheric mass fraction, mass, age, heating efficiency, XUV history,
composition, ...) and, by re-running posterior samples, a posterior over evolutionary histories.

It replaces `isofate/isofate_MCMC.py` (left untouched). The bugs found in that script are fixed here; the list is
at the end of section 10.

Contents
1. Quickstart
2. Config reference
3. The likelihood
4. Mass, age and temperature: inputs, so priors
5. Where is the posterior? `scan`, the bisection initializer, and solve mode for thin ridges
6. Choosing n_steps and n_atmodeller: the resolution study
7. Comparing runs
8. Outputs and file formats
9. Performance
10. Caveats and known upstream issues
11. Python use, tests, layout

## 1. Quickstart

Use the repo's virtual environment. All commands are `python -m isofate.mcmc COMMAND`, and `--help` works on
every command. The results root is `/Users/collin/Documents/Chicago/isofate_nick/mcmc_results/`. Keep working
configs in its `configs/` folder with `output_root = ".."`, so runs land in `mcmc_results/<name>/`.

```bash
cd /Users/collin/Documents/Chicago/isofate_nick/IsoFATE
PY=.venv/bin/python
C=/Users/collin/Documents/Chicago/isofate_nick/mcmc_results/configs/lhs1140b.toml
```

**0. A config.** `lhs1140b.toml` is the physics of `isofate_MCMC.py` with its bugs fixed (legacy composition,
step-function XUV with F_final = 0.17 W/m² after t_sat = 3.128 Gyr), fitted to the outflow composition and the
H+He mass-loss rate of Cherubim et al. 2026 and the radius and mass of Cadieux et al. 2024:

| observable | data | space |
|---|---|---|
| `outflow_H_fraction`, H/(H+He) of the escaping gas (p-winds' h_fraction) | 0.00115 (−0.052/+0.085 %), σ = [0.261, 0.240] dex | log10 |
| `Mdot_HHe`, the H+He mass-loss rate | 2.03e5 kg/s, σ = [0.137, 0.121] dex | log10 |
| `Rp` | 1.730 ± 0.025 R⊕ (`sigma_model = 0`) | linear |
| `Mp` (an input: a prior on Mp) | 5.60 ± 0.19 M⊕ | linear |

The free parameters are `log_f_atm` U[−5, −1.3], `age` U[3.128e9, 1e10] yr (from t_sat; the paper's minimum
age is > 3.12 Gyr; section 4), `Mp` U[4.5, 6.7] M⊕ and `eps` U[0.01, 1]. The outflow datum pins log_f_atm to a
ridge about 0.003 dex thin, so the config samples in **solve mode** (section 5): `[solve] param = "log_f_atm",
observable = "outflow_H_fraction"`.
The sampler then moves in u = log10 of the outflow H fraction in place of log_f_atm, and each evaluation solves
for the log_f_atm that gives u, with the exact Jacobian in the density. The walkers start with `init = "solve"`,
and `reset_stuck_at = 0`. `lhs1140b_static_phi.toml` next to it is identical except `dynamic_phi = false` and
`name = "lhs1140b_static_phi"`, for a side-by-side comparison (section 7).

`configs/toi406b.toml` is a generic example with only a radius and a mass (TOI-406 b, the setup of
`isofate_MCMC_v2.py`), without solve mode. `--set 'solve.param="log_f_atm"'` would sample it in u = Rp, but there
82 % of the posterior are stripped planets at the bare-core radius, a plateau that solve mode leaves out (section
5, "TOI-406 b"); its working copy is `mcmc_results/configs/toi406b.toml`.

For a new planet, start from the commented template, which lists every option. Written into a `configs/` folder,
it gets `output_root = ".."` (so the runs go next to `configs/`, not into the working directory):

```bash
$PY -m isofate.mcmc template /Users/collin/Documents/Chicago/isofate_nick/mcmc_results/configs/my_planet.toml
```

**1. check: one evaluation, in this process.** It prints the resolved star and planet (L, d, Fp, Teq, F0), the
parameter table, each active observable's model value against the data with its log-likelihood term, every model
output, the warnings and an MCMC cost estimate. The model runs twice so that the quoted time excludes
compilation. `--theta name=value` evaluates elsewhere (it is the same as `--set params.name.value=...`).
In solve mode it then solves once from that point, u at the datum (or `--theta u_<observable>=value`), and prints
every model run of the root search, the solution, ln|J| and the density. `--bisect` runs one bisection and prints
every round (section 5).

```bash
$PY -m isofate.mcmc check $C
$PY -m isofate.mcmc check $C --theta log_f_atm=-2.8 --theta age=5e9
$PY -m isofate.mcmc check $C --theta u_outflow_H_fraction=-3.2   # solve for another u
$PY -m isofate.mcmc check $C --bisect                         # does the bisection find the island from here?
```

At the centre of `lhs1140b.toml` (log_f_atm −3.0175, Mp 5.6, 7.6 Gyr, eps 0.15, on the He-dominated island) this
gave status 0, outflow H fraction 1.079e-3 (term −0.006), Rp = 1.6079 R⊕ (term −11.9), Mdot_HHe = 4.01e4 kg/s
(term −13.2), 0.155 s per evaluation after a 1.6 s first call. The solve for u = log10 0.00115 converged at
log_f_atm* = −3.017355 in 15 model runs plus 2 for the central-difference derivative (the forward default since
2026-10-29 needs 1; dT/dlog_f_atm = 186, ln|J| = −5.227, 3.6 s);
with `dynamic_phi = false` at −3.017845 (dT/dlog_f_atm = 190). The whole command took 7.9 s and 0.44 GB. With
`--bisect` the bisection converged at −3.017313 in 11 rounds (13 evaluations).

**2. scan: where does the posterior live?** One or two free parameters on a grid, the others at their values
(section 5):

```bash
$PY -m isofate.mcmc scan $C --x log_f_atm=-3.1:-2.9:41 --theta age=7.6e9 --workers 2
$PY -m isofate.mcmc scan $C --x log_f_atm=-3.25:-2.85:41 --y age=3.5e9:1e10:14 --workers 6
```

**3. resolution: choose `model.n_steps` and `model.n_atmodeller`** (section 6). Look at the plan first, then run
the quick grid. The test points come from where the walkers will start: init-style solves in solve mode (the
study then re-solves at every cell, `--solve`), the bisection with `init = "bisect"`.

```bash
$PY -m isofate.mcmc resolution $C --grid quick --dry-run
$PY -m isofate.mcmc resolution $C --grid quick --points 4 --workers 6
```

It prints `table.txt` and ends with two recommendations, each as options like `--set model.n_steps=30000 --set
model.n_atmodeller=100` plus a copy-paste run command: the conservative one at fixed θ, and one by the
equivalent parameter shift (section 6).

**4. run: the MCMC.**

```bash
$PY -m isofate.mcmc run $C --workers 6
$PY -m isofate.mcmc run $C --workers 6 --max-steps 2000 --set model.n_steps=30000 --set model.n_atmodeller=100
$PY -m isofate.mcmc run $C --bisect-observable Rp            # = --set sampler.bisect_observable=Rp
```

**Without solve mode the walkers do not mix on lhs1140b** (acceptance below 1 %, every walker stuck in 400 steps;
section 5, "Measured on LHS 1140 b"). The configs therefore use solve mode, in which the same walkers move in
(u, Mp, age, eps), where the ridge is a smooth, wide coordinate: both LHS 1140 b configs mixed (acceptance
0.19–0.20, τ ≈ 19 steps, N/τ = 68–83 after 1300–1600 steps, about 2 h on 3 workers each at 3e4/100; use 1e5/100
for a final posterior, section 5). Solve mode is for thin ridges: a planet whose radius is the datum and whose
bare core also fits it (e.g. TOI-406 b) needs plain sampling (section 10).

If it prints `reducing workers: 2 of 6` on a Mac with a browser open, the memory check is the cause. Add
`--set parallel.check_available=false`, which budgets `max_memory_gb` alone (section 9). Both mcmc_results
configs already set `check_available = false` and `max_memory_gb = 6`.

The run goes to `mcmc_results/lhs1140b/mcmc_<YYYYmmdd-HHMMSS>/`; an existing run directory is never overwritten.
It starts a pool of spawned workers (each compiles the model once, about 5 s with the compile caches), evaluates
the centre once as a preflight (a bug or a model failure there stops the run with a clear message), and
initializes the walkers (`sampler.init`: solved draws here, otherwise e.g. the bisection or the best of a
Latin-hypercube prescan).
It then samples, printing a progress line every `print_every` steps (also appended to `run.log`), writing
`chain.npz` atomically every `checkpoint_every` steps, and estimating the autocorrelation time every
`check_every` steps. It stops at `max_steps` or once converged (N ≥ `min_steps`, N > 50 τ, and τ stable within
max(1 %, its statistical noise), with every walker in the estimate, at 2 checks in a row; section 2), then writes
`summary.json`, re-runs 200 posterior samples as evolutionary tracks, and draws the figures.

**Ctrl-C** saves the chain, writes `summary.json` (state "interrupted") and prints the resume command; so do
`kill` (SIGTERM, also to the whole process group) and closing the terminal (SIGHUP; under `nohup` the run goes on).
A worker that dies (killed by the OS for memory, a crash in compiled code), during an evaluation or between two
steps, stops the run with a message within 2 s instead of a silent hang, and the chain up to the last step is
saved. While a session runs it holds `RUN_DIR/.lock`, so a second `run --resume` of the same directory is
refused. Resuming gives the same chain, bit for bit, as an uninterrupted run (also in solve mode):

```bash
$PY -m isofate.mcmc run --resume RUN_DIR                    # up to sampler.max_steps
$PY -m isofate.mcmc run --resume RUN_DIR --max-steps 10000  # or further
```

**5. Look at it.** `summary` works during a run too: when `summary.json` describes fewer steps than the last
checkpoint (e.g. a resumed run is going), it recomputes it from the chain:

```bash
$PY -m isofate.mcmc summary RUN_DIR             # --update recomputes it, --json prints the raw file
$PY -m isofate.mcmc plot RUN_DIR                # corner_params (and corner_sampled), corner_observables, traces, ...
$PY -m isofate.mcmc tracks RUN_DIR --n 100      # re-run posterior tracks (e.g. after a resume), redraws figures
$PY -m isofate.mcmc plot RUN_DIR/../resolution_<stamp>    # the resolution figures (or a scan_<stamp>)
$PY -m isofate.mcmc compare RUN_A RUN_B --labels dynamic_phi,static_phi                    # section 7
```

**6. synth: injection–recovery and forecasts.** It writes a copy of the config whose active observables have the
model prediction at θ as their value, keeping sigma (optionally scaled). The params' values in the copy are θ,
its `name` is the file stem (so its runs get their own folder) and `output_root` is made absolute. Running the
MCMC on it tests whether the data can recover θ; `--sigma-scale 0.5` asks what twice the precision would give.

```bash
$PY -m isofate.mcmc synth $C --theta log_f_atm=-2.7 --theta age=6e9 --sigma-scale 0.5 \
    --out /Users/collin/Documents/Chicago/isofate_nick/mcmc_results/configs/lhs1140b_inj.toml
$PY -m isofate.mcmc check /Users/collin/Documents/Chicago/isofate_nick/mcmc_results/configs/lhs1140b_inj.toml
```

`check` on such a copy gives every term exactly 0 (the model is deterministic). An observable the model leaves
undefined at θ (e.g. a VMR of a stripped atmosphere, or the outflow H fraction without any wind) stops `synth`
with a message.

An earlier injection–recovery test, with the previous radius-and-mass setup, is in
`mcmc_results/configs/lhs1140b_injection.toml` and `mcmc_results/lhs1140b_injection/`. Its truth is log_f_atm
−2.8, age 5 Gyr, Mp 5.6. The data are Rp and Mp, H/He (log10, 0.3 dex), C/O (±10 %), Mdot (log10, 0.5 dex) and
Φ_He/Φ_H < 0 ± 0.01. After 200 steps (not converged, N/τ = 28) the posterior is:
- log_f_atm = −2.781 (+0.051/−0.060): the truth lies at the 37th percentile, twice as tight as with Rp and Mp
  alone;
- Mp = 5.63 ± 0.18 (45th percentile);
- age = 6.7 (+2.1/−2.3) Gyr (25th percentile): almost the whole prior, so these data barely constrain the age.

**Testing the tool without the model.** `--set model.backend="toy"` swaps isocalc for a fast analytic stand-in
with the same inputs and outputs (`toy.py`; not physics: e.g. its outflow stays H-rich, so a bisection on the
outflow H fraction finds no root at the LHS 1140 b centre, and it ignores dynamic_phi). Every command works with
it.

## 2. Config reference

A config is TOML with the tables `[star]`, `[planet]`, `[model]`, `[params.NAME]`, `[observables.NAME]`,
`[solve]`, `[sampler]`, `[parallel]` and `[resolution]`, plus top-level `name`, `description` and `output_root`.
`configs/template.toml` lists every option with its default and unit; `python -m isofate.mcmc template` prints it.
Unknown keys raise an error that suggests the closest valid name (`unknown key 'n_step' in [model]; did you mean
'n_steps'?`).

- **Overrides.** Every command that reads a config takes `--set dotted.key=value`, repeatable. Values are TOML
  (`--set model.n_steps=3e4`, `--set params.age.bounds=[3e9,1e10]`, `--set 'model.mechanism="XUV+CPML"'`); text
  that is not valid TOML is taken as a string (`--set model.backend=toy`). A run records them in `config.json`
  and at the top of its `config.toml` copy, written so they parse back.
- **Units.** Values are written in the units below and converted to SI internally. Parameters and observables keep
  the units of their tables (Mp in M⊕, age in yr, Mdot in kg/s).
- **name** (default: the file stem) is used in paths. **output_root** is relative to the config file's directory;
  without it, results go to `./mcmc_results` in the working directory, and a warning appears if that lies inside
  the IsoFATE repository. Compile caches go to `<output_root>/.cache/` (about 1.5 MB).

`[star]`: `mass` [M☉] (required), `radius` [R☉], `teff` [K], `luminosity` [L☉] (default 4πR²σTeff⁴ from radius
and teff). `[planet]`: `period` [days], `semi_major_axis` [AU] (default from Kepler's law; a warning if both are
given and differ by more than 1 %), `Fp` [W/m²] (default L/4πd²).

`[model]` (defaults): `n_steps = 100000` (at most 1e6 unless `allow_large_n_steps = true`), `n_atmodeller = 100`
(steps between Atmodeller calls; 0 switches the coupling off, with a warning), `mechanism = "XUV"` ("XUV",
"CPML", "XUV+CPML", "phi kill"), `flux_model = "power law"` ("phoenix", "Johnstone"), `stellar_type = "M1"`,
`activity = "medium"`, `step_fn = false`, `RR = true`, `rad_evol = true`, `thermal = true`, `dynamic_phi = true`
(fractionate the two most abundant species; false: always H and He, as the old script), `t0 = 1e6` [yr],
`rho_rcb = 1.0`, `melt_fraction_override = false` (or a value in (0, 1]; 0.0 is rejected because isocalc treats
it as false), `mantle_iron = false` (or `{type = "static"|"dynamic", Fe_mass_fraction = 0.1}`),
`composition = "ratios"` (or "legacy"), `backend = "isocalc"` (or "toy").

**Composition.** `"ratios"` sets number ratios to H: protosolar He/H, solar D/H, and O, C, N, S protosolar ×
Z = 10^log_metallicity, with extra factors 10^log_OtoH_enhancement on O and 10^log_CtoH_enhancement on C. Then
N_H = M_atm/(μ_H + Σ r_X μ_X), N_X = r_X N_H and μ = M_atm/ΣN (the mean particle mass isocalc expects). It is
linear in its inputs and keeps the requested ratios exactly. `"legacy"` reproduces `isofate_MCMC.py:139-149`
operation for operation (mass ratios, enhanced O taken out of the H budget), for continuity with earlier runs.
It accepts only `log_OtoH_enhancement`, and it is nonlinear (section 10).

### Parameters: `[params.NAME]`

Keys: `vary` (default false), `value` (the fixed value; also the centre of `init = "ball"` and the point of
`check`; default for free parameters: the bounds midpoint, or the log-midpoint for loguniform), `prior`
(`"uniform"`, `"loguniform"` or `"normal"`), `bounds = [lo, hi]` (required for uniform and loguniform, an optional
truncation for normal), `mu` and `sigma` (normal), `init_scale` (spread of `init = "ball"`, default 1 % of the
box). A bare number, `NAME = x`, fixes a parameter. Bounds must lie inside the valid range below. Free
parameters, in this order, form θ.

<!-- BEGIN params table: generated by isofate.mcmc.config.markdown_tables(); tests/test_mcmc_cli.py checks it -->
| name | unit | default | valid range | meaning |
|---|---|---|---|---|
| `log_f_atm` | dex | -2 | < 0 | initial atmospheric mass fraction, f_atm = 10**x |
| `Mp` | M_earth | required | [1, 21] with Atmodeller and no melt_fraction_override (melt grid), else > 0 | planet mass (required) |
| `age` | yr | 5e+09 | (t0, 1.4e+10] | planet age = end of the run |
| `log_metallicity` | dex | 0 | any | scales O, C, N, S relative to H (ratios composition only) |
| `log_OtoH_enhancement` | dex | 0 | any | extra O/H |
| `log_CtoH_enhancement` | dex | 0 | any | extra C/H (ratios composition only) |
| `eps` | - | 0.15 | (0, 1] | heating efficiency |
| `log_F0_ratio` | dex | -3 | any | initial XUV flux F0 = Fp*10**x |
| `F_final` | W/m2 | 0 | >= 0 | XUV flux after t_sat when step_fn = true |
| `t_sat` | yr | 5e+08 | > 0 | XUV saturation time |
| `t_pms` | yr | 0 | >= 0 | end of the pre-main-sequence phase |
| `pms_factor` | - | 100 | > 0 | XUV flux boost before t_pms |
| `beta` | - | -1.23 | any | XUV power-law index (step_fn = false) |
| `albedo` | - | 0 | [0, 1) | Bond albedo, T = EqTemp(Fp, A) |
| `Rp_fixed` | R_earth | unset | > 0 | constant radius (isocalc Rp_override); needs rad_evol = false, which is then set. Unset: off |
<!-- END params table -->

**`eps` and `F_final`.** With `step_fn = true` the present-day energy-limited escape scales as eps·F_final, so the
two are degenerate today; they differ in the history (F_final only acts after t_sat, eps at all times, and the
radiation-recombination limited branch does not depend on eps). The LHS 1140 b config frees eps and keeps
F_final = 0.17 W/m². On the island at eps = 0.15 the model gives Mdot_HHe = 4.0e4 kg/s, 0.70 dex below the
observed 2.03e5 kg/s; if the rate scaled linearly with eps, matching it would need eps ≈ 0.75, but eps also
changes the whole escape history, and with it where the island lies.

**`Rp_fixed`: an optional constant radius** (unset by default, i.e. off). Setting it, fixed or free, passes
`Rp_override = Rp_fixed·R⊕` to isocalc and switches `model.rad_evol` to false, with a warning, because isocalc
applies the override only without radius evolution (`isofate_coupler.py:148`). The radius then stays at Rp_fixed
for the whole run (from the second time step on, also after the atmosphere is lost; the first step uses the core
radius, `escape_kernel.py:140`) instead of the Lopez & Fortney radius, and the `Rp` observable acts as a prior on
Rp_fixed. It decouples the escape history from the radius
model:

```toml
[params.Rp_fixed]             # [R_earth]
vary = true
bounds = [1.5, 2.0]
```

The LHS 1140 b configs leave it unset (radius from Lopez & Fortney, decision of 2026-10-28).

### Observables: `[observables.NAME]`

Keys: `use` (default false; only active observables enter the likelihood), `value` (always in the unit below,
even for log10 space), `sigma` (a number, or `[lo, hi]` for asymmetric errors; in the unit of value for linear
space and in dex for log10 space), `space` (`"linear"` or `"log10"`), `kind` (`"detection"`, `"upper"` or
`"lower"`), `floor` (override of the log10 floor below) and `sigma_model` (an extra model error in the same space,
added in quadrature, default 0). An inactive observable may omit value and sigma. Observables that need
Atmodeller raise an error with `n_atmodeller = 0`.

Every observable is the value at the last step of the run. Escape rates use A = 4πRp[-1]² and the per-atom
masses of `isofate.constants`. Molecules come only from `sol['atmodeller_final']`, the Atmodeller solve at the
last step; the `n_*_a` arrays lag by up to n_atmodeller − 1 steps. Atmospheric He is not in `atmodeller_final`,
so it comes from conservation: (N_He + N_He_int)/N_A − He_mantle. Composition observables are NaN once the
atmosphere is exhausted.

<!-- BEGIN observables table: generated by isofate.mcmc.config.markdown_tables(); tests/test_mcmc_cli.py checks it -->
| name | unit | definition | log10 floor | needs Atmodeller |
|---|---|---|---|---|
| `Mp` | M_earth | planet mass (input: acts as a prior on Mp) | input | no |
| `age` | yr | age (input: acts as a prior on age) | input | no |
| `Teq` | K | equilibrium temperature EqTemp(Fp, albedo), full redistribution (input) | input | no |
| `Rp` | R_earth | final radius Rp[-1] (Lopez & Fortney; the core radius when stripped) | - | no |
| `fatm` | - | final atmospheric mass fraction fatm[-1] (0 when stripped) | 1e-12 | no |
| `Mdot` | kg/s | total escape rate, sum over X of Phi_X mu_X 4 pi Rp^2 | 0.001 (no ceiling) | no |
| `Mdot_H` | kg/s | H escape rate Phi_H mu_H 4 pi Rp^2 | 0.001 (no ceiling) | no |
| `Mdot_He` | kg/s | He escape rate Phi_He mu_He 4 pi Rp^2 (0 when the flux is subcritical) | 0.001 (no ceiling) | no |
| `Mdot_HHe` | kg/s | H+He escape rate (Phi_H mu_H + Phi_He mu_He) 4 pi Rp^2, the mass-loss rate of p-winds | 0.001 (no ceiling) | no |
| `outflow_H_fraction` | - | H number fraction of the H+He outflow, Phi_H/(Phi_H + Phi_He) (p-winds h_fraction); 1 when He does not escape, NaN without an H/He wind | 1e-15 | no |
| `outflow_He_fraction` | - | He number fraction of the outflow, Phi_He/(Phi_H + Phi_He); NaN without an H/He wind | 1e-15 | no |
| `escape_He_H` | - | legacy: Phi_He/Phi_H (0/0: NaN); prefer outflow_He_fraction | 1e-12 | no |
| `escape_H_He` | - | legacy: Phi_H/Phi_He, the old script's flux_ratio (inf when He does not escape); prefer outflow_H_fraction | 1e-12 | no |
| `H_He` | - | (N_H + N_D)/N_He in the atmosphere [atoms] | 1e-12 | no |
| `C_O` | - | atmospheric C/O: Atmodeller gas totals of the final call, else kernel N_C/N_O | 1e-12 | no |
| `D_H` | - | atmospheric N_D/N_H from the escape kernel (Atmodeller lumps D into H) | 1e-12 | no |
| `mmw` | g/mol | mean molecular weight of the 10 Atmodeller gases plus He | - | yes |
| `VMR_H2` | - | volume mixing ratio of H2 at the surface (Atmodeller) | 1e-15 | yes |
| `VMR_He` | - | volume mixing ratio of He at the surface (Atmodeller), He from conservation | 1e-15 | yes |
| `VMR_H2O` | - | volume mixing ratio of H2O at the surface (Atmodeller) | 1e-15 | yes |
| `VMR_CO` | - | volume mixing ratio of CO at the surface (Atmodeller) | 1e-15 | yes |
| `VMR_CO2` | - | volume mixing ratio of CO2 at the surface (Atmodeller) | 1e-15 | yes |
| `VMR_CH4` | - | volume mixing ratio of CH4 at the surface (Atmodeller) | 1e-15 | yes |
| `VMR_N2` | - | volume mixing ratio of N2 at the surface (Atmodeller) | 1e-15 | yes |
| `VMR_O2` | - | volume mixing ratio of O2 at the surface (Atmodeller) | 1e-15 | yes |
| `VMR_S2` | - | volume mixing ratio of S2 at the surface (Atmodeller) | 1e-15 | yes |
| `VMR_SO2` | - | volume mixing ratio of SO2 at the surface (Atmodeller) | 1e-15 | yes |
| `VMR_H2SO4` | - | volume mixing ratio of H2SO4 at the surface (Atmodeller) | 1e-15 | yes |
| `T_surf` | K | surface temperature at the base of the adiabat (not directly observable) | - | yes |
<!-- END observables table -->

**The outflow observables.** `outflow_H_fraction` x = Φ_H/(Φ_H + Φ_He) at the last step is p-winds' `h_fraction`,
which Cherubim et al. 2026 quote as 0.115 (+0.085/−0.052) % for LHS 1140 b; `outflow_He_fraction` is 1 − x.
`Mdot_HHe` = (Φ_H μ_H + Φ_He μ_He)·4πRp² is the mass-loss rate of an H+He Parker wind as p-winds defines it
(`Mdot` also counts D, O, C, N and S). The division needs no floor:
- an H-only wind (He not escaping because the flux is subcritical) gives exactly x = 1. That is a finite, honest
  prediction: in log10 space it lies 2.94 dex from the LHS 1140 b datum, a term of −75;
- a He-only wind gives x = 0, which the log10 floor turns into 1e-15;
- no H/He wind (stripped, or nothing escaping) gives NaN, which rules out a detection (status 3).

`escape_H_He` (the old script's `flux_ratio`, Φ_H/Φ_He) and `escape_He_H` are legacy. Their zero denominators make
them ill-defined: Φ_He = 0 whenever the flux is subcritical, which covers most of the LHS 1140 b prior, so
escape_H_He is +inf there, and in log10 space the ceiling 1e12 turns that into a huge but arbitrary penalty. For
x ≪ 1 escape_H_He equals x to 0.1 %; use the outflow fractions instead.

Every model output is also stored in the blobs (section 8), active or not, plus five extras: `exhausted` (1 if
fatm[-1] ≤ 0), `t_lost` [yr] (first step without atmosphere, NaN if never), `Matm` [kg], `subcritical` (1 if
phi < phic at the end; NaN when exhausted) and `n_atmod_calls`.

### `[solve]`, `[sampler]`, `[parallel]`, `[resolution]`

The defaults are listed below; every key is explained in `template.toml`.
- `[solve]` (section 5): `param = ""` (off; e.g. `"log_f_atm"`), `observable = "Rp"`, `range_sigma = 6`,
  `tol = 0.01` (σ), `max_evals = 40`, `warm_start = true`, `refit_steps = [25, 50, 100, 200, 400]` (`[]`: never
  refit), `derivative = "forward"` (or `"central"`).
- `[sampler]`: `nwalkers = "auto"` (or an even number ≥ 2 ndim), `max_steps = 5000`, `checkpoint_every = 50`
  (and at most once a minute), `print_every = 10`, `moves = "de"`, `seed = 12345`, `init = "prescan"` (`"solve"`
  in solve mode; or `"bisect"`, `"prior"`, `"ball"`), `prescan_factor = 8`, `bisect_param = "log_f_atm"`,
  `bisect_observable = "Rp"`, `bisect_draws = "auto"` (3 × nwalkers per batch), `bisect_iters = 22`,
  `bisect_tol = 0.1`, `bisect_fallback = true`, `reset_stuck_at = 0`, `stop_on_convergence = true`,
  `check_every = 100`, `n_tau = 50`, `tau_rtol = 0.01`, `tau_checks = 2`, `min_steps = 200`, `burn_factor = 3`,
  `tracks = 200`,
  `track_points = 300`, `max_unexpected_errors = 10`.
- `[parallel]`: `workers = 6`, `max_memory_gb = 8.0`, `check_available = true`, `cache = true`, `warmup = true`.
- `[resolution]`: `n_steps = [1e4, 3e4, 1e5, 3e5, 1e6]`, `n_atmodeller = [1000, 300, 100, 30, 10, 3, 1]`,
  `points = 6`, `source = "auto"` (solve in solve mode; bisect when `init = "bisect"`; else prescan),
  `tol = 0.1`, `dlogl_tol = 0.5`, `perturb = 1e-6`, `shift_tol = 1e-3`, `max_eval_seconds = 300`.

**Moves.**
- `"de"` (default) is DE 0.7 + DE(γ0 = 1) 0.1 + DESnooker 0.2. γ0 = 1 lets walkers jump between modes. In tests on
  bimodal and strongly correlated targets this mix beat the stretch move: tau 19 against 53, and it kept 50/50 mode
  occupancy where stretch never moved walkers between modes.
- `"de_only"` is DE 0.9 + DE(γ0 = 1) 0.1.
- `"stretch"` is emcee's default move.

No move finds a mode with no walkers in it, which is why the initialization matters.

**Initialization.**
- `"prescan"` (default without solve mode) evaluates `prescan_factor × nwalkers` Latin-hypercube prior points
  (saved in `prescan.npz`) and starts the walkers from a random draw, without replacement, of the best
  `2 × nwalkers`.
- `"solve"` (the default in solve mode; section 5) draws the other parameters from the prior and u from the data
  distribution of the solve observable, solves each draw, and starts from nwalkers of the best `2 × nwalkers`.
- `"bisect"` moves `bisect_param` of prior draws until `bisect_observable` matches its data, and starts the walkers
  there, without jitter (section 5). It is the way onto a thin ridge such as the LHS 1140 b outflow island, which
  a prescan almost never hits. Starting there is not sampling there: on that island the walkers barely mix
  (section 5, "Measured on LHS 1140 b"); solve mode does. In solve mode `"bisect"` is redundant and becomes
  `"solve"`, with a note.
- `"prior"` uses random prior draws with a finite posterior.
- `"ball"` starts at the params' values plus `init_scale` noise.

In solve mode `"prescan"`, `"prior"` and `"ball"` work in the sampler's coordinates, with u uniform on its support
(`"ball"` centred on the datum).

**Jitter of prescan starts (and of reset copies).** Each start is moved by 1e-3 of the prior box and re-evaluated.
If its lp falls by more than 5 below the unjittered point's, the jitter is halved and drawn again, at most 8 times;
a point that still loses more than 5 keeps its unjittered position if no other walker sits there (a copy takes
its best finite try). This keeps walkers on thin ridges, where a fixed jitter of 1e-3 of the box can cost tens in
lp. The log line says how many were halved or left unjittered. Bisection starts are never jittered: they are
distinct already.

**Stuck walkers.** A walker is *stuck* when its acceptance is below min(0.05, 0.2 × the median walker's), or it
accepted nothing. The threshold is relative: on some posteriors every walker accepts only 1–3 % of its proposals
and still mixes, slowly; an absolute 0.05 (the rule before 2026-10-28) called all of them stuck, left them out of τ
and made τ NaN, so such a run could never be declared converged. With `reset_stuck_at = k > 0`, at step k the stuck
walkers and those with lp < median − 20 are replaced by jittered copies of good ones. Steps 1–k are kept in
`burnin.npz` and sampling restarts on a fresh chain. `max_steps` counts from the start. The LHS 1140 b configs use
k = 0 (no reset).
- The progress lines report both kinds from step 50 on: `stuck 0/24, behind 2`. "Behind" walkers have lp more
  than 20 below the median, which acceptance alone does not catch. The summary lists them, with a warning,
  because they are part of the posterior samples.
- When the mean acceptance is below 0.05 and τ shows no convergence, the summary warns that the moves barely mix
  and the samples may still be close to the starting points, not a posterior (and points to solve mode). It says
  nothing about acceptance or stuck walkers before step 50.
- A reset that would replace more than half of the walkers replaces none and says so: copying the few others onto
  them would collapse the ensemble onto a few points. On the LHS 1140 b island without solve mode, 23–24 of 24
  walkers had accepted at most 4 of 100 proposals at step 100 (section 5).
- In the earlier radius-and-mass setup, 4 of 24 prescan-started walkers began on the flat stripped-core plateau
  (log f_atm < −4, Rp = the bare core, lp ≈ −24). In a 150-step test run, 3 of them left it within 18–56 steps.
  The fourth was still there at step 150, with an acceptance of 0.25 (not "stuck"), and made up 4 % of the
  posterior samples.

**Convergence.** Every `check_every` steps τ is estimated over the last 2/3 of the chain, using only walkers that
are not stuck and moved in that window. N is the number of steps per walker, and τ the integrated
autocorrelation time: a walker gives about one independent sample every τ steps, so a run holds about
walkers × N/τ independent samples. The run stops once
1. N ≥ `min_steps`;
2. N > `n_tau`·max τ: long compared with its own memory;
3. τ is stable at `tau_checks` (2) checks in a row: every walker is in the estimate, and every coordinate's τ
   changed since the previous check by less than max(`tau_rtol`, its statistical noise). A stuck walker, or one
   that did not move in the window, is left out of τ but stays in the posterior samples, and the noise of an
   estimate from a few walkers is large, so such a check never counts (`reset_stuck_at` is the remedy). The
   noise is the relative 1σ error of the estimate,
   √(2(2M + 1)/(L·W)) with emcee's window M = 5τ, L the steps used and W the walkers (Sokal 1997), printed as
   `tau = ... (+-x%)` in `run.log`; it is about 10–15 % for 24 walkers and N ≈ 1000–1500. On independent AR(1)
   walkers the scatter of τ is 0.75 times it; on the LHS 1140 b chain the spread of τ between random halves of
   the walkers was 0.6–1 times it.

In solve mode τ is that of the sampler's coordinates (u in place of the
solved parameter). The summary discards `burn_factor`·max τ steps (at most half the chain) and thins by τ_min/2.
Converged means N/τ > 50, which takes about 1000–2500 steps for τ ≈ 20–50. Short chains underestimate τ (300 steps
gave τ ≈ 20 where 3000 gave 50), so a "converged" flag after a few hundred steps deserves a second look at
`traces.png`.
- **What the rule cannot see** (simulations of 2026-10-29). τ̂ is biased low, by about 15–30 % at N = 50 τ̂, so at
  the stop N/τ is about 40–45 rather than 50. From N ≈ 1000 on, successive estimates share most of their window
  and the stability test almost always passes on a stationary chain, so the stop is then set mainly by N > n_tau·τ.
  A slow component that the 5τ window cannot see (rare switches between regions, a plateau) is not detected at
  all. So look at `traces.png` and compare the two halves of the chain before trusting a "converged".
- A resume redoes a convergence check (and a table refit) that the end of the previous session cut short, so a
  resumed run checks, and stops, where the uninterrupted one would have.
- **Why the noise term (2026-10-29).** Before it, τ had to change by less than `tau_rtol` = 1 % alone, which is
  far below the estimate's own noise: with 16–24 walkers τ moved by 2.6–7.8 % per check in the LHS 1140 b runs
  from step 1100 on (N/τ 52–83), and by 1.2–5.3 % in a 3000-step TOI-406 b run (N/τ 134–163). None of those runs
  stopped by itself. Replayed on the final LHS 1140 b chain, the rule above stops at step 1000 (N/τ = 55); it held
  the run at steps 200–400 and 800, where τ grew by 22–40 %, more than its noise. A chain stopped by hand
  (Ctrl-C saves it; then `tracks` and `plot`) is labelled "not converged by the stopping rule", with the missing
  condition.

## 3. The likelihood

For each active observable with model value m, data value v and effective sigma s:

| kind | term |
|---|---|
| detection | −½ ((m − v)/s)² |
| upper limit at v | log Φ((v − m)/s) |
| lower limit at v | log Φ((m − v)/s) |

Φ is the normal CDF (`scipy.special.log_ndtr`, accurate far into the tails). The log-likelihood is the sum over
active observables; the log-prior is added for the posterior.

- **Spaces.** In `"linear"` space m and v are compared as they are. In `"log10"` space the comparison is between
  log10 v and log10 of m clipped to [floor, ceiling]. A model value of 0 (no He escape, a stripped atmosphere)
  then becomes log10(floor) instead of −inf, i.e. far below any detection but consistent with an upper limit
  above the floor. The ceiling is 1/floor for the dimensionless observables, so the inf of a ratio x/0 becomes
  log10(1/floor). The escape rates [kg/s] have no ceiling. Sigma is in dex in log10 space.
- **Asymmetric errors** `sigma = [lo, hi]` give a split normal: hi when the model is above the data, lo below.
- **sigma_model** is added in quadrature on the side used: s = √(σ² + σ_model²). Use it to encode model
  systematics, e.g. 0.05–0.1 R⊕ on Rp, or 0.5–1 dex on surface VMRs compared with photospheric retrievals. The
  LHS 1140 b configs use 0 on Rp (observational error only), although the radius model is trusted less than the
  fractionation physics.
- **Undefined and infinite model values.** NaN means undefined, e.g. the composition of an exhausted atmosphere,
  or the outflow H fraction when nothing escapes. It rules out a detection and a lower limit and satisfies an
  upper limit (the species is absent). In linear space +inf rules out a detection and an upper limit and
  satisfies a lower limit. NaN never reaches emcee: any undefined total is −inf.
- **Status codes.** Every evaluation records a status in its blob:
  - 0 ok;
  - 1 outside the prior (model not run);
  - 2 expected model failure (ValueError, RuntimeError, JAX runtime errors, ...);
  - 3 likelihood −inf from the data;
  - 4 invalid inputs (legacy N_H ≤ 0);
  - 5 unexpected exception, i.e. a bug. The traceback goes to `RUN_DIR/worker_errors/<pid>.log`. The run
    aborts after `max_unexpected_errors` of them.

The progress lines count statuses over all proposals, rejected ones included.

## 4. Mass, age and temperature: inputs, so priors

Mass, age and equilibrium temperature are isocalc inputs, not outputs:
- `Mp` enters as the planet mass;
- `age` is the end time of the run;
- Teq = EqTemp(Fp, A) is set by the Bond albedo parameter `albedo`, assuming full redistribution.

Comparing them with data is therefore the same as a prior on the parameter. The observables `Mp`, `age` and `Teq`
are flagged as inputs and evaluated before isocalc runs, so a point they rule out costs nothing.

- **A measured mass:** make Mp free with wide uniform bounds and switch on the Mp observable (LHS 1140 b:
  Mp ∈ U[4.5, 6.7], observed 5.60 ± 0.19 M⊕). Alternatively, give `params.Mp` a normal prior; the two are
  equivalent. Mp must stay inside the Atmodeller melt grid, [1, 21] M⊕, unless `melt_fraction_override` is set.
  The bare-core radius (Mp/M⊕)^¼ R⊕ makes the radius very sensitive to Mp: ±0.19 M⊕ moves the final Rp of
  LHS 1140 b by ±0.017 R⊕, 0.7 σ.
- **Age:** give it bounds, or a normal prior, or an age observable (an asymmetric sigma or a lower limit works).
  It must be > t0 and ≤ 14 Gyr. The time step is (age − t0)/n_steps, so numerical error varies across the age
  prior. LHS 1140 b uses U[3.128e9, 1e10] yr: the paper's minimum age is > 3.12 Gyr (spin-down), and the prior
  starts at t_sat = 3.128 Gyr (the user's decision of 2026-10-28). With `step_fn = true` the XUV flux drops by
  3.5× at t_sat, so ages between 3.1 and 3.128 Gyr would still see the saturated flux today, which is unphysical
  for this quiet star (131-d rotation, L_X ≈ 6.5e-6 L_bol); that 28-Myr sliver made a spurious second mode with
  the best fit of the whole prior (section 5, "Measured on LHS 1140 b"). For any planet the config warns when
  t_sat lies inside the age prior with `step_fn = true`.
- **Temperature:** free `albedo` in [0, 1) with a Teq observable. A dayside or brightness temperature needs a
  redistribution model first.
- An input observable on a *fixed* parameter only adds a constant, and the config warns about it.

## 5. Where is the posterior? `scan`, the bisection initializer, and solve mode for thin ridges

On LHS 1140 b only tracks whose outflow is very H-poor and He-dominated reproduce the observed outflow H fraction.
At 7.6 Gyr and eps 0.15 that "island" is a sliver of log_f_atm: x rises from 1e-13 at −3.08 to 1.08e-3 at −3.0175
and 3.2e-3 at −3.015, i.e. d log10 x/d log_f_atm ≈ 190. Below it the atmosphere is lost (no wind: x undefined),
above it He stops escaping (x = 1). The band where x is within 1 σ of the datum is about 2 × 0.25/190 ≈ 0.0026
dex of log_f_atm wide, 0.07 % of the 3.7 dex prior: a prescan of 192 points expects at most 0.13 points in it,
and emcee does not find a mode without walkers. Three tools help: `scan` shows it, the bisection puts walkers on
it, and solve mode samples it.

### scan: the model on a grid

```bash
$PY -m isofate.mcmc scan CONFIG --x NAME=LO:HI:N [--y NAME=LO:HI:N] [--obs a,b,...] [--theta NAME=VALUE ...] \
    [--workers N] [--out DIR] [--set ...]
```

It evaluates a 1-D or 2-D grid of one or two free parameters (N evenly spaced values from LO to HI, inside the
prior), the other free parameters at their `value` (or `--theta`), through a worker pool, and writes
`<output_root>/<name>/scan_<stamp>/` (never overwritten) with `config.json`, `scan.npz` (θ, lp and every blob in
grid order) and `scan.png`:
- 1-D: the log-probability and one panel per observable against x, in its likelihood space, with the observed
  value ± σ as a band (a limit as a dashed line with an arrow). Undefined values are crosses along the bottom.
  It also prints the table.
- 2-D: a heat map of lp − max (floored at −50) and one per observable, with the observed value and ±1 σ as
  contours; undefined cells are hatched. Cells at an observable's log10 floor (a value of 0, e.g. the escape rate
  of a stripped planet) are grey, and the colour scale spans the other cells.

`--obs` draws any blob field, active or not (e.g. `fatm`, `exhausted`, `Mdot_HHe`); the default is the active
model observables. On lhs1140b, `--x log_f_atm=-3.1:-2.9:41 --theta age=7.6e9 --workers 2` took 9.2 s (5.5 s of it
warm-up) and reproduced the island scan of 2026-10-28 to 4–5 digits: x = 9.84e-14 at −3.08, 3.54e-4 at −3.02,
3.19e-3 at −3.015 (the best point, lp −26.7) and 1 from −2.925 up. Rp is 1.57–1.65 R⊕ across the scan and
Mdot_HHe 3.75–4.35e4 kg/s.

In solve mode the scan also checks the one-root assumption of section 5's solve (below): when an axis is the
solved parameter, it reports whether T(solve observable) is monotonic in it inside the u range, line by line
(`solve check: ... monotonic on every line of the scan`), or where it turns or enters the range more than once.
The lines are printed and stored in `scan.npz` (`solve_check`).

### The root finder (bisection initializer and solve mode)

Both the bisection initializer and solve mode (below) move one free parameter p inside its prior bounds, with the
other parameters fixed, until one observable k reaches a target in its likelihood space (linear, or log10 with the
floor):

    g(p) = T(model value of k at p) − target

The bisection's target is the data value of k, solve mode's the sampled u. One root finder (`search.Root`) serves
both:
1. **Cold start:** evaluate both prior bounds of p. If g changes sign between them the search is bracketed.
   - An undefined end (NaN, e.g. the outflow fraction of a stripped planet) takes the sign opposite to the
     defined end: the undefined region is treated as lying on the other side of the target. A NaN later inside
     the bracket takes the same sign. A NaN inside a bracket whose ends are both defined stops the search
     ("undefined inside the bracket"). An infinite g is defined (it has a sign).
   - No root: g has the same sign at both ends, or both ends are undefined. A model failure also ends it.
   - An end that already fits ends the search there.
2. **Warm start** (solve mode, from the warm-start table): evaluate the table's guess, then step away from it toward
   the root (from the sign of g and of dT/dp), doubling the step until g changes sign; at a prior bound without a
   sign change, try the other bound.
3. **Refine:** each step evaluates a point inside the bracket and keeps the part whose ends differ in sign: the
   Illinois (modified regula falsi) point while both ends are finite, else the midpoint (the geometric mean for a
   loguniform prior). When 3 Illinois steps in a row have not halved the bracket, the next step bisects.
4. **Stop** when |g| ≤ tol × σ (σ of k on the side of the target the model is on): `bisect_tol` = 0.1 for the
   bisection, `solve.tol` = 0.01 for solve mode. A bracket narrower than 1e-9 of the prior width without that is a
   jump of g across the target, not a root ("jump across the target"). The bisection stops after `bisect_iters`
   rounds, a solve after `solve.max_evals` model runs.

**Outflow fractions need an H/He wind.** For `outflow_H_fraction`, `outflow_He_fraction` and the legacy
`escape_H_He`, `escape_He_H`, the root finder counts a point as undefined when H and He carry less than half of
the escaping mass (Mdot_HHe < 0.5 Mdot). With `dynamic_phi = true` the stripping edge of LHS 1140 b has a
metal-dominated wind in which H and He are nearly gone: within 0.01 dex of log_f_atm their ratio swings through
any value (at Mp 6.006, 6.98 Gyr, eps 0.688: x = 1.9e-7, 1, 0.9996, 0.68, 0.0097, 4.9e-5 ... with Mdot_HHe 2e-10
to 1600 kg/s against Mdot 1.6e5). The bisection used to converge there (log_f_atm −2.384267, lp −1858.9, Mdot_HHe
4e-8 kg/s) instead of on the island beyond it; now those points join the stripped side and both the bisection and
the solve converge on the island root, −2.375334 (lp −15.34, Mdot_HHe 1.70e5 kg/s). That is not the H/He wind
the p-winds fractions describe, so it is excluded from the search only. The likelihood itself is unchanged
(such points keep their very low likelihood, about lp −1860 on LHS 1140 b, when the walkers move in log_f_atm
itself). The review of 2026-10-28 found such edge roots in 24 % of the dynamic-phi bisections of LHS 1140 b,
mostly at eps > 0.6, and every one of those draws also had an island root.

**One root per search.** The root finder needs only a sign change, not monotonicity, and finds one root (if g
crosses the target several times, which one depends on the steps). For the bisection that only decides where a
walker starts. For solve mode it is an assumption about the posterior (below).

### init = "bisect": starting the walkers where one observable fits

For a draw θ from the prior, the bisection moves `bisect_param` until `bisect_observable` matches its data, by the
root finder above, from both prior bounds. All draws run their steps together, one pool map per round. A draw
has converged when |g| ≤ `bisect_tol` × σ; its point must also have a finite lp.

**Worked example (LHS 1140 b, `check --bisect` at Mp 5.6, 7.6 Gyr, eps 0.15).** k is the outflow H fraction in
log10 space, so T(v) = log10 0.00115 = −2.939 and σ = 0.261 below, 0.240 above:

    round        log_f_atm     outflow_H_fraction g = T(m) - target   g/sigma  status
    lo                  -5                    nan              NaN         -  likelihood -inf
    hi                -1.3                      1           +2.939     +12.2  ok
    1                -3.15                    nan              NaN         -  likelihood -inf
    2               -2.225                      1           +2.939     +12.2  ok
    3              -2.6875                      1           +2.939     +12.2  ok
    4             -2.91875                      1           +2.939     +12.2  ok
    5            -3.034375             2.9078e-07           -3.597     -13.8  ok
    6            -2.970744               0.030161           +1.419     +5.91  ok
    ...
    10           -3.014919              0.0032955          +0.4572     +1.91  ok
    11           -3.017313               0.001171        +0.007858   +0.0327  ok
    converged: log_f_atm = -3.017313 (11 rounds, 13 evaluations, 3.0 s), lp = -25.1264

At −5 the planet is stripped by 7.6 Gyr: there is no H/He wind, x is undefined, and the NaN end takes the sign
opposite to the +2.94 of the H-rich end. That is the physical order: stripped, then the island, then H-rich. Rounds
1–5 bisect, because one end is undefined or x = 1 on a plateau; from round 6 both ends are finite and Illinois
steps take over. Plain bisection took 14 rounds here (0.1 σ in log10 x is 0.026 dex, 1.4e-4 dex of log_f_atm on a
slope of 190, so log2(3.7/1.4e-4) ≈ 15 halvings). The 22-round default allows a bracket 3.7/2²² ≈ 9e-7 dex
wide. The result agrees with the island scan of 2026-10-28 (x = 1.079e-3 at −3.0175, 3.185e-3 at −3.015).

**Why only an initializer.** The bisection fits one observable along one parameter and ignores everything else.
At the converged point above Rp is 1.608 R⊕ (term −11.9) and Mdot_HHe 4.0e4 kg/s (term −13.2): the island at
eps 0.15 does not fit the radius or the rate. The points are spread over the other parameters by the prior
draws, so the walkers start on the whole surface where k fits, which is a starting set, not a posterior. emcee
then samples with every observable. The starts are not jittered, because they are distinct already and the
ridge is narrower than any sensible jitter (1e-3 of the log_f_atm box, 0.0037 dex, moves log10 x by 0.7 dex).
On a ridge that thin, starting the walkers there is not enough: see solve mode.

**Generic, for any planet and observable.** Detections target their value. Upper and lower limits target the
limit value, so the walkers start on the boundary of the allowed region (the config warns).
- The defaults are `bisect_observable = "Rp"`, which nearly every planet has, and `bisect_param = "log_f_atm"`:
  the radius grows with the atmosphere, from the bare core up.
- On LHS 1140 b the config sets `bisect_observable = "outflow_H_fraction"`. Radius bisection
  (`check $C --bisect --bisect-observable Rp`) converges at log_f_atm = −2.731, where Rp = 1.7295 but the wind is
  pure H (x = 1, term −75, lp −84.8 against −25.1 on the island): it would start the walkers off the island.
- To choose another pair: any active, non-input observable and any free parameter that moves it across its data
  within the prior. Look at a `scan` of the parameter first, then try `check CONFIG --bisect --bisect-observable K
  --bisect-param P`. On the command line `run` and `resolution` take the same `--bisect-observable` and
  `--bisect-param` (equivalent to `--set sampler.bisect_observable=K`); they are an error when the run does not
  use the bisection (`sampler.init` other than "bisect", or a study whose source is not "bisect").

**Batches, choice and failures.**
- Each batch draws `bisect_draws` Latin-hypercube prior points (auto: 3 × nwalkers). Batches are added, at most
  3, until nwalkers draws converge. nwalkers of the 2 × nwalkers converged points with the highest lp are chosen
  at random, without replacement. The log line after each batch gives the totals so far: draws, bracketed,
  converged, evaluations and seconds. Every draw, its bracket, its final point, g, lp, blobs and the history of
  every evaluation go to `bisect.npz`.
- If too few converge, the message counts why, and suggests another observable or parameter, `check --bisect`
  and `scan`:
  - "no sign change: g > 0 at both ends in N draws" (the model is above the data across the whole prior, e.g.
    an observed radius below the bare-core radius), and the mirror case "g < 0 at both ends";
  - undefined at both ends (e.g. stripped at both ends), model failures, undefined inside the bracket, a jump
    across the data, not converged after `bisect_iters` rounds, converged but with lp = −inf.
- `bisect_fallback = true` (default) then falls back to `init = "prescan"` with a loud warning ("BISECTION FAILED:
  falling back ...") in `run.log` and in the summary's warnings. With false the run stops with that message.

**check --bisect** runs one bisection from the check point (the params' values, or `--theta`), in this process,
and prints every round as above: the parameter value, the model value, g and g/σ. The exit code is 1 when it
does not converge. A bad bisection target (a parameter that is not free, an inactive observable) is an error
before anything runs.

### Solve mode: sampling a thin ridge (`[solve]`)

When one precise datum pins one parameter to a thin, curved ridge, emcee's moves cannot follow it: they combine
walker positions linearly, and such a combination stays inside a curved surface 0.003 dex thick only when the
walkers involved are close together (on LHS 1140 b: acceptance below 1 %, every walker stuck, section 5 below).
Solve mode samples the observable instead of the parameter it pins down. A config chooses one free parameter P
(`solve.param`, e.g. `"log_f_atm"`) and one active, non-input observable O (`solve.observable`, default `"Rp"`):

```toml
[solve]
param = "log_f_atm"
observable = "outflow_H_fraction"   # LHS 1140 b; the default "Rp" suits a planet with a precise radius
```

- **The sampled coordinate** u_<O> (e.g. `u_outflow_H_fraction`, `u_Rp`) replaces P in θ: u is O's model value
  in O's likelihood space (log10 x for a log10-space observable, Rp in R⊕ for a linear one). `config.sampled_names`
  lists the sampler's coordinates; `cfg['free']` stays the physical free parameters.
- **Each evaluation solves**: for a proposal (u, φ), φ the other free parameters, the root finder above finds P*
  inside P's prior support with T(O_model(P*, φ)) = u, to `solve.tol` = 0.01 σ of O, and the model run at P*
  gives every observable, so the blobs belong to the physical point (P*, φ).
- **The density is the exact change of variables**

      p(u, φ) = prior_P(P*) · prior_φ(φ) · |∂P*/∂u| · L(model(P*, φ)),   ∂P*/∂u = 1/(∂T/∂P at P*)

  so the posterior over (P, φ) is unchanged. ∂T/∂P is a finite difference with h = 1e-5 of P's prior width.
  `solve.derivative = "forward"` (the default since 2026-10-29) costs 1 more model run, at P* + h (P* − h at the
  upper prior bound; where T is undefined at P* + h, a second run at P* − h). `"central"` costs 2, and both
  one-sided slopes must then have the same sign. A run whose `config.json` predates the option resumes with the
  scheme of its last session, and `run.log` says so: central for the runs of 2026-10-28, forward for the final LHS
  1140 b chain (central for its steps 1–250, forward from step 251 on). On 20 LHS 1140 b posterior points at
  1e5/100, forward and central ln|J| differ by 0.009 (median), at most 0.021, about the central difference's own
  numerical noise (section 10), and far below the resolution error. The slope must have the sign of the bracket; a
  derivative that is 0, undefined or (central) changes sign within ±h gives lp = −inf (status 3) with the reason in
  the error string. ln|∂P*/∂u| is the blob field `log_jac`, and lp = `log_prior` + `log_jac` + `log_like`, where
  `log_prior` is the prior of the physical point.
- **The u support** is [v − k σ_lo, v + k σ_hi], v the datum, σ its effective sigma on each side and k =
  `solve.range_sigma` (6). The truncation removes only points whose O term is below −18. It also keeps the sampler
  off plateaus where T cannot be inverted when they lie outside it: x = 1 (He not escaping) and the 1e-15 floor
  of the outflow H fraction. A plateau inside it, such as the bare-core radius of a stripped planet for u = Rp,
  is lost to solve mode (section 10; the config warns). The support is logged at the start of a run
  (`u_outflow_H_fraction in [-4.5053, -1.4993]` on LHS 1140 b). A limit uses [v − k σ, v] (upper) or
  [v, v + k σ] (lower), with a warning: solve mode is meant for detections.
- **No root** in P's prior for (u, φ): lp = −inf with status 1 ("outside prior: no log_f_atm in its prior gives
  u_Rp = 1.6"). A solve that does not converge within `solve.max_evals` model runs, or undefined inside its bracket,
  is status 3; a model failure keeps its status.
- **One root per bracket is assumed.** With several roots the density would be a sum over them, and solve mode
  finds one. On LHS 1140 b the scans support it for log_f_atm against the outflow H fraction (given the H/He-wind
  rule above) and against Rp; `scan` reports non-monotonic stretches inside the u range so that it can be checked
  for a new planet.
- **init = "solve"** (the default in solve mode): batches of 3 × nwalkers draws, the other parameters from their
  priors (a Latin hypercube) and u from O's data distribution (the split normal of the datum, truncated to the
  support), each solved from both prior bounds; until 2 × nwalkers have a finite lp, at most 5 batches. nwalkers of
  the best 2 × nwalkers are chosen at random, without replacement and without jitter (the draws are distinct in
  every coordinate, u included). The log counts "ok, no root, lp = −inf, failed" per batch; the draws go to
  `solve_init.npz`.
- **Warm start, deterministic.** After the initialization the parent fits a table of P* over the sampler's
  coordinates and writes `solve_table.npz` before the first step: a local linear fit on the 2(ndim + 1) nearest
  solved points in the unit cube, the coordinates scaled by how strongly P* depends on each (a global linear fit).
  The initialization's draws cover the whole prior, so after each step in `solve.refit_steps` (default 25, 50, 100,
  200, 400) the parent refits the table to them plus every distinct solved state of the chain so far (at most 5000
  points, the most recent), between two steps, while no evaluation runs, and replaces the file. Workers reload it
  when it changes. On the final LHS 1140 b chain (1e5/100, 6 workers) a refit after step 250 from 1161 solved
  points cut the median error of the guess from 0.0067 to 0.00024 dex. With the forward derivative, the model runs
  per solve fell from a mean of 10.5 (median 8, 90 % 19) to about 5.6 (median 4, 90 % 7), and the wall time per
  step and evaluation inside the prior from about 0.072 to 0.043 s (1.7× faster; steps 200–220 against 250–300).
  The table changes only the path of a solve, not its root (to `solve.tol`). Every checkpoint holds the active
  table (or `table_version = -1` while the run has none), which a resume writes back (or removes); a resume also
  redoes a refit that the end of the previous session cut short, and a checkpoint from before refits existed gets
  one catch-up refit. A refit that fails is a warning: the solves go on with the table they have. A solve then
  starts from the table's guess and expands a bracket around it: the first step is 1.5 Newton steps with the slope
  |dT/dP| of the neighbouring solves (their `log_jac`), within 1/16 and 16 times the 68th percentile of the table's
  leave-one-out errors; a first step that takes g away from u turns the expansion around, and meeting an undefined
  region the far prior bound decides, by the cold search's NaN-end rule, which side the root is on. Between refits
  the result depends only on θ and that file, so a resumed run is bit-identical (tested on the toy, with refits
  before and after the interruption; before refits existed, on the real model a 5-step LHS 1140 b run at 2e4/100
  resumed to 10 steps, with 6 workers instead of 2, reproduced a straight 10-step run bit for bit: chain, lp, RNG
  state, the table and every blob field but the runtime). Without the table (the initialization itself, `check`,
  `resolution`) a solve starts from both prior bounds.
- **Outputs** report the physical parameters: `summary.json` and `summary` show P from the blob `solve_value`
  next to u (`sampled`, and `solve` with the model runs per solve); `corner_params.png`, the tracks and `compare`
  use the physical parameters; `corner_sampled.png` shows (u, φ); `traces.png` shows the sampler's coordinates plus
  P. The tracks re-run (P*, φ) directly, without solving.
- **check** in solve mode prints one solve at the centre, u at the datum or `--theta u_<O>=value`, model run by
  model run. `resolution` evaluates its cells in solve mode (`--solve`, the default; `--no-solve` at fixed physical
  θ; section 6).
- **Cost.** A cold solve on LHS 1140 b took 15 model runs plus 2 for the central-difference derivative (17 runs,
  3.9 s at 1e5/100; check above; 16 with the forward default). In a real 2-worker smoke run (16 walkers, 10 steps,
  2e4/100) the 48 cold solves of the initialization took 17.2 runs each (14.8 s); the warm starts of the sampling
  took 11.1 on average (median 10, 10 % 6, 90 % 18, max 23), and cheaper runs too (0.037 s per run against 0.050 s
  in the centre's cold solve), since they stay near the ridge instead of evaluating both prior bounds. The
  acceptance was 0.21 with no stuck walker, 1.8 s/step; 81 of the 160 proposals fell outside the support (prior or
  u range) and cost no model run. The progress lines give `model runs/solve mean ... p50 ... p90 ...` and the
  isocalc time per run; the summary the same over the whole run.
- **Without solve mode** a config with `[solve]` runs with `--set 'solve.param=""' --set sampler.init=bisect` (the
  LHS 1140 b configs keep their bisection settings for that).

**Tested on a thin ridge** (`tests/test_mcmc_solve.py::test_thin_ridge_posterior`, the toy backend): log_f_atm and Mp
free, Rp known to 2e-4 R⊕, so the ridge is 0.0003–0.0008 dex thin in log_f_atm and curves from −1.4 (Mp 4) to −3.1
(Mp 8) while dRp/dlog_f_atm changes 2.7×. Plain sampling started on it (init = bisect) accepted 1.7 % of its
proposals; solve mode accepted 32–34 % with τ of 8–9 steps. In 1000 steps its 16th, 50th and 84th percentiles of
log_f_atm matched a brute-force grid posterior (16,200 model runs across the ridge, no Jacobian involved) within
0.5–2.0 standard errors for two seeds, while the same samples reweighted without the Jacobian missed it by 6–14
standard errors.

### Measured on LHS 1140 b (2026-10-28): the ridge, and why emcee does not mix on it in log_f_atm

Both configs (`lhs1140b`, dynamic_phi true; `lhs1140b_static_phi`, false) gave nearly the same numbers; the
results are in `mcmc_results/lhs1140b*/` (scan_*, resolution_*_bisect-4pts, mcmc_*_stuck_plain-sampling,
compare_*_stuck_plain-sampling, ridge-importance_*, check_bisect_*.txt). They were measured before solve mode
existed; the solve-mode runs follow in the next subsection.

**The ridge.** The scans (1e5 steps, n_atmodeller 100) show a thin surface in the 4-D prior:
- Along log_f_atm (7.6 Gyr, eps 0.15, Mp 5.6; 161 points from −3.2 to −2.8): no H/He wind below −3.1175 (static:
  −3.1125, where a frozen metal atmosphere of f_atm 6–8e-6 keeps Rp at 1.552 instead of the bare core's 1.538);
  then x climbs from the floor through the datum at −3.0174 (static −3.0178) with a slope of 190 dex per dex, so
  the 1 σ band is 0.0026 dex wide; above −3.0125 the slope drops to 16 (x from 0.007 to 0.05), and x = 1 (a pure H
  wind) from −2.925 on. The best point has lp −25.1: Rp 1.608 (term −11.9), Mdot_HHe 4.0e4 kg/s (term −13.2).
- Along age (log_f_atm × age, 41 × 12): the ridge moves from −3.138 at 3.7 Gyr to −2.959 at 10 Gyr (+0.029
  dex/Gyr) with lp flat at −25.1 on it. At 3.1 Gyr, below t_sat = 3.128 Gyr where the XUV is still saturated, it
  lies at −3.183 with Mdot_HHe 1.37e5 kg/s and lp −14.0; that sliver is 0.4 % of the age prior.
- Along eps (log_f_atm −3.6 to −2.0 × eps 0.05 to 1, 81 × 12, 7.6 Gyr): the ridge runs from −3.48 (eps 0.05)
  through −3.02 (0.15) to −2.09 (1.0), curved (about −1.9 dex per unit eps² near eps 0.45). On it Mdot_HHe grows
  roughly linearly with eps (1.3e4, 1.05e5, 2.5e5 kg/s at eps 0.05, 0.40, 1.0) while Rp falls (1.610 at 0.14,
  1.579 at 1.0), so lp along the ridge peaks near eps 0.4–0.5 at about −15.4. The Rp datum (1.73) is reached only
  in the H-rich region (x = 1), from log_f_atm −3.0 at eps 0.05 to −2.0 at eps 0.95.

**The MCMC in log_f_atm barely mixes.** `run` with init = bisect, 24 walkers on 3 workers each, the same seed,
n_steps 1e5 and n_atmodeller 300 (the shift-based recommendation of section 6), 400 steps: the bisection converged
70 and 72 of 72 draws (1160–1190 evaluations, 51 s), but the acceptance was 0.008 and 0.006 (78 and 56 accepted
moves in 9600 walker-steps; 5 and 8 walkers never moved), the reset at step 100 replaced none and τ could not be
estimated. Of the sampling proposals 60–64 % fell outside the prior, 14–15 % had no H/He wind (lp = −inf) and the
rest left the ridge. The rare accepted moves are long jumps (`traces.png`), as made by DE with γ = 1. So the run's
"posterior" is close to the bisection starts (prior draws moved onto the ridge), not a posterior.
- That is slow mixing, not none. The review of 2026-10-28 fitted a surrogate of the ridge to the 1200 solved draws
  below and ran the tool's DE moves on it: from the real starts it reproduced the real acceptance (0.003–0.006
  over 400 steps); later the acceptance rose to 0.016–0.024, and τ was about 1000–2000 steps (rerun here: 30,000
  steps, acceptance 0.020, τ still growing at 900–1000 steps). A converged chain in log_f_atm would need about 1e5
  steps, 6–7 h on 6 workers.
- Two tool problems made it look worse, both fixed: a walker accepting less than 0.05 counted as stuck and was
  left out of τ, so with every walker at 0.02 τ was NaN forever and a mixed chain could never be declared
  converged (the threshold is now relative, section 2); and a stuck-walker reset with 23 of 24 walkers stuck would
  have copied the one good walker onto all others (it now replaces none).
- Solve mode removes the problem: in (u, Mp, age, eps) the ridge is a smooth, wide coordinate. A real 2-worker,
  16-walker, 10-step smoke run at 2e4/100 accepted 21 % of its proposals with no stuck walker, and the full
  runs of the next subsection reached N/τ = 68–83 with every walker mixing.

**A ridge-marginalized estimate** (`ridge-importance_*`, a script next to the results, not a tool command):
bisect log_f_atm onto the outflow datum for each of 1200 prior draws of (Mp, age, eps), and integrate the narrow
outflow term along log_f_atm analytically: with slope s = dT/dlog_f_atm across the band, the marginal posterior of
(Mp, age, eps) is prior × exp(lp − term_x)/|s|, so each draw gets the weight exp(lp − term_x)/|s|. This is what
solve mode does exactly, with the Jacobian 1/|s|. 1087 and 1200 of 1200 draws converged, effective sample size 88
and 103:

| | dynamic_phi = true | dynamic_phi = false | data |
|---|---|---|---|
| log_f_atm | −2.624 (+0.133/−0.079) | −2.618 (+0.150/−0.080) | |
| Mp [M⊕] | 6.01 (+0.15/−0.13) | 6.01 (+0.16/−0.14) | 5.60 ± 0.19 |
| age [Gyr] | 5.65 (+2.9/−1.9) | 5.65 (+2.9/−1.9) | |
| eps | 0.47 (+0.09/−0.10) | 0.48 (+0.10/−0.11) | |
| Mdot_HHe [kg/s] | 1.20e5 (+0.21/−0.24 e5) | 1.23e5 (+0.22/−0.27 e5) | 2.03e5 (−0.137/+0.121 dex) |
| Rp [R⊕] | 1.621 (+0.010/−0.011) | 1.620 (+0.009/−0.012) | 1.730 ± 0.025 |
| D/H (not fitted; dynamic: before the He–D fix, section 10) | 5.6e-4 | 0.047 | |

The best draws have lp −13.0 and −13.1 (log_f_atm −2.62, Mp 5.97, 4.2 Gyr, eps 0.49: Rp 1.618, Mdot_HHe 1.26e5).
So eps rises from 0.15 to about 0.47 and removes most of the Mdot tension (0.23 dex low, 1.7 σ, against 0.70 dex
at eps 0.15), but no higher: along the ridge a larger eps lowers Rp. The radius stays 4.4 σ low (1.62 against
1.73) and pulls Mp 2.2 σ above its measurement. Two corrections from the review of 2026-10-28:
- **The dynamic-phi column was biased by stripping-edge roots.** 178 of its draws converged at the metal-wind
  edge (lp < −100) and 113 more did not converge, mostly at eps > 0.6; every one of them also has an island root
  (the H/He-wind rule above now finds it). Re-bisected, the dynamic estimate becomes eps 0.482 (+0.112/−0.107),
  log_f_atm −2.618 (+0.153/−0.078), log10 Mdot_HHe 5.089, ESS 104, with 15 % of the weight at eps ≥ 0.6 (the
  table: 10 %); it then matches the static column, so the dynamic-versus-static difference in eps was produced by
  the tool, not by dynamic_phi. The 85× between the D/H of the two columns was the He–D bug of
  `get_binary_diffusion_coeff` (section 10), not dynamic_phi.
- **The pre-t_sat sliver holds the best fit.** With `step_fn = true` ages from 3.100 to 3.128 Gyr still see the
  saturated XUV today (F0 = 0.588 against F_final = 0.17 W/m², 3.5×). There the island fits Mdot completely at eps
  ≈ 0.2: at log_f_atm −3.108, Mp 6.0, 3.11 Gyr, eps 0.2, log_like is −11.20 (Rp 1.6245, term −8.9; Mdot_HHe
  1.78e5, term −0.09; checked again here), against −13.0 for the best draw above. The 1200 draws had only 5 points
  in the sliver, none at eps 0.15–0.3, so their "no weight before t_sat" was an undersampling artefact; a
  stratified sample gave it roughly 8 % of the posterior (ESS 6 inside it, so rough). A saturated present-day
  XUV is unphysical for this quiet star, so the user decided (2026-10-28) that the LHS 1140 b age prior starts
  at t_sat: U[3.128e9, 1e10] in all three LHS 1140 b configs, which removes the sliver. The estimate above had no
  weight there, so it stands for the new prior. For any planet the config warns whenever t_sat lies inside the
  age prior with `step_fn = true`.

Assumptions of the estimate: one root per draw, T linear in log_f_atm across the band, and prior draws as the
proposal.

### Measured on LHS 1140 b in solve mode (2026-10-28/29): the posterior

Both configs, sampled in (u_outflow_H_fraction, Mp, age, eps) with `init = "solve"`, at the resolution the
solve-mode study of section 6 recommended for both, 3e4/100 (`--set model.n_steps=30000 --set
model.n_atmodeller=100 --set sampler.tracks=48`), 24 walkers on 3 workers each, the two runs at once, seed 12345.
Results in `mcmc_results/lhs1140b*/mcmc_20261028-214248_solve-mode_*`, compared in
`mcmc_results/lhs1140b/compare_*_dynamic-vs-static-phi_solve-mode`.
- **Initialization:** 72 of 72 draws solved (no root 0, lp = −inf 0), 16.8 and 17.3 model runs each, 38 and 42 s;
  the walkers started from 24 of the best 48 (lp −19.8 to −37.5).
- **Mixing:** acceptance 0.19 and 0.20 (every walker between 0.17 and 0.22), no stuck walker, none behind. τ grew
  from 5 steps at step 100 to 18–19 at step 900 and then stayed there: at the end τ(u, Mp, age, eps) = 18.0, 18.3,
  18.7, 19.4 steps (dynamic_phi = true) and 17.9, 18.8, 14.3, 19.1 (false). The stopping rule of then (τ changed
  by less than 1 % between checks) never fired: from step 1100 on the change was 2.6–7.8 % per check, the
  statistical noise of τ with 24 walkers (section 2). The runs were stopped by hand (Ctrl-C) once N/τ > 50 with a
  change below 5 % at two checks in a row: at step 1604 (N/τ = 82.6, 1 h 56 min) and 1303 (68.3, 1 h 43 min).
- **Cost:** 4.3 and 4.7 s/step on 3 workers each, with both runs and a busy machine (2.4–3 s/step once the other
  run had stopped); 9.9 and 9.8 model runs per solve (median 8, 10 % 6, 90 % 18, max 41), 0.088–0.095 s per model
  run; 45 % of the proposals fell outside the support (mostly beyond the age and eps bounds) and cost nothing.
  Memory 1.44 GB for 3 workers, parent 0.1 GB.
- **The 48 tracks** reproduce the chain's lp (49/49, the MAP included).

The posterior at 3e4/100 (median and 16/84 %), and reweighted to 1e5/100 (below):

| | dynamic_phi = true | dynamic_phi = false | reweighted to 1e5/100, 240 samples (true / false) | data |
|---|---|---|---|---|
| log_f_atm | −2.561 (+0.185/−0.133) | −2.573 (+0.170/−0.124) | −2.573 (+0.130/−0.141) / −2.589 (+0.143/−0.095) | |
| Mp [M⊕] | 6.03 (+0.18/−0.17) | 6.03 (+0.17/−0.16) | 6.03 / 6.01 | 5.60 ± 0.19 |
| age [Gyr] | 6.30 (+2.47/−2.05) | 6.28 (+2.35/−2.07) | 6.27 / 6.07 | |
| eps | 0.505 (+0.196/−0.114) | 0.490 (+0.181/−0.104) | 0.482 (+0.150/−0.110) / 0.488 (+0.124/−0.102) | |
| u = log10 x | −2.877 (+0.252/−0.264) | −2.869 (+0.250/−0.258) | | −2.939 (−0.261/+0.240) |
| outflow H fraction | 1.33e-3 (+1.05/−0.61 e-3) | 1.35e-3 (+1.05/−0.61 e-3) | 1.35e-3 / 1.28e-3 | 1.15e-3 |
| Mdot_HHe [kg/s] | 1.28e5 (+0.46/−0.28 e5) | 1.25e5 (+0.42/−0.25 e5) | 1.23e5 / 1.24e5 | 2.03e5 |
| Rp [R⊕] | 1.619 (±0.012) | 1.619 (±0.012) | 1.620 / 1.618 | 1.730 ± 0.025 |
| D/H (not fitted; dynamic: before the He–D fix) | 5.0e-4 (+1.6/−3.0 e-4) | 0.037 (+0.024/−0.021) | | |

- **Where the data sit.** The outflow H fraction is reproduced (its median 0.06 dex, 0.26 σ above the datum);
  Mdot_HHe is 0.20 dex (1.5 σ) low, with 30 % (static: 27 %) of the samples within 1 σ of it; Rp is 4.5 σ low
  (no sample above 1.665 R⊕), and it pulls Mp 2.2 σ above its measurement. The tensions of the ridge estimate
  (section above) stand.
- **eps** is 0.48–0.50, correlated with log_f_atm along the ridge (r = 0.91); almost no sample has eps < 0.2
  (the smallest are 0.21 and 0.19). The age is barely constrained (nearly flat over the prior, 12 % below 4 Gyr).
- **The MAP** (the maximum of the sampled density p(u, φ), which includes ln|J|; section 10) has lp −18.74 at step
  1368: log_f_atm −2.636, Mp 6.029, 3.99 Gyr, eps 0.501 (u −2.915); log_jac −5.70 and log_like −13.04 (terms Mp
  −2.54, Rp −9.41, Mdot_HHe −1.09, outflow H fraction −0.004; Rp 1.6216, Mdot_HHe 1.28e5 kg/s). The best physical
  point of the chain (the maximum of log_prior + log_like) is log_f_atm −2.606, Mp 5.985, 4.73 Gyr, eps 0.499
  (log_like −13.038). With dynamic_phi = false the MAP is log_f_atm −2.607, Mp 6.008, 4.89 Gyr, eps 0.496
  (log_like −13.088), which is also its best physical point.
- **dynamic_phi**: the 74× in D/H was mostly the He–D bug (section 10). After the fix of 2026-10-29 the final
  dynamic chain (1e5/100) gives D/H 0.028 against 0.037 for the static one (3e4/100); at the same posterior
  points and resolution dynamic/static is 0.7–1.07, and 10 re-solved posterior samples of the dynamic run rose
  37–90× (`diffusion-fix-ab_*`). `subcritical` is 1 against 0 at the end; the fitted quantities agree:
  the medians differ by 0.012 dex in log_f_atm and 0.015 in eps, about 2 standard errors of the two chains
  (0.0045 each, from about 2000 and 1600 independent samples).

**The resolution, checked on the posterior.** 240 posterior samples of each run were solved again, cold, at
3e4/100 and at finer cells, and the posterior was reweighted by w = exp(lp_fine − lp_3e4/100) (a script next to
the results, `resolution-reweight_*`; the cold re-solves at 3e4/100 matched the chain's lp to 0.04, median 0.002):
- the bulk barely moves: Mp, age, Rp, Mdot_HHe and the outflow fraction change their medians by at most 3 % at
  1e5/100 and 3e5/100 (Mdot_HHe by −2.5 % and −2.9 %; up to 5 % at 1e5/300), and the log-weights have a standard
  deviation of 0.30–0.41 (ESS 223–230 of 240);
- but the tail to high eps does: the log-weight falls with eps (r = −0.72), by 0.87 on average at eps ≥ 0.8 for
  1e5/100 (1.24 for 1e5/300). It is the Jacobian, ln|dT/dlog_f_atm| of the steep outflow fraction, that depends
  on the resolution there; Rp and Mdot_HHe change by less than 3e-3 and 2e-3 dex. 3e4/100 puts 27 % (dynamic)
  and 25 % (static) of the posterior at eps ≥ 0.6, against 20 % and 19 % at 1e5/100, 18.5 % at 3e5/100 and 16 %
  at 1e5/300 (dynamic). On the same 240 samples the 84th percentile of eps falls from 0.72 to 0.63 (static:
  0.68 to 0.61), and that of log_f_atm from −2.37 to −2.44; the medians move by −0.013 and −0.007 in eps.
- 1e5/100 and 3e5/100 agree (eps 0.482 (+0.150/−0.110) and 0.482 (+0.141/−0.111), log_f_atm −2.573 and −2.577),
  so the reweighted column is the converged posterior. With 1e5/300 the dynamic estimate becomes eps 0.478
  (+0.131/−0.110), log_f_atm −2.583, which matches the ridge estimate above at the same resolution (0.482
  (+0.112/−0.107), 15 % at eps ≥ 0.6).
- The 3-point study of section 6 missed this because none of its points was in the tail (eps 0.47, 0.58, 0.69:
  |Δ log L| ≤ 0.16 at 3e4/100). `resolution --source chain:RUN_DIR` on the run with 6 random posterior points
  (eps 0.44–0.62) again called 3e4/100 stable (|Δ log L| ≤ 0.16); with 30 points it failed it (|Δ log L| 1.3, d
  0.16 of Rp) and 1e5/100 too (|Δ log L| 0.76), against 3e5/100. So: check a run on 30 or more posterior points,
  and read the result as the size of the reweighting, not as a strict pass.
- For a final posterior, start a new run at 1e5/100 (the configs' own setting), about 3× the cost per step, or
  reweight as above. `run --resume` continues a run at its own resolution (from its `config.json`), so it extends
  the 3e4/100 chains but does not change their resolution.

### TOI-406 b: a planet with only Rp and Mp (generality check, 2026-10-28/29)

`mcmc_results/configs/toi406b.toml` (the package example with `output_root = ".."` and 2 workers; free log_f_atm
U[−5, −1.3], Mp U[1, 8], age U[3.5, 10] Gyr; Rp 1.55 ± 0.06 R⊕ and Mp 4.57 ± 0.70 M⊕; ratios composition,
dynamic_phi, 1e5/100), with the tool's defaults. Results in `mcmc_results/toi406b/`.
- **check --bisect** (`check_bisect_*.txt`): at the centre (log_f_atm −2.7, Mp 4.57, 5 Gyr) Rp = 1.5841 (term
  −0.16), 0.156 s per evaluation. The bisection on Rp converged at log_f_atm −2.7736 in 6 rounds (8
  evaluations, 1.9 s); it met Rp = 1.4621 R⊕, the bare core, at every log_f_atm from −5 to −2.92.
- **Plain sampling** (`mcmc_*_plain_3000steps`; init = prescan, 16 walkers on 2 workers, 600 steps resumed to
  3000): acceptance 0.20, no stuck walker, τ = 17.1, 18.4, 14.5 steps, N/τ = 163, 0.40 s/step (20 min). Posterior:
  log_f_atm −3.72 (+0.89/−0.87), Mp 5.00 (+0.55/−0.59) M⊕, age 6.8 (+2.2/−2.2) Gyr, Rp 1.509 (+0.046/−0.044).
  82 % of the samples are stripped planets at the bare-core radius, which fits the radius for Mp ≈ 5 (hence Mp
  above its datum); log_f_atm is flat over the whole stretch that strips the planet. The 48 tracks reproduce
  the chain (49/49).
- **Solve mode on Rp** (`mcmc_*_solve-Rp_600steps`; `--set 'solve.param="log_f_atm"'`, 300 steps resumed to 600):
  the initialization solved 35 of 48 draws (13 had no root: u below the bare core of their Mp; 1 failed), 9.3
  model runs each; acceptance 0.23, τ = 11.0, 11.0, 12.7 steps, N/τ = 47, 6.0 s/step (4–9), 6.0 model runs per
  solve (median 5, 90 % 8), 0.21 s per model run; 18 of 9600 proposals (0.2 %) failed their solve (status 3).
  Posterior: log_f_atm −2.667 (+0.167/−0.163), Mp 4.48 (+0.61/−0.56), age 6.5 (+2.5/−2.2) Gyr, Rp 1.566
  (+0.048/−0.044).
- **The two agree where solve mode applies, and solve mode misses the plateau.** The plain run's samples that
  keep an atmosphere give log_f_atm −2.662 (+0.157/−0.150), Mp 4.48 (+0.65/−0.53), Rp 1.569 (+0.053/−0.045):
  the medians agree within 0.005 dex and 0.01 M⊕, about 0.4 and 0.2 standard errors, and the widths within 10 %.
  Their age medians (7.04 against 6.54 Gyr) differ by about the scatter of either chain (the plain one's moves
  by 0.35 Gyr between its two halves); a 2-D scan at Mp 4.5 (`scan_*_2d-log_f_atm-age_Mp4.5`) gives the exact
  branch integral ∫L dlog_f_atm flat in age to 2 %, so the true age marginal is flat there and both are noise
  around it. The 82 % of the posterior on the plateau are not in the solve-mode run at all (section 10). For
  this planet plain sampling is the right mode: with σ = 0.06 R⊕ the radius ridge is wide.
- The config now warns about this case (bare core 1–1.68 R⊕ over the Mp prior, inside the u range [1.19, 1.91],
  −1.5 σ from the datum at Mp 4.57), and the scan's solve check lists the plateau on every line (Rp = 1.45648
  for log_f_atm −3.2 to −2.96 at 3.5 Gyr, to −2.72 at 10 Gyr).
- `compare` of the two runs: `compare_*_plain-vs-solve-Rp`.

## 6. Choosing n_steps and n_atmodeller: the resolution study

Every number the MCMC compares to data carries numerical error, from the time step (age/n_steps) and from how
often Atmodeller re-equilibrates the atmosphere with the magma ocean (every n_atmodeller steps). The study
measures that error in the units that matter, the measurement sigmas, and picks the cheapest resolution where it
is negligible.

**What it does.**
1. Picks test points θ (`source`):
   - `auto` (default): `solve` in solve mode; `bisect` when `sampler.init` is `"bisect"` (or `"solve"` with
     `--no-solve`); else `prescan`;
   - `solve` (solve mode): init-style draws, the other parameters from the prior and u from the data
     distribution (max(20, 5 × points) per batch), solved at the config resolution; `points` of the better half by
     lp, by farthest-point sampling in the unit cube of the sampler's coordinates;
   - `bisect`: the bisection of section 5 at the config resolution (max(20, 5 × points) draws per batch), then
     `points` of the better half of the converged points by lp (at least `points`), by farthest-point sampling
     in the unit cube, starting from the best lp. It puts the test points where the posterior lives. The half
     matters: a converged draw can still lie far off the posterior. On LHS 1140 b one matched the outflow H
     fraction at the stripping edge (eps 1, Mdot_HHe 5e-8 kg/s, lp −1859), where the outflow is numerically
     ill-defined (x from 1e-3 to 0.89 across the grid); as a test point it alone failed every cell up to 1e6/30.
     If fewer converge it uses those (a warning); if none, prescan points when `bisect_fallback`, else an error;
   - `prescan`: the best quarter of 20 × points Latin-hypercube prior draws, spread out by farthest-point
     sampling;
   - `prior`: prior draws that keep an atmosphere;
   - `chain:RUN_DIR`: random posterior samples of a previous run (the best check once a posterior exists; use
     `--points 30` or more and only the cells in question, so that the tails are represented: on LHS 1140 b 6
     points passed a cell that 30 points failed).
2. Runs every θ at every cell (n_steps, n_atmodeller) of the grid, longest first. Cells whose estimated cost
   exceeds `max_eval_seconds` are skipped and listed. Cells up to 3e5 steps share one pool; larger cells get a
   second pool with fewer workers if memory requires it.
3. The **reference** is the finest computed cell: the most steps, then the smallest n_atmodeller.
4. For each active, non-input observable k, compares each cell with the reference:
   d_k = |T_k(cell) − T_k(reference)| / s_k. T_k is the likelihood's space transform (with floors) and s_k the
   effective sigma (the mean of lo and hi when asymmetric). d = 0 when the value is undefined at both resolutions
   and inf when it is defined at only one, e.g. when the atmosphere is lost at only one of them.
5. A cell **passes** when max d ≤ `tol` (0.1 σ) and |Δ log L| ≤ `dlogl_tol` (0.5) at every test point. It is
   **stable** when it passes and every finer cell (as many or more steps, and n_atmodeller as small or smaller)
   passes too. The **recommendation** is the stable cell with the smallest measured time per evaluation.
6. **Noise floor:** every run is repeated with the parameter p_j of step 7 (in solve mode: u) moved by
   `perturb` (1e-6) of its prior width, and the change a smooth model has over that step, ∂T_k/∂p_j × the step
   (from the derivative runs of step 7), is subtracted. What is left over is numerical noise: if it is ≥ tol, the
   likelihood is rough at that level, and the MCMC will see it whatever the resolution. (Before 2026-10-28 the raw
   difference was reported; for a steep observable it measured the slope, e.g. the "0.011, smooth" of the LHS
   1140 b studies below was 750 dex/dex × 3.7e-6 dex, and a slope above about 7e3 would have been called rough.)
7. **Equivalent parameter shift.** Each test point is also run at the reference cell at θ + h e_j, with h = 1e-4
   of the prior width W_j of p_j (`solve.param` in solve mode, else `sampler.bisect_param` if free, else the first
   free parameter). That gives the derivative ∂T_k/∂p_j of every observable, and each cell's error becomes a
   shift of p_j:
   shift_k = (T_k(cell) − T_k(reference)) / (∂T_k/∂p_j), and shift_frac = |shift_k|/W_j.

**Two recommendations.** On a thin ridge a steep observable fails the fixed-θ criterion although its error is
negligible. At the LHS 1140 b island (log_f_atm −3.0175, 7.6 Gyr, Mp 5.6) log10 of the outflow H fraction was:

| n_steps | n_atmodeller | log10 x |
|---|---|---|
| 1e5 | 1000 | −2.337 |
| 1e5 | 100 | −2.967 |
| 1e5 | 30 | −3.114 |
| 1e5 | 10 | −3.185 |
| 3e5 | 300 | −3.010 |
| 3e5 | 100 | −3.152 |
| 3e5 | 30 | −3.238 |
| 1e6 | 100 | −3.255 |
| 1e6 | 30 | −3.291 |
| 1e6 | 10 | −3.303 |

The fiducial 1e5/100 is 0.34 dex from 1e6/10, about 1.3 σ at fixed θ, but with d log10 x/d log_f_atm ≈ 190 that
equals a shift of log_f_atm by only 0.0018 dex, far below the posterior's extent along the other parameters. Rp
and Mdot are converged to < 1e-3 relative. So:
- the first recommendation (the conservative one) keeps the fixed-θ criterion above;
- the second, "SHIFT-BASED RECOMMENDATION", is the cheapest cell that is stable when each observable passes at
  each point with d ≤ tol OR shift_frac ≤ `shift_tol` (1e-3), and the ridge weights stay put. For an observable
  that is not steep (moving p_j by shift_tol·W_j changes it by less than tol σ) the shift condition implies
  d ≤ tol, so the non-steep observables must still pass at fixed θ. Instead of |Δ log L| (the same fixed-θ
  difference) it requires the **ridge-weight spread** ≤ `dlogl_tol`: at each test point the cell's values are moved
  back along p_j until the steepest observable matches the reference again (linearized), and the other terms'
  change there, which reweights the posterior along the ridge, may differ between the test points by at most
  dlogl_tol. The worst shift it reports is over the (point, observable) pairs that need the shift criterion
  (d > tol).

Use the second when the posterior is much wider than shift_tol·W_j (0.0037 dex of log_f_atm for LHS 1140 b) along
the ridge, or after checking that both give the same posterior.

**In solve mode** (`--solve`, the default when `[solve]` is active; `--no-solve` turns it off) the test points
are points of the sampler's coordinates (u, φ), and every cell solves for the solved parameter again, from both
prior bounds, as the sampler does. So d of the solve observable is about 0 by construction (within `solve.tol`),
and the other observables show the error that matters for the posterior; |Δ log L| compares the sampler's density,
log_prior + log_jac + log_like. The noise floor moves u (raw differences: no observable is steep in u). A solve
costs about 15 model runs: the plan counts 15 per evaluation, and the estimate is printed again with the number
measured on the test-point selection. The equivalent-shift block stays as a cross-check: its derivative runs are
made at (P* + h, φ) of the reference cell, and it lists the worst shift of P* itself between each cell and the
reference (`solve_shift` in `results.npz`), which is the error a study without solve mode expresses as the
equivalent shift of the solve observable.

**How to read `table.txt`.**
- The grid shows, per cell, the worst d over the test points and observables, with the mean isocalc time per
  evaluation on the line below. Marks: S stable, p passes but a finer cell fails, x fails, R reference, − skipped,
  and [brackets] for the recommended cell.
- Then come the recommendation (why, the cost, the speed-up against the reference, the `--set` options and a
  copy-paste run command), the verdicts for the fiducial cell (1e5, 100) and for the config's own cell.
- The block "Equivalent shift of <param>": the grid of the worst |shift| (in the parameter's unit) with the
  observable responsible below it, the derivatives per test point (* marks the steep ones), and the shift-based
  recommendation with its `--set` options.
- The noise floor, then warnings (each line of a multi-line warning on its own): a recommendation at the coarse
  edge of the grid, or one equal to the reference
  (the grid shows no convergence), unfitted outputs that change by more than 10 % at the recommended cell,
  failures, and too few test points.
- At the end: the test points, a per-cell table per observable, and the relative change of *every* output
  (fitted or not) at the recommended, fiducial and config cells.
- `resolution_heatmap.png` shows the same grid, with n_atmodeller decreasing to the right (more calls, higher
  cost) and n_steps growing downwards. Solid outlines are stable, dashed ones pass, the orange border is the
  recommendation, the dashed orange border the shift-based one, and hatched cells were skipped.
  `resolution_convergence.png` plots worst d against the number of Atmodeller calls per run, one line per n_steps.

**Measured with the outflow observables (LHS 1140 b, 2026-10-28).** `--source bisect --points 4`, n_steps 3e4,
1e5, 3e5, 1e6 × n_atmodeller 300, 100, 30, 10 (16 cells, 132 evaluations, estimated 1.5 min, took 2.3–2.5 min on
5 then 3 workers; no cell skipped), in `mcmc_results/lhs1140b*/resolution_*_bisect-4pts`:

| | dynamic_phi = true | dynamic_phi = false |
|---|---|---|
| fixed-θ recommendation | 1e6/100: d 0.053, 1.87 s/eval | 1e6/30: d 0.041, 5.18 s/eval |
| fiducial 1e5/100 | fails: d 4.0 (outflow fraction), \|Δ log L\| 8, 0.28 s | fails: d 4.5, \|Δ log L\| 9.9, 0.37 s |
| shift-based recommendation | 1e5/300: worst shift 0.0035 dex (9.4e-4 of the width), 0.12 s | 1e5/300: 0.0033 dex (8.8e-4), 0.11 s |
| noise floor | 0.011: smooth | 0.011: smooth |

- At fixed θ the outflow H fraction passes only at 1e6 steps with n_atmodeller ≤ 100 (dynamic) or ≤ 30
  (static); its error falls with Atmodeller calls until the time step dominates (at 1e5 steps it stalls at d
  1.2–1.3, at 3e5 at 0.3). Rp has d 0.20–0.24 at 1e5/300, 0.10–0.14 at 1e5/100 and ≤ 0.04 from 1e5/30; Mdot_HHe
  passes everywhere (d ≤ 0.073).
- At each test point the three errors are the same shift of log_f_atm, to 30 % (e.g. 0.0033, 0.0033 and 0.0024
  dex for Rp, Mdot_HHe and x at one point at 1e5/300): the resolution moves the track as a slightly different
  initial atmosphere would. The derivatives of x are 95–855 dex per dex at the test points. Every cell with at
  least 1e5 steps passes the shift criterion, 1e5/300 only just (steep observables up to 0.0031 and 0.0033 dex,
  against 0.0037; the table then reported 0.0035 for dynamic_phi, the shift of Mdot_HHe, which passes at fixed θ),
  and 1e5/100 with 0.0015–0.0018 dex. The MCMC runs of section 5 used 1e5/300.
- Recomputed from these studies (2026-10-28), the ridge-weight spread is 0.33 (dynamic) and 0.29 (static) at
  1e5/300, 0.10 at 1e5/100 and 0.01 at 1e6/100: 1e5/300 distorts the weights along the ridge 3× more than the
  fiducial, still below dlogl_tol on these 4 points. Their noise floors (0.011) were the slope of x, not noise.
- These studies predate solve mode. In solve mode the same question is asked at fixed (u, φ) directly (next
  paragraph).
- D/H (not fitted) changes by 43 % (dynamic, before the He–D fix of section 10) and 84 % (static) between the
  fixed-θ recommendation and the
  reference.
- The first attempt of the dynamic study (kept as `resolution_*_unfiltered-incl-stripping-edge-point`) took its
  test points from all converged draws. One had converged at the stripping edge (eps 1.0, Mdot_HHe 5e-8 kg/s,
  lp −1859), where x is numerically ill-defined (1e-3 to 0.89 across the grid, d up to 42), and alone pushed both
  recommendations to 1e6/30. Since then the points come from the better half by lp (step 1 above).

**Measured in solve mode (LHS 1140 b, 2026-10-28).** `resolution CONFIG --points 3 --n-steps 3e4,1e5,3e5,1e6
--n-atmodeller 300,100,30 --set resolution.max_eval_seconds=5` (the 3 × 3 grid plus 1e6/100 and, from the same
lists, 1e6/300; 1e6/30 skipped by the cost limit), 69 solve evaluations, 3.4 min on 5 then 3 workers against an
estimate of 2.0–2.3 min; in `mcmc_results/lhs1140b*/resolution_*_solve-3pts_3e4-1e6`. The two configs drew the
same 3 points (u −2.93, −2.62, −2.54; eps 0.58, 0.69, 0.47) and a solve took 16.8 model runs:

| | dynamic_phi = true | dynamic_phi = false |
|---|---|---|
| recommendation (both criteria) | 3e4/100: d 0.041 (Rp), \|Δ log L\| 0.16, 1.37 s per solve, 20× faster than 1e6/100 | 3e4/100: d 0.037, \|Δ log L\| 0.25, 1.39 s, 21× |
| fails | 3e4/300 (d 0.11, \|Δ log L\| 1.2), 1e5/300 (\|Δ log L\| 0.59) | 3e4/300 (0.096, 1.0), 1e5/300 (0.69) |
| fiducial 1e5/100 | stable: d 0.011, \|Δ log L\| 0.037, 3.86 s | stable: d 0.010, 0.030, 5.21 s |
| ridge-weight spread at 3e4/100, 1e5/100 | 0.27, 0.07 | 0.38, 0.05 |
| noise floor (u moved by 1e-6 of its width) | 1.3e-5: smooth | 1.3e-5: smooth |

- In solve mode d of the outflow fraction is ≤ 0.012 everywhere (the solve tolerance), so the steep observable no
  longer dominates: the cell is set by Rp (d 0.11 at 3e4/300) and by |Δ log L|, which includes ln|J|.
- The table warned that 3e4/100 lies at the coarse edge of the grid, and that D/H (16 % and 37 %; the dynamic value
  before the He–D fix) and some VMRs change by more than 10 % between it and the reference.
- **On the posterior 3e4/100 is not good enough in the tail.** The runs used it; reweighting 240 posterior samples
  to finer cells (section 5, "Measured on LHS 1140 b in solve mode") moved the 84th percentile of eps from 0.72 to
  0.62–0.63 and the share of eps ≥ 0.6 from 25–27 % to 19–20 %, because ln|J| at eps > 0.8 changes by about 0.9 between
  3e4/100 and 1e5/100. Three test points did not contain such a sample. `--source chain:RUN_DIR --points 30`
  (`resolution_*_chain-30pts_3e4-3e5`) did: it fails 3e4/100 (|Δ log L| 1.3) and 1e5/100 (0.76) against 3e5/100;
  with 6 points (`resolution_*_chain-6pts_3e4-3e5`) it passed both. Use 30 or more posterior points for that
  check; 1e5/100 is the resolution for a final LHS 1140 b posterior (its marginals agree with 3e5/100).

**Measured so far (LHS 1140 b, the earlier radius-and-mass setup).**
- Rp is insensitive to resolution. It moves 1.7348 → 1.7312 R⊕ from 2000 to 1e5 steps at the centre, and
  1.6222 → 1.6206 from 1e4 to 1e6 steps elsewhere. The fiducial 1e5/100 is within 0.02 % of 1e6/1 in Rp, 0.25 %
  in fatm and 0.05 % in T_surf.
- Composition converges much more slowly. At one prior point, Rp at 2000/100 was within 0.09 σ of 5000/10, but C/O
  differed by 14 % and the VMRs by 13–50 %. At the old centre (log_f_atm −2.8, 5 Gyr) C/O is 0.614 at 2000/100 and
  0.571 at 1e5/100.
- **So a recommendation holds only for the observables that were active in the study.** Before fitting a new
  composition or outflow observable, rerun the study with it switched on; the table's "outputs that are not
  fitted" warning points this out.
- Real studies with Rp and Mp active (6 workers and then 4 at 1e6, `parallel.check_available = false`), in
  `mcmc_results/lhs1140b/resolution_20261028-*`:
  - The quick grid with 3 points took 20 s. Only 1e4/1000 fails (d = 0.14), so it recommends 3e4/1000 (d = 0.072,
    0.014 s/eval). The fiducial 1e5/100 is stable at d = 0.02.
  - The default grid with 1 point took 4.1 min against an estimate of 2.6; the 1e6/1 reference took 163 s per
    evaluation. Every cell is stable and the fiducial has d = 0.013, so it recommends the cheapest cell,
    1e4/1000 (d = 0.074). That is at the coarse edge of the grid, and 1 point is not enough to trust it.
  - D/H differs by 39 % between the fiducial and the reference (before the He–D fix), and VMR_SO2 and VMR_H2SO4 by
    up to 14 %
    at 1e4/1000. Fitting either needs its own study.
  - The noise floor is 5e-5 σ: smooth.
  - The peak worker footprint was 0.95 GB at 1e6 steps, 3.1 GB for 4 workers, against an estimate of 1.85 GB.
- Cost: the default grid is 35 cells × 6 points × 2 runs + 6 shift runs, plus the test-point selection. The
  quick grid needs 12 cells × points × 2 + points. `--dry-run` prints the estimate from the cost model, which can
  be up to 2× off.

## 7. Comparing runs

```bash
$PY -m isofate.mcmc compare RUN_A RUN_B [RUN_C ...] [--labels A,B] [--out DIR]
```

It overlays two or more MCMC runs, of two settings (e.g. `lhs1140b` and `lhs1140b_static_phi`) or two planets,
in `compare_<stamp>/` (default: under the first run's `<output_root>/<name>/`; never overwritten):
- `compare_corner.png`: the free parameters both runs share, one categorical colour and line style per run
  (39 % and 86 % contours, a legend with the sample counts);
- `compare_predictive.png`: the posterior of every shared active observable per run, with the data band of the
  first run (a note says so if the runs' data differ);
- `compare_tracks.png`: the evolutionary tracks side by side, one column per run, the same panels and the same y
  limits in each row;
- `compare.txt` (also printed): per run the path, the model settings (n_steps, n_atmodeller, dynamic_phi, backend,
  init, seed), steps × walkers, posterior samples, mean acceptance, τ max, N/τ and the convergence flag, s/step,
  the status counts of the sampling evaluations; then the median and 16/84 interval of every free parameter and
  observable of either run, next to the data.

Labels default to the run directory names when the runs share a config name, else the config names (both when
that is still ambiguous); long labels wrap in the legends. A solve-mode run enters with its physical parameters
(the solved one from `solve_value`), so it can be compared with a run without solve mode. Earlier runs whose
blobs lack fields added since (e.g. the outflow fractions) show "-" for those.

## 8. Outputs and file formats

A run directory `<output_root>/<name>/mcmc_<YYYYmmdd-HHMMSS>/` holds the files below. It is claimed atomically: a
run started in the same second gets `mcmc_<stamp>_1`, `_2`, ... (scan, resolution and compare directories
likewise), so two runs never share one.

| file | content |
|---|---|
| `config.json` | the resolved config (`config.load_resolved` reads it back) |
| `config.toml` | a copy of the TOML, with the `--set` overrides listed at the top |
| `provenance.json` | git commit, dirty flag and status of the IsoFATE repo, package versions, host, start of each session |
| `run.log` | every progress line, time-stamped; tracebacks of failures |
| `prescan.npz` | `theta`, `lp`, `blobs` of the prescan points, `start` (the chosen ones) |
| `bisect.npz` | `init = "bisect"`: every draw (below) |
| `solve_init.npz` | `init = "solve"`: `theta` (the sampler's coordinates), `lp`, `blobs` of every draw, `free` |
| `solve_table.npz` | solve mode: the warm-start table (`x` scaled unit-cube coordinates, `p` the solved values, `lslope` ln\|dT/dP\|, `scale`, `k`, `step`, `loo` errors, the box and bounds); written before the first step and replaced by each refit (`solve.refit_steps`); `chain.npz` holds the active one as `table_*` with `table_version` (the step it was fitted after; −1: none yet) |
| `chain.npz` | the chain (rewritten atomically at each checkpoint) |
| `burnin.npz` | the chain before a stuck-walker reset (same format) |
| `summary.json` | posterior summary |
| `tracks.npz` | posterior evolutionary tracks |
| `*.png` | `corner_params`, `corner_sampled` (solve mode), `corner_observables`, `traces`, `predictive`, `tracks` |
| `worker_errors/` | tracebacks of status-5 evaluations, one file per worker |
| `.lock` | the pid and host of the session running in the directory (removed at its end) |

**bisect.npz** (`search.save`): per draw `theta` (the prior draw), `point` (the final point), `x` (its value of
the parameter), `g`, `a` and `b` (the final bracket), `outcome` (an index into `outcome_names`: converged, g > 0
at both ends, g < 0 at both ends, undefined at both ends, model failed, undefined inside the bracket, not
converged, converged with lp = −inf), `rounds`, `lp`, `blobs`, `exhausted_lo`/`exhausted_hi`, and the history of
every evaluation, `hist_x`, `hist_value`, `hist_g`, `hist_status` (draws × (iters + 2), the two ends first, NaN
padded); and the settings `param`, `observable`, `target`, `tol`, `iters`, `lo`, `hi`, `evaluations`, `seconds`.

**Figure axes.** These rules keep a few extreme samples from flattening the rest of a panel.
- Histograms and corner panels use the full range of the samples. When a few outliers make that range more than
  3× wider than the 0.5–99.5 percentile range, they use the percentile range instead, and `predictive.png` notes
  "N samples beyond the axis".
- `corner_observables.png` leaves out observables that are constant over the posterior (e.g. Φ_He/Φ_H = 0
  throughout). `predictive.png` shows every observable.
- The log-prob row of `traces.png` reaches at most 3 spans of the second half of the chain below its maximum. The
  title says which fraction of the values lies below the axis.
- Each log panel of `tracks.png` spans the 95 % band, the MAP and the data, plus at most one decade more for the
  individual tracks. A track whose atmosphere is being lost (VMRs falling by 30 decades) runs off the axis. The
  panels are Rp, fatm, Mdot (Mdot_HHe when that is fitted), the outflow H fraction, H/He, C/O, D/H, T_surf and the
  VMRs of the 4 most abundant species.

**chain.npz:**
- `chain` (nsteps, nwalkers, ndim), `log_prob` (nsteps, nwalkers), `accepted` (nwalkers), `iteration`;
- `blobs` (nsteps, nwalkers): a structured array with one float64 field per name in `model.BLOB_FIELDS`. These
  are `status`, `runtime` [s] (all model runs of the evaluation), `log_prior` and `log_like`, then every
  observable of the table above (in its unit), then the five extras, then the three fields of solve mode:
  `solve_value` (the solved parameter P*; NaN without solve mode), `solve_evals` (model runs of the evaluation:
  0 or 1 without solve mode) and `log_jac` (ln|∂P*/∂u|; 0 without solve mode). The chain's log-probability is
  log_prior + log_jac + log_like. Blobs hold accepted states only; rejected proposals are counted in the
  evaluation tallies;
- the RNG state (`rs_*`);
- bookkeeping, cumulative over sessions: `reset_step`, `replaced`, the τ history (per check: `tau_steps`, `tau`,
  `tau_stable`, `tau_change` and `tau_tol` of the coordinate closest to failing, `tau_walkers` used), `converged`
  (at the last check), `converged_at` (the first converged step), in solve mode `derivative` and `refit_step` (the
  last refit step handled), `max_steps`, `workers`, `warmup_s`, evaluations per status (`counts_init`,
  `counts_sampling`), histograms of the model runs per evaluation (`evals_init`, `evals_sampling`), isocalc time
  (`runtime_*`), timing (`sampling_s`, `steps_timed`, `init_s`, `wall_s`) and `run_warnings`.

The file grows with the chain and every checkpoint rewrites all of it: at 24 walkers × 5000 steps it is about
34 MB and a save takes 1.1 s, at 20,000 steps 134 MB and 4.7 s (measured 2026-10-28). Periodic checkpoints
therefore come every `checkpoint_every` steps but at most once a minute and at most every 20 times the last save's
duration, so that they cost at most 5 % of a long run; an interrupt, a failure, a reset and the end of sampling
always checkpoint. The blob fields grew from 35 to 38 with the outflow observables and to 41 with solve mode:
runs written before can still be summarized, plotted and compared, but not resumed (resume checks the blob fields
and says so).

**summary.json:**
- `state` (finished, interrupted, partial), `provenance`, `settings` (nwalkers, ndim, free, n_steps,
  n_atmodeller, backend, workers, moves, init, seed, max_steps, tau_checks, observables);
- `steps`, `total_steps`, `reset_step`, `replaced_walkers`;
- `timing` (wall, warm-up, s/step, isocalc s per evaluation and per model run) and `evaluations` per status,
  for initialization and sampling;
- `acceptance` (mean, min, n_stuck, behind: the walkers ending with lp < median − 20, per walker);
- `tau` per parameter, `tau_max`, `n_over_tau`, `tau_history` (per check: steps, tau, stable, change, tol and
  walkers; change and tol are NaN at a first check, and the checks of a checkpoint from before 2026-10-29 get
  False, NaN and −1), `converged`, `converged_at`, `discard`, `thin`, `n_samples`;
- `params`: per physical parameter (in solve mode the solved one from `solve_value`), the 2.5/16/50/84/97.5
  percentiles, mean and std; `sampled` (solve mode): the same for the sampler's coordinates;
- `solve` (solve mode): param, observable, the u name and bounds, tol, range_sigma, warm_start, derivative,
  refit_steps, and the model runs
  per solve (`evals_init`, `evals_sampling`: mean, 10/50/90 %, max, n, and `none`, the proposals without a model
  run; `evals_posterior`);
- `blobs`: the same for every blob field over the posterior, NaN ignored, plus `frac_nan`;
- `map`: step, walker, lp, θ (physical), `sampled` (solve mode), log_prior, log_like, log_jac, the per-observable
  terms and all observables;
- `warnings` (including a bisection fallback).

**tracks.npz** (`tracks.load_tracks(run_dir)`):
- `grid_yr`: `track_points` log-spaced times [yr] from t0 to the oldest sample's age, plus the age of every
  sample, so that each track ends on the grid at the state the likelihood compared with the data (before
  2026-10-28 a track ended at the last log-spaced time before its age, up to 150 Myr early at 5 Gyr; on LHS 1140 b
  the outflow H fraction falls by 0.5 dex in that time, so the MAP seemed to miss its datum);
- `theta` (n, ndim) and `free`, physical (in solve mode P* from the blobs: the tracks re-run (P*, φ) without
  solving); row 0 is the MAP (`is_map`). The others are posterior samples drawn without replacement after burn-in
  and thinning;
- `status`, `lp`, `error` per track, and `final` (the end-state blob of each re-run). Each re-run reproduces
  its chain log-probability exactly, which is checked and logged;
- one (n, grid size) array per key of `model.TRACK_KEYS`:
  - `Rp` [R⊕], `fatm`, `Matm` [kg];
  - `Mdot`, `Mdot_H`, `Mdot_He`, `Mdot_HHe` [kg/s], `outflow_H_fraction`, `escape_He_H`, `subcritical`,
    `T_surf` [K];
  - `N_X` and `N_X_int` [atoms] for X in H, He, D, O, C, N, S;
  - `H_He`, `C_O` (= N_C/N_O; inf once all O is dissolved), `D_H`;
  - `n_<mol>_atm` [mol] and `VMR_<mol>` for the 11 molecules, and `mmw`.

  The arrays are float32, except N_X (float64, since atom numbers overflow float32). Values are NaN after a
  sample's age and, for composition, once the atmosphere is gone. The molecule tracks come from the `n_*_a`
  arrays, so they step every n_atmodeller steps. 200 tracks × 300 points take about 5–10 MB.

A resolution directory `<output_root>/<name>/resolution_<stamp>/` holds:
- `config.json`;
- `points.npz`: θ (in solve mode the sampler's coordinates; `free` names them, `physical_free` the physical
  parameters, `solve` says which), the perturbed θ, `theta_shifted` (θ + h e_j; in solve mode (P* + h, φ) of the
  reference cell), the selection candidates (with `candidates_runs`, model runs each) and, for
  `source = "bisect"`, the bisection arrays (`bisect_theta`, `bisect_point`, `bisect_x`, `bisect_g`,
  `bisect_outcome`, `bisect_lp`, `bisect_draws`, `bisect_converged`, `bisect_evaluations`);
- `results.npz`:
  - `blobs` (points, cells, 2), with [..., 1] the perturbed run, and `lp`;
  - the cells and the grid;
  - `d`, `noise` (points, cells, observables), `dlogl`, `rel_diff`;
  - per cell: `worst_d`, `worst_dlogl`, `worst_noise`, `mean_runtime`, `mean_runs` (model runs per evaluation),
    `passes`, `stable`, `verdict`;
  - `reference`, `recommended`, `solve` (a solve-mode study), and in solve mode `solve_shift` (points, cells: P* at
    the cell minus P* at the reference) and per cell `worst_solve_shift`;
  - the equivalent shift: `shift_blobs` (points), `shift_h`, `deriv` (points, observables), `steep`, `shift` and
    `shift_frac` (points, cells, observables), `ridge_dlogl` (points, cells), per cell `worst_shift` (over the
    observables with d > tol), `ridge_spread`, `passes_shift`, `stable_shift`, `verdict_shift`, and `ridge_obs`
    (the steepest observable), `recommended_shift`, `shift_param`, `shift_width`, `shift_tol`;
- `table.txt`, `recommendation.json` (the recommended cell, its `--set` options and run command, every cell's
  verdict, and `recommended_shift` and `equivalent_shift` for the second recommendation) and the two figures.

A scan directory `scan_<stamp>/` holds `config.json`, `scan.npz` (`x_name`, `x`, and `y_name`, `y` for 2-D;
`theta` (n, ndim) in grid order with y slowest, physical also in solve mode, `lp`, `blobs`, `shape`, `obs`,
`free`, and in solve mode `solve_check`, the monotonicity report) and `scan.png`. A
comparison directory `compare_<stamp>/` holds `compare.txt` and the three figures of section 7.

## 9. Performance

All numbers were measured on this machine, an Apple M1 with 4 performance and 4 efficiency cores and 16 GB of
RAM, shared with other work (the load reached 1.3–4.9 busy cores), so they vary by about ±15 %.

| quantity | measured |
|---|---|
| one evaluation, LHS 1140 b, 1e5 steps / n_atmodeller 100 | 0.15–0.16 s (31 MB solution dict) |
| stripped track / O-rich or high f_atm track | 0.01 s / 0.4–0.6 s |
| n_steps 1e6 / n_atmodeller 10 (at 1e5 steps) | 1.4–1.7 s / 1.2–1.6 s |
| worker warm-up (import, Numba and JAX compiles, Atmodeller fallback paths) | 6–8 s cold; 2.4–5 s with the compile caches |
| 6-worker pool ready | 20 s without caches, 7.5 s with; 2 workers 5.5 s |
| throughput, 1 worker / 6 workers | 7.4–8.1 / 22–22.5 evaluations/s |
| emcee, 6 workers, 24 walkers | 0.8–1.1 s/step, about 3,600 steps/h |
| the same in real `run`s of lhs1140b (Rp, Mp / 6 observables) | 0.83 / 0.74 s/step, 4,300 / 4,900 steps/h; about half the proposals land outside the prior and cost nothing |
| peak physical footprint per worker in those runs | 0.55 GB (6 workers 3.3 GB, parent 0.15–0.2 GB); RSS reads less (0.45 GB) when macOS compresses idle pages |
| emcee, 6 workers, 48 walkers | 1.7–2.3 s/step, about 1,950 steps/h |
| memory per worker, 1e5 steps / 1e6 steps | 0.5–0.86 GB / 1.2–1.9 GB |
| parent process | about 0.1 GB (it never imports JAX) |
| `check --bisect` on lhs1140b (in process) | 13 evaluations, 3.0 s (Illinois steps; plain bisection took 16) |
| `check` on lhs1140b in solve mode (in process, 1e5/100) | one cold solve: 15 + 2 model runs (central difference; 15 + 1 with the forward default), 3.6–3.9 s; 7.9–8.1 s and 0.44–0.47 GB for the whole command |
| solve-mode evaluation on lhs1140b, 2e4/100 (in process, 16 points near a chain's walkers) | cold (both prior bounds): 17.8 model runs, 1.2 s; warm (the run's table of 48 solves): 10.6 runs (median 9, max 20), 0.35 s; the same P* to 1e-5 dex |
| `run` lhs1140b in solve mode, 2e4/100, 16 walkers on 2 workers, 10 steps (smoke run) | pool 12.1–12.7 s; initialization 48 cold solves, 807–824 model runs, 14.3–14.8 s; sampling 1.8 s/step, acceptance 0.21, 79 solves at 11.0–11.1 runs (median 9–10, 90 % 18), 81 proposals outside the support, 0.037 s per model run; 50 s in total, parent 0.18 GB |
| `scan`, 41 points, 2 workers | 9.2 s including 5.5 s of warm-up |
| `scan`, 6 workers: 161 / 492 / 972 points (1e5/100) | 18–19 s / 36–46 s / 42–61 s, including 8–12 s of warm-up |
| `run` lhs1140b, 1e5/300, 24 walkers on 3 workers, two runs at once | 0.41–0.43 s/step (60–64 % of the proposals outside the prior), isocalc 0.044–0.046 s/eval; the bisection 1160–1190 evaluations in 51 s; 3.7–3.8 min for 400 steps with 48 tracks |
| bisection of 1200 prior draws, 1e5/300, 6 workers | 19,000–20,400 evaluations in 420–500 s |
| `run` lhs1140b in solve mode, 3e4/100, 24 walkers on 3 workers, two runs at once | 4.3–4.7 s/step (2.4–3 s/step once one run was left), 9.8–9.9 model runs per solve (median 8, 90 % 18, max 41), 0.088–0.095 s per model run, 45 % of the proposals outside the support; 1303 and 1604 steps (N/τ 68 and 83) in 1 h 43 min and 1 h 56 min; 1.44 GB for 3 workers, parent 0.1 GB; 48 tracks 16 s |
| `resolution` lhs1140b in solve mode, 3 points, 11 cells from 3e4/300 to 1e6/100 (69 solves) | 3.4 min on 5 then 3 workers (estimate 2.0–2.3 min); 16.8 model runs per solve; one solve 0.5–0.6 s at 3e4/300, 1.4 s at 3e4/100, 3.9–5.2 s at 1e5/100, 12–13 s at 3e5/100, 27–29 s at 1e6/100 |
| re-solving 240 posterior samples of lhs1140b (the resolution reweighting, cold solves, 17 model runs each) | 71–95 s at 3e4/100 and 285 s at 1e5/100 on 4 workers, 601 s at 3e5/100 on 5 |
| `run` toi406b, 1e5/100, 16 walkers on 2 workers | plain: 0.30–0.47 s/step (3000 steps in 20 min); solve mode on Rp: 4–9 s/step with 5.9 model runs per solve (median 5), 0.19–0.21 s per model run |

The bisection costs at most `bisect_draws` × (2 + `bisect_iters`) evaluations per batch (1728 for 72 draws), and
one pool map per round, so a round waits for its slowest draw. A draw that converges in 14 rounds costs 16
evaluations; draws without a sign change cost 2.

The cost of one evaluation follows t ≈ 120 µs × (n_steps/n_atmodeller + 1) + 0.2 µs × n_steps, within about
10 % (`resolution.cost_model`):

| n_steps \ n_atmodeller | 1000 | 100 | 10 | 1 |
|---|---|---|---|---|
| 1e4 | 0.004 s | 0.017 s | 0.13 s | 1.2 s |
| 1e5 | 0.033 s | 0.15 s | 1.2 s | 12 s |
| 1e6 | 0.32 s | 1.4 s | 12 s | 120 s |

**Why 6 workers.**
- Four performance cores do most of the work. Workers 5 and 6 add 8–18 % throughput over 4, the 6th only 0–2 %
  over the 5th.
- Under load, emcee at 4 and 6 workers was nearly tied (24 walkers: 1.04–1.08 against 0.98–1.05 s/step).
- 6 of 8 cores is the requested ~75 % of the machine.
- Memory: 6 workers at 1e5 steps need about 3–4 GB. At 1e6 steps, use 4 workers or fewer. `choose_workers` cuts
  the count to fit min(`max_memory_gb`, 70 % of the available memory) and says so.
- On macOS, psutil's "available" memory is only free + inactive pages. It leaves out what the system frees by
  compressing idle memory, so with a browser open it reads 2–4 GB on a 16 GB Mac, and the rule above allows only
  2 workers. In that state (2026-10-28: 2.9 GB "available", 16.5 GB of swap in use), 6 workers at 1e5 steps held
  2.5 GB in total and pushed 0.8 GB to swap. `--set parallel.check_available=false` (or the key in `[parallel]`)
  budgets `max_memory_gb` alone, which still limits 1e6-step cells to 4 workers. The two mcmc_results LHS 1140 b
  configs set it, with `max_memory_gb = 6` (6 real workers measured 2.5–3.3 GB).

**Why nwalkers is a multiple of 4 × workers.**
- emcee evaluates the ensemble in sequential groups and waits for the slowest evaluation of each group. Stretch
  and DE use 2 groups; DESnooker forces 4.
- With W walkers and P workers a step costs about groups × ⌈W/groups/P⌉ evaluations, so W should be a multiple
  of 4P with Snooker in the mix (2P otherwise).
- The auto rule takes max(4 ndim, 16) rounded up to that multiple, which gives 24 for 6 workers and 3–6 free
  parameters, 24 for 3 workers and 4 free parameters, and 16 for 1–2 workers and up to 4 free parameters. Walker
  utilization is about 80 % at 24 walkers and 90 % at 48, but only 51 % at 12.
- Small ensembles mix poorly: in an 8-walker test DESnooker split the walkers into groups of 2, and every
  proposal of the first steps landed outside the prior. That is why the rule starts at 16 (8 before
  2026-10-28). Keep `nwalkers = "auto"`.

**Things that were measured to matter, and are set automatically.**
- `NPROC=1` and the other single-thread variables: the default XLA threading cost a third of the throughput at 6
  workers.
- The `spawn` start method: fork deadlocks once JAX is imported.
- Warm-up behind a barrier: a cold worker's first calls cost 6–8 s, plus about 2 s each for the two Atmodeller
  fallback compiles.
- `chunksize = 1`: evaluation times vary 50×.
- The Numba and JAX on-disk caches, in `<output_root>/.cache/`. The Numba cache directory is keyed on a hash of
  every `isofate/*.py` and the numba version, because the kernel inlines `isofunks.py`. Editing any isofate
  module triggers one recompile.

Blobs cost nothing measurable per step. `save_molecules` is off during sampling; tracks turn it on.

## 10. Caveats and known upstream issues

None of these are changed by this tool, except the diffusion-coefficient fix below (made in IsoFATE itself on
2026-10-29, at the user's request).

- **He–D diffusion coefficient with `dynamic_phi = true`: fixed on 2026-10-29.**
  `isofunks.get_binary_diffusion_coeff` sorted each pair by mass, so ('He', 'D') became ('D', 'He'), which was not
  in its table, and fell back to b_H_He: 5.50e21 instead of b_He_D = 2.63e21 at 226 K. Every D–metal and
  metal–metal pair fell back to b_H_He too. The lookup is now symmetric. Pairs without their own coefficient use
  the Genda & Ikoma (2008) prescription the other approximated coefficients in `constants.py` follow, b ∝
  1/√(reduced mass) from b_H_O: D–metal pairs 1.98–2.08e21, metal–metal pairs 0.83–1.07e21 at 226 K. Unknown
  species raise an error instead of falling back. Runs with dynamic_phi = true made before the fix (every
  dynamic_phi = true run of 2026-10-28: LHS 1140 b and TOI-406 b) used the old coefficients, and most of the 74–85×
  dynamic-versus-static D/H difference of section 5 was this bug; do not resume those runs (their earlier steps
  used the old coefficients), start new ones. dynamic_phi = false never used this function; with false, tracks that
  lose H and He keep a frozen O/C/N/S atmosphere (no escape with μ = 0). D/H depends steeply on b_He_D itself
  (`constants.py`, scaled from b_H_D by the reduced mass, in both modes): the isotopic substitution of the measured
  H–He coefficient, b_H_He·√(μ_HHe/μ_HeD) = 8.06e19 T^0.732, is 1.62× larger and lowers D/H 6–13× on the LHS 1140 b
  posterior (5–11× with dynamic_phi = false); the fit does not change.
- **μ is per atom** (`escape_kernel.py:153-156`). R_atm = 9kT/(μg) and the atmosphere descent use the mean mass
  per atom, not per molecule: 1.29 against 2.35 g/mol for protosolar H₂ gas. R_atm is then about 1.8× larger
  than with molecular μ; at the end of a typical LHS 1140 b run (3.39 against 3.83 g/mol) the difference is 13 %.
  This may be intended; it matters for Rp.
- **`flux_model = "Johnstone"`** reads files from a hard-coded `/Users/collin/Documents/Harvard/...` path
  (`isofunks.py:92-96`). Its φ_E omits the 1/(4 V_pot) factor (`isofunks.py:219-220`, `escape_kernel.py:74`), so
  the units are wrong. The config warns; do not fit with it. With `"phoenix"`, the 45 Myr median activity value
  3661 looks like a typo next to 20,000–42,000 (`isofunks.py:148`; a guess).
- **Legacy composition is nonlinear.** The enhanced O mass is taken out of the H budget, so the actual O/H is not
  10^x × protosolar:
  - x = 1 gives O/H 6.4e-3 (target 5.8e-3);
  - x = 2 gives about 0.8 (target 0.058, i.e. 13–15× too high), and He/H rises to 1.31;
  - N_H < 0 above x = 2.03, which is status 4.

  `composition = "ratios"` is linear and exact. The legacy mode refuses log_metallicity and log_CtoH and warns on
  any O/H enhancement.
- **Surface, not photospheric, chemistry.** Atmodeller speciates at the surface (T_surf ≈ 1600–1800 K for
  LHS 1140 b), not at the photosphere (Teq ≈ 226 K). Comparing VMRs with retrievals assumes a well-mixed
  atmosphere with quenched surface chemistry; condensation is not modelled. Prefer the elemental quantities (C/O,
  mmw, H/He) and give VMRs a `sigma_model` of 0.5–1 dex. `T_surf` is the base of the adiabat, capped at 6000 K
  (`atmodeller_coupler.py:120`), and is not directly observable (the config warns). O-rich runs hit the cap.
- **Missing species.** NH₃, H₂S, OCS and HCN are not in Atmodeller's species list (`atmodeller_coupler.py:64-79`),
  so N sits in N₂ and S in S₂/SO₂/H₂SO₄. Atmodeller never sees less than 1e9 kg of an element, which sets a floor
  of about 1e9 kg/M_atm on trace VMRs in very thin atmospheres.
- **D is lumped into H in Atmodeller.** D/H therefore comes from the escape-kernel rows, not from the molecules.
- **He has no molecule key in `atmodeller_final`.** It comes from conservation, which matched N_He/N_A to 1.4e-5.
  When the atmosphere is exhausted, `atmodeller_final` is all NaN and lacks `log10dIW_1_bar`.
- **C/O from Atmodeller and from the kernel differ slightly.** They agree to 0.03 % at 1e5/100 but differ by 2 % at
  2000/100: the gap scales with n_atmodeller × dt, the staleness of the last regular Atmodeller split. Without
  Atmodeller, C_O falls back to the kernel N_C/N_O. If `atmodeller_final` is non-finite (a silent
  non-convergence), VMRs and mmw are NaN.
- **The time convention.** isocalc's time grid ends one step past the age (t = dt·(1 … n+1) + t0), so final
  values refer to age·(1 + ~1/n_steps), and tracks stay finite up to age + dt.
- **Memory.** isocalc allocates its molecule arrays even with save_molecules = False. A solution dict is 31 MB at
  1e5 steps and 312–480 MB at 1e6, which is why only final scalars travel between processes.
- **Φ_He = 0 is physics.** He stops escaping once the flux is subcritical: in one LHS 1140 b run after 65 Myr.
  With `step_fn = true`, t_sat = 3.1 Gyr and ages ≥ 3.5 Gyr, every H-rich track has Φ_He = 0 today, so its outflow
  H fraction is 1.
- **The outflow fraction is steep.** On the LHS 1140 b island log10 x changes by 190 per dex of log_f_atm (200–1400
  at the bisection points of the posterior) and by up to 1 dex between 1e5/100 and 1e6/10 (section 6). Its
  posterior is a curved ridge 0.0026 dex thick in log_f_atm, on which emcee's moves in log_f_atm mix far too
  slowly (acceptance below 1 % in 400 steps, section 5); solve mode samples it.
- **Solve mode assumes one root per bracket** (section 5). With several crossings of the u level along the solved
  parameter it finds one of them and the density misses the others; `scan` along the solved parameter reports it.
  The outflow fractions count as undefined for the root finder where H and He carry less than half of the
  escaping mass (the metal-dominated wind at the stripping edge with dynamic_phi), so that root is never used; the
  likelihood there is unchanged (on LHS 1140 b about lp −1860). Defining the outflow fractions as NaN there in
  `model.extract` itself would make the two modes agree exactly but change the definition of the observable (SPEC
  amendment 1, A1); the user decided (2026-10-28) to keep the definitions and apply the rule in the root finder
  only.
- **Solve mode cannot sample a plateau of the solve observable.** Where O does not change with P over a stretch
  of P's prior, that whole stretch maps to one value of u, so the change of variables gives it no mass: the
  posterior there is silently left out. The common case is Rp along log_f_atm on a planet that can lose its
  atmosphere: every log_f_atm below the stripping edge gives Rp = the bare-core radius (Mp/M⊕)^¼ R⊕. If the bare
  core fits the radius, plain sampling puts mass on it and solve mode does not. On TOI-406 b 82 % of the plain
  posterior was stripped; solve mode on Rp sampled only the planets with an atmosphere, and agreed with the plain
  run's samples that have one (section 5, "TOI-406 b"). The tool points it out three ways: the config warns when
  the bare core over the Mp prior lies inside the u range (solve mode on Rp along log_f_atm, without Rp_fixed);
  `check` prints a note when its solve evaluates the same value of O at several P inside the u support; and `scan`
  along the solved parameter lists such plateaus. On LHS 1140 b nothing is lost: the plateaus of the outflow
  fraction (x = 1, the 1e-15 floor) lie outside the u range, and a stripped planet has no wind (lp = −inf).
- **The MAP in solve mode** is the maximum of the sampled density p(u, φ) = prior · |∂P*/∂u| · L, so it includes
  ln|J|, and it is not the maximum of prior · L over the physical parameters. `summary`, `tracks` (row 0) and
  `tracks.png` use it. On LHS 1140 b the two were close (log_like −13.04 at the MAP against −13.038 at the best
  physical point of the chain); where O is flat in P, ln|J| is large and the MAP drifts toward that region (on
  TOI-406 b ln|J| reached +0.9 there).
- **`tracks.png` with a free age.** The bands are time slices over the tracks still running at each time. Near
  the end they mix older planets that are still in an earlier state with others at their end: on LHS 1140 b the
  band median of the outflow fraction is about 0.3 at 8 Gyr, while every track ends near the datum, 1e-3. Compare
  the data with the ends of the individual tracks and of the MAP; since 2026-10-29 each track ends on the time
  grid at its own age (section 8).
- **The solve-mode Jacobian is a finite difference**, and the model has numerical structure at the 1e-5 dex
  scale, so ln|J| carries an error of about 0.01–0.02 (the target was 0.01). On LHS 1140 b at 1e5/100 (5 points
  across Mp, age and eps): central differences with steps of 3e-6, 1e-5 (the tool's) and 3e-5 of the prior width
  agree to 0.002–0.037 in ln|J| (median 0.014); the two one-sided slopes differ by 0.4–2.4 %; a root moved by the
  solve tolerance (what a warm start may return instead of a cold one) changes ln|J| by 0.003–0.010. At 2e4/100
  the spread was 0.004–0.014, warm and cold roots 1e-5 dex apart differed by up to 0.021, and steps of 1e-4 of
  the width by 0.2 (structure on the 4e-4 dex scale). A density error of 1–2 % is far below the Monte Carlo error
  of any run; it is deterministic, so it does not break the Markov chain.
- **The step-function XUV and the age prior.** With `step_fn = true` ages below t_sat still see the saturated flux
  F0 today (3.5× F_final on LHS 1140 b). When t_sat lies inside the age prior the config warns: those ages can form
  a separate, narrow mode that prior draws rarely hit. On LHS 1140 b the 28 Myr below t_sat = 3.128 Gyr held the
  best fit of the old prior (lp −11.2 at eps ≈ 0.2, where the Mdot term is −0.09; section 5); its age prior now
  starts at t_sat (section 4).
- **The toy backend is not physics.** It is for testing the plumbing; its Rp carries a deliberate
  resolution bias of 1e3/n_steps, it ignores dynamic_phi, its radius jumps from R_core × the bias to the bare core
  when the atmosphere is lost, and its outflow never becomes He-dominated.

**What was wrong with `isofate_MCMC.py`, and what replaced it here.**
- The flux-ratio likelihood with a 1e-20 floor on Φ_He gave ratio ≈ 1e36 and χ² ≈ 1e78 for every H-rich track
  (emcee then compared meaningless differences between enormous numbers), gave a fully stripped track (no wind)
  ratio 0 and χ² ≈ 3, so a planet with no atmosphere "fitted" a detected He wind, and, as a side effect, pulled
  the walkers toward the H-depleted island by minimising Φ_H. Here the outflow H fraction is finite (1) for an
  H-only wind, undefined (−inf for a detection) without a wind, limits, log10 spaces, floors and NaN rules are
  explicit, and the guidance toward the island is the initialization (solved draws in solve mode, otherwise the
  bisection).
- With n_atmodeller > 0 and save_molecules = False every evaluation silently raised KeyError and returned −inf.
  Molecules now come from `atmodeller_final`, and a preflight evaluation fails loudly.
- n_atmodeller = 0 meant the run was not coupled at all. The fiducial is now 100, and 0 gives a warning.
- Mdot was phi·A, the available flux, instead of Σ Φ_X μ_X A.
- All walkers started in a 1e-4 ball, including age in years. Walkers now start from a prescan, a bisection or
  solved draws.
- There were no convergence checks, no checkpoints, no seed, and every exception was swallowed. Now: τ-based
  stopping, burn-in and thinning; atomic checkpoints with bit-identical resume; a seeded RNG; status codes; and
  tracebacks for bugs.
- `Pool()` used 8 workers including the efficiency cores, ran without NPROC=1 and with 12 walkers, so 2 workers sat
  idle. See section 9 for what replaced it.
- Configuration lived in module globals, and isocalc mutated mantle_iron_dict in place. The config is now passed
  explicitly, and each call gets its own copy.
- Blobs were a pickled object array. They are now a structured float array in npz with a JSON config, an
  automatic run name, and runs are never overwritten.

## 11. Python use, tests, layout

```python
import numpy as np
from isofate.mcmc import config, sampler, tracks, resolution, plots, scan, search, solve

cfg = config.load_config('/Users/collin/Documents/Chicago/isofate_nick/mcmc_results/configs/lhs1140b.toml',
                         ['model.n_steps=3e4'])
run_dir = sampler.run_mcmc(cfg, workers = 6, max_steps = 2000)   # needs an `if __name__ == '__main__':` guard
run = sampler.load_run(run_dir)          # cfg, chain, log_prob, blobs (structured), accepted, summary, ...
samples, blobs = sampler.posterior(run)  # after the summary's burn-in and thinning
rp = blobs['Rp']                         # any blob field, e.g. 'outflow_H_fraction', 'exhausted', 'status'
tr = tracks.load_tracks(run_dir)         # tr['grid_yr'], tr['Rp'] (n, grid size), ...
data = scan.load_scan(SCAN_DIR)          # data['x'], data['lp'], data['blobs'], data['cfg']
out_dir, paths = plots.compare_runs([RUN_A, RUN_B], labels = ['dynamic', 'static'])

from isofate.mcmc import model
result = search.bisect_points(cfg, model.LocalPool(cfg), 8, np.random.default_rng(1))  # in this process
points = result['point'][search.good(result)]                                       # the converged points

physical = sampler.physical_samples(samples, blobs, cfg)  # solve mode: P* from the blobs in place of u
res = solve.evaluate([np.log10(0.00115), 5.6, 7.6e9, 0.15], cfg, history = True)   # one solve in this process
print(res['solve_value'], res['log_jac'], res['solve_evals'])
```

Scripts that start pools (`run_mcmc`, `run_resolution`, `run_scan`, `compute_tracks`) must protect their entry
point with `if __name__ == '__main__':` because workers are spawned. Without the guard each worker re-runs the
script. Notebooks work for reading runs and plotting. Do not import `isofate.isofate_coupler` in a process that
then starts a pool: the pool always uses spawn, but the parent would carry an extra 0.6 GB of JAX. `search` and
`model.LocalPool` run the model in the calling process, so they do import it.

**Tests:**

```bash
.venv/bin/python -m pytest tests/test_mcmc_*.py -q -x                 # about 2 min
.venv/bin/python -m pytest tests/test_mcmc_*.py -q -x -m "not slow"   # without the real-isocalc tests
```

The sampler, search, solve, scan, compare, robustness and resolution tests use the toy backend in one process or
with 2 spawned workers. `test_mcmc_solve.py::test_thin_ridge_posterior` (25 s) is the check of the solve-mode
Jacobian: the marginal of the solved parameter against a brute-force grid posterior (section 5). The real-model
tests (marked slow) run isocalc in one process at n_steps = 2000; they include the outflow observables (x = 1 for
an H-only wind, NaN without a wind) and one solve on LHS 1140 b. The `slow` marker is registered in
`pyproject.toml` (`[tool.pytest.ini_options]`).

**Layout of `isofate/mcmc/`:**

| module | role |
|---|---|
| `__main__.py` | the command line |
| `config.py` | TOML loading, validation, the PARAMS/OBSERVABLES registries, `dump_toml`, resolved JSON |
| `params.py` | priors, Latin-hypercube prior draws, θ ↔ parameters, jitter |
| `likelihood.py` | the terms of section 3 |
| `model.py` | isocalc inputs, compositions, observables of the final state, blobs, status codes, worker functions |
| `env.py` | single-thread environment, compile caches |
| `pool.py` | spawn pool with warm-up barrier, memory-aware worker count and dead-worker detection |
| `sampler.py` | the run: initialization (prescan, bisect, solve, prior, ball; safe jitter), sampling loop, checkpoints, resume, convergence, summary |
| `search.py` | the root finder (bisection, Illinois, warm-start expansion) and the bisection initializer (init = "bisect", resolution source = "bisect", check --bisect) |
| `solve.py` | solve mode: the solved evaluation with its Jacobian, the warm-start table |
| `scan.py` | grid scans (the scan command) |
| `tracks.py` | posterior evolutionary tracks |
| `resolution.py` | the resolution study, with the equivalent shift |
| `plots.py` | all figures (Agg `Figure` objects, PNG at 150 dpi), scans and run comparisons |
| `toy.py` | the fast fake isocalc for tests |
| `configs/` | `template.toml`, `lhs1140b.toml` (solve mode on the outflow H fraction) and `toi406b.toml` (Rp and Mp only) |
