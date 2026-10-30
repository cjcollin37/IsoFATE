# isofate.mcmc

isofate.mcmc fits coupled IsoFATE–Atmodeller evolutionary tracks (`isocalc` in isofate_coupler.py) to the
observables of one planet with emcee (Foreman-Mackey et al 2013). Each sample is a whole evolution, from the
initial atmosphere to today, so the result is a posterior over evolutionary histories. A TOML config sets the star,
the planet and the priors, and switches each observable on or off: radius, mass, escape rates, the outflow
composition, atmospheric ratios and abundances. The walkers run in parallel worker processes, and runs checkpoint
and resume. Solve mode handles posteriors that sit on a thin ridge, and a resolution study picks `n_steps` and
`n_atmodeller`.

## Quick start

isofate.mcmc comes with IsoFATE, whose dependencies include emcee, corner and psutil. I strongly recommend uv: in a
clone of the repository, `uv sync` sets up `.venv` with everything. That should do it!

Every command is `python -m isofate.mcmc COMMAND` (with `.venv/bin/python`, or the environment active), and
`--help` works on each one. From the repo root, with the config in a results folder next to the repository:

```bash
mkdir -p ../mcmc_results/configs; C=../mcmc_results/configs/my_planet.toml
python -m isofate.mcmc template $C                       # the commented template: every option and its default
python -m isofate.mcmc check $C                          # one evaluation here: model vs data, warnings, cost
python -m isofate.mcmc scan $C --x log_f_atm=-4:-1:61    # the model along one parameter (--y for a second)
python -m isofate.mcmc resolution $C --grid quick --dry-run    # the plan of a resolution study and its cost
python -m isofate.mcmc resolution $C --grid quick --points 4   # choose n_steps and n_atmodeller
python -m isofate.mcmc run $C --workers 6                # the MCMC
python -m isofate.mcmc summary RUN_DIR                   # the posterior (also while the run is going)
python -m isofate.mcmc plot RUN_DIR                      # redraw the figures
python -m isofate.mcmc tracks RUN_DIR --n 100            # re-run posterior samples as evolutionary tracks
```

`run` writes the summary, 200 posterior tracks and the figures into RUN_DIR, the directory it prints. `check`
evaluates at the params' values, or elsewhere with `--theta NAME=VALUE`.

**Where results go.** Every run gets a new directory, `<output_root>/<name>/mcmc_<YYYYmmdd-HHMMSS>/`, with scans,
studies and comparisons beside it. `output_root` is relative to the config file (default: `./mcmc_results` in the
working directory). `template` sets it to ".." when it writes into a `configs/` folder, so the runs of the config
above land in `../mcmc_results/my_planet/`. The config warns when results would land inside the repository.

Ctrl-C saves the chain and prints the `run --resume RUN_DIR` command that continues it, bit for bit.
`--set section.key=value` overrides any option (TOML values). `synth $C --theta NAME=VALUE --out OUT.toml` writes a
copy of the config whose data are the model prediction at θ, for injection–recovery tests, and
`--set model.backend=toy` swaps isocalc for a fast stand-in to try the plumbing (not physics).

## How it works

One evaluation at θ is one isocalc run. θ gives the inputs (the initial atmosphere's mass and composition, the XUV
history, ...). isocalc integrates the planet from `t0` to its age in `n_steps` steps, and Atmodeller
re-equilibrates the atmosphere with the magma ocean every `n_atmodeller` steps. The observables at the last step
give the log-likelihood. `Mp`, `age` and `Teq` are inputs, compared with their data before the model runs.

Each evaluation also returns a blob, a row of numbers kept with the chain: the status, the runtime, `log_prior`,
`log_like`, every observable (active or not), the extras `exhausted`, `t_lost`, `Matm`, `subcritical` and
`n_atmod_calls`, and in solve mode `solve_value`, `solve_evals` and `log_jac`, so the posterior of any model output
comes for free. The status is 0 ok, 1 outside the prior, 2 a model failure, 3 log-likelihood −inf, 4 invalid inputs
or 5 a bug (traceback in `RUN_DIR/worker_errors/`; the run stops once there are more than
`max_unexpected_errors`).

The walkers are evaluated in a pool of worker processes. They are spawned, not forked, because JAX starts threads
when it is imported and forking a process with live threads can deadlock; the parent never imports JAX and stays at
about 0.1 GB. Each worker is single-threaded (`NPROC=1` and its relatives; XLA's default threading cost a third of
the throughput of 6 workers) and compiles the model once before sampling: 6–8 s, or 2.4–5 s with the compile
caches in `<output_root>/.cache/`. A worker needs about 0.75 GB plus 1.1 GB per 1e6 steps, and the worker count is
cut to fit the memory budget. A dead worker stops the run within 2 s, with the chain saved. Checkpoints of
`chain.npz` (atomic, every `checkpoint_every` steps but at most once a minute, and at every stop) hold the chain,
the blobs, the random-number state and the bookkeeping, so a resume is bit-identical.

## The config file

A config is a TOML file with the tables below and the top-level keys `name` (default: the file stem; used in
paths), `description` and `output_root`. The template lists every option with its unit; values are converted to SI
inside, but parameters and observables keep the units of their tables (Mp in M⊕, age in yr, Mdot in kg/s). A
misspelt key is an error that names the closest one. The examples in `isofate/mcmc/configs/` are `lhs1140b.toml`
(see "Example: LHS 1140 b") and `toi406b.toml`, a planet with only a radius and a mass.

### [star] and [planet]

- `[star]`: `mass` [M☉] (required), `radius` [R☉], `teff` [K] and `luminosity` [L☉] (default 4πR²σTeff⁴ from
  radius and teff; one or the other is needed).
- `[planet]`: `period` [days] or `semi_major_axis` [AU] (default from Kepler's law; a warning when both are given
  and disagree by more than 1 %), and the bolometric insolation `Fp` [W/m²] (default L/4πd²).

### Free parameters: [params.NAME]

```toml
[params.log_f_atm]
vary = true
bounds = [-5.0, -1.3]         # a uniform prior
value = -3.0                  # the point of `check`, and the centre of init = "ball"
```

Keys: `vary` (default false: fixed at `value`), `value`, `prior` ("uniform", "loguniform" or "normal"),
`bounds = [lo, hi]` (required for uniform and loguniform, an optional truncation for normal), `mu`, `sigma` and
`init_scale` (the spread of init = "ball"; default 1 % of the box). A free parameter's value defaults to mu or the
midpoint of its bounds (geometric for loguniform). A bare number under `[params]`, e.g. `F_final = 0.17`, fixes a
parameter, and the others keep the defaults below. The free ones, in this order, form θ.

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
| `t_pms` | yr | 0 | >= 0 | end of the pre-main-sequence XUV boost (0: no boost) |
| `pms_factor` | - | 0 | >= 0 | XUV boost at t0, falling to 1 at t_pms (0: none; must be > 0 when t_pms > t0) |
| `beta` | - | -1.23 | any | XUV power-law index (step_fn = false) |
| `albedo` | - | 0 | [0, 1) | Bond albedo, T = EqTemp(Fp, A) |
| `Rp_fixed` | R_earth | unset | > 0 | constant radius (isocalc Rp_override); needs rad_evol = false, which is then set. Unset: off |
<!-- END params table -->

With `step_fn = true`, eps and F_final are degenerate today (the escape scales as eps·F_final) but not over the
history: F_final acts only after t_sat, and the radiation-recombination limited branch ignores eps. The
pre-main-sequence XUV boost is off by default (`t_pms = 0`, `pms_factor = 0`); with `t_pms > t0` the flux starts
at pms_factor × F0 at t0 and falls as a power law to F0 at t_pms, and pms_factor must then be > 0. `Rp_fixed`
(unset: off) gives isocalc a constant radius and switches `rad_evol` to false, as isocalc applies `Rp_override`
only without radius evolution; the `Rp` observable then acts as a prior on it.

### Observables: [observables.NAME]

Keys: `use` (default false; only active observables enter the likelihood), `value` (in the unit of the table, also
in log10 space), `sigma` (a number, or `[lo, hi]` for asymmetric errors; in dex in log10 space), `space` ("linear",
the default, or "log10"), `kind` ("detection", the default, "upper" or "lower"), `floor` (overrides the log10 floor
of the table) and `sigma_model` (default 0). The H+He mass-loss rate of LHS 1140 b, for example, is `value = 2.03e5`,
`sigma = [0.137, 0.121]` and `space = "log10"`. For a model value m, a datum v and an effective sigma s, a
detection adds −½((m − v)/s)² to the log-likelihood, an upper limit at v adds log Φ((v − m)/s) and a lower limit
log Φ((m − v)/s), with Φ the normal CDF (scipy's `log_ndtr`, accurate far into the tails).
- **log10 space** suits quantities known to a factor, such as escape rates and abundances: a Gaussian in dex is a
  lognormal in the value. The model is first clipped to [floor, ceiling], so a value of 0 (He not escaping, a
  stripped atmosphere) becomes log10 of the floor, far below any detection but consistent with an upper limit above
  the floor. Dimensionless observables also have a ceiling, 1/floor, so a ratio x/0 stays finite.
- **Asymmetric sigmas** give a split normal (hi above the datum, lo below). **sigma_model** adds a model error in
  quadrature: e.g. 0.05–0.1 R⊕ on Rp, or 0.5–1 dex on surface VMRs against retrievals.
- **Undefined values** (NaN: a stripped atmosphere's composition, the outflow fraction when nothing escapes) rule
  out a detection and a lower limit and satisfy an upper limit; +inf satisfies only a lower limit.
- **Inputs are priors.** `Mp`, `age` and `Teq` = EqTemp(Fp, albedo) are isocalc inputs, not outputs, so comparing
  them with data is the same as a prior on the parameter (the `Mp` observable equals a normal prior on
  `params.Mp`). Mp must stay inside the Atmodeller melt grid, [1, 21] M⊕, unless `melt_fraction_override` is set.
  A literature Teq usually assumes albedo 0, so with a free albedo (as in the template and the examples) leave
  `Teq` off unless it is a measured temperature: it would pin the albedo near 0.

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

Every observable is its value at the last step; molecules come from the final Atmodeller call, and atmospheric He
from conservation. The legacy `escape_H_He` and `escape_He_H` are ill-defined where a flux is 0.

### [model]

- `n_steps = 100000`: time steps of a run; at most 1e6 unless `allow_large_n_steps = true` (default false).
- `n_atmodeller = 100`: steps between Atmodeller calls; 0 switches the coupling off (escape only), with a warning.
- `mechanism = "XUV"`: or "CPML", "XUV+CPML", "phi kill". `flux_model = "power law"`: or "phoenix", "Johnstone".
- `stellar_type = "M1"`: for "Johnstone" only. `activity = "medium"`: or "low", "high", for "phoenix" only.
- `step_fn = false`: true drops the XUV flux to `F_final` at `t_sat` instead of following the power law.
- `RR = true`: radiation-recombination limited escape (Murray-Clay et al 2009).
- `rad_evol = true`: the radius evolves (Lopez & Fortney 2014). `thermal = true`: with the age term of that radius.
- `dynamic_phi = true`: fractionate the two most abundant species; false always uses H and He.
- `t0 = 1e6`: start of the run [yr]; `rho_rcb = 1.0`: CPML density at the radiative-convective boundary [kg/m³].
- `melt_fraction_override = false`: the melt fraction from the grid (1–21 M⊕), or a value in (0, 1].
- `mantle_iron = false`: or `{type = "static" | "dynamic", Fe_mass_fraction = 0.1}`: mantle Fe reacts with O2.
- `composition = "ratios"`: number ratios to H: protosolar He/H, solar D/H, and O, C, N, S
  protosolar × 10^log_metallicity, times 10^log_OtoH_enhancement on O and 10^log_CtoH_enhancement on C; linear and
  exact. "legacy" is the recipe of isofate_MCMC.py (see "Caveats").
- `backend = "isocalc"`: or "toy", a fast fake model for testing the tool.

### [sampler] and [parallel]

`[sampler]`:
- `nwalkers = "auto"`: max(4 ndim, 16) rounded up to a multiple of 4 × workers (2 × workers without DESnooker),
  e.g. 24 on 6 workers, as emcee maps the walkers in groups and waits for the slowest of each; or an even
  number ≥ 2 ndim.
- `max_steps = 5000`: the most steps of a run, counted from its start; `checkpoint_every = 50`, `print_every = 10`.
- `moves = "de"`: DE (ter Braak 2006) 0.7, DE with γ0 = 1 (mode jumps) 0.1 and DESnooker (ter Braak & Vrugt 2008)
  0.2, which beat emcee's stretch move (Goodman & Weare 2010) on bimodal and correlated test targets (τ 19 against
  53 steps); or "de_only" (DE 0.9 + DE(γ0 = 1) 0.1) or "stretch".
- `seed = 12345`; `init = "prescan"` ("solve" in solve mode) with `prescan_factor = 8`: see "Starting the walkers".
- `bisect_param = "log_f_atm"`, `bisect_observable = "Rp"`: what init = "bisect" moves, and the observable it
  brings to its datum, with `bisect_draws = "auto"` (3 × nwalkers) draws per batch, `bisect_iters = 22` rounds at
  most and `bisect_tol = 0.1` σ; `bisect_fallback = true` falls back to "prescan" when too few converge.
- `reset_stuck_at = 0`: the step at which stuck walkers are replaced by copies of good ones (0: never).
- `stop_on_convergence = true`: stop at convergence, checked every `check_every = 100` steps with `n_tau = 50.0`,
  `tau_rtol = 0.01`, `tau_checks = 2` and `min_steps = 200` (see "Running, resuming and convergence").
- `burn_factor = 3.0`: discard burn_factor × max τ steps (at most half the chain). `tracks = 200`,
  `track_points = 300`: posterior tracks at the end (0: none) and their time points.
- `max_unexpected_errors = 10`: stop once more evaluations than this hit a bug (status 5).

`[parallel]`:
- `workers = 6` and `max_memory_gb = 8.0`: worker processes, cut to fit this memory budget [GB].
- `check_available = true`: also fit 70 % of the memory the system reports available, which on a Mac with a browser
  open is often only 2–4 GB (macOS counts compressible memory as used); false budgets max_memory_gb alone.
- `cache = true`: compile caches in `<output_root>/.cache/`. `warmup = true`: compile before sampling.

### [solve]

- `param = ""`: the free parameter to solve for, e.g. "log_f_atm"; "" is off.
- `observable = "Rp"`: the active observable, computed by the model (not an input), whose model value is sampled
  in its place; meant for a detection.
- `range_sigma = 6.0`: u is sampled within range_sigma sigmas of the datum on each side.
- `tol = 0.01` and `max_evals = 40`: a solve stops within tol × σ of u, after at most max_evals model runs.
- `warm_start = true`: start each solve from a table fitted to earlier solves;
  `refit_steps = [25, 50, 100, 200, 400]`: refit it to the chain after these steps (`[]`: never).
- `derivative = "forward"`: the Jacobian's finite difference, "forward" (one more model run) or "central" (two).

### [resolution]

- `n_steps = [10000, 30000, 100000, 300000, 1000000]` and `n_atmodeller = [1000, 300, 100, 30, 10, 3, 1]`: the
  default grid (`--grid quick` is 1e4–3e5 × 1000, 100, 10).
- `points = 6` test points from `source = "auto"`: "solve" in solve mode, "bisect" when init = "bisect", else
  "prescan"; or "prior", or "chain:RUN_DIR" (posterior samples of a run).
- `tol = 0.1`, `dlogl_tol = 0.5`: a cell passes when every observable is within tol × σ and log L within dlogl_tol.
- `perturb = 1e-6`: the step of the noise-floor runs (of the prior width). `shift_tol = 1e-3`: the largest
  equivalent shift (of the prior width) of the second recommendation.
- `max_eval_seconds = 300.0`: skip cells whose estimated cost per model run is above this.

## Choosing n_steps and n_atmodeller

Numerical error comes from the time step, (age − t0)/n_steps, and from how often Atmodeller is called; a model run
costs about 120 µs per Atmodeller call plus 0.2 µs per step (0.15 s at 1e5/100, 12 s at 1e5/1). The study measures
that error in units of the data's sigmas and picks the cheapest cell where it is negligible.

```bash
python -m isofate.mcmc resolution $C --grid quick --dry-run      # the plan and its estimated time
python -m isofate.mcmc resolution $C --grid quick --points 4     # n_steps 1e4-3e5 x n_atmodeller 1000, 100, 10
python -m isofate.mcmc resolution $C --points 3 --n-steps 3e4,1e5,3e5,1e6 --n-atmodeller 300,100,30
```

**What it runs.** Test points (`--source`; by default where the walkers start: solved draws in solve mode, the
bisection with init = "bisect", else the best of a prescan) run at every cell of the grid and are compared with the
reference, the finest cell. For each active observable, d = |T(cell) − T(reference)|/σ in the likelihood's space is
the error in units of the measurement error (inf when the observable is defined at one resolution only). A cell
**passes** when d ≤ `tol` and |Δ log L| ≤ `dlogl_tol` at every point (in solve mode Δ log L includes ln|J|), and is
**stable** when every finer cell passes too; the **recommendation** is the fastest stable cell.

**The equivalent shift.** A steep observable can fail at fixed θ although its error does not matter. At the
LHS 1140 b island without solve mode, log10 of the outflow H fraction moves by 0.34 dex (1.3 σ) between 1e5/100 and
1e6/10, but with d log10 x/d log_f_atm ≈ 190 that is a shift of log_f_atm by only 0.0018 dex. So each point also
runs at θ + h e_j, with h = 1e-4 of the prior width W_j of p_j (`solve.param` in solve mode, else `bisect_param` if
free, else the first free parameter), and each error becomes a shift of p_j. The **shift-based recommendation** is
the fastest cell that is stable when each observable passes with d ≤ tol or a shift ≤ `shift_tol` × W_j, and the
ridge weights hold: with the steepest observable moved back onto the reference, the other terms differ between the
points by at most dlogl_tol. Use it when the posterior is much wider than shift_tol × W_j along the ridge. In solve
mode, where each cell solves again at fixed (u, φ), it is a cross-check.

**The noise floor.** Each run is repeated with p_j (in solve mode u) moved by `perturb` of its prior width: what
changes beyond the smooth trend is numerical noise, which above tol makes the likelihood rough at any resolution.

**table.txt** (also printed) shows the grid of the worst d per cell with the time per evaluation (`S` stable, `p`
passes but a finer cell fails, `x` fails, `R` reference, `-` skipped, `[ ]` recommended), the recommendation with
the `--set` options and run command to use it, the shift grid with its derivatives (`*` steep) and the SHIFT-BASED
RECOMMENDATION, the noise floor, the warnings and the details; `resolution_heatmap.png` draws the grid,
`resolution_convergence.png` the worst d against the Atmodeller calls per run.

**Cost.** `--dry-run` prints the plan (cells × points × 2 runs, the second for the noise floor, plus a shift run
per point and the test-point selection) and its time, good to 2×. In solve mode each evaluation is a solve of about
15 model runs, so keep the grid small; cells above `max_eval_seconds` are skipped. Rerun it for a new observable.

**Re-check on the posterior**, on the cells in question, with 30 or more of its samples (on LHS 1140 b, 6 passed a
cell that 30 failed), and read a failure as the size of a reweighting, not a verdict (see "Caveats"):

```bash
python -m isofate.mcmc resolution $C --source chain:RUN_DIR --points 30 --n-steps 1e5,3e5,1e6 --n-atmodeller 100
```

## Starting the walkers

emcee's moves combine walker positions, so no move finds a mode without walkers, and `sampler.init` sets the start:
- `"prescan"` (the default without solve mode): `prescan_factor` × nwalkers Latin-hypercube prior points, and
  nwalkers of the best 2 × nwalkers at random. Each start is jittered by 1e-3 of the prior box; the jitter is
  halved, up to 8 times, while a start loses more than 5 in lp, so starts on a thin ridge stay on it.
- `"prior"`: random prior draws with a finite posterior; `"ball"`: the params' values plus noise of `init_scale`.
- `"bisect"`: prior draws moved along `bisect_param` (default log_f_atm) until `bisect_observable` (default Rp)
  matches its datum, to put walkers on a thin surface that no prior draw hits by chance.
- `"solve"` (the default in solve mode): u drawn from the datum's distribution, the others from the prior, each
  solved (see "Solve mode"). In solve mode "prescan", "prior" and "ball" use the sampler's coordinates.

**The bisection.** For each draw the root finder evaluates both prior bounds; a sign change of
g = T(model) − T(datum) brackets a root, which Illinois (regula falsi) and bisection steps narrow
until |g| ≤ `bisect_tol` × σ. An undefined end (NaN, e.g. no wind from a stripped planet) counts as lying beyond
the datum, and the outflow fractions count as undefined in the metal-dominated wind at the stripping edge (H and He
below half of the escaping mass), where their ratio swings through any value. Batches of `bisect_draws` run, at
most 3, until nwalkers converge, and the walkers start, unjittered, from nwalkers of the best 2 × nwalkers
(`bisect.npz` keeps every draw). If too few converge, the message counts why, and `bisect_fallback` falls back to
"prescan" with a loud warning. `check $C --bisect` prints one bisection round by round.

The default, Rp along log_f_atm, suits most planets: the radius grows with the atmosphere from the bare core up. On
LHS 1140 b only a pure H wind reaches the radius datum, so its config bisects on the outflow H fraction
(`--bisect-observable`, `--bisect-param`). Starting on a thin ridge is not sampling it, though: see "Solve mode".

## Solve mode

**Why.** One precise datum can pin a parameter to a thin, curved ridge. On LHS 1140 b only very H-poor,
He-dominated outflows reproduce the outflow H fraction: at 7.6 Gyr and eps 0.15, x climbs from 1e-13 at
log_f_atm = −3.08 to 3e-3 at −3.015, so its 1σ band is about 0.003 dex of a 3.7 dex prior, and it moves with age,
mass and eps. emcee's moves combine walker positions, which stay on such a ridge only when the walkers are very
close: plain sampling accepted under 1 % of its proposals there (τ would have been 1000–2000 steps). Solve mode
samples the observable instead. In that coordinate the ridge is smooth and wide, and the walkers accept 20 % of
their proposals (τ of 14–18 steps on the final LHS 1140 b chain).

```toml
[solve]
param = "log_f_atm"
observable = "outflow_H_fraction"
```

**How.** With P = `param` and O = `observable`, the sampler moves in u = T(O), O's model value in its likelihood
space (here log10 x, named `u_outflow_H_fraction`), in place of P. For a proposal (u, φ), φ being the other
parameters, each evaluation finds the P* inside P's prior with T(O(P*, φ)) = u, to `tol` × σ, with the root finder
of the bisection, and the model run at P* gives every observable. The density is the change of variables

    p(u, φ) = prior_P(P*) prior_φ(φ) |∂P*/∂u| L(P*, φ),    ∂P*/∂u = 1/(∂T/∂P at P*)

The Jacobian keeps the posterior over (P, φ) unchanged: where O is steep in P, a short stretch of P covers a wide
range of u, and without |∂P*/∂u| that stretch would be weighted by its length in u instead of in P. Without it, a
test on a thin toy ridge misses a brute-force grid posterior by 6–14 standard errors.
- ∂T/∂P is a finite difference (step 1e-5 of P's prior width): "forward" costs one more model run, "central" two
  (their ln|J| differ by 0.01 on LHS 1140 b). A slope of 0, of the wrong sign or undefined gives lp = −inf.
- u stays within `range_sigma` (6) sigmas of the datum on each side, which drops only points whose O term is below
  −18 and keeps the sampler off the plateaus of T outside it (x = 1, the 1e-15 floor). A (u, φ) with no root in P's
  prior is outside the prior. Solve mode is meant for detections: on a limit the range is one-sided.
- **One root per bracket.** The root finder returns one root; if O(P) crosses u several times, the density should
  sum over them. A `scan` along the solved parameter reports where T is not monotonic inside the u range.
- **Warm starts.** A cold solve from both prior bounds takes about 15 model runs. The parent fits a table of P*
  over the sampler's coordinates to the solved draws (`solve_table.npz`) and refits it to the chain after each step
  in `refit_steps`; solves then start from its guess, which changes the path, not the root. On LHS 1140 b a warm
  solve takes about 5 model runs (median 4) once the table is refitted, the derivative included, and checkpoints
  keep the table.
- The blobs keep P* (`solve_value`), the model runs (`solve_evals`) and ln|∂P*/∂u| (`log_jac`). The summary, tracks
  and `compare` report physical parameters, `corner_sampled.png` shows (u, φ), and the MAP maximizes the sampled
  density, ln|J| included. `check $C` prints one solve run by run.

**Where it fails: plateaus.** Where O does not change with P over a stretch of P's prior, the whole stretch maps to
a single u and gets no mass: that part of the posterior is silently left out. The common case is Rp along log_f_atm
on a planet that can lose its atmosphere, since every log_f_atm below the stripping edge gives the bare-core
radius. On TOI-406 b (Rp 1.55 ± 0.06 R⊕, Mp 4.57 ± 0.70 M⊕) 82 % of the plain posterior were stripped planets at
that radius, and solve mode on Rp returned only planets that keep an atmosphere. The config warns about this case,
`check` and `scan` report plateaus, and such a planet needs plain sampling (`--set 'solve.param=""'`). On
LHS 1140 b nothing is lost: the plateaus of x lie outside the u range, and a stripped planet has no wind.

## Running, resuming and convergence

`run` starts the pool, evaluates the centre once (a bug or a model failure there stops it with a clear message),
starts the walkers and samples, printing a progress line every `print_every` steps (also in `run.log`):

    step 1400/3000  5.83 s/step  acc 0.20  lp max -18.793 med -20.431  stuck 0/24, behind 0  240 evals: ok 174, ...

**N and τ.** N is the number of steps per walker, and τ, the integrated autocorrelation time, is how many steps a
walker takes to forget where it was, so W walkers hold about W·N/τ independent samples. τ is estimated every
`check_every` steps over the last 2/3 of the chain, and the run stops at `max_steps` or once
1. N ≥ `min_steps` (200),
2. N > `n_tau` × max τ (50), and
3. τ is stable at `tau_checks` (2) checks in a row: every coordinate's τ changed by less than max(`tau_rtol`, its
   statistical noise) since the previous check, with every walker in the estimate.

The noise is the relative error of the estimate, √(2(2M + 1)/(L W)) with emcee's window M = 5τ, L steps and W
walkers (Sokal 1997): 10–15 % for 24 walkers at N ≈ 1000–1500. A stuck walker is left out of τ but stays in the
samples, so such a check never counts. The summary drops `burn_factor` × max τ steps and thins by τ_min/2.

**What the rule cannot see.** The estimate of τ is biased low, by about 15–30 % at N = 50 τ, so at the stop the
true N/τ is nearer 40–45. From N ≈ 1000 on, successive estimates share most of their window, so the stability test
nearly always passes and the stop comes mainly from N > 50 τ. A slow component the 5τ window cannot see (e.g. rare
mode switches) goes unnoticed: look at `traces.png`, and compare its two halves, before trusting "converged".

**Stopping by hand and resuming.** Ctrl-C saves the chain, writes `summary.json` (state "interrupted") and prints
the resume command; `kill` (SIGTERM) and closing the terminal (SIGHUP) do the same, and under `nohup` the run goes
on. A run started with `&` from a script ignores Ctrl-C: stop it with `kill PID`. `run --resume RUN_DIR` continues
from `chain.npz` with the run's own `config.json`, giving the same chain, bit for bit, as a run never stopped, and
`--max-steps N` (counted from the start) takes it further, past the convergence stop. While it runs,
`RUN_DIR/.lock` refuses a second resume. A chain stopped by hand is "not converged by the stopping rule" in the
summary, which names the missing condition; the final LHS 1140 b chain was stopped so at N/τ = 79.

**Stuck walkers.** A walker is stuck when its acceptance is below min(0.05, 0.2 × the median walker's), and behind
when its lp is more than 20 below the median (e.g. on a stripped-core plateau, where it still accepts moves); the
progress lines and the summary report both. With `reset_stuck_at = k > 0`, at step k they are replaced by jittered
copies of good walkers, steps 1–k go to `burnin.npz`, and sampling restarts on a fresh chain. A reset that would
replace more than half of the walkers replaces none, since copying the few good ones would collapse the ensemble:
such a posterior is thinner than the moves can follow, and solve mode is the fix.

## Outputs

A run directory, `<output_root>/<name>/mcmc_<stamp>/`, holds:

| file | content |
|---|---|
| `config.json`, `config.toml` | the resolved config, and a copy of the TOML with the `--set` overrides on top |
| `provenance.json` | the git commit and status of IsoFATE, package versions, the host, the start and each resume |
| `run.log` | every progress line, time-stamped, and the tracebacks of failures |
| `prescan.npz`, `bisect.npz`, `solve_init.npz`, `solve_table.npz` | the starting draws; the warm-start table |
| `chain.npz`, `burnin.npz` | the chain with its blobs, random-number state and bookkeeping; the chain before a reset |
| `summary.json`, `tracks.npz` | the posterior summary, and the posterior evolutionary tracks |
| `worker_errors/` | tracebacks of status-5 evaluations, one file per worker |

`summary.json` (which `summary RUN_DIR` prints) holds the settings, timing, evaluations per status, acceptance, τ
and its history, convergence, burn-in and thinning, the 2.5/16/50/84/97.5 percentiles of every parameter and blob
field, the MAP with its terms, and the warnings. `tracks.npz` (`tracks.load_tracks(RUN_DIR)`) holds the MAP and
`tracks` posterior samples re-run with the molecules saved, on `track_points` log-spaced times plus each sample's
own age, so every track ends at the state the likelihood saw: Rp, fatm, escape rates, outflow fractions, N_X, H/He,
C/O, D/H, and the amounts and VMRs of the 11 gases.

The figures (`plot RUN_DIR` redraws them) are `corner_params.png`, `corner_sampled.png` (solve mode: the sampler's
coordinates), `corner_observables.png`, `traces.png` (the walkers and the log-probability against step, with the
burn-in), `predictive.png` (every observable against its data) and `tracks.png` (Rp, fatm, Mdot, the outflow H
fraction, H/He, C/O, D/H, T_surf and the main VMRs over time, with 68 and 95 % bands, the MAP and the data).

## Comparing runs

```bash
python -m isofate.mcmc compare RUN_A RUN_B --labels dynamic_phi,static_phi
```

`compare` overlays two or more runs, of two settings or two planets, in a new `compare_<stamp>/` (under the first
run's `<output_root>/<name>/`, or `--out DIR`): `compare_corner.png` (the shared free parameters, 39 and 86 %
contours), `compare_predictive.png` (the shared active observables and the first run's data), `compare_tracks.png`
(one column per run, the same panels and y limits in each row) and `compare.txt`, also printed (per run the
settings, acceptance, τ, N/τ, convergence and cost, then the median and 16/84 % of every parameter and observable
next to the data). A solve-mode run enters with its physical parameters, so it compares directly with a plain run.

## Example: LHS 1140 b

`isofate/mcmc/configs/lhs1140b.toml` fits the outflow H fraction of LHS 1140 b, 0.115 (+0.085/−0.052) % (Cherubim
et al 2026; p-winds' `h_fraction`), and its H+He mass-loss rate, 2.03e5 kg/s (+0.121/−0.137 dex), both in log10
space, with the radius, 1.730 ± 0.025 R⊕, and the mass, 5.60 ± 0.19 M⊕ (Cadieux et al 2024). The physics is that of
isofate_MCMC.py: the legacy composition and a step-function XUV that drops to F_final = 0.17 W/m² at
t_sat = 3.128 Gyr, without a pre-main-sequence boost. The free parameters are log_f_atm U[−5, −1.3], Mp
U[4.5, 6.7] M⊕, age U[3.128, 10] Gyr, eps U[0.01, 1] and the Bond albedo U[0, 0.8] (Teq 151–226 K); the age prior
starts at t_sat, as younger ages would still see the saturated XUV today, which is unphysical for this quiet star
(a 131-day rotation) and made a spurious narrow mode. The outflow datum makes a
ridge about 0.003 dex thin in log_f_atm, hence solve mode on the outflow H fraction (see "Solve mode"). To run it,
copy it to `../mcmc_results/configs/` and uncomment its `output_root = ".."` line.

The final posterior so far (dynamic_phi = true, n_steps 1e5, n_atmodeller 100) took 1409 steps × 24 walkers on 6
workers, about 2 h, and reached N/τ = 79 with an acceptance of 0.20. It predates two changes of the config: it had
the albedo fixed at 0 and the pre-main-sequence boost of isofate_MCMC.py (t_pms = 1e8 yr, pms_factor = 100).
Medians and 16/84 %:

| | posterior | data |
|---|---|---|
| eps | 0.49 (+0.16/−0.11) | |
| log_f_atm | −2.57 (+0.16/−0.13) | |
| Mp [M⊕] | 6.02 (+0.18/−0.17) | 5.60 ± 0.19: 2.2σ high |
| age [Gyr] | 6.2 (+2.5/−2.1) | nearly flat over its prior |
| Rp [R⊕] | 1.62 (±0.01) | 1.730 ± 0.025: 4.4σ low |
| Mdot_HHe [kg/s] | 1.25e5 (+0.38/−0.26 e5) | 2.03e5: 1.5σ low |
| outflow H fraction | 1.31e-3 (+1.00/−0.60 e-3) | 1.15e-3 (fitted) |
| D/H | 0.028 | not fitted (see "Caveats") |

The outflow composition is reproduced. Along the ridge a larger eps raises the escape rate but shrinks the radius,
so eps settles near 0.5 with Mdot_HHe 1.5σ low. The radius stays 4.4σ below the measured one and pulls Mp 2.2σ
above its measurement, since a heavier core is larger. The age is barely constrained.

## Caveats

- **The resolution study is stricter than the posterior needs.** A cell passes only when every test point does, so
  one tail point can fail it while the posterior does not change: on LHS 1140 b, 1e5/100 failed at one of 30
  posterior samples (eps 0.91, |Δ log L| = 1.1), yet re-solving 240 posterior samples at 3e5/100 and reweighting
  them by exp(lp_fine − lp) moved every marginal by far less than its error (eps 0.496 to 0.493, effective sample
  size 239 of 240). The tool does not do that reweighting itself.
- **Solve mode** leaves out a plateau of its observable (TOI-406 b) and keeps one root per bracket (see "Solve
  mode"). Its ln|J| carries a numerical error of about 0.01–0.02, far below the Monte Carlo error.
- **The stopping rule** misses slow components, and its τ is biased low (see "Running, resuming and convergence").
- **μ is the mean mass per atom**, not per molecule (escape_kernel.py), so R_atm = 9kT/(μg) is about 1.8× larger
  than with the molecular μ of protosolar H₂ (13 % at the end of a typical LHS 1140 b run). It may be intended.
- **The step-function XUV** (`step_fn = true`) drops 3.5× at t_sat on LHS 1140 b: ages just below t_sat still see
  the saturated flux today and can form a narrow mode, so the config warns when t_sat lies inside the age prior.
- **Surface, not photospheric, chemistry.** Atmodeller speciates at the surface (T_surf ≈ 1600–1800 K on
  LHS 1140 b, against Teq ≈ 226 K) without condensation, so comparing VMRs with retrievals assumes a well-mixed
  atmosphere: prefer C/O, mmw and H/He, and give VMRs a `sigma_model` of 0.5–1 dex. NH₃, H₂S, OCS and HCN are not
  among Atmodeller's species, and `T_surf` is capped at 6000 K.
- **D/H depends steeply on b_He_D** in isofate/constants.py, scaled from b_H_D by the reduced mass (Genda & Ikoma
  2008): the isotopic substitution of the measured H–He coefficient would be 1.62× larger and lower D/H 6–13× on
  the LHS 1140 b posterior; the fit does not change. D/H comes from the escape kernel (Atmodeller lumps D into H).
- **He stops escaping** when the flux is subcritical (Φ_He = 0 is physics): with `step_fn = true`, t_sat = 3.1 Gyr
  and ages ≥ 3.5 Gyr, every H-rich track has an outflow H fraction of 1 today.
- **`flux_model = "Johnstone"`** reads its data from hard-coded paths in isofunks.py, and its φ_E lacks the
  1/(4 V_pot) factor: the config warns; do not fit with it. In "phoenix", the 45 Myr median flux (3661 W/m²) lies
  below its own lower quartile (30288): it looks like a typo.
- **The legacy composition** takes the enhanced O mass out of the H budget, so O/H is not 10^x × protosolar: x = 2
  gives about 0.8 instead of 0.058, and N_H < 0 above x = 2.03 (status 4). It refuses log_metallicity and
  log_CtoH_enhancement; "ratios" is linear and exact.
- **isocalc's time grid** ends one step past the age (final values refer to age × (1 + 1/n_steps)). With a free age
  the bands of `tracks.png` near the end mix older planets still in an earlier state with others at their end:
  compare the data with the ends of the tracks.
- **isofate_MCMC.py**, the older script, has problems fixed here: its flux-ratio likelihood gave every H-rich track
  χ² ≈ 1e78 and let a planet with no wind fit a detected He outflow; with n_atmodeller > 0 and
  save_molecules = False every evaluation silently returned −inf; its Mdot was phi·A, not Σ Φ_X μ_X A; and all its
  walkers started in a 1e-4 ball, without convergence checks, checkpoints or a seed.

## Python use and tests

```python
from isofate.mcmc import config, model, plots, sampler, solve, tracks

cfg = config.load_config('../mcmc_results/configs/lhs1140b.toml', ['model.n_steps=3e4'])  # as --set strings
run_dir = sampler.run_mcmc(cfg, workers = 6, max_steps = 2000)           # returns the run directory
run = sampler.load_run(run_dir)               # cfg, chain, log_prob, blobs, accepted, summary, ...
samples, blobs = sampler.posterior(run)       # after burn-in and thinning; blobs['Rp'], blobs['status'], ...
physical = sampler.physical_samples(samples, blobs, cfg)  # solve mode: P* in place of u
tr = tracks.load_tracks(run_dir)              # tr['grid_yr'], tr['Rp'] (n, grid size), ...
out_dir, paths = plots.compare_runs([run_dir, other_run_dir], labels = ['a', 'b'])
res = model.evaluate([-2.6, 6.0, 6e9, 0.5], cfg)      # one model run at a physical theta: res['obs'], res['lp']
res = solve.evaluate([-2.94, 5.6, 7.6e9, 0.15], cfg)  # one solve at (u, Mp, age, eps): res['solve_value']
```

`scan.run_scan`, `resolution.run_resolution` and `tracks.compute_tracks` run the other commands. A script that
starts a worker pool (these and `run_mcmc`) needs an `if __name__ == '__main__':` guard, as the spawned workers
would run it again. `model.evaluate` and `solve.evaluate` run the model in the calling process, which then holds
JAX (about 0.6 GB). A pool started from that process afterwards still works, since its workers are spawned, but
the parent keeps the 0.6 GB.

```bash
.venv/bin/python -m pytest tests -m "not slow"   # the MCMC tests on the toy backend, and IsoFATE's unit tests
.venv/bin/python -m pytest tests -m slow         # the real coupled model, at n_steps = 2000
```

The MCMC tests run the toy backend in one process or with 2 spawned workers; `test_thin_ridge_posterior` in
tests/test_mcmc_solve.py checks the solve-mode Jacobian against a brute-force grid posterior, and the slow tests
include one LHS 1140 b solve. tests/test_mcmc_cli.py checks that the tables above match `config.markdown_tables()`
and that every option appears here and in the template.
