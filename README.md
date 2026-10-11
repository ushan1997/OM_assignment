# Build guide — Delivery Route Optimisation (IT5082)

Kavindu Ariyasinghe (MS26900186) · Ushan Gunaratne (MS26901930)

Every code block below is the **exact final file**. Create the file at the path shown, paste the block, run the check, commit. Open this guide in IntelliJ (**Markdown preview**) — each code block has a copy button.

| Step | What | Who | Branch |
|---|---|---|---|
| 1 ✅ | Project setup | Kavindu | develop |
| 2 ✅ | Data + extraction script | Kavindu | develop |
| 3 | `instance.py` | Kavindu | develop |
| 4 | Colombo instance generator | Kavindu | develop |
| 5 | Exact solvers `exact.py` | Kavindu | develop |
| 6 | Exact-solver tests | Kavindu | develop |
| 7 | Heuristics + tests | **Ushan** | feature/heuristics → PR into develop |
| 8 | Experiments + analysis + results | **Ushan** | feature/heuristics → PR into develop |
| 9 | Notebook | Joint | develop |
| 10 | Report, submission files, release to main | Joint | develop → PR into main |

**Before every step:** check IntelliJ shows branch **develop** (Ushan: his feature branch), and run `git pull` if the other person pushed something.

**Commands** are for the IntelliJ terminal (PowerShell) in the project root with `.venv` active.

---

## Steps 1–2 — two small updates

You already did these. Two improvements since then:

**a) `.gitignore`** — make sure it contains these lines (IntelliJ files stay out of GitHub):

```text
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
.DS_Store
Optimization-assignment.zip
.idea/
*.iml
```

**b) `experiments/extract_census_data.py`** — the auto-clone version (if you already pasted it, skip):

**`experiments/extract_census_data.py`**

```python
"""Extract Colombo District GND data from the open nuuuwan/gig-data repository.

Reproduces data/raw/colombo_gnd.csv (557 Grama Niladhari Divisions with
centroid coordinates and 2012 Census population).

Usage:
    python experiments/extract_census_data.py [path-to-gig-data]

If the folder does not contain the census repository yet, it is cloned
automatically (about 300 MB) into the given path, or ~/gig-data by default.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLOMBO_DISTRICT = "LK-11"
GIG_DATA_URL = "https://github.com/nuuuwan/gig-data.git"


def ensure_source(gig_dir: Path) -> None:
    """Clone the census repository if it is not present yet."""
    if not (gig_dir / "ents" / "gnd.tsv").exists():
        print(f"cloning {GIG_DATA_URL} -> {gig_dir} (about 300 MB, please wait)")
        subprocess.run(["git", "clone", "--depth", "1", GIG_DATA_URL, str(gig_dir)], check=True)


def main(gig_dir: Path) -> None:
    ents = gig_dir / "ents"
    gnd = pd.read_csv(ents / "gnd.tsv", sep="\t")
    dsd = pd.read_csv(ents / "dsd.tsv", sep="\t")

    col = gnd[gnd.district_id == COLOMBO_DISTRICT].copy()
    col[["lat", "lon"]] = col.centroid.str.strip("[]").str.split(",", expand=True).astype(float)
    col = col.merge(dsd[["id", "name"]].rename(columns={"id": "dsd_id", "name": "dsd_name"}),
                    on="dsd_id")
    out = ROOT / "data" / "raw" / "colombo_gnd.csv"
    col[["gnd_id", "name", "dsd_name", "lat", "lon", "population"]].to_csv(out, index=False)
    print(f"wrote {len(col)} GNDs -> {out}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "gig-data"
    ensure_source(target)
    main(target)
```

```powershell
git add .gitignore experiments/extract_census_data.py
git commit -m "chore: ignore IDE files; auto-clone census source repository"
git push
```

---

## Step 3 — `src/cvrp/instance.py` (Kavindu)

The foundation every other file imports:
* **`Instance`** — one delivery problem: coordinates, demands, capacity Q, distance matrix. **Index 0 is always the depot**, customers are 1…n.
* **`load_vrplib`** — reads CVRPLIB `.vrp` files (and the `Cost` line of the `.sol` file as `inst.optimum`). Uses the official benchmark distance rule: Euclidean distance rounded to the nearest integer.
* **`solution_cost` / `check_feasible`** — the objective function and the constraints, evaluated on a concrete plan. *Every* method's output is measured with these two.

**`src/cvrp/instance.py`**

```python
"""CVRP instance model, loaders and solution utilities.

Node 0 is always the depot; customers are 1..n.
A solution is a list of routes; each route is a list of customer indices
(depot omitted), e.g. [[3, 1, 7], [2, 5]].
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Instance:
    name: str
    coords: np.ndarray            # (n+1, 2) — depot first
    demand: np.ndarray            # (n+1,)  — demand[0] == 0
    capacity: int
    dist: np.ndarray              # (n+1, n+1) travel cost matrix
    labels: list[str] = field(default_factory=list)
    optimum: float | None = None  # proven optimum / best-known value, if any
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of customers (excluding depot)."""
        return len(self.demand) - 1

    @property
    def min_vehicles(self) -> int:
        """Bin-packing lower bound on the number of vehicles."""
        return math.ceil(self.demand.sum() / self.capacity)

    # ------------------------------------------------------------------ io
    def to_json(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "capacity": int(self.capacity),
            "coords": self.coords.tolist(),
            "demand": self.demand.astype(int).tolist(),
            "dist": np.round(self.dist, 3).tolist(),
            "labels": self.labels,
            "optimum": self.optimum,
            "meta": self.meta,
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def from_json(cls, path: str | Path) -> "Instance":
        p = json.loads(Path(path).read_text())
        return cls(
            name=p["name"],
            coords=np.array(p["coords"], dtype=float),
            demand=np.array(p["demand"], dtype=int),
            capacity=int(p["capacity"]),
            dist=np.array(p["dist"], dtype=float),
            labels=p.get("labels", []),
            optimum=p.get("optimum"),
            meta=p.get("meta", {}),
        )


# ---------------------------------------------------------------- VRPLIB
def _euc2d_matrix(coords: np.ndarray) -> np.ndarray:
    """TSPLIB EUC_2D convention: Euclidean distance rounded to nearest int."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.floor(np.sqrt((diff ** 2).sum(-1)) + 0.5)


def load_vrplib(vrp_path: str | Path, sol_path: str | Path | None = None) -> Instance:
    """Minimal parser for CVRPLIB .vrp files (EUC_2D, single depot)."""
    lines = [ln.strip() for ln in Path(vrp_path).read_text().splitlines()]
    header, section, coords, demand, depot, weights = {}, None, {}, {}, [], []
    for ln in lines:
        if not ln or ln == "EOF":
            continue
        if ln.endswith("SECTION"):
            section = ln
            continue
        if ":" in ln and section is None:
            k, v = ln.split(":", 1)
            header[k.strip()] = v.strip().strip('"')
            continue
        parts = ln.split()
        if section == "NODE_COORD_SECTION":
            coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif section == "EDGE_WEIGHT_SECTION":
            weights.extend(float(x) for x in parts)
        elif section == "DEMAND_SECTION":
            demand[int(parts[0])] = int(parts[1])
        elif section == "DEPOT_SECTION" and int(parts[0]) > 0:
            depot.append(int(parts[0]))

    ewt = header.get("EDGE_WEIGHT_TYPE", "EUC_2D")
    d = depot[0] if depot else 1
    order = [d] + [i for i in sorted(demand) if i != d]  # depot -> index 0
    q = np.array([demand[i] for i in order])
    if ewt == "EUC_2D":
        xy = np.array([coords[i] for i in order])
        dist = _euc2d_matrix(xy)
    elif ewt == "EXPLICIT" and header.get("EDGE_WEIGHT_FORMAT") == "LOWER_ROW":
        if d != 1:
            raise ValueError("EXPLICIT instances assume depot = node 1")
        m = len(order)
        dist = np.zeros((m, m))
        it = iter(weights)
        for i in range(1, m):
            for j in range(i):
                dist[i, j] = dist[j, i] = next(it)
        xy = np.zeros((m, 2))  # no coordinates available
    else:
        raise ValueError(f"unsupported EDGE_WEIGHT_TYPE {ewt}")

    optimum = None
    if sol_path and Path(sol_path).exists():
        m = re.search(r"Cost\s+([\d.]+)", Path(sol_path).read_text())
        optimum = float(m.group(1)) if m else None

    return Instance(
        name=header.get("NAME", Path(vrp_path).stem),
        coords=xy,
        demand=q,
        capacity=int(header["CAPACITY"]),
        dist=dist,
        labels=[str(i) for i in order],
        optimum=optimum,
        meta={"source": "CVRPLIB", "comment": header.get("COMMENT", "")},
    )


# ------------------------------------------------------------- solutions
def route_load(inst: Instance, route: list[int]) -> int:
    return int(inst.demand[route].sum())


def route_cost(inst: Instance, route: list[int]) -> float:
    if not route:
        return 0.0
    D = inst.dist
    c = D[0, route[0]] + D[route[-1], 0]
    for a, b in zip(route, route[1:]):
        c += D[a, b]
    return float(c)


def solution_cost(inst: Instance, routes: list[list[int]]) -> float:
    return sum(route_cost(inst, r) for r in routes)


def check_feasible(inst: Instance, routes: list[list[int]]) -> tuple[bool, str]:
    """Every customer visited exactly once and no route exceeds capacity."""
    seen = [c for r in routes for c in r]
    if sorted(seen) != list(range(1, inst.n + 1)):
        return False, "customers not visited exactly once"
    for r in routes:
        if route_load(inst, r) > inst.capacity:
            return False, f"capacity exceeded on route {r}"
    return True, "ok"
```

**Check** — reproduces the published optimum of A-n32-k5 (one line):

```powershell
$env:PYTHONPATH="src"; python -c "from cvrp.instance import load_vrplib, solution_cost, check_feasible; import re; i=load_vrplib('data/raw/cvrplib/A-n32-k5.vrp','data/raw/cvrplib/A-n32-k5.sol'); r=[[int(x) for x in m.split()] for m in re.findall(r'Route #\d+:\s*([\d ]+)', open('data/raw/cvrplib/A-n32-k5.sol').read())]; print(i.n, i.capacity, i.min_vehicles, i.optimum, solution_cost(i,r), check_feasible(i,r))"
```

Expected: `31 100 5 784.0 784.0 (True, 'ok')`

```powershell
git add src/cvrp/instance.py
git commit -m "feat(data): instance model, CVRPLIB parser and solution feasibility checks"
git push
```

---

## Step 4 — Colombo instances (Kavindu)

* **`colombo.py`** turns the census CSV into delivery instances: depot = Kelaniya DSD centroid; demand = ⌈population/1000⌉ crates; Q = 30; distance = haversine km × 1.3 circuity. `build_instance(n, seed)` samples *n* ordering outlets reproducibly.
* **`build_instances.py`** generates all 28 instances (8–30 outlets × 3 seeds, plus 50/100/200/557), a dataset summary and the outlet map figure.

**`src/cvrp/colombo.py`**

```python
"""Real-world CVRP instances for Colombo District, Sri Lanka.

Data source
-----------
`data/raw/colombo_gnd.csv` — the 557 Grama Niladhari Divisions (GNDs) of
Colombo District with centroid coordinates and 2012 Census population,
extracted from github.com/nuuuwan/gig-data (Department of Census and
Statistics, Sri Lanka).

Scenario
--------
A distributor runs a distribution centre (DC) in Kelaniya (Kelaniya DSD
centroid), across the Kelani River from Colombo city in neighbouring Gampaha
District. Each GND of Colombo District has one retail outlet.
On a given day a subset of outlets places orders; the DC must serve all of
them with a fleet of identical light trucks, minimising total km driven.

Modelling assumptions (documented in the report)
------------------------------------------------
* Demand  d_i = ceil(population_i / 1000) crates  (1 crate / 1,000 residents / day)
* Truck capacity Q = 30 crates
* Road distance = great-circle (haversine) distance x 1.3 circuity factor
  (typical road-network circuity, Ballou et al., 2002)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .instance import Instance

ROOT = Path(__file__).resolve().parents[2]
GND_CSV = ROOT / "data" / "raw" / "colombo_gnd.csv"

DEPOT_NAME = "DC - Kelaniya"
DEPOT_LATLON = (6.970155, 79.909850)  # Kelaniya DSD centroid
CAPACITY = 30                         # crates per truck
PEOPLE_PER_CRATE = 1000
CIRCUITY = 1.3
EARTH_R_KM = 6371.0088


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None, :]) * np.sin(dlo / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_gnd() -> pd.DataFrame:
    return pd.read_csv(GND_CSV)


def build_instance(n: int | None, seed: int = 0) -> Instance:
    """Sample `n` ordering outlets (all 557 if n is None) and build an instance."""
    gnd = load_gnd()
    if n is not None and n < len(gnd):
        gnd = gnd.sample(n=n, random_state=seed).reset_index(drop=True)
    n_eff = len(gnd)

    lat = np.r_[DEPOT_LATLON[0], gnd["lat"].to_numpy()]
    lon = np.r_[DEPOT_LATLON[1], gnd["lon"].to_numpy()]
    dist = np.round(haversine_matrix(lat, lon) * CIRCUITY, 3)
    demand = np.r_[0, np.ceil(gnd["population"].to_numpy() / PEOPLE_PER_CRATE)].astype(int)

    return Instance(
        name=f"COL-n{n_eff}-s{seed}",
        coords=np.c_[lon, lat],
        demand=demand,
        capacity=CAPACITY,
        dist=dist,
        labels=[DEPOT_NAME] + [f"{r.name} ({r.dsd_name})" for r in gnd.itertuples()],
        meta={
            "source": "Colombo District GND centroids, Census 2012 (nuuuwan/gig-data)",
            "seed": seed,
            "distance": f"haversine x {CIRCUITY} (km)",
            "demand": f"ceil(population / {PEOPLE_PER_CRATE}) crates",
        },
    )


def describe(inst: Instance) -> dict:
    q = inst.demand[1:]
    return {
        "instance": inst.name,
        "customers": inst.n,
        "total_demand": int(q.sum()),
        "capacity": inst.capacity,
        "min_vehicles": inst.min_vehicles,
        "demand_min": int(q.min()),
        "demand_median": float(np.median(q)),
        "demand_max": int(q.max()),
        "max_depot_km": float(inst.dist[0, 1:].max()),
        "mean_depot_km": float(inst.dist[0, 1:].mean()),
    }
```

**`experiments/build_instances.py`**

```python
"""Generate all Colombo instances + dataset summary + map figure.

Usage:  python experiments/build_instances.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cvrp.colombo import DEPOT_LATLON, build_instance, describe, load_gnd

OUT = ROOT / "data" / "instances"
FIG = ROOT / "results" / "figures"

SMALL_SIZES = [8, 10, 12, 15, 18, 20, 25, 30]   # exact + heuristic
LARGE_SIZES = [50, 100, 200, None]               # heuristic only (None = all 557)
SEEDS = [1, 2, 3]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in SMALL_SIZES:
        for s in SEEDS:
            inst = build_instance(n, s)
            inst.to_json(OUT / f"{inst.name}.json")
            rows.append(describe(inst))
    for n in LARGE_SIZES:
        inst = build_instance(n, 1)
        inst.to_json(OUT / f"{inst.name}.json")
        rows.append(describe(inst))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "dataset_summary.csv", index=False)
    print(df.to_string(index=False))

    # map of all outlets, marker size ~ demand
    gnd = load_gnd()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(gnd.lon, gnd.lat, s=gnd.population / 400, alpha=0.45, c="#2a6f97",
               edgecolors="none", label="Outlet (GND centroid), size ∝ demand")
    ax.scatter(*DEPOT_LATLON[::-1], marker="s", s=120, c="#c1121f", label="Distribution centre (Kelaniya)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Colombo District – 557 GND outlets (Census 2012 population)")
    ax.set_aspect(1 / 0.993)  # ~equal-distance aspect at 7°N
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "colombo_outlets_map.png", dpi=160)


if __name__ == "__main__":
    main()
```

**Run:**

```powershell
python experiments/build_instances.py
```

Expected: a 28-row table ending with `COL-n557-s1  557  2604  30  87 ...`, plus new files:
* `data/instances/` — 28 JSON files
* `results/dataset_summary.csv`
* `results/figures/colombo_outlets_map.png` (open it — the Colombo District map)

```powershell
git add src/cvrp/colombo.py experiments/build_instances.py data/instances results/dataset_summary.csv results/figures/colombo_outlets_map.png
git commit -m "feat(data): Colombo delivery instance generator, 28 instances and dataset figure"
git push
```

---

## Step 5 — exact solvers `src/cvrp/exact.py` (Kavindu)

Two MILP models solved with HiGHS (`scipy.optimize.milp`) — see report §3–4:
* **Model A (`solve_exact`)** — compact MTZ formulation: variables x_ij, load variables u_i, constraints (A1)–(A6). Simple but weak LP bound.
* **Model B (`solve_exact_cuts`)** — one variable per edge, degree constraints, and **rounded capacity cuts** added only when violated (root LP cutting loop, then MILP + lazy cuts on integer solutions). Much stronger.

It is one file; paste it in three segments in order (the split points are only for readability).


**Segment 5a — header, result type and Model A (MTZ)**

```python
"""Exact method: Mixed-Integer Linear Programme (two-index vehicle-flow
formulation with Miller-Tucker-Zemlin load constraints), solved with the
HiGHS branch-and-bound/branch-and-cut solver via scipy.optimize.milp.

Model (see report, Section 3)
-----------------------------
    min   sum_{i != j} c_ij x_ij
    s.t.  sum_j x_ij = 1                         for all customers i     (1) out-degree
          sum_i x_ij = 1                         for all customers j     (2) in-degree
          sum_j x_0j = sum_i x_i0                                        (3) depot balance
          sum_j x_0j >= ceil(sum d / Q)                                  (3b) min fleet (valid ineq.)
          u_j >= u_i + d_j - Q (1 - x_ij)        for customers i != j    (4) MTZ load / subtour
          d_i <= u_i <= Q                        for all customers i     (5) capacity
          x_ij in {0, 1}                                                 (6)

Optional strengthening (lifted=True): Desrochers & Laporte (1991) lifting of (4)
          u_i - u_j + Q x_ij + (Q - d_i - d_j) x_ji <= Q - d_j
Pre-processing: arc (i, j) between customers is removed if d_i + d_j > Q.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .instance import Instance, solution_cost


@dataclass
class ExactResult:
    routes: list[list[int]] | None
    cost: float | None          # objective of best feasible solution found
    lower_bound: float | None   # proven lower bound (dual bound)
    gap: float | None           # relative optimality gap (0 = proven optimal)
    status: str                 # 'optimal' | 'time_limit' | 'no_solution' | ...
    runtime: float
    nodes: int | None
    n_vars: int
    n_cons: int


def build_model(inst: Instance, lifted: bool = False):
    n, Q, d, D = inst.n, inst.capacity, inst.demand, inst.dist
    V = range(n + 1)

    # ---- variables: x_ij for every admissible arc, then u_1..u_n
    arcs = [(i, j) for i in V for j in V
            if i != j and not (i > 0 and j > 0 and d[i] + d[j] > Q)]
    a_idx = {a: k for k, a in enumerate(arcs)}
    nx = len(arcs)
    u_idx = {i: nx + i - 1 for i in range(1, n + 1)}
    nvar = nx + n

    c = np.zeros(nvar)
    for (i, j), k in a_idx.items():
        c[k] = D[i, j]

    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(coefs: dict[int, float], lb: float, ub: float):
        nonlocal r
        for col, v in coefs.items():
            rows.append(r); cols.append(col); vals.append(v)
        lo.append(lb); hi.append(ub); r += 1

    # (1) out-degree and (2) in-degree of each customer
    for i in range(1, n + 1):
        add({a_idx[(i, j)]: 1 for j in V if (i, j) in a_idx}, 1, 1)
        add({a_idx[(j, i)]: 1 for j in V if (j, i) in a_idx}, 1, 1)

    # (3) depot balance and (3b) minimum number of vehicles
    bal = {a_idx[(0, j)]: 1 for j in range(1, n + 1)}
    for i in range(1, n + 1):
        bal[a_idx[(i, 0)]] = -1
    add(bal, 0, 0)
    add({a_idx[(0, j)]: 1 for j in range(1, n + 1)}, inst.min_vehicles, n)

    # (4) MTZ load constraints:  u_i - u_j + Q x_ij [+ (Q-d_i-d_j) x_ji] <= Q - d_j
    for (i, j), k in a_idx.items():
        if i == 0 or j == 0:
            continue
        coefs = {u_idx[i]: 1, u_idx[j]: -1, k: Q}
        if lifted and (j, i) in a_idx:
            coefs[a_idx[(j, i)]] = Q - d[i] - d[j]
        add(coefs, -np.inf, Q - d[j])

    A = coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsr()

    # (5) bounds and (6) integrality
    lb = np.zeros(nvar); ub = np.ones(nvar)
    for i in range(1, n + 1):
        lb[u_idx[i]], ub[u_idx[i]] = d[i], Q
    integrality = np.r_[np.ones(nx), np.zeros(n)]

    return c, LinearConstraint(A, lo, hi), Bounds(lb, ub), integrality, arcs


def extract_routes(x: np.ndarray, arcs: list[tuple[int, int]]) -> list[list[int]]:
    succ = {}
    starts = []
    for k, (i, j) in enumerate(arcs):
        if x[k] > 0.5:
            if i == 0:
                starts.append(j)
            else:
                succ[i] = j
    routes = []
    for s in starts:
        route, cur = [], s
        while cur != 0:
            route.append(cur)
            cur = succ[cur]
        routes.append(route)
    return routes


def solve_exact(inst: Instance, time_limit: float = 300.0, lifted: bool = False,
                verbose: bool = False) -> ExactResult:
    c, cons, bounds, integrality, arcs = build_model(inst, lifted=lifted)
    t0 = time.perf_counter()
    res = milp(c, constraints=cons, bounds=bounds, integrality=integrality,
               options={"time_limit": time_limit, "disp": verbose, "mip_rel_gap": 1e-6})
    runtime = time.perf_counter() - t0

    status = {0: "optimal", 1: "time_limit", 2: "infeasible", 3: "unbounded"}.get(res.status, "error")
    routes = cost = None
    if res.x is not None:
        routes = extract_routes(res.x, arcs)
        cost = solution_cost(inst, routes)
    elif status == "time_limit":
        status = "no_solution"
    lb = getattr(res, "mip_dual_bound", None)
    gap = getattr(res, "mip_gap", None)
    if status == "optimal":
        gap = 0.0
    return ExactResult(routes, cost, lb, gap, status, runtime,
                       getattr(res, "mip_node_count", None), len(c), cons.A.shape[0])
```


**Segment 5b — Model B helpers: components, route extraction, cut separation**

```python
# =========================================================================
# Exact method 2: two-index undirected formulation + lazy Rounded Capacity
# Inequalities (RCI), solved by iterative row generation (cutting planes).
#
#     min  sum_e c_e x_e
#     s.t. x(delta(i)) = 2                       for all customers i
#          x(delta(0)) = 2K,  K >= ceil(d(C)/Q)
#          x(delta(S)) >= 2 ceil(d(S)/Q)         for all S subset of C   (RCI, added lazily)
#          x_e in {0,1} (customer edges), x_0j in {0,1,2}
#
# The RCI family is exponential, so we start without it, solve the MILP,
# find violated sets S in the integer solution (subtours or overloaded
# routes = connected components of the customer subgraph) and re-solve.
# Every intermediate optimum is a valid lower bound; the first solution
# that violates no RCI is a proven optimum.
# =========================================================================
def _components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        parent[find(i)] = find(j)
    comps: dict[int, list[int]] = {}
    for v in range(1, n + 1):
        comps.setdefault(find(v), []).append(v)
    return list(comps.values())


def _routes_from_edges(n: int, chosen: dict[tuple[int, int], int]) -> list[list[int]]:
    adj = {v: [] for v in range(n + 1)}
    for (i, j), m in chosen.items():
        for _ in range(m):
            adj[i].append(j); adj[j].append(i)
    routes, used = [], set()
    for start in list(adj[0]):
        if start in used:
            continue
        route, prev, cur = [], 0, start
        while cur != 0:
            route.append(cur); used.add(cur)
            nxt = [w for w in adj[cur] if w != prev]
            if not nxt:          # single-customer route uses edge (0,j) twice
                break
            prev, cur = cur, nxt[0]
        routes.append(route)
    return routes


def _separate_rci(x: np.ndarray, e_idx: dict, n: int, d: np.ndarray, Q: int,
                  incident, seen: set, max_cuts: int | None = None) -> list[tuple[dict, float, float]]:
    """Heuristic separation of Rounded Capacity Inequalities on (fractional) x.

    Uses x(delta(S)) = 2|S| - 2 x(E(S)), valid because every customer has
    degree exactly 2, so a candidate set is checked in O(1) while it grows.
      1. connected components of the customer support graph at several thresholds;
      2. greedy growth from every customer seed (add the customer most strongly
         connected to S, test the cut after each addition).
    """
    W = np.zeros((n + 1, n + 1))
    for (i, j), k in e_idx.items():
        W[i, j] = W[j, i] = x[k]
    Wc = W[1:, 1:]
    dc = d[1:].astype(float)

    def slack(members: list[int], inner: float, demand: float) -> float:
        """x(delta(S)) - rhs; negative means the cut is violated."""
        return 2 * len(members) - 2 * inner - 2 * np.ceil(demand / Q - 1e-9)

    def violated(members, inner, demand) -> bool:
        return slack(members, inner, demand) < -1e-4

    found: dict[frozenset, float] = {}
    for t in (1e-6, 0.25, 0.5, 0.75):
        edges = [(i, j) for (i, j) in e_idx if i > 0 and W[i, j] > t]
        for S in _components(n, edges):
            idx = np.array(S) - 1
            inner = Wc[np.ix_(idx, idx)].sum() / 2
            if violated(S, inner, dc[idx].sum()):
                found[frozenset(S)] = slack(S, inner, dc[idx].sum())
    for seed in range(1, n + 1):
        S = [seed]
        inner, demand = 0.0, dc[seed - 1]
        conn = Wc[seed - 1].copy()
        conn[seed - 1] = -1
        while len(S) < n - 1:
            j = int(np.argmax(conn))
            if conn[j] <= 1e-6:
                break
            inner += conn[j]
            demand += dc[j]
            S.append(j + 1)
            conn += Wc[j]
            conn[[s - 1 for s in S]] = -1
            if violated(S, inner, demand):
                found[frozenset(S)] = slack(S, inner, demand)
    order = list(found)                           # discovery order
    if max_cuts is not None:                      # keep only the most violated
        order = [S for S in sorted(found, key=found.get) if S not in seen][:max_cuts]
    new = []
    for S in order:
        if S in seen:
            continue
        seen.add(S)
        rhs = 2 * int(np.ceil(d[list(S)].sum() / Q))
        new.append(({k: 1 for k in incident(S)}, rhs, np.inf))
    return new
```


**Segment 5c — Model B solver**

```python
def solve_exact_cuts(inst: Instance, time_limit: float = 300.0, max_rounds: int = 200,
                     max_cuts_per_round: int | None = None, verbose: bool = False) -> ExactResult:
    """max_cuts_per_round=None adds every violated cut found (used for all
    instances up to n=100); a cap keeps only the most violated cuts and bounds
    memory on very large instances (used for n=200)."""
    n, Q, d, D = inst.n, inst.capacity, inst.demand, inst.dist
    edges = [(i, j) for i in range(n + 1) for j in range(i + 1, n + 1)
             if not (i > 0 and d[i] + d[j] > Q)]
    e_idx = {e: k for k, e in enumerate(edges)}
    ne = len(edges)
    kvar = ne                       # fleet-size variable K
    nvar = ne + 1

    c = np.zeros(nvar)
    for (i, j), k in e_idx.items():
        c[k] = D[i, j]
    lb = np.zeros(nvar)
    ub = np.array([2.0 if i == 0 else 1.0 for i, _ in edges] + [n])
    lb[kvar] = inst.min_vehicles

    node_edges: list[list[tuple[int, int]]] = [[] for _ in range(n + 1)]
    for (i, j), k in e_idx.items():
        node_edges[i].append((j, k)); node_edges[j].append((i, k))

    def incident(S) -> list[int]:
        S = set(S)
        return [k for v in S for (w, k) in node_edges[v] if w not in S]

    base_rows: list[tuple[dict, float, float]] = []
    for v in range(1, n + 1):
        base_rows.append(({k: 1 for k in incident({v})}, 2, 2))
    depot = {k: 1 for (i, j), k in e_idx.items() if i == 0}
    depot[kvar] = -2
    base_rows.append((depot, 0, 0))
    cuts: list[tuple[dict, float, float]] = []
    seen: set = set()

    def solve(integer: bool, tlim: float):
        rows = base_rows + cuts
        R, C, Vv, lo, hi = [], [], [], [], []
        for r, (coefs, l, h) in enumerate(rows):
            for col, v in coefs.items():
                R.append(r); C.append(col); Vv.append(v)
            lo.append(l); hi.append(h)
        A = coo_matrix((Vv, (R, C)), shape=(len(rows), nvar)).tocsr()
        return milp(c, constraints=LinearConstraint(A, lo, hi), bounds=Bounds(lb, ub),
                    integrality=np.ones(nvar) if integer else np.zeros(nvar),
                    options={"time_limit": max(tlim, 0.1), "disp": False, "mip_rel_gap": 1e-6})

    t0 = time.perf_counter()
    elapsed = lambda: time.perf_counter() - t0
    best_lb, nodes_total, status = -np.inf, 0, "time_limit"
    routes = cost = None

    # ---- phase 1: root cutting-plane loop on the LP relaxation
    for rnd in range(max_rounds):
        if elapsed() > 0.3 * time_limit:
            break
        res = solve(False, time_limit - elapsed())
        if res.status != 0:
            break
        best_lb = max(best_lb, res.fun)
        new = _separate_rci(res.x, e_idx, n, d, Q, incident, seen, max_cuts_per_round)
        if verbose:
            print(f"LP round {rnd}: bound={res.fun:.2f} new cuts={len(new)}")
        if not new:
            break
        cuts += new
    root_lb = best_lb

    # ---- phase 2: MILP + lazy cuts on integer solutions
    for it in range(10_000):
        remaining = time_limit - elapsed()
        if remaining <= 0.5:
            break
        res = solve(True, remaining)
        nodes_total += getattr(res, "mip_node_count", 0) or 0
        if res.status != 0:
            db = getattr(res, "mip_dual_bound", None)
            if db is not None and np.isfinite(db):
                best_lb = max(best_lb, db)
            break
        best_lb = max(best_lb, res.fun)
        x = res.x
        new = _separate_rci(x, e_idx, n, d, Q, incident, seen, max_cuts_per_round)
        if verbose:
            print(f"MILP iter {it}: obj={res.fun:.2f} new cuts={len(new)}")
        if not new:
            chosen = {e: int(round(x[k])) for e, k in e_idx.items() if x[k] > 0.5}
            routes = _routes_from_edges(n, chosen)
            cost = solution_cost(inst, routes)
            status = "optimal"
            break
        cuts += new

    gap = 0.0 if status == "optimal" else None
    r = ExactResult(routes, cost, None if not np.isfinite(best_lb) else best_lb, gap, status,
                    elapsed(), nodes_total, nvar, len(base_rows) + len(cuts))
    r.root_lb = root_lb
    return r
```


**Check** — both models reproduce the proven optimum 450 of P-n16-k8 (a few seconds):

```powershell
$env:PYTHONPATH="src"; python -c "from cvrp.instance import load_vrplib; from cvrp.exact import solve_exact, solve_exact_cuts; i=load_vrplib('data/raw/cvrplib/P-n16-k8.vrp','data/raw/cvrplib/P-n16-k8.sol'); a=solve_exact(i,60); b=solve_exact_cuts(i,60); print('MTZ', a.status, a.cost, round(a.runtime,2)); print('Cuts', b.status, b.cost, round(b.runtime,2))"
```

Expected: `MTZ optimal 450.0 …` and `Cuts optimal 450.0 …`

```powershell
git add src/cvrp/exact.py
git commit -m "feat(exact): MILP solvers - MTZ model and two-index model with lazy capacity cuts"
git push
```

---

## Step 6 — exact-solver tests (Kavindu)

Regression tests: the exact models must reproduce published proven optima. If anyone breaks the solver later, `pytest` fails.

**`tests/test_exact.py`**

```python
"""Exact solvers must reproduce proven optima of CVRPLIB benchmarks."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvrp.exact import solve_exact, solve_exact_cuts  # noqa: E402
from cvrp.instance import check_feasible, load_vrplib  # noqa: E402

RAW = ROOT / "data" / "raw" / "cvrplib"


def _load(name):
    return load_vrplib(RAW / f"{name}.vrp", RAW / f"{name}.sol")


@pytest.mark.parametrize("name", ["E-n13-k4", "P-n16-k8"])
def test_mtz_reaches_optimum(name):
    inst = _load(name)
    res = solve_exact(inst, time_limit=60)
    assert res.status == "optimal"
    assert check_feasible(inst, res.routes)[0]
    assert res.cost == pytest.approx(inst.optimum)


@pytest.mark.parametrize("name", ["E-n13-k4", "P-n16-k8", "B-n31-k5"])
def test_cutting_plane_reaches_optimum(name):
    inst = _load(name)
    res = solve_exact_cuts(inst, time_limit=120)
    assert res.status == "optimal"
    assert check_feasible(inst, res.routes)[0]
    assert res.cost == pytest.approx(inst.optimum)


if __name__ == "__main__":  # allows `python tests/test_exact.py` without pytest
    for n in ["E-n13-k4", "P-n16-k8"]:
        test_mtz_reaches_optimum(n)
    for n in ["E-n13-k4", "P-n16-k8", "B-n31-k5"]:
        test_cutting_plane_reaches_optimum(n)
    print("all exact-solver tests passed")
```

**Run:**

```powershell
pytest -q
```

Expected: `5 passed` (about 30 s).

```powershell
git add tests/test_exact.py
git commit -m "test: exact solvers reproduce CVRPLIB proven optima"
git push
```

**Hand-over to Ushan:** tell him Steps 1–6 are pushed to `develop`. Make sure he is added as a collaborator (GitHub → Settings → Collaborators).

---

## Step 7 — heuristics (Ushan)

**Setup on Ushan's machine (once):**

```powershell
git clone https://github.com/kavndu/colombo-delivery-optimization.git
cd colombo-delivery-optimization
git checkout develop
git checkout -b feature/heuristics
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
git config user.name  "Ushan Gunaratne"
git config user.email "ushananuka5@gmail.com"
```

In IntelliJ: open the folder, select `.venv` as interpreter, right-click `src` → **Mark Directory as → Sources Root**.

* **`clarke_wright`** — greedy savings heuristic: start with one route per outlet, merge routes in order of saving s_ij = c_0i + c_0j − c_ij while capacity allows.
* **`simulated_annealing`** — starts from Clarke–Wright; each iteration picks an outlet and one of its 10 nearest neighbours and tries relocate / swap / 2-opt / 2-opt*; accepts worse moves with probability e^(−Δ/T); geometric cooling; final local-search polish. Every visited plan is feasible.

**`src/cvrp/heuristics.py`**

```python
"""Heuristic and metaheuristic methods for the CVRP.

* clarke_wright      — Clarke & Wright (1964) parallel savings algorithm (greedy constructive)
* simulated_annealing — SA over a route-based representation with four
                        neighbourhood moves (relocate, swap, 2-opt, 2-opt*),
                        started from the Clarke–Wright solution.

All moves keep the solution feasible (capacity is checked before a move is
accepted), so every solution SA visits is a valid delivery plan.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from .instance import Instance, check_feasible, route_cost, solution_cost


# =====================================================================
# Clarke–Wright savings
# =====================================================================
def clarke_wright(inst: Instance) -> list[list[int]]:
    """Start with one out-and-back route per customer, then repeatedly merge
    the two routes whose joining saves the most distance:
        s_ij = c_0i + c_0j - c_ij
    A merge is allowed only if i and j are route end-points, they lie on
    different routes, and the merged load fits in the vehicle."""
    n, D, d, Q = inst.n, inst.dist, inst.demand, inst.capacity
    routes: dict[int, list[int]] = {i: [i] for i in range(1, n + 1)}
    load = {i: int(d[i]) for i in range(1, n + 1)}
    route_of = {i: i for i in range(1, n + 1)}

    iu, ju = np.triu_indices(n, k=1)
    iu, ju = iu + 1, ju + 1
    savings = D[0, iu] + D[0, ju] - D[iu, ju]
    order = np.argsort(-savings, kind="stable")

    for k in order:
        if savings[k] <= 0:
            break
        i, j = int(iu[k]), int(ju[k])
        ri, rj = route_of[i], route_of[j]
        if ri == rj or load[ri] + load[rj] > Q:
            continue
        a, b = routes[ri], routes[rj]
        if a[-1] != i and a[0] != i or b[0] != j and b[-1] != j:
            continue                      # i or j is interior to its route
        if a[-1] != i:
            a.reverse()                   # orient: ... -> i
        if b[0] != j:
            b.reverse()                   # orient: j -> ...
        a.extend(b)
        load[ri] += load[rj]
        for c in b:
            route_of[c] = ri
        del routes[rj], load[rj]
    return list(routes.values())


# =====================================================================
# Simulated Annealing
# =====================================================================
@dataclass
class SAResult:
    routes: list[list[int]]
    cost: float
    initial_cost: float
    runtime: float
    iterations: int
    accepted: int
    history: list[tuple[int, float, float]] = field(default_factory=list)  # (iter, current, best)


def _cost(D, r):
    if not r:
        return 0.0
    c = D[0, r[0]] + D[r[-1], 0]
    for a, b in zip(r, r[1:]):
        c += D[a, b]
    return c


def _neighbours(inst: Instance, k: int) -> list[list[int]]:
    """k nearest customers of every customer (granular neighbourhood)."""
    D = inst.dist
    nb = [[]]
    for c in range(1, inst.n + 1):
        order = np.argsort(D[c, 1:]) + 1
        nb.append([int(v) for v in order if v != c][:k])
    return nb


def simulated_annealing(inst: Instance, init: list[list[int]] | None = None,
                        iterations: int | None = None, t0: float | None = None,
                        p_accept0: float = 0.3, t_final_ratio: float = 1e-3,
                        k_neighbours: int = 10, time_limit: float | None = None,
                        seed: int = 0, log_every: int = 0, polish: bool = True) -> SAResult:
    """Metropolis acceptance with geometric cooling T_k = T0 * alpha^k.

    Granular neighbourhood: every move is built around a random customer c
    and one of its k nearest neighbours v (moves joining far-apart customers
    are almost never improving, so they are not sampled):
      relocate  — move c next to v (or, rarely, to a new empty route)
      swap      — exchange c and v
      2-opt     — c, v on the same route: reverse the segment between them
      2-opt*    — c, v on different routes: exchange the route tails after c and v
    T0 is calibrated so that an average uphill move is accepted with
    probability p_accept0; the best solution found is finally polished by
    first-improvement local search with the same moves.
    """
    rng = random.Random(seed)
    D, d, Q, n = inst.dist, inst.demand, inst.capacity, inst.n
    routes = [list(r) for r in (init if init is not None else clarke_wright(inst))]
    if iterations is None:
        iterations = max(50_000, 4_000 * n)
    NB = _neighbours(inst, min(k_neighbours, n - 1)) if n > 1 else [[], []]

    routes = [r for r in routes if r] + [[]]   # one spare empty route = option to open a vehicle
    costs = [_cost(D, r) for r in routes]
    loads = [int(d[r].sum()) for r in routes]
    where = {}

    def reindex():
        where.clear()
        for k, r in enumerate(routes):
            for c in r:
                where[c] = k

    reindex()
    cur = sum(costs)
    init_cost = cur

    def propose():
        c = rng.randint(1, n)
        a = where[c]
        ra = routes[a]
        i = ra.index(c)
        mtype = rng.random()
        if mtype < 0.05:                       # relocate c to the empty route
            b = len(routes) - 1 if not routes[-1] else next(k for k, r in enumerate(routes) if not r)
            if len(ra) == 1:
                return None
            na, nb_ = ra[:i] + ra[i + 1:], [c]
            return (a, b), (na, nb_), _cost(D, na) + _cost(D, nb_) - costs[a] - costs[b]
        v = rng.choice(NB[c])
        b = where[v]
        rb = routes[b]
        j = rb.index(v)
        if mtype < 0.40:                       # relocate c next to v
            after = rng.random() < 0.5
            if a == b:
                new = ra[:i] + ra[i + 1:]
                jj = new.index(v) + (1 if after else 0)
                new.insert(jj, c)
                if new == ra:
                    return None
                return (a,), (new,), _cost(D, new) - costs[a]
            if loads[b] + d[c] > Q:
                return None
            na = ra[:i] + ra[i + 1:]
            nb_ = list(rb); nb_.insert(j + (1 if after else 0), c)
            return (a, b), (na, nb_), _cost(D, na) + _cost(D, nb_) - costs[a] - costs[b]
        if mtype < 0.65:                       # swap c and v
            if a == b:
                new = list(ra); new[i], new[j] = new[j], new[i]
                return (a,), (new,), _cost(D, new) - costs[a]
            if loads[a] - d[c] + d[v] > Q or loads[b] - d[v] + d[c] > Q:
                return None
            na, nb_ = list(ra), list(rb)
            na[i], nb_[j] = v, c
            return (a, b), (na, nb_), _cost(D, na) + _cost(D, nb_) - costs[a] - costs[b]
        if a == b:                             # 2-opt: make c and v adjacent
            lo, hi = (i + 1, j) if i < j else (j + 1, i)
            if hi - lo < 1:
                return None
            new = ra[:lo] + ra[lo:hi + 1][::-1] + ra[hi + 1:]
            return (a,), (new,), _cost(D, new) - costs[a]
        # 2-opt*: c -> successor of v,  v -> successor of c  (exchange tails)
        na, nb_ = ra[:i + 1] + rb[j + 1:], rb[:j + 1] + ra[i + 1:]
        if int(d[na].sum()) > Q or int(d[nb_].sum()) > Q:
            return None
        return (a, b), (na, nb_), _cost(D, na) + _cost(D, nb_) - costs[a] - costs[b]

    def apply(idx, new):
        nonlocal routes, costs, loads
        for k, r in zip(idx, new):
            routes[k] = r
            costs[k] = _cost(D, r)
            loads[k] = int(d[r].sum()) if r else 0
            for c in r:
                where[c] = k
        n_empty = sum(1 for r in routes if not r)
        if n_empty != 1:                       # keep exactly one spare empty route
            keep = [k for k, r in enumerate(routes) if r]
            routes = [routes[k] for k in keep] + [[]]
            costs = [costs[k] for k in keep] + [0.0]
            loads = [loads[k] for k in keep] + [0]
            reindex()

    # ---------- initial temperature from a sample of uphill moves
    if t0 is None:
        ups = []
        for _ in range(1000):
            m = propose()
            if m and m[2] > 1e-9:
                ups.append(m[2])
        t0 = (np.mean(ups) / -math.log(p_accept0)) if ups else 1.0
    alpha = t_final_ratio ** (1.0 / iterations)

    best, best_routes = cur, [list(r) for r in routes if r]
    history = [(0, cur, best)]
    T, accepted = t0, 0
    start = time.perf_counter()
    it = 0
    for it in range(1, iterations + 1):
        m = propose()
        if m is not None:
            idx, new, delta = m
            if delta < 0 or rng.random() < math.exp(-delta / T):
                apply(idx, new)
                cur += delta
                accepted += 1
                if cur < best - 1e-9:
                    best = cur
                    best_routes = [list(r) for r in routes if r]
        T *= alpha
        if log_every and it % log_every == 0:
            history.append((it, cur, best))
        if time_limit is not None and (it & 1023) == 0 and time.perf_counter() - start > time_limit:
            break

    # ---------- polish: first-improvement descent from the best solution
    if polish:
        routes = [list(r) for r in best_routes] + [[]]
        costs = [_cost(D, r) for r in routes]
        loads = [int(d[r].sum()) for r in routes]
        reindex()
        cur = sum(costs)
        stall = 0
        while stall < 20 * n * len(NB[1]):
            m = propose()
            if m is not None and m[2] < -1e-9:
                apply(m[0], m[1])
                cur += m[2]
                stall = 0
            else:
                stall += 1
        if cur < best - 1e-9:
            best_routes = [list(r) for r in routes if r]
    runtime = time.perf_counter() - start

    best_routes = [r for r in best_routes if r]
    ok, msg = check_feasible(inst, best_routes)
    assert ok, msg
    return SAResult(best_routes, solution_cost(inst, best_routes), init_cost, runtime,
                    it, accepted, history)
```

**`tests/test_heuristics.py`**

```python
"""Heuristics must return feasible plans; SA must reach small proven optima."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvrp.heuristics import clarke_wright, simulated_annealing  # noqa: E402
from cvrp.instance import check_feasible, load_vrplib, solution_cost  # noqa: E402

RAW = ROOT / "data" / "raw" / "cvrplib"
NAMES = ["E-n13-k4", "P-n16-k8", "A-n32-k5"]


def _load(name):
    return load_vrplib(RAW / f"{name}.vrp", RAW / f"{name}.sol")


@pytest.mark.parametrize("name", NAMES)
def test_clarke_wright_feasible(name):
    inst = _load(name)
    routes = clarke_wright(inst)
    assert check_feasible(inst, routes)[0]
    assert solution_cost(inst, routes) >= inst.optimum - 1e-9


@pytest.mark.parametrize("name", ["E-n13-k4", "P-n16-k8"])
def test_sa_reaches_optimum(name):
    inst = _load(name)
    res = simulated_annealing(inst, seed=1)
    assert check_feasible(inst, res.routes)[0]
    assert res.cost == pytest.approx(inst.optimum)


def test_sa_never_worse_than_start():
    inst = _load("A-n32-k5")
    cw = clarke_wright(inst)
    res = simulated_annealing(inst, init=cw, seed=7)
    assert res.cost <= solution_cost(inst, cw) + 1e-9


if __name__ == "__main__":
    for n in NAMES:
        test_clarke_wright_feasible(n)
    for n in ["E-n13-k4", "P-n16-k8"]:
        test_sa_reaches_optimum(n)
    test_sa_never_worse_than_start()
    print("all heuristic tests passed")
```

**Run:**

```powershell
pytest -q
```

Expected: `11 passed` (5 exact + 6 heuristic).

```powershell
git add src/cvrp/heuristics.py tests/test_heuristics.py
git commit -m "feat(heuristics): Clarke-Wright savings and simulated annealing with granular neighbourhood"
git push -u origin feature/heuristics
```

---

## Step 8 — experiments and analysis (Ushan)

* **`run_experiments.py`** — runs Model B, Model A (MTZ), Clarke–Wright + SA (5 seeds) and the SA sensitivity study; writes one CSV row per run (resumable).
* **`analyse_results.py`** — builds the comparison tables (`results/tables/`) and all figures (`results/figures/`).

**`experiments/run_experiments.py`**

```python
"""Run all experiments and write results/*.csv + results/solutions/*.json.

Usage:
    python experiments/run_experiments.py exact        # Model B (cutting planes) on small instances
    python experiments/run_experiments.py mtz          # Model A (MTZ) baseline on small instances
    python experiments/run_experiments.py heuristics   # Clarke-Wright + SA (5 seeds) on all instances
    python experiments/run_experiments.py sensitivity  # SA iteration-budget study
    python experiments/run_experiments.py all

Every run appends one row to its CSV immediately, so an interrupted
experiment keeps the rows it has finished and resumes where it stopped.
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvrp.exact import solve_exact, solve_exact_cuts  # noqa: E402
from cvrp.heuristics import clarke_wright, simulated_annealing  # noqa: E402
from cvrp.instance import Instance, check_feasible, load_vrplib, solution_cost  # noqa: E402

RES = ROOT / "results"
SOL = RES / "solutions"
INST = ROOT / "data" / "instances"
RAW = ROOT / "data" / "raw" / "cvrplib"

SMALL_COL = [f"COL-n{n}-s{s}" for n in (8, 10, 12, 15, 18, 20, 25, 30) for s in (1, 2, 3)]
LARGE_COL = ["COL-n50-s1", "COL-n100-s1", "COL-n200-s1", "COL-n557-s1"]
BENCH_SMALL = ["E-n13-k4", "P-n16-k8", "B-n31-k5", "A-n32-k5"]
BENCH_LARGE = ["X-n101-k25", "X-n148-k46", "X-n200-k36"]

EXACT_TL = 300.0     # seconds per instance, exact methods
EXACT_LB_ONLY = ["COL-n50-s1", "COL-n100-s1", "COL-n200-s1"]  # exact run only to obtain a lower bound
SA_SEEDS = [1, 2, 3, 4, 5]


def load(name: str) -> Instance:
    if name.startswith("COL-"):
        return Instance.from_json(INST / f"{name}.json")
    return load_vrplib(RAW / f"{name}.vrp", RAW / f"{name}.sol")


def done_keys(path: Path, keys: tuple[str, ...]) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {tuple(row[k] for k in keys) for row in csv.DictReader(f)}


def append(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def save_solution(tag: str, name: str, routes, cost) -> None:
    SOL.mkdir(parents=True, exist_ok=True)
    (SOL / f"{name}__{tag}.json").write_text(json.dumps({"routes": routes, "cost": cost}))


# --------------------------------------------------------------------- exact
def run_exact(model: str) -> None:
    out = RES / f"exact_{model}.csv"
    names = BENCH_SMALL + SMALL_COL + (EXACT_LB_ONLY if model == "cuts" else [])
    if model == "mtz":   # MTZ baseline: stop at n=20, beyond that it only times out
        names = BENCH_SMALL[:2] + [x for x in SMALL_COL if int(x.split("-n")[1].split("-")[0]) <= 20]
    seen = done_keys(out, ("instance",))
    for name in names:
        if (name,) in seen:
            continue
        inst = load(name)
        if model == "cuts":
            # n >= 200: keep the 2n most violated cuts per round (memory bound, see report §6)
            cap = 2 * inst.n if inst.n >= 200 else None
            r = solve_exact_cuts(inst, time_limit=EXACT_TL, max_cuts_per_round=cap)
        else:
            r = solve_exact(inst, time_limit=EXACT_TL)
        feasible = check_feasible(inst, r.routes)[0] if r.routes else False
        row = {
            "instance": name, "n": inst.n, "model": model, "status": r.status,
            "cost": round(r.cost, 3) if r.cost is not None else "",
            "lower_bound": round(r.lower_bound, 3) if r.lower_bound is not None else "",
            "root_lb": round(getattr(r, "root_lb", float("nan")), 3) if model == "cuts" else "",
            "gap_pct": round(100 * r.gap, 3) if r.gap is not None else "",
            "runtime_s": round(r.runtime, 2), "vehicles": len(r.routes) if r.routes else "",
            "feasible": feasible, "n_vars": r.n_vars, "n_cons": r.n_cons,
            "known_opt": inst.optimum if inst.optimum is not None else "",
        }
        append(out, row)
        if r.routes:
            save_solution(f"exact_{model}", name, r.routes, r.cost)
        print(f"[{model}] {name}: {r.status} cost={row['cost']} lb={row['lower_bound']} "
              f"t={row['runtime_s']}s", flush=True)


# ---------------------------------------------------------------- heuristics
def run_heuristics() -> None:
    out = RES / "heuristics.csv"
    names = BENCH_SMALL + SMALL_COL + LARGE_COL + BENCH_LARGE
    seen = done_keys(out, ("instance", "method", "seed"))
    for name in names:
        inst = load(name)
        if (name, "clarke_wright", "0") not in seen:
            t = time.perf_counter()
            cw = clarke_wright(inst)
            tcw = time.perf_counter() - t
            append(out, {"instance": name, "n": inst.n, "method": "clarke_wright", "seed": 0,
                         "cost": round(solution_cost(inst, cw), 3), "runtime_s": round(tcw, 4),
                         "vehicles": len(cw), "feasible": check_feasible(inst, cw)[0],
                         "iterations": "", "known_opt": inst.optimum or ""})
            save_solution("cw", name, cw, solution_cost(inst, cw))
        else:
            cw = clarke_wright(inst)
        best = None
        for s in SA_SEEDS:
            if (name, "sa", str(s)) in seen:
                continue
            t = time.perf_counter()
            r = simulated_annealing(inst, init=cw, seed=s)
            total = time.perf_counter() - t      # includes T0 calibration and polish
            append(out, {"instance": name, "n": inst.n, "method": "sa", "seed": s,
                         "cost": round(r.cost, 3), "runtime_s": round(total, 3),
                         "vehicles": len(r.routes), "feasible": check_feasible(inst, r.routes)[0],
                         "iterations": r.iterations, "known_opt": inst.optimum or ""})
            if best is None or r.cost < best.cost:
                best = r
        if best is not None:
            save_solution("sa_best", name, best.routes, best.cost)
        print(f"[heur] {name} done", flush=True)


def run_sensitivity() -> None:
    """Quality vs iteration budget, plus one convergence trace."""
    out = RES / "sa_sensitivity.csv"
    seen = done_keys(out, ("instance", "iters_per_customer", "seed"))
    for name in ["COL-n100-s1", "COL-n200-s1", "X-n101-k25"]:
        inst = load(name)
        cw = clarke_wright(inst)
        for mult in (500, 1000, 4000, 16000, 32000):
            for s in (1, 2, 3):
                if (name, str(mult), str(s)) in seen:
                    continue
                t = time.perf_counter()
                r = simulated_annealing(inst, init=cw, seed=s, iterations=mult * inst.n)
                append(out, {"instance": name, "n": inst.n, "iters_per_customer": mult, "seed": s,
                             "cost": round(r.cost, 3), "runtime_s": round(time.perf_counter() - t, 3),
                             "known_opt": inst.optimum or ""})
        print(f"[sens] {name} done", flush=True)

    trace = RES / "sa_convergence_COL-n100-s1.csv"
    if not trace.exists():
        inst = load("COL-n100-s1")
        r = simulated_annealing(inst, seed=1, log_every=2000, polish=False)
        with trace.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "current", "best"])
            w.writerows(r.history)


def write_env() -> None:
    import numpy, scipy
    (RES / "environment.txt").write_text(
        f"python {platform.python_version()}\nnumpy {numpy.__version__}\nscipy {scipy.__version__}\n"
        f"platform {platform.platform()}\nprocessor {platform.processor() or 'x86_64'}\n")


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    write_env()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("exact", "all"):
        run_exact("cuts")
    if what in ("mtz", "all"):
        run_exact("mtz")
    if what in ("heuristics", "all"):
        run_heuristics()
    if what in ("sensitivity", "all"):
        run_sensitivity()
```

**`experiments/analyse_results.py`**

```python
"""Turn results/*.csv into the report tables (results/tables/*.csv|.md) and
figures (results/figures/*.png).

Usage:  python experiments/analyse_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cvrp.instance import Instance

RES = ROOT / "results"
FIG = RES / "figures"
TAB = RES / "tables"
SOL = RES / "solutions"

# fixed method -> colour/marker mapping (colour follows the method on every chart)
STYLE = {
    "Exact B (capacity cuts)": dict(color="#2a78d6", marker="o"),
    "Simulated Annealing":     dict(color="#eb6834", marker="s"),
    "Clarke-Wright":           dict(color="#1baf7a", marker="^"),
    "Exact A (MTZ)":           dict(color="#eda100", marker="D"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 5,
    "legend.frameon": False,
})


def load():
    cuts = pd.read_csv(RES / "exact_cuts.csv")
    mtz = pd.read_csv(RES / "exact_mtz.csv") if (RES / "exact_mtz.csv").exists() else None
    heur = pd.read_csv(RES / "heuristics.csv")
    return cuts, mtz, heur


def reference_values(cuts: pd.DataFrame, heur: pd.DataFrame) -> pd.DataFrame:
    """Best reference per instance: proven optimum > best-known (CVRPLIB) > exact lower bound."""
    rows = []
    for name, g in heur.groupby("instance"):
        n = int(g.n.iloc[0])
        known = g.known_opt.dropna()
        c = cuts[cuts.instance == name]
        ref, kind, lb = np.nan, "", np.nan
        if len(c):
            lb = c.lower_bound.iloc[0]
            if c.status.iloc[0] == "optimal":
                ref, kind = c.cost.iloc[0], "proven optimum"
        if np.isnan(ref) and len(known) and known.iloc[0] != "":
            ref, kind = float(known.iloc[0]), "best-known (CVRPLIB)"
        if np.isnan(ref) and not np.isnan(lb):
            ref, kind = lb, "lower bound"
        rows.append({"instance": name, "n": n, "reference": ref, "ref_type": kind, "lower_bound": lb})
    return pd.DataFrame(rows)


def instance_order(names):
    def key(x):
        if x.startswith("COL"):
            n, s = x[5:].split("-s")
            return (0, int(n), int(s))
        return (1, int(x.split("-n")[1].split("-")[0]), 0)
    return sorted(names, key=key)


def build_tables(cuts, mtz, heur, ref):
    TAB.mkdir(parents=True, exist_ok=True)
    cw = heur[heur.method == "clarke_wright"].set_index("instance")
    sa = heur[heur.method == "sa"].groupby("instance").agg(
        sa_best=("cost", "min"), sa_mean=("cost", "mean"), sa_std=("cost", "std"),
        sa_time=("runtime_s", "mean"), sa_vehicles=("vehicles", "min"),
        sa_feasible=("feasible", "all"))
    r = ref.set_index("instance")
    rows = []
    for name in instance_order(r.index):
        R = r.loc[name]
        c = cuts[cuts.instance == name]
        m = mtz[mtz.instance == name] if mtz is not None else pd.DataFrame()
        refv = R.reference

        def gap(v):
            return 100 * (v / refv - 1) if pd.notna(v) and pd.notna(refv) else np.nan

        rows.append({
            "instance": name, "n": int(R.n), "reference": refv, "ref_type": R.ref_type,
            "B_status": c.status.iloc[0] if len(c) else "not run",
            "B_cost": c.cost.iloc[0] if len(c) else np.nan,
            "B_lb": c.lower_bound.iloc[0] if len(c) else np.nan,
            "B_time": c.runtime_s.iloc[0] if len(c) else np.nan,
            "A_status": m.status.iloc[0] if len(m) else "not run",
            "A_cost": m.cost.iloc[0] if len(m) else np.nan,
            "A_gap_to_ref": gap(m.cost.iloc[0]) if len(m) else np.nan,
            "A_time": m.runtime_s.iloc[0] if len(m) else np.nan,
            "CW_cost": cw.loc[name, "cost"], "CW_gap": gap(cw.loc[name, "cost"]),
            "CW_time": cw.loc[name, "runtime_s"],
            "SA_best": sa.loc[name, "sa_best"], "SA_mean": sa.loc[name, "sa_mean"],
            "SA_std": sa.loc[name, "sa_std"],
            "SA_best_gap": gap(sa.loc[name, "sa_best"]), "SA_mean_gap": gap(sa.loc[name, "sa_mean"]),
            "SA_time": sa.loc[name, "sa_time"], "SA_vehicles": sa.loc[name, "sa_vehicles"],
            "all_feasible": bool(sa.loc[name, "sa_feasible"] and cw.loc[name, "feasible"]),
        })
    full = pd.DataFrame(rows)
    full.to_csv(TAB / "comparison_full.csv", index=False)

    # compact markdown tables for the report
    def fmt(v, p=2):
        if pd.isna(v):
            return "–"
        return f"{0.0 if abs(v) < 0.5 * 10 ** -p else v:.{p}f}"

    lines = ["| Instance | n | Reference | Exact B | B (s) | MTZ (s) | CW gap % | SA mean % | SA best % | SA (s) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for x in full.itertuples():
        bstat = {"optimal": "optimal", "time_limit": f"LB {fmt(x.B_lb)}", "not run": "–"}.get(x.B_status, x.B_status)
        mtz_t = "–" if x.A_status == "not run" else (fmt(x.A_time, 1) if x.A_status == "optimal" else f">{fmt(x.A_time, 0)} ({fmt(x.A_gap_to_ref, 1)}%)")
        reft = {"proven optimum": "*", "best-known (CVRPLIB)": "†", "lower bound": "‡"}.get(x.ref_type, "")
        lines.append(f"| {x.instance} | {x.n} | {fmt(x.reference)}{reft} | {bstat} | {fmt(x.B_time, 2)} | {mtz_t} | "
                     f"{fmt(x.CW_gap)} | {fmt(x.SA_mean_gap)} | {fmt(x.SA_best_gap)} | {fmt(x.SA_time, 2)} |")
    lines.append("")
    lines.append("\\* proven optimum (Exact B); † best-known solution from CVRPLIB; ‡ Exact B lower bound "
                 "(time limit reached), so the gap is an upper bound on the true gap. B (s), MTZ (s), SA (s): "
                 "runtime in seconds; for MTZ, >300 means the time limit was reached and the bracket gives "
                 "the gap of its best solution to the reference. SA columns: mean / best gap over 5 seeds.")
    (TAB / "comparison.md").write_text("\n".join(lines))
    return full


def summary_by_size(full: pd.DataFrame) -> pd.DataFrame:
    col = full[full.instance.str.startswith("COL")]
    g = col.groupby("n").agg(
        instances=("instance", "count"),
        B_solved=("B_status", lambda s: int((s == "optimal").sum())),
        B_time_median=("B_time", "median"),
        A_solved=("A_status", lambda s: int((s == "optimal").sum())),
        A_time_median=("A_time", "median"),
        CW_gap_mean=("CW_gap", "mean"), SA_gap_mean=("SA_mean_gap", "mean"),
        SA_time_mean=("SA_time", "mean"), CW_time_mean=("CW_time", "mean"))
    g.to_csv(TAB / "summary_by_size.csv")
    return g


# ------------------------------------------------------------------ figures
def fig_runtime(full: pd.DataFrame):
    col = full[full.instance.str.startswith("COL")]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    a = col[col.A_status != "not run"]
    series = {
        "Exact B (capacity cuts)": (col.groupby("n").B_time.median(), col.groupby("n").B_status.apply(lambda s: (s == "optimal").mean() >= 0.5)),
        "Exact A (MTZ)": (a.groupby("n").A_time.median(), a.groupby("n").A_status.apply(lambda s: (s == "optimal").mean() >= 0.5)),
        "Simulated Annealing": (col.groupby("n").SA_time.mean(), None),
        "Clarke-Wright": (col.groupby("n").CW_time.mean(), None),
    }
    for label, (s, solved) in series.items():
        s = s.dropna()
        st = STYLE[label]
        ax.plot(s.index, s.values, label=label, color=st["color"], marker=st["marker"])
        if solved is not None:
            lim = [n for n in s.index if not solved.get(n, True)]
            if lim:   # median run stopped at the time limit: hollow marker
                ax.scatter(lim, s.loc[lim], s=70, facecolors="white", edgecolors=st["color"],
                           linewidths=2, zorder=5, marker=st["marker"])
    ax.axhline(300, color=MUTED, lw=1, ls="--")
    ax.text(557, 380, "exact time limit (300 s)", color=MUTED, fontsize=8, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ticks = [8, 15, 30, 50, 100, 200, 557]
    ax.set_xticks(ticks); ax.set_xticklabels(ticks); ax.minorticks_off()
    ax.set_ylim(top=1500)
    ax.set_xlabel("Number of outlets n (log scale)")
    ax.set_ylabel("Runtime (s, log scale)")
    ax.set_title("Runtime vs problem size – Colombo instances")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "runtime_vs_size.png"); plt.close(fig)


def fig_gap(full: pd.DataFrame):
    """(a) Colombo instances with a proven optimum; (b) large benchmarks vs best-known."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.5), gridspec_kw={"width_ratios": [1.6, 1]})
    col = full[full.instance.str.startswith("COL") & (full.ref_type == "proven optimum")]
    for label, colname, dx in (("Clarke-Wright", "CW_gap", -0.6), ("Simulated Annealing", "SA_mean_gap", 0.6)):
        st = STYLE[label]
        s = col.groupby("n")[colname].mean()
        ax1.plot(s.index, s.values, label=f"{label} (mean)", color=st["color"], marker=st["marker"])
        ax1.scatter(col.n + dx, col[colname], s=12, alpha=0.45, color=st["color"], linewidths=0)
    ax1.set_xlabel("Number of outlets n")
    ax1.set_ylabel("Gap to proven optimum (%)")
    ax1.set_title("(a) Colombo, 8–30 outlets (23 proven optima)")
    ax1.legend(fontsize=7.5)

    big = full[full.instance.str.startswith("X-")].copy()
    xs = np.arange(len(big))
    w = 0.38
    ax2.bar(xs - w / 2 - 0.01, big.CW_gap, w, color=STYLE["Clarke-Wright"]["color"], label="Clarke-Wright")
    ax2.bar(xs + w / 2 + 0.01, big.SA_mean_gap, w, color=STYLE["Simulated Annealing"]["color"], label="SA (mean of 5)")
    for x, v in zip(xs, big.SA_mean_gap):
        ax2.text(x + w / 2, v + 0.08, f"{v:.1f}", ha="center", fontsize=7, color=INK)
    for x, v in zip(xs, big.CW_gap):
        ax2.text(x - w / 2, v + 0.08, f"{v:.1f}", ha="center", fontsize=7, color=INK)
    ax2.set_xticks(xs); ax2.set_xticklabels([x.split("-k")[0] for x in big.instance], fontsize=8)
    ax2.set_ylabel("Gap to best-known (%)")
    ax2.set_title("(b) CVRPLIB, 100–199 customers")
    ax2.legend(fontsize=7.5); ax2.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "gap_vs_size.png"); plt.close(fig)


def fig_sensitivity():
    p = RES / "sa_sensitivity.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    heur = pd.read_csv(RES / "heuristics.csv")
    # reference: CVRPLIB best-known for X instances, else best cost found in ANY run
    best_any = pd.concat([s[["instance", "cost"]], heur[["instance", "cost"]]]).groupby("instance").cost.min()
    ref = {}
    for name in s.instance.unique():
        k = s[s.instance == name].known_opt.dropna()
        ref[name] = float(k.iloc[0]) if len(k) else best_any[name]
    s["gap"] = 100 * (s.cost / s.instance.map(ref) - 1)
    g = s.groupby(["instance", "iters_per_customer"]).agg(gap=("gap", "mean"), t=("runtime_s", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for (name, h), c in zip(g.groupby("instance"), colors):
        lab = name + (" (vs best-known)" if name.startswith("X") else " (vs best found)")
        ax.plot(h.t, h.gap, color=c, marker="o", label=lab)
        for x in h.itertuples():
            ax.annotate(f"{x.iters_per_customer / 1000:g}k", (x.t, x.gap), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("Mean SA runtime (s, log scale); labels = iterations per customer")
    ax.set_ylabel("Mean gap (%)")
    ax.set_title("SA quality–time trade-off (3 seeds per point)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "sa_sensitivity.png"); plt.close(fig)


def fig_convergence():
    p = RES / "sa_convergence_COL-n100-s1.csv"
    if not p.exists():
        return
    h = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(h.iteration, h.current, color="#9b9a95", lw=1, label="current solution")
    ax.plot(h.iteration, h.best, color=STYLE["Simulated Annealing"]["color"], label="best so far")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Total distance (km)")
    ax.set_title("Simulated Annealing convergence – COL-n100-s1")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "sa_convergence.png"); plt.close(fig)


def _draw_routes(ax, inst: Instance, routes, title, single_color=None):
    xy = inst.coords
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    for k, r in enumerate(routes):
        path = [0] + r + [0]
        col = single_color or palette[k % len(palette)]
        ax.plot(xy[path, 0], xy[path, 1], color=col, lw=1.2 if single_color else 1.6, alpha=0.9)
    ax.scatter(xy[1:, 0], xy[1:, 1], s=6 + 2 * inst.demand[1:], color="#52514e", zorder=3, linewidths=0)
    ax.scatter(xy[0, 0], xy[0, 1], marker="s", s=60, color="#e34948", zorder=4)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect(1 / 0.993)


def fig_routes(full: pd.DataFrame):
    name = "COL-n30-s2"
    p_cw, p_ex = SOL / f"{name}__cw.json", SOL / f"{name}__exact_cuts.json"
    if p_cw.exists() and p_ex.exists():
        inst = Instance.from_json(ROOT / "data" / "instances" / f"{name}.json")
        cw, ex = json.loads(p_cw.read_text()), json.loads(p_ex.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0))
        _draw_routes(axes[0], inst, cw["routes"], f"Clarke-Wright: {cw['cost']:.1f} km, {len(cw['routes'])} trucks")
        _draw_routes(axes[1], inst, ex["routes"], f"Exact optimum: {ex['cost']:.1f} km, {len(ex['routes'])} trucks")
        fig.suptitle(f"{name}: delivery routes from the Kelaniya DC (red square)", fontsize=10)
        fig.tight_layout(); fig.savefig(FIG / "routes_n30.png"); plt.close(fig)

    name = "COL-n557-s1"
    p = SOL / f"{name}__sa_best.json"
    if p.exists():
        inst = Instance.from_json(ROOT / "data" / "instances" / f"{name}.json")
        sol = json.loads(p.read_text())
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        _draw_routes(ax, inst, sol["routes"],
                     f"All 557 outlets – SA solution: {sol['cost']:.0f} km, {len(sol['routes'])} truck routes",
                     single_color="#2a78d6")
        fig.tight_layout(); fig.savefig(FIG / "routes_n557.png"); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cuts, mtz, heur = load()
    ref = reference_values(cuts, heur)
    full = build_tables(cuts, mtz, heur, ref)
    summary = summary_by_size(full)
    fig_runtime(full); fig_gap(full); fig_sensitivity(); fig_convergence(); fig_routes(full)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string())
    print((TAB / "comparison.md").read_text())


if __name__ == "__main__":
    main()
```

**Results.** The full experiment run takes **~2 hours** (exact solver time limit 300 s per instance). The report's numbers come from our run, so **use the provided `results_for_step8.zip`** to keep report and repo consistent:

1. Unzip `results_for_step8.zip` into the project root (it contains `results/…` paths).
2. Regenerate tables and figures from those CSVs (seconds) — this proves the analysis pipeline works:

```powershell
python experiments/analyse_results.py
```

Expected: a per-size summary table and the markdown comparison table; figures appear in `results/figures/` (runtime_vs_size, gap_vs_size, sa_sensitivity, sa_convergence, routes_n30, routes_n557).

*Optional reproduction check (any time later):* `python experiments/run_experiments.py heuristics` re-runs the heuristic part in a few minutes — but it **appends** to the CSVs, so do it on a copy of the project, not before committing.

```powershell
git add experiments/run_experiments.py experiments/analyse_results.py
git commit -m "feat(experiments): experiment runner (exact, MTZ, heuristics, sensitivity) and analysis"
git add results
git commit -m "results: experiment results, comparison tables and figures"
git push
```

**Pull request:** on GitHub → **Compare & pull request** → base `develop` ← compare `feature/heuristics` → title *"Heuristics, experiments and results"* → Kavindu reviews → **Merge pull request** (normal merge, not squash). Then both run:

```powershell
git checkout develop
git pull
```

---

## Step 9 — notebook (joint, on `develop`)

`build_notebook.py` writes **and executes** `notebooks/delivery_optimization.ipynb` (data → worked example → exact vs heuristic → results), embedding outputs and figures so it renders on GitHub. You can also open the generated notebook in IntelliJ/Jupyter and use it in the video.

**`notebooks/build_notebook.py`**

```python
"""Build and execute notebooks/delivery_optimization.ipynb.

The notebook is the guided walkthrough of the project (data -> model ->
exact -> heuristic -> results). This script writes it cell by cell and
executes it, embedding text and figure outputs, so it renders fully on
GitHub. If Jupyter is installed you can instead open the notebook and run
"Restart & Run All".

Usage:  python notebooks/build_notebook.py
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "delivery_optimization.ipynb"

CELLS: list[tuple[str, str]] = [
("md", """# Delivery Route Optimisation for a Distribution Centre in Colombo District
**IT5082 Optimization Methods – Programming Assignment**
Kavindu Ariyasinghe (MS26900186) · Ushan Gunaratne (MS26901930)

A distribution centre (DC) in **Kelaniya** delivers crates to retail outlets across **Colombo District** with identical light trucks (capacity **30 crates**). Every day a subset of outlets orders; we must decide **which truck serves which outlets and in what order**, so that every order is delivered once, no truck is overloaded and the **total distance driven is minimal**.

Mathematically this is the **Capacitated Vehicle Routing Problem (CVRP)**, an NP-hard problem. This notebook compares:
* **Exact:** MILP – Model A (MTZ, compact) and Model B (two-index with lazy rounded capacity cuts), solved with HiGHS (`scipy.optimize.milp`)
* **Heuristic:** Clarke–Wright savings (greedy) → **Simulated Annealing** (metaheuristic)"""),
("code", """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from cvrp.colombo import build_instance, describe, load_gnd, CAPACITY, CIRCUITY
from cvrp.instance import Instance, load_vrplib, solution_cost, check_feasible
from cvrp.exact import solve_exact, solve_exact_cuts
from cvrp.heuristics import clarke_wright, simulated_annealing
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)"""),
("md", """## 1. Data
Outlet locations and demand come from **real data**: the 557 Grama Niladhari Divisions (GNDs) of Colombo District with their centroid coordinates and **2012 Census population** (Department of Census & Statistics, via the open `nuuuwan/gig-data` repository).

Modelling assumptions:
* one outlet per GND; demand $d_i = \\lceil \\text{population}_i / 1000 \\rceil$ crates per day
* road distance = haversine distance × 1.3 (typical road circuity factor)
* truck capacity $Q = 30$ crates, unlimited identical fleet"""),
("code", """gnd = load_gnd()
print(f"{len(gnd)} GNDs, total population {gnd.population.sum():,}")
gnd.head()"""),
("code", """gnd.population.describe().round(0)"""),
("code", """from IPython.display import Image, display
display(Image(filename=str(ROOT / "results/figures/colombo_outlets_map.png"), width=520))"""),
("md", """Daily instances are created by sampling `n` ordering outlets with a fixed random seed (8–30 outlets × 3 seeds for the exact study; 50, 100, 200 and all 557 for scalability). CVRPLIB benchmarks with **proven optima** are used to validate both solvers."""),
("code", """summary = pd.read_csv(ROOT / "results/dataset_summary.csv")
summary[["instance","customers","total_demand","min_vehicles","demand_max","mean_depot_km"]].round(1)"""),
("md", """## 2. A worked example – 15 outlets
We build one instance and solve it with every method."""),
("code", """inst = build_instance(15, seed=2)
print(describe(inst))
pd.DataFrame({"outlet": inst.labels[1:], "demand (crates)": inst.demand[1:],
              "km from DC": inst.dist[0, 1:].round(1)}).head(8)"""),
("md", """### 2.1 Exact – Model A (MTZ) vs Model B (capacity cuts)
Both are the same CVRP; they differ only in how capacity and sub-tours are written (see report §3). Model B's LP relaxation is much tighter, so branch-and-bound has far less work to do."""),
("code", """a = solve_exact(inst, time_limit=60)
b = solve_exact_cuts(inst, time_limit=60)
for label, r in (("Model A (MTZ)", a), ("Model B (cuts)", b)):
    print(f"{label:15s} status={r.status:10s} cost={r.cost} lower bound={r.lower_bound:.2f} time={r.runtime:.2f}s")
print("optimal routes:", b.routes)"""),
("md", """### 2.2 Heuristic – Clarke–Wright savings + Simulated Annealing"""),
("code", """cw = clarke_wright(inst)
sa = simulated_annealing(inst, init=cw, seed=1)
print(f"Clarke-Wright : {solution_cost(inst, cw):.2f} km  ({len(cw)} trucks)  feasible={check_feasible(inst, cw)[0]}")
print(f"Sim. Annealing: {sa.cost:.2f} km  ({len(sa.routes)} trucks)  time={sa.runtime:.2f}s")
print(f"Exact optimum : {b.cost:.2f} km")"""),
("code", """def plot_routes(ax, inst, routes, title):
    xy = inst.coords
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    for k, r in enumerate(routes):
        p = [0] + r + [0]
        ax.plot(xy[p, 0], xy[p, 1], color=colours[k % 8], lw=1.8)
    ax.scatter(xy[1:, 0], xy[1:, 1], s=10 + 3 * inst.demand[1:], color="#52514e", zorder=3)
    ax.scatter(*xy[0], marker="s", s=80, color="#e34948", zorder=4)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
plot_routes(axes[0], inst, cw, f"Clarke-Wright {solution_cost(inst, cw):.1f} km")
plot_routes(axes[1], inst, b.routes, f"Exact optimum {b.cost:.1f} km")
plt.tight_layout(); plt.show()"""),
("md", """### 2.3 Validation on CVRPLIB benchmarks with proven optima"""),
("code", """rows = []
for name in ["E-n13-k4", "P-n16-k8", "B-n31-k5", "A-n32-k5"]:
    bi = load_vrplib(ROOT / f"data/raw/cvrplib/{name}.vrp", ROOT / f"data/raw/cvrplib/{name}.sol")
    r = solve_exact_cuts(bi, time_limit=120)
    s = simulated_annealing(bi, seed=1)
    rows.append({"instance": name, "published optimum": bi.optimum, "Exact B": r.cost,
                 "Exact B time (s)": round(r.runtime, 2), "SA": s.cost, "SA time (s)": round(s.runtime, 2)})
pd.DataFrame(rows)"""),
("md", """## 3. Experimental comparison
All experiments are produced by `experiments/run_experiments.py` (exact limit 300 s; SA 5 seeds) and summarised by `experiments/analyse_results.py`. Machine: 2-core Intel Xeon 2.8 GHz, Python 3.13, SciPy 1.18 (HiGHS)."""),
("code", """full = pd.read_csv(ROOT / "results/tables/comparison_full.csv")
cols = ["instance","n","reference","ref_type","B_status","B_time","A_status","A_time","CW_gap","SA_mean_gap","SA_best_gap","SA_time"]
full[cols].round(2)"""),
("code", """pd.read_csv(ROOT / "results/tables/summary_by_size.csv").round(3)"""),
("code", """for f in ["runtime_vs_size", "gap_vs_size", "sa_sensitivity", "sa_convergence", "routes_n30", "routes_n557"]:
    p = ROOT / f"results/figures/{f}.png"
    if p.exists():
        display(Image(filename=str(p), width=620))"""),
("md", """## 4. Findings
* **Exact B** proves optimality on 23 of the 24 Colombo instances with 8–30 outlets (and on all four classic benchmarks, including A-n32-k5), but runtime grows sharply and erratically with *n*; from 50 outlets on it only returns a lower bound within 5 minutes, which is the NP-hardness wall.
* **Formulation matters:** Model A (MTZ) with the same solver times out on one 15-outlet instance and on all 18–20-outlet instances, because its LP relaxation is weak.
* **Clarke–Wright** is near-instant (< 0.2 s even for 557 outlets) but typically several % above optimal.
* **Simulated Annealing** closes most of that gap: 0.13% above the proven optimum on average (90 of 115 runs exactly optimal), within 3% of the best-known solutions on the 100–199-customer benchmarks, and all 557 outlets planned in under 20 s.
* **Every solution produced by every method was checked feasible** (each outlet once, no truck over 30 crates).

See the report for the full discussion and future work."""),
]


def _new_cell(kind: str, src: str) -> dict:
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def build_and_execute() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.chdir(HERE)
    ns: dict = {"__name__": "__main__"}
    display_buffer: list[dict] = []

    # minimal IPython.display stand-in so the notebook code runs headless
    class _Image:
        def __init__(self, filename, width=None):
            self.data = base64.b64encode(Path(filename).read_bytes()).decode()

    def _display(obj):
        if isinstance(obj, _Image):
            display_buffer.append({"output_type": "display_data", "metadata": {},
                                   "data": {"image/png": obj.data, "text/plain": ["<Image>"]}})

    # headless stand-ins for IPython.display (the import line is skipped when executing here;
    # in real Jupyter the genuine IPython.display is used)
    ns["Image"], ns["display"] = _Image, _display

    def _show(*a, **k):
        for num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
            display_buffer.append({"output_type": "display_data", "metadata": {},
                                   "data": {"image/png": base64.b64encode(buf.getvalue()).decode(),
                                            "text/plain": ["<Figure>"]}})
        plt.close("all")
    plt.show = _show

    cells, count = [], 0
    for kind, src in CELLS:
        cell = _new_cell(kind, src)
        if kind == "code":
            count += 1
            cell["execution_count"] = count
            display_buffer.clear()
            out = io.StringIO()
            last_value = None
            try:
                with contextlib.redirect_stdout(out):
                    import ast
                    run_src = src.replace("from IPython.display import Image, display", "")
                    tree = ast.parse(run_src)
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        last = ast.Expression(tree.body.pop().value)
                        exec(compile(tree, "<cell>", "exec"), ns)
                        last_value = eval(compile(last, "<cell>", "eval"), ns)
                    else:
                        exec(compile(tree, "<cell>", "exec"), ns)
            except Exception:
                out.write(traceback.format_exc())
            outputs = []
            if out.getvalue():
                outputs.append({"output_type": "stream", "name": "stdout",
                                "text": out.getvalue().splitlines(keepends=True)})
            outputs += list(display_buffer)
            if last_value is not None:
                data = {"text/plain": repr(last_value).splitlines(keepends=True)}
                if hasattr(last_value, "_repr_html_"):
                    data["text/html"] = last_value._repr_html_().splitlines(keepends=True)
                outputs.append({"output_type": "execute_result", "execution_count": count,
                                "metadata": {}, "data": data})
            cell["outputs"] = outputs
            print(f"cell {count} done", file=sys.stderr)
        cells.append(cell)

    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]}},
        "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build_and_execute()
```

**Run** (≈1 minute — it solves the benchmarks live):

```powershell
python notebooks/build_notebook.py
```

Expected: `cell 1 done … cell 13 done` then `wrote …delivery_optimization.ipynb`.

```powershell
git add notebooks
git commit -m "docs(notebook): executed walkthrough notebook"
git push
```

---

## Step 10 — report, submission files, release (joint, on `develop`)

### 10.1 Report sources

Create these files (section text: **read it and reword it in your own style** — Turnitin and the viva).


**`report/metadata.yaml`**

```yaml
---
title: "Delivery Route Optimisation for a Distribution Centre in Colombo District"
subtitle: "IT5082 Optimization Methods — Programming Assignment: Solving a Real-World Optimization Problem"
author:
  - "Kavindu Ariyasinghe (MS26900186)"
  - "Ushan Gunaratne (MS26901930)"
date: "October 2026"
documentclass: article
classoption: [11pt, a4paper]
geometry: "margin=2.2cm"
fontfamily: fontspec   # avoids the template default lmodern (not installed)
mainfont: "DejaVu Serif"
sansfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
fontsize: 11pt
linestretch: 1.12
colorlinks: true
linkcolor: "blue"
urlcolor: "blue"
toc: true
toc-depth: 2
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \usepackage{caption}
  - \captionsetup{font=small,labelfont=bf}
  - \setlength{\tabcolsep}{4pt}
  - \renewcommand{\arraystretch}{1.15}
abstract: |
  We model the daily deliveries of a distribution centre in Kelaniya to retail outlets across Colombo District as a Capacitated Vehicle Routing Problem (CVRP). Outlet locations and demand come from the 557 Grama Niladhari Divisions of the district and their 2012 Census population. We compare two exact mixed-integer programming formulations, solved with HiGHS (a compact Miller–Tucker–Zemlin model, and a two-index model with lazily generated rounded capacity cuts), against the Clarke–Wright savings heuristic and a Simulated Annealing metaheuristic with a granular neighbourhood. On 27 instances with proven optima, the capacity-cut model proves optimality up to 30 outlets, usually within seconds, while the MTZ model already times out on one 15-outlet instance and on every instance from 18 outlets. Simulated Annealing reaches the optimum in 90 of 115 runs (0.13% mean gap), stays within 3% of best-known solutions on 100–199-customer benchmarks, and plans all 557 outlets in under 20 seconds. Exact methods certify optimality and bound heuristic quality on small instances; the metaheuristic is the only practical option at district scale.
---
```


**`report/table_widths.lua`**

```lua
-- Tables whose cells are all short get natural column widths (no wrapping);
-- tables with long text keep pandoc's relative widths so they wrap inside the page.
local function cell_len(cell)
  return #pandoc.utils.stringify(cell.contents)
end

function Table(tbl)
  local maxlen = 0
  local function scan(rows)
    for _, row in ipairs(rows) do
      for _, cell in ipairs(row.cells) do
        maxlen = math.max(maxlen, cell_len(cell))
      end
    end
  end
  scan(tbl.head.rows)
  for _, body in ipairs(tbl.bodies) do scan(body.body) end
  if maxlen <= 24 then
    for i, spec in ipairs(tbl.colspecs) do
      tbl.colspecs[i] = {spec[1], pandoc.ColWidthDefault}
    end
  end
  -- wide tables: smaller font and tighter column spacing
  local size = (#tbl.colspecs >= 8) and "\\footnotesize\\setlength{\\tabcolsep}{3pt}" or "\\small"
  return {
    pandoc.RawBlock("latex", "\\begingroup" .. size),
    tbl,
    pandoc.RawBlock("latex", "\\endgroup"),
  }
end
```


**`report/build_report.sh`**

````bash
#!/usr/bin/env bash
# Build report/Optimization_Report.pdf and report/code_appendix.txt
# Requires: pandoc + a LaTeX engine (xelatex).
set -euo pipefail
cd "$(dirname "$0")"

# Appendix A is regenerated from the analysis output so it always matches results/
{
  echo '\newpage'
  echo
  echo '# Appendix A. Full results per instance'
  echo
  echo 'Costs in km (Colombo) or CVRPLIB units; gaps are relative to the Reference column.'
  echo
  cat ../results/tables/comparison.md
  echo
  echo
  echo '# Appendix B. Reproducing the results'
  echo
  echo '```bash'
  echo 'pip install -r requirements.txt'
  echo 'python experiments/build_instances.py      # instances + dataset figure'
  echo 'python experiments/run_experiments.py all  # ~2 h on 2 cores (exact limit 300 s)'
  echo 'python experiments/analyse_results.py      # tables + figures in results/'
  echo 'python notebooks/build_notebook.py         # executed walkthrough notebook'
  echo 'pytest -q                                  # solver regression tests'
  echo '```'
} > sections/10_appendix.md

pandoc metadata.yaml sections/0[1-8]_*.md sections/09_references.md sections/10_appendix.md \
  --resource-path=.:sections:.. \
  --pdf-engine=xelatex --lua-filter=table_widths.lua \
  -o Optimization_Report.pdf
echo "wrote report/Optimization_Report.pdf"

# Code appendix (plain text) - same script works on Windows: python report/make_code_appendix.py
python3 make_code_appendix.py
````


**`report/make_code_appendix.py`**

```python
"""Write report/code_appendix.txt: every source file as plain text with headers.

Usage:  python report/make_code_appendix.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "src/cvrp/__init__.py", "src/cvrp/instance.py", "src/cvrp/colombo.py",
    "src/cvrp/exact.py", "src/cvrp/heuristics.py",
    "experiments/extract_census_data.py", "experiments/build_instances.py",
    "experiments/run_experiments.py", "experiments/analyse_results.py",
    "notebooks/build_notebook.py", "tests/test_exact.py", "tests/test_heuristics.py",
    "requirements.txt",
]

parts = [
    "CODE APPENDIX - Delivery Route Optimisation for a Distribution Centre in Colombo District",
    "Kavindu Ariyasinghe (MS26900186), Ushan Gunaratne (MS26901930)",
    "",
]
for rel in FILES:
    parts += ["=" * 82, f"FILE: {rel}", "=" * 82,
              (ROOT / rel).read_text(encoding="utf-8"), ""]
out = ROOT / "report" / "code_appendix.txt"
out.write_text("\n".join(parts), encoding="utf-8")
print(f"wrote {out} ({len(FILES)} files)")
```


**`report/sections/01_introduction.md`**

```markdown
# 1. Introduction and Problem Selection

Last-mile and outlet distribution is one of the largest variable costs in retail supply chains. Fuel, driver hours and vehicle wear all scale with the distance driven. Colombo District is Sri Lanka's most densely populated district. A distributor serving its retail outlets every day must decide:

* **which truck** delivers to **which outlets**;
* **in what order** each truck visits them.

The decision must satisfy two rules: every order is delivered exactly once, and no truck carries more than its capacity. Small differences in route design add up across hundreds of deliveries per day.

We study a distributor operating a **distribution centre (DC) in Kelaniya**, just across the Kelani River from Colombo city and close to the main arterial roads. On a given day a subset of the district's outlets places orders measured in crates. The DC has an unlimited pool of identical light trucks, each carrying 30 crates. The objective is to **minimise total distance driven**. Because each truck route also starts at and returns to the DC, the number of trucks used falls out of the optimisation as a by-product.

This is the **Capacitated Vehicle Routing Problem (CVRP)** of Dantzig and Ramser (1959), the core model behind delivery optimisation. We chose it because it is:

* **Realistic.** The data are real Sri Lankan administrative units and census populations, and the scenario mirrors daily FMCG distribution.
* **Non-trivial.** CVRP is NP-hard: it generalises both the Travelling Salesman Problem and Bin Packing. Real instances with a few dozen customers already challenge exact solvers.
* **Well suited to comparing methods.** Published benchmark instances with *proven* optimal solutions let us validate our implementations. Instance size can also be scaled smoothly from 8 to 557 customers to expose the trade-off between exact and heuristic methods.

The remainder of the report is organised as follows:

* Section 2 describes the data and assumptions.
* Section 3 gives the mathematical formulation.
* Section 4 presents and justifies the two exact and two heuristic methods.
* Section 5 describes the implementation.
* Section 6 reports the experimental comparison.
* Section 7 discusses the results and future work.
* Section 8 states individual contributions.
```


**`report/sections/02_data.md`**

```markdown
# 2. Problem Context and Data

## 2.1 Data sources

| Source | Content | Use |
|---|---|---|
| Department of Census and Statistics, Census of Population and Housing 2012, via the open repository `github.com/nuuuwan/gig-data` | 557 Grama Niladhari Divisions (GNDs) of Colombo District: identifier, name, Divisional Secretariat, centroid latitude/longitude, population | Real outlet locations and demand |
| CVRPLIB (Uchoa et al., 2017), via `github.com/PyVRP/VRPLIB` and `github.com/PyVRP/Instances` | E-n13-k4, P-n16-k8, B-n31-k5, A-n32-k5 (proven optima); X-n101-k25, X-n148-k46, X-n200-k36 (best-known solutions) | Validation of both solvers against published optimal costs |

As the assignment brief specifies, the data are used **only to instantiate and validate the optimisation model**; no model is trained on them. The extraction script `experiments/extract_census_data.py` reproduces `data/raw/colombo_gnd.csv` from the source repository.

![Colombo District: 557 GND outlets (marker size proportional to demand) and the Kelaniya DC.](../results/figures/colombo_outlets_map.png){width=62%}

## 2.2 From census data to delivery instances

**Customers.** Each GND represents one retail outlet located at its centroid. GND populations range from 533 to 28,003 residents (median 3,358). The outlets are dense in the western, coastal part of the district (Colombo city, Dehiwala, Moratuwa) and sparse in the eastern rural divisions (Seethawaka, Padukka). This spatial unevenness is what makes the instances realistic rather than uniformly random.

**Depot.** The DC is placed at the centroid of Kelaniya Divisional Secretariat (6.9702° N, 79.9099° E). Kelaniya lies just across the Kelani River in neighbouring Gampaha District, immediately north-east of Colombo city. The DC is therefore just outside the district it serves, 3–45 km (road distance) from the outlets.

**Demand.** Daily demand is assumed proportional to the population an outlet serves: $d_i=\lceil \text{population}_i/1000\rceil$ crates, i.e. one crate per 1,000 residents per day. This gives 1–29 crates per outlet (median 4). The proportional rule is a simple, transparent assumption for everyday consumer goods. The constant only scales demand relative to capacity and is easy to change in `src/cvrp/colombo.py`.

**Vehicles.** Light trucks carry $Q=30$ crates. This makes the largest outlet (Mattakkuliya, 29 crates) just feasible. It also gives about six stops per route, a plausible workload for a day of urban outlet deliveries. An earlier setting of $Q=40$ let one or two trucks serve the small instances, which reduced them to near-TSPs, so it was rejected as too easy.

**Distances.** Road distance is approximated as the great-circle (haversine) distance multiplied by a circuity factor of 1.3. Ballou, Rahardja and Sakai (2002) report country-level circuity factors between 1.12 and 2.10. Studies of metropolitan road networks typically find values around 1.2–1.4; Giacomin and Levinson (2015), for example, report a mean of 1.34 across 51 US metropolitan areas. A live road-network matrix (e.g. OpenStreetMap/OSRM) is a natural refinement and is listed as future work.

**Daily instances.** On a given day only some outlets order. An instance is created by sampling $n$ outlets uniformly with a fixed seed:

* $n\in\{8,10,12,15,18,20,25,30\}$ with three seeds each (24 instances) for the exact-versus-heuristic comparison;
* $n\in\{50,100,200,557\}$ for scalability, where $n=557$ is the whole district.

: Summary of selected Colombo instances (full table in `results/dataset_summary.csv`).

| Instance | Outlets | Total demand (crates) | Min. trucks $\lceil\sum d/Q\rceil$ | Mean DC distance (km) |
|---|---|---|---|---|
| COL-n15-s2 | 15 | 51 | 2 | 23.5 |
| COL-n30-s3 | 30 | 162 | 6 | 17.8 |
| COL-n100-s1 | 100 | 520 | 18 | 20.5 |
| COL-n557-s1 | 557 | 2,604 | 87 | 21.4 |

## 2.3 Constraints in business terms

1. Every ordering outlet is visited **exactly once** by **one** truck (no split deliveries).
2. Every truck **starts and ends at the DC**.
3. The crates on a truck never exceed **30**.
4. Any number of trucks may be used. Their count is a result, not an input.
5. The goal is to **minimise total kilometres driven**.

Section 3 states these rules formally.
```


**`report/sections/03_formulation.md`**

```markdown
# 3. Mathematical Formulation

The delivery problem is modelled as a **Capacitated Vehicle Routing Problem (CVRP)** with a single depot, a homogeneous unlimited fleet and symmetric travel distances.

## 3.1 Sets and parameters

| Symbol | Meaning | Colombo instances |
|---|---|---|
| $V=\{0,1,\dots,n\}$ | all locations; $0$ is the depot | Kelaniya DC + $n$ outlets |
| $C=V\setminus\{0\}$ | customers (outlets) | GND centroids |
| $c_{ij}$ | travel distance between $i$ and $j$ (symmetric) | haversine km $\times 1.3$ |
| $d_i$ | demand of customer $i$ | $\lceil \text{population}_i/1000\rceil$ crates |
| $Q$ | vehicle capacity | 30 crates |
| $K_{\min}=\lceil \sum_{i\in C} d_i / Q\rceil$ | lower bound on fleet size | derived |

For $S\subseteq C$ we write $d(S)=\sum_{i\in S} d_i$ and $\delta(S)$ for the set of edges with exactly one end in $S$.

## 3.2 Model A — compact directed formulation (MTZ)

**Decision variables**

* $x_{ij}\in\{0,1\}$ — 1 if a vehicle travels directly from $i$ to $j$ ($i\neq j$);
* $u_i\in[d_i,Q]$ — cumulative load delivered on the vehicle's route up to and including customer $i$.

**Objective** — minimise total distance driven:
$$\min \sum_{i\in V}\sum_{j\in V, j\neq i} c_{ij}\,x_{ij} \tag{A0}$$

**Constraints**
$$\sum_{j\in V} x_{ij}=1 \quad \forall i\in C \tag{A1}$$
$$\sum_{i\in V} x_{ij}=1 \quad \forall j\in C \tag{A2}$$
$$\sum_{j\in C} x_{0j}=\sum_{i\in C} x_{i0} \ \ge K_{\min} \tag{A3}$$
$$u_j \ge u_i + d_j - Q\,(1-x_{ij}) \quad \forall i,j\in C,\ i\neq j \tag{A4}$$
$$d_i \le u_i \le Q \quad \forall i\in C \tag{A5}$$
$$x_{ij}\in\{0,1\} \tag{A6}$$

(A1)–(A2) ensure every outlet is entered and left exactly once. (A3) balances vehicles leaving and returning to the depot, and imposes the bin-packing bound on the fleet size, which is a valid inequality that tightens the LP relaxation. (A4) is the Miller–Tucker–Zemlin (MTZ) constraint. When $x_{ij}=1$ it forces the load to accumulate ($u_j \ge u_i + d_j$). When $x_{ij}=0$ it is relaxed by the big-M term $Q$. Together with (A5), it does two jobs:

* it enforces vehicle capacity;
* it eliminates sub-tours: a cycle $i\to j\to i$ that avoids the depot would require $u_i \ge u_i + d_i + d_j$, which is impossible.

The number of vehicles is not fixed. It is implied by $K=\sum_j x_{0j}$.

*Size:* $n(n+1)$ binary and $n$ continuous variables, and $O(n^2)$ constraints. For $n=30$ that is 930 binaries.

## 3.3 Model B — two-index undirected formulation with capacity cuts

Because $c_{ij}=c_{ji}$, the direction of travel is irrelevant. We therefore use one variable per edge.

* $x_e\in\{0,1\}$ for edges between two customers, and $x_{0j}\in\{0,1,2\}$ for depot edges ($x_{0j}=2$ is a single-customer out-and-back route);
* $K\in\mathbb{Z}_{\ge K_{\min}}$ — number of vehicles used.

$$\min \sum_{e\in E} c_e\,x_e \tag{B0}$$
$$\sum_{e\in\delta(\{i\})} x_e = 2 \quad \forall i\in C \tag{B1}$$
$$\sum_{e\in\delta(\{0\})} x_e = 2K \tag{B2}$$
$$\sum_{e\in\delta(S)} x_e \ \ge\ 2\left\lceil \frac{d(S)}{Q}\right\rceil \quad \forall S\subseteq C,\ S\neq\emptyset \tag{B3}$$

(B1) is a degree constraint: every outlet has one edge in and one edge out. (B2) gives each vehicle two depot edges. (B3) are the **rounded capacity inequalities (RCI)**: any group of outlets $S$ needs at least $\lceil d(S)/Q\rceil$ vehicles, and each vehicle crosses the boundary of $S$ twice. With $|S|\ge 2$ they also forbid sub-tours, because a sub-tour has $\sum_{\delta(S)}x_e=0$.

Family (B3) has $2^n-1$ members, so it cannot be written out in full. It is generated **lazily**:

1. Solve the LP relaxation and add violated RCIs found by connected-component and greedy-growth separation heuristics. Repeat until no more cuts are found (root cutting-plane phase).
2. Solve the MILP. Check the integer solution: every connected component of the customer sub-graph whose crossing count is below $2\lceil d(S)/Q\rceil$ is a sub-tour or an overloaded route, and gets a cut. Re-solve.
3. The first integer solution that violates no RCI is **provably optimal**. Every intermediate objective value is a valid lower bound.

A pair of outlets with $d_i+d_j>Q$ can never share a vehicle, so that edge is removed from both models during pre-processing.

## 3.4 Complexity

CVRP is NP-hard because it contains two classic NP-hard problems as special cases:

* when $Q\ge d(C)$, a single vehicle suffices and CVRP reduces to the **Travelling Salesman Problem**;
* deciding whether $K$ vehicles are enough is the **Bin Packing Problem**.

The number of feasible route plans grows faster than $n!$. Exact methods are therefore expected to hit a wall as $n$ grows, which motivates the heuristic approach in Section 4.
```


**`report/sections/04_methods.md`**

```markdown
# 4. Optimisation Methods

The assignment requires one exact method and one heuristic or metaheuristic. We implement two of each. In each pair, the simpler method serves as a baseline that explains *why* the stronger one is needed.

| Role | Method | Type | Guarantee |
|---|---|---|---|
| Exact (main) | **Model B:** two-index MILP with lazily generated rounded capacity cuts, solved by HiGHS branch-and-cut | Exact | Proven optimum, or a proven lower bound if stopped |
| Exact (baseline) | Model A: compact MTZ MILP, solved by HiGHS | Exact | Same, but weaker bounds |
| Heuristic (baseline) | Clarke–Wright savings | Greedy constructive | None |
| Metaheuristic (main) | **Simulated Annealing**, started from Clarke–Wright | Stochastic local search | None (quality measured against exact bounds) |

## 4.1 Exact method: Mixed-Integer Linear Programming

**Why MILP.** The CVRP of Section 3 is naturally a linear model in binary variables. Modern MILP solvers combine LP relaxations, branch-and-bound, cutting planes and presolve. They return either a **certified optimum** or, if stopped early, a **lower bound** on the optimal cost. That bound is valuable even when the instance is too large to solve, because it certifies how far any heuristic solution can be from optimal. Dynamic programming over subsets would need $O(2^n)$ states and is impractical beyond about 20 customers. Pure combinatorial branch-and-bound would mean re-implementing what a MILP solver already does well.

**Solver.** HiGHS (Huangfu and Hall, 2018) is a state-of-the-art open-source LP/MIP solver, called through `scipy.optimize.milp`. It needs nothing beyond SciPy, which keeps the pipeline reproducible with a single `pip install`.

**Model A (MTZ, Section 3.2)** is compact, with $O(n^2)$ variables and constraints. It is easy to write down and explain. Its weakness is the big-M term in (A4): in the LP relaxation, fractional $x_{ij}$ make the load constraints almost inactive, so the LP bound sits far below the optimum. Branch-and-bound must then explore a huge tree.

**Model B (Section 3.3)** replaces MTZ with the rounded capacity inequalities (B3). These are among the strongest known valid inequalities for the CVRP (Lysgaard, Letchford and Eglese, 2004). Because there are exponentially many, they are added only when violated (row generation):

1. **Root cutting-plane loop.** Solve the LP relaxation and separate violated RCIs heuristically. The separation uses connected components of the support graph at thresholds 0, 0.25, 0.5 and 0.75, plus greedy growth of a customer set from every seed customer. Add the cuts and repeat until no cut is found.
2. **Integer phase.** Solve the MILP and take the connected components of the integer solution. Each sub-tour or overloaded route violates an RCI exactly, so add the cut and re-solve.
3. **Stop** when an integer solution violates no RCI (provably optimal) or the time limit is reached (report the best lower bound).

The separation heuristics can find tens of thousands of violated sets per round on large instances. For $n\ge 200$ only the $2n$ most violated cuts are kept per round, to bound memory use; below that size every violated cut found is added.

Model B also uses one variable per undirected edge, which halves the variable count for symmetric distances. Both models remove edges between outlet pairs whose combined demand exceeds $Q$.

## 4.2 Heuristic: Clarke–Wright savings

The savings algorithm (Clarke and Wright, 1964) is the classical constructive heuristic for vehicle routing, and still a common baseline in practice:

1. Start with one out-and-back route per outlet.
2. For every pair of outlets compute the saving from serving them consecutively, $s_{ij}=c_{0i}+c_{0j}-c_{ij}$.
3. Process pairs in decreasing order of saving. Merge the two routes when $i$ and $j$ are at route ends, on different routes, and the merged load fits within $Q$.

It runs in $O(n^2\log n)$ time, has no parameters, and is deterministic. That makes it the natural "what a planner could do by hand" baseline and a good starting point for the metaheuristic.

## 4.3 Metaheuristic: Simulated Annealing

**Why SA.** Simulated Annealing (Kirkpatrick, Gelatt and Vecchi, 1983) explores the space of feasible delivery plans with small moves. It always accepts improvements and accepts worsening moves with probability $e^{-\Delta/T}$, which lets it escape the local optima where greedy improvement stalls. It has few parameters, is easy to explain and reproduce, and works directly on routes. A genetic algorithm would need a crossover operator that preserves capacity feasibility, which is notoriously awkward for the CVRP.

**Representation.** A solution is a list of routes. A spare empty route is always kept, so moving an outlet into it opens a new truck.

**Neighbourhood (granular).** Each iteration picks a random outlet $c$ and one of its $k=10$ nearest outlets $v$, then applies one of these moves:

| Move | Probability | Effect |
|---|---|---|
| relocate | 35% | move $c$ next to $v$ |
| swap | 25% | exchange $c$ and $v$ |
| 2-opt | 35% (shared with 2-opt\*) | $c$, $v$ on the same route: reverse the segment between them |
| 2-opt\* | (shared) | $c$, $v$ on different routes: exchange the route tails after $c$ and $v$ |
| open truck | 5% | move $c$ to a new route |

Moves that would overload a truck are rejected, so **every visited solution is feasible**. Restricting moves to nearby outlets follows the granular-neighbourhood idea of Toth and Vigo (2003). Our first version picked moves uniformly at random and could not improve on Clarke–Wright for $n\ge 100$, because almost all random moves join distant outlets.

**Cooling schedule.** Geometric cooling $T_{k+1}=\alpha T_k$:

* $T_0$ is calibrated from the worsening moves among 1,000 sampled proposals, so that an average worsening move is initially accepted with probability 0.3.
* $T$ decays to $10^{-3}T_0$ over $\max(50\,000,\,4000n)$ iterations.

After annealing, the best solution found is **polished** by first-improvement descent with the same moves. Each run is seeded, so results are reproducible. We report the mean and best of 5 seeds.
```


**`report/sections/05_implementation.md`**

````markdown
# 5. Implementation

## 5.1 Environment and repository

The project is written in Python 3. Its dependencies are only NumPy, SciPy (HiGHS MILP), pandas and matplotlib, plus pytest and Jupyter for testing and the walkthrough notebook. The GitHub repository is organised as a small package with reproducible scripts:

```
src/cvrp/instance.py     instance model, CVRPLIB parser, cost and feasibility checks
src/cvrp/colombo.py      census data -> Colombo delivery instances
src/cvrp/exact.py        Model A (MTZ) and Model B (capacity cuts)
src/cvrp/heuristics.py   Clarke-Wright savings, Simulated Annealing
experiments/             extract_census_data.py, build_instances.py,
                         run_experiments.py, analyse_results.py
notebooks/               delivery_optimization.ipynb (guided walkthrough)
tests/                   regression tests against CVRPLIB optima
results/                 CSV results, tables, figures, solution files
```

## 5.2 Design decisions

* **Single source of truth for cost and feasibility.** All four methods return a plain list of routes. A single function, `check_feasible`, verifies every result: each outlet visited exactly once and no route above capacity. A single function, `solution_cost`, computes every reported cost. Before any experiment was run, these functions were checked against the published reference solutions of all seven CVRPLIB instances. They reproduce every published cost exactly (247, 450, 672, 784, 27,591, 43,448 and 58,578).
* **Exact solvers build sparse constraint matrices directly** (`scipy.sparse`) and call `scipy.optimize.milp`. Model B re-solves with the accumulated cut pool. Each call receives only the remaining time budget, so the total limit is respected.
* **SA uses incremental bookkeeping.** It keeps per-route costs and loads, so a move is evaluated by re-costing only the one or two routes it touches. It also keeps an outlet→route index for the granular moves.
* **Reproducibility.** Instances are generated from fixed seeds. SA runs are seeded. The experiment script appends one CSV row per run and skips rows already present, so an interrupted run resumes. Every table and figure in this report is regenerated by `python experiments/analyse_results.py`.
* **Testing.** `pytest` checks two things. First, the exact models reproduce proven optima: Model A on E-n13-k4 and P-n16-k8, and Model B on those two plus B-n31-k5. Second, the heuristics return feasible solutions, SA reaches the optimum on the two smallest benchmarks, and SA never ends worse than its starting point.

## 5.3 Version control

Development followed the order of this report: data, then the exact methods, then the heuristics, then the experiments and the report. Each step was committed with a descriptive message, so the repository's commit history records how the project evolved and who contributed each part (Section 8).

## 5.4 Experimental protocol

* **Exact methods:** time limit of 300 s per instance.
  * Model B runs on all 24 small Colombo instances, the four small benchmarks, and COL-n50, n100 and n200 (to obtain lower bounds).
  * Model A runs on the Colombo instances up to 20 outlets and the two smallest benchmarks. It already failed to finish at 18–20 outlets, so larger instances were not attempted.
* **Heuristics:** Clarke–Wright once per instance (it is deterministic), and SA with 5 seeds on all 28 Colombo and 7 benchmark instances.
* **Quality** is measured as the gap to a reference value: the proven optimum where Exact B proved one; otherwise the CVRPLIB best-known value; otherwise Exact B's lower bound. A gap against a lower bound *overstates* the true gap, so it is conservative.
* **Machine:** a 2-core Intel Xeon at 2.8 GHz with 8 GB RAM, Python 3.13, SciPy 1.18. At most two experiment processes ran at the same time, one per core.
````


**`report/sections/06_results.md`**

```markdown
# 6. Results

All numbers below are produced by `experiments/run_experiments.py` and summarised by `experiments/analyse_results.py`. The complete per-instance table is in Appendix A.

## 6.1 Validation on benchmarks with proven optima

: CVRPLIB instances with proven optimal values. SA statistics over 5 seeds.

| Instance | Cust. | Optimum | Exact B | B time (s) | MTZ time (s) | CW gap | SA mean gap | SA best | SA time (s) |
|---|---|---|---|---|---|---|---|---|---|
| E-n13-k4 | 12 | 247 | 247 (opt.) | 0.23 | 1.02 | 11.34% | 0.00% | 247 | 0.28 |
| P-n16-k8 | 15 | 450 | 450 (opt.) | 1.49 | 3.75 | 7.11% | 0.00% | 450 | 0.34 |
| B-n31-k5 | 30 | 672 | 672 (opt.) | 8.34 | not run | 0.74% | 0.30% | 672 | 0.77 |
| A-n32-k5 | 31 | 784 | 784 (opt.) | 10.40 | not run | 7.40% | 1.28% | 784 | 0.85 |

Exact B reproduces every published optimum. This confirms that the formulation, the separation routine and the cost function are correct. MTZ was run only on the two smallest benchmarks; on the Colombo instances it already fails at 15–18 outlets (Section 6.3). SA's best of 5 runs reaches the optimum on all four benchmarks in under a second.

## 6.2 Solution quality on the Colombo instances

: Colombo instances by size (3 instances per size for n ≤ 30). The reference is the proven optimum.

| n | Exact B solved | B median time (s) | MTZ solved | CW mean gap | SA mean gap | SA time (s) |
|---|---|---|---|---|---|---|
| 8 | 3/3 | 0.02 | 3/3 | 0.18% | 0.00% | 0.33 |
| 10 | 3/3 | 0.03 | 3/3 | 0.17% | 0.00% | 0.32 |
| 12 | 3/3 | 0.05 | 3/3 | 3.46% | 0.00% | 0.36 |
| 15 | 3/3 | 0.04 | 2/3 | 1.35% | 0.04% | 0.40 |
| 18 | 3/3 | 6.74 | 0/3 | 2.67% | 0.13% | 0.48 |
| 20 | 3/3 | 5.61 | 0/3 | 3.97% | 0.31% | 0.54 |
| 25 | 3/3 | 60.53 | not run | 1.58% | 0.45% | 0.66 |
| 30 | 2/3 | 19.69 | not run | 2.06%\* | 0.26%\* | 0.80 |

\* For COL-n30-s3 the reference is Exact B's lower bound (283.32).

![(a) Gap to the proven optimum on the 23 Colombo instances that Exact B solved; dots are individual instances. (b) Gap to the best-known solution on the large CVRPLIB benchmarks.](../results/figures/gap_vs_size.png){width=100%}

* **Exact B proved the optimum on 23 of the 24 small Colombo instances** (all up to 25 outlets and two of three at 30). On COL-n30-s3 it stopped at the 300 s limit with a lower bound of 283.32 km.
* **Clarke–Wright** is optimal on 6 of those 23 instances and on average **1.96% above optimal**, with a worst case of 8.42% (COL-n20-s3).
* **Simulated Annealing** is on average **0.13% above optimal**:
  * 90 of its 115 runs (23 instances × 5 seeds) hit the proven optimum exactly;
  * its best-of-5 is optimal on 21 of the 23 instances;
  * its worst single-instance mean is 1.34% (COL-n25-s2).
* **Feasibility:** all 210 heuristic runs and every exact solution passed the independent feasibility check (each outlet once, no truck above 30 crates). SA's best plans use the minimum possible number of trucks, $\lceil\sum d/Q\rceil$, on 24 of the 28 Colombo instances, and at most two more on the rest.

## 6.3 Runtime and scalability

![Runtime against the number of outlets (log–log). Exact methods: median of 3 instances per size; a hollow marker means the median run hit the 300 s limit.](../results/figures/runtime_vs_size.png){width=88%}

* **Exact A (MTZ)** grows fastest. It needs up to 22 s at 12 outlets and 39–245 s at 15 outlets (one 15-outlet instance timed out). **None of the 18- or 20-outlet instances finished within 300 s**. Even when MTZ had already found the optimal plan (COL-n20-s1: 212.74 km), it could not prove it: its lower bound was still 10% short (191.19). On instances both models solved, Model B was between 2.5 and more than 6,000 times faster (e.g. COL-n15-s3: 245 s versus 0.04 s).
* **Exact B** solves most instances up to 30 outlets in seconds, but its runtime is **erratic**. At $n=18$ it ranges from 0.7 s to 94 s, and at $n=25$ from 7 s to 120 s, depending on how the outlets cluster. This unpredictability is typical of NP-hard problems.
  * At $n=50$ and $n=100$ it returns only a lower bound (405.06 and 697.22).
  * At $n=200$ the first run was stopped by the operating system after the cut pool exceeded the available memory (8 GB). With the cut cap (Section 4.1) only a weak root bound (974.35) was obtained.
* **Clarke–Wright** stays below 0.25 s even for all 557 outlets.
* **Simulated Annealing** grows roughly linearly with $n$, because the iteration budget is $\max(50\,000,\,4000n)$: 0.3 s at $n=8$, 2.5 s at $n=100$, and **18.6 s for the full 557-outlet district**.

## 6.4 Large instances

: Large instances. Gap to CVRPLIB best-known for X-instances; for Colombo, improvement of SA over Clarke–Wright.

| Instance | n | CW | SA mean (± sd) | SA best | Reference | SA mean gap | SA time (s) |
|---|---|---|---|---|---|---|---|
| X-n101-k25 | 100 | 28,986 | 28,369 (± 176) | 28,160 | 27,591 (BKS) | 2.82% | 2.23 |
| X-n148-k46 | 147 | 45,009 | 44,262 (± 162) | 44,026 | 43,448 (BKS) | 1.87% | 3.20 |
| X-n200-k36 | 199 | 61,167 | 60,268 (± 127) | 60,061 | 58,578 (BKS) | 2.88% | 4.64 |
| COL-n50-s1 | 50 | 417.47 | 414.17 (± 1.39) | 413.00 | LB 405.06 | ≤ 2.25% | 1.38 |
| COL-n100-s1 | 100 | 775.06 | 763.43 (± 7.67) | 755.10 | LB 697.22 | ≤ 9.50% | 2.50 |
| COL-n200-s1 | 200 | 1,438.82 | 1,418.48 (± 5.81) | 1,409.20 | LB 974.35 | – | 5.16 |
| COL-n557-s1 | 557 | 3,664.92 | 3,633.59 (± 9.57) | 3,622.68 | none | — | 18.58 |

On the benchmarks with best-known solutions, **SA stays within 1.9–2.9% of the best-known value on 100–199 customers** and cuts Clarke–Wright's gap by 35–48%.

For the large Colombo instances the only reference is Exact B's lower bound. The "gap" against it is therefore an *upper bound* on SA's true gap. It is informative at $n=50$ (at most 2.25%) but not at $n\ge 100$, where the bound itself is far from the optimum. For the full district, SA's mean saving over Clarke–Wright is 31 km per day (0.85%). Its best plan (3,623 km, 42 km shorter than Clarke–Wright) uses 89 trucks; the bin-packing lower bound is 87, and the individual runs used 89–90.

![SA plan for all 557 outlets: 3,623 km over 89 routes.](../results/figures/routes_n557.png){width=62%}

## 6.5 Sensitivity of Simulated Annealing to its iteration budget

![Mean gap against mean runtime for iteration budgets of 0.5k–32k iterations per customer (3 seeds each). Colombo gaps are relative to the best solution found in any run.](../results/figures/sa_sensitivity.png){width=85%}

More annealing time buys better solutions with clearly diminishing returns:

* On X-n101-k25 the mean gap falls from 3.7% (0.5k iterations per customer, 0.5 s) to 2.9% (4k, the default, 2.4 s) and 1.7% (32k, 18 s).
* At very small budgets the curve is noisy: 1k was no better than 0.5k on any of the three instances. With fast cooling the search behaves almost like pure descent from the Clarke–Wright start.

The default of 4k iterations per customer is a reasonable middle point. A planner with time to spare gains a further 0.4–1.3 percentage points at 16k iterations (about 10–20 s) and 0.8–1.8 points at 32k (about 20–40 s).

![SA trajectory on COL-n100-s1 (no final polish). The current cost first rises well above the Clarke–Wright start, then cools back; improvements over the start appear only in the second half.](../results/figures/sa_convergence.png){width=85%}

The convergence trace shows that, at the calibrated starting temperature, roughly the first half of the run explores solutions worse than the starting point. Only after about 200,000 iterations does the best-so-far improve. This motivates the future-work item on adaptive temperature control (Section 7).

## 6.6 Example: optimal versus greedy routes

![COL-n30-s2: Clarke–Wright (289.9 km) versus the proven optimum (277.9 km), both with 5 trucks.](../results/figures/routes_n30.png){width=95%}

Both plans use five trucks and serve the same outlets, but the optimum regroups the central and eastern outlets into different routes. The same fleet drives 12 km (4.1%) less every day, which is the kind of saving that compounds over a year of operation.
```


**`report/sections/07_discussion.md`**

```markdown
# 7. Discussion

## 7.1 Exact versus heuristic: what the experiments show

**Exact optimisation is worth it while it is fast, and it stops being fast abruptly.** For daily plans of up to about 25–30 outlets, Exact B delivers a *certified* optimum, usually in seconds. A distributor with a small daily order book can simply use it. The NP-hardness of the CVRP shows up not as a smooth slowdown but as erratic jumps: two 18-outlet days took 0.7 s and 94 s. On the Colombo instances with 50 or more outlets, no run reached a proven optimum within five minutes. A planning tool cannot rely on an exact solver alone, because some days it simply will not finish.

**Formulation matters as much as the solver.** Models A and B describe the same problem and use the same HiGHS solver. They differ only in how capacity and sub-tours are expressed, yet the stronger model was more than three orders of magnitude faster on the hardest instance both solved, and it solved every instance that MTZ could not. The reason is the LP relaxation. MTZ's big-M constraints are almost inactive at fractional solutions, so its lower bound is weak; on COL-n20-s1 it had already found the optimal plan but could not prove it, with the bound still 10% short. Rounded capacity cuts tighten the relaxation so much that the root LP bound alone was within 1% of the optimum on 21 of the 27 instances Model B solved (mean 0.6%). MTZ's bounds on the instances it could not solve were 10–28% below the best plan it had found. For the viva this is the most important lesson: *how you write the model decides whether it can be solved*.

**Simulated Annealing gives near-optimal plans at any scale, in seconds.** Where the optimum is known, SA was on average 0.13% above it, and 90 of 115 runs found it exactly. On the 100–199-customer benchmarks it stayed within 3% of the best-known solutions from the research literature, and it planned all 557 outlets in under 20 seconds. Its price is the absence of any guarantee: without a lower bound we cannot say how good a given plan is. This is exactly where the two families complement each other. The exact method's lower bound **certifies** the heuristic (at $n=50$, SA is provably within 2.25%), and the heuristic supplies plans where the exact method cannot.

**Greedy is not enough.** Clarke–Wright is effectively instantaneous and often decent, but averaged 2% above optimal on Colombo with a worst case of 8.4%. For a fleet driving thousands of kilometres a week, that difference is real money. SA removes most of it for under a second of extra computation.

**Neighbourhood design beats brute force.** The first SA version chose moves uniformly at random and could not improve on Clarke–Wright for $n\ge 100$. Restricting moves to each outlet's ten nearest neighbours fixed this without any extra iterations. The convergence trace (Section 6.5) also shows the calibrated starting temperature is too hot for warm starts: half the run is spent recovering from early uphill moves.

**Practical recommendation.** For daily operations, run SA every time. On days with up to about 30 orders, additionally run Exact B with a short time limit, both to obtain the optimum and to measure how close SA came. This hybrid gives fast plans every day and an optimality certificate whenever one is cheap.

## 7.2 Limitations

* **Distances** are haversine distances with a uniform 1.3 circuity factor rather than real road distances. Rivers, the Kelani bridge crossings and one-way systems make real circuity uneven. Traffic is ignored.
* **Demand** is a proportional function of 2012 population, not actual order data. The population figures are over a decade old.
* **Single-day, single-depot, homogeneous fleet.** Time windows, driver shift limits, heterogeneous vehicles and split deliveries are not modelled.
* **Exact method scope.** Our cut separation is heuristic and we only add capacity cuts. State-of-the-art exact CVRP codes (branch-cut-and-price) prove optimality for hundreds of customers.
* **Runtimes** come from a 2-core cloud VM. Absolute numbers will differ on other hardware, but the relative trends are what matter.

## 7.3 Future work

1. **Real road distances:** replace the haversine × 1.3 matrix with an OpenStreetMap/OSRM distance-and-duration matrix for Colombo.
2. **Richer operational constraints:** delivery time windows (VRPTW) and a maximum shift duration per driver, both natural extensions of the MTZ load variables to time variables.
3. **Stronger exact method:** exact separation of capacity cuts (e.g. with the CVRPSEP library of Lysgaard et al., 2004) and additional cut families (comb, multistar). Warm-start the MILP with the SA solution as an incumbent, which also gives the exact method an upper bound when it times out.
4. **Better metaheuristic control:** lower or adaptive initial temperature for warm starts, reheating, or a population-based hybrid such as Hybrid Genetic Search (Vidal, 2022), one of the strongest published CVRP heuristics.
5. **Multi-day planning:** couple daily routing with inventory decisions (inventory-routing), so that outlet visit frequency itself becomes a decision.
```


**`report/sections/08_contribution.md`**

```markdown
# 8. Individual Contribution

| Member | Main responsibilities | Repository evidence |
|---|---|---|
| **Kavindu Ariyasinghe** (MS26900186) | Problem selection and scenario design; census data extraction and instance generator; mathematical formulation; exact methods (Model A MTZ, Model B capacity cuts); exact-solver tests; report Sections 1–3 | commits on `src/cvrp/instance.py`, `colombo.py`, `exact.py`, `tests/test_exact.py`, `experiments/extract_census_data.py`, `build_instances.py` |
| **Ushan Gunaratne** (MS26901930) | Clarke–Wright and Simulated Annealing (including the granular neighbourhood redesign); heuristic tests; experiment runner and analysis; result tables and figures; report Sections 4.2–4.3 and 6 | commits on `src/cvrp/heuristics.py`, `tests/test_heuristics.py`, `experiments/run_experiments.py`, `analyse_results.py` |
| **Both** | Discussion and future work (Section 7), notebook walkthrough, video presentation, final review | joint commits |

Both members reviewed each other's code before it was merged and can explain both methods.
```


**`report/sections/09_references.md`**

```markdown
# References

Augerat, P., Belenguer, J. M., Benavent, E., Corberán, A., Naddef, D., & Rinaldi, G. (1995). *Computational results with a branch and cut code for the capacitated vehicle routing problem* (Research Report 949-M). Université Joseph Fourier, Grenoble.

Ballou, R. H., Rahardja, H., & Sakai, N. (2002). Selected country circuity factors for road travel distance estimation. *Transportation Research Part A: Policy and Practice, 36*(9), 843–848.

Clarke, G., & Wright, J. W. (1964). Scheduling of vehicles from a central depot to a number of delivery points. *Operations Research, 12*(4), 568–581.

Dantzig, G. B., & Ramser, J. H. (1959). The truck dispatching problem. *Management Science, 6*(1), 80–91.

Department of Census and Statistics, Sri Lanka. (2012). *Census of Population and Housing 2012*. GND-level data accessed via Senaratna, N., *gig-data* (GitHub repository), https://github.com/nuuuwan/gig-data.

Desrochers, M., & Laporte, G. (1991). Improvements and extensions to the Miller–Tucker–Zemlin subtour elimination constraints. *Operations Research Letters, 10*(1), 27–36.

Giacomin, D. J., & Levinson, D. M. (2015). Road network circuity in metropolitan areas. *Environment and Planning B: Planning and Design, 42*(6), 1040–1053.

Huangfu, Q., & Hall, J. A. J. (2018). Parallelizing the dual revised simplex method. *Mathematical Programming Computation, 10*(1), 119–142.

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science, 220*(4598), 671–680.

Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004). A new branch-and-cut algorithm for the capacitated vehicle routing problem. *Mathematical Programming, 100*(2), 423–445.

Miller, C. E., Tucker, A. W., & Zemlin, R. A. (1960). Integer programming formulation of traveling salesman problems. *Journal of the ACM, 7*(4), 326–329.

Toth, P., & Vigo, D. (2003). The granular tabu search and its application to the vehicle-routing problem. *INFORMS Journal on Computing, 15*(4), 333–346.

Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., & Subramanian, A. (2017). New benchmark instances for the capacitated vehicle routing problem. *European Journal of Operational Research, 257*(3), 845–858.

Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source implementation and SWAP\* neighborhood. *Computers & Operations Research, 140*, 105643.
```


**`report/sections/09_references.md`**

```markdown
# References

Augerat, P., Belenguer, J. M., Benavent, E., Corberán, A., Naddef, D., & Rinaldi, G. (1995). *Computational results with a branch and cut code for the capacitated vehicle routing problem* (Research Report 949-M). Université Joseph Fourier, Grenoble.

Ballou, R. H., Rahardja, H., & Sakai, N. (2002). Selected country circuity factors for road travel distance estimation. *Transportation Research Part A: Policy and Practice, 36*(9), 843–848.

Clarke, G., & Wright, J. W. (1964). Scheduling of vehicles from a central depot to a number of delivery points. *Operations Research, 12*(4), 568–581.

Dantzig, G. B., & Ramser, J. H. (1959). The truck dispatching problem. *Management Science, 6*(1), 80–91.

Department of Census and Statistics, Sri Lanka. (2012). *Census of Population and Housing 2012*. GND-level data accessed via Senaratna, N., *gig-data* (GitHub repository), https://github.com/nuuuwan/gig-data.

Desrochers, M., & Laporte, G. (1991). Improvements and extensions to the Miller–Tucker–Zemlin subtour elimination constraints. *Operations Research Letters, 10*(1), 27–36.

Giacomin, D. J., & Levinson, D. M. (2015). Road network circuity in metropolitan areas. *Environment and Planning B: Planning and Design, 42*(6), 1040–1053.

Huangfu, Q., & Hall, J. A. J. (2018). Parallelizing the dual revised simplex method. *Mathematical Programming Computation, 10*(1), 119–142.

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science, 220*(4598), 671–680.

Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004). A new branch-and-cut algorithm for the capacitated vehicle routing problem. *Mathematical Programming, 100*(2), 423–445.

Miller, C. E., Tucker, A. W., & Zemlin, R. A. (1960). Integer programming formulation of traveling salesman problems. *Journal of the ACM, 7*(4), 326–329.

Toth, P., & Vigo, D. (2003). The granular tabu search and its application to the vehicle-routing problem. *INFORMS Journal on Computing, 15*(4), 333–346.

Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., & Subramanian, A. (2017). New benchmark instances for the capacitated vehicle routing problem. *European Journal of Operational Research, 257*(3), 845–858.

Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source implementation and SWAP\* neighborhood. *Computers & Operations Research, 140*, 105643.
```


`report/sections/10_appendix.md` is generated automatically by `build_report.sh` from `results/tables/comparison.md` — don't create it by hand.

**`report/video_plan.md`**

```markdown
# 15-minute video plan

Record as a screen share (slides or the notebook plus the PDF report) with both faces on camera if possible. Each member presents the part they built, because the video is evidence for the *Individual Contribution* criterion. Times are targets; the total is 15:00.

| Time | Speaker | Segment | Show on screen | Key points to say |
|---|---|---|---|---|
| 0:00–1:00 | Kavindu | Introduction | Title slide, both names and IDs | Problem in one sentence: a Kelaniya DC delivering to Colombo outlets with 30-crate trucks, minimise km. Why it matters: fuel, driver hours. Who did what. |
| 1:00–3:00 | Kavindu | Problem and data | `colombo_outlets_map.png`, dataset table | Real data: 557 GNDs, Census 2012 population. Assumptions: demand = ⌈pop/1000⌉, Q = 30, haversine × 1.3. Why Q = 30 and not 40 (40 made instances trivial). CVRPLIB benchmarks for validation. |
| 3:00–5:30 | Kavindu | Formulation | Report §3 equations | Decision variables, objective, constraints. Explain MTZ (A4) with the 3→5→3 sub-tour example. Then Model B and the rounded capacity cuts. NP-hard: generalises TSP and bin packing. |
| 5:30–7:30 | Kavindu | Exact method demo | Notebook §2.1, benchmark table | Run Model A vs Model B on the 15-outlet example. Show that Model B proves A-n32-k5 optimal (784) in about 10 s, and the MTZ timeouts from 15–18 outlets in the results table. Key message: formulation strength matters as much as the solver. |
| 7:30–9:30 | Ushan | Heuristics | Notebook §2.2, route plot | Clarke–Wright savings formula and merge rule. SA: moves, Metropolis acceptance, cooling, granular neighbourhood, and why the uniform-random version failed on n ≥ 100. |
| 9:30–12:30 | Ushan | Results | `runtime_vs_size`, `gap_vs_size`, `sa_sensitivity`, `routes_n30`, `routes_n557` | Read the numbers from the comparison table: where exact stops scaling, SA's gap to optimum, the time–quality trade-off, all solutions feasible, the full 557-outlet plan in seconds. |
| 12:30–14:15 | Both | Discussion | Report §7 | When to use which method; limitations (haversine distance, single day, no time windows); future work (OSRM road distances, time windows, shift limits, branch-cut-and-price, warm-starting the MILP with SA). |
| 14:15–15:00 | Both | Wrap-up | GitHub repo page, commit history | Reproduce with `pip install -r requirements.txt` → `run_experiments.py` → `analyse_results.py`. Thank you. |

**Recording tips:** rehearse once with a timer. Run the notebook beforehand so no cell waits on camera. Upload to YouTube as *Unlisted* and put the link in `submission.txt`.
```

### 10.2 Code appendix (works on Windows)

```powershell
python report/make_code_appendix.py
```

### 10.3 The PDF

`build_report.sh` needs **pandoc + XeLaTeX** (Linux/WSL or Git Bash with pandoc and MiKTeX installed). On Windows the easiest route: use the provided **`Optimization_Report.pdf`** and copy it to `report/`. If you reword sections and want to rebuild yourself: install [pandoc](https://pandoc.org/installing.html) and [MiKTeX](https://miktex.org/download), then in **Git Bash**: `cd report && ./build_report.sh`. (If the DejaVu fonts are missing, change the three font lines in `metadata.yaml` to fonts you have, e.g. *Cambria*, *Calibri*, *Consolas*.)

### 10.4 Submission files

**`submission.txt`** — paste your links later

```text
IT5082 Optimization Methods - Programming Assignment
Delivery Route Optimisation for a Distribution Centre in Colombo District

Members
  Kavindu Ariyasinghe - MS26900186
  Ushan Gunaratne     - MS26901930

Dataset
  Colombo District GND centroids and 2012 Census population (557 Grama Niladhari Divisions)
    Source repository : https://github.com/nuuuwan/gig-data  (ents/gnd.tsv, ents/dsd.tsv)
    Extracted copy    : data/raw/colombo_gnd.csv in our GitHub repository
  CVRPLIB benchmark instances (validation)
    http://vrp.galgos.inf.puc-rio.br/index.php/en/
    copies via https://github.com/PyVRP/VRPLIB and https://github.com/PyVRP/Instances
    (data/raw/cvrplib/ in our GitHub repository)

GitHub repository
  <PASTE GITHUB REPOSITORY URL HERE>

YouTube video (15 min)
  <PASTE YOUTUBE URL HERE>
```

**`make_submission.py`** — builds the ZIP; refuses while placeholders remain

```python
"""Package Optimization-assignment.zip for upload.

Run after pasting the GitHub and YouTube links into submission.txt:
    python make_submission.py
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ["members.txt", "submission.txt",
         "report/Optimization_Report.pdf", "report/code_appendix.txt"]

if "<PASTE" in (ROOT / "submission.txt").read_text(encoding="utf-8"):
    sys.exit("submission.txt still contains placeholder links - add the GitHub and YouTube URLs first.")
missing = [f for f in FILES if not (ROOT / f).exists()]
if missing:
    sys.exit(f"missing files: {missing}")

out = ROOT / "Optimization-assignment.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(ROOT / f, arcname=Path(f).name)
print(f"wrote {out}")
for info in zipfile.ZipFile(out).infolist():
    print(f"  {info.filename:28s} {info.file_size:>10,d} bytes")
```

**`README.md`** — final version

````markdown
# Delivery Route Optimisation — Colombo District

IT5082 Optimization Methods, group programming assignment.
Kavindu Ariyasinghe & Ushan Gunaratne.

Daily deliveries from a distribution centre in Kelaniya to retail outlets across
Colombo District are modelled as a **Capacitated Vehicle Routing Problem (CVRP)** and
solved with:

* **Exact:** MILP (MTZ compact model, and a two-index model with lazily generated
  rounded capacity cuts) solved by HiGHS via `scipy.optimize.milp`
* **Heuristic / metaheuristic:** Clarke–Wright savings + Simulated Annealing

## Data
* `data/raw/colombo_gnd.csv` — 557 Grama Niladhari Divisions of Colombo District:
  centroid coordinates and 2012 Census population
  (Department of Census & Statistics, via github.com/nuuuwan/gig-data)
* `data/raw/cvrplib/` — CVRPLIB benchmark instances with proven optimal / best-known
  solutions (Augerat, Eilon, Uchoa et al.), used to validate the solvers
* `data/instances/` — generated Colombo instances (`python experiments/build_instances.py`)

## Results at a glance
| | Exact B (capacity cuts) | Exact A (MTZ) | Clarke–Wright | Simulated Annealing |
|---|---|---|---|---|
| Colombo 8–30 outlets | optimal on 23/24 | times out from 15–18 outlets | 1.96% above optimal | 0.13% above optimal |
| Large (100–557) | lower bound only | – | < 0.25 s | within 3% of best-known (CVRPLIB), 557 outlets in 19 s |

Full analysis: `report/Optimization_Report.pdf` and the executed notebook
`notebooks/delivery_optimization.ipynb`.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python experiments/build_instances.py   # regenerate Colombo instances + dataset figure
python experiments/run_experiments.py all   # all experiments (~2 h on 2 cores)
python experiments/analyse_results.py       # tables + figures in results/
python notebooks/build_notebook.py          # re-execute the walkthrough notebook
pytest -q                               # solvers reproduce CVRPLIB optima
python report/make_code_appendix.py     # plain-text code appendix
(cd report && ./build_report.sh)        # report PDF (needs pandoc + xelatex)
python make_submission.py               # Optimization-assignment.zip
```

## Layout
```
src/cvrp/        instance model, data loaders, solvers
experiments/     reproducible scripts
notebooks/       analysis notebook
results/         CSV results and figures
report/          report sources
tests/           regression tests
```
````

```powershell
git add report README.md submission.txt make_submission.py
git commit -m "docs(report): report sources, PDF, code appendix, video plan and submission files"
git push
```

### 10.5 Release to `main`

GitHub → **Pull requests → New** → base `main` ← compare `develop` → title *"Release: final submission"* → **Merge pull request** (normal merge — keeps every commit).

### 10.6 After the video

1. Paste the GitHub URL and YouTube URL into `submission.txt`, commit (`docs: add repository and video links`), push to `develop`, merge to `main` again.
2. Build the ZIP:

```powershell
python make_submission.py
```

3. Upload `Optimization-assignment.zip` **once**.

---

## Final checklist

* [ ] `pytest -q` → 11 passed
* [ ] GitHub `main` shows commits by **both** of you and the merged pull requests
* [ ] Repo is public (or lecturer added) — open it in an incognito window
* [ ] Video is Unlisted and opens in an incognito window
* [ ] Report reworded in your own words; Section 8 (contributions) is accurate
* [ ] ZIP contains: members.txt, submission.txt, Optimization_Report.pdf, code_appendix.txt
