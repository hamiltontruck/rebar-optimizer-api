from collections import defaultdict
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ortools.linear_solver import pywraplp


class Piece(BaseModel):
    size: float = Field(gt=0)
    length: float = Field(gt=0)
    quantity: int = Field(gt=0)


class OptimizeRequest(BaseModel):
    stock_length: float = Field(default=12.0, gt=0)
    kerf_mm: float = Field(default=0, ge=0)
    pieces: List[Piece]


app = FastAPI(title="Adil Rebar Optimizer API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cutting-optimizer-pro.adilabdu52.chatgpt.site"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def patterns(lengths, stock, kerf, limit=10000):
    found = []

    def walk(i, used, counts):
        if len(found) >= limit:
            return
        if i == len(lengths):
            if any(counts):
                found.append(tuple(counts))
            return
        cuts_before = sum(counts)
        unit = lengths[i]
        max_count = int((stock - used + (kerf if cuts_before else 0)) // (unit + kerf))
        for n in range(max_count, -1, -1):
            extra = n * unit + (n * kerf if cuts_before else max(0, n - 1) * kerf)
            counts.append(n)
            walk(i + 1, used + extra, counts)
            counts.pop()

    walk(0, 0.0, [])
    return list(dict.fromkeys(found))


def solve_size(size, requirements, stock, kerf):
    lengths = sorted(requirements, reverse=True)
    demand = [requirements[x] for x in lengths]
    scale = 1000
    capacity = int(round(stock * scale))
    weights = [int(round((length + kerf) * scale)) for length in lengths]
    # Every bar gets one kerf allowance back because only gaps between cuts use kerf.
    priced_capacity = capacity + int(round(kerf * scale))

    # Start with homogeneous patterns so every demand is covered.
    pats = []
    for j, weight in enumerate(weights):
        pat = [0] * len(lengths)
        pat[j] = max(1, priced_capacity // weight)
        pats.append(tuple(pat))

    # Gilmore-Gomory column generation: LP dual prices a new knapsack pattern.
    for _ in range(500):
        lp = pywraplp.Solver.CreateSolver("GLOP")
        variables = [lp.NumVar(0, lp.infinity(), f"p{i}") for i in range(len(pats))]
        constraints = []
        for j, qty in enumerate(demand):
            constraints.append(lp.Add(sum(pats[i][j] * variables[i] for i in range(len(pats))) >= qty))
        lp.Minimize(sum(variables))
        if lp.Solve() != pywraplp.Solver.OPTIMAL:
            break
        dual = [c.dual_value() for c in constraints]

        best = [0.0] * (priced_capacity + 1)
        choice = [-1] * (priced_capacity + 1)
        for used in range(1, priced_capacity + 1):
            value, selected = best[used - 1], -1
            for j, weight in enumerate(weights):
                if weight <= used:
                    candidate = best[used - weight] + dual[j]
                    if candidate > value + 1e-10:
                        value, selected = candidate, j
            best[used], choice[used] = value, selected

        used = max(range(priced_capacity + 1), key=lambda c: best[c])
        if best[used] <= 1.000001:
            break
        pat = [0] * len(lengths)
        while used > 0 and choice[used] >= 0:
            j = choice[used]
            pat[j] += 1
            used -= weights[j]
        pat = tuple(pat)
        if not any(pat) or pat in pats:
            break
        pats.append(pat)

    # Solve the integer master problem over generated cutting patterns.
    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC")
    x = [solver.IntVar(0, solver.infinity(), f"p{i}") for i in range(len(pats))]
    for j, qty in enumerate(demand):
        solver.Add(sum(pats[i][j] * x[i] for i in range(len(pats))) >= qty)
    total_bars = sum(x)
    solver.Minimize(total_bars)
    solver.SetTimeLimit(60000)
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise HTTPException(500, f"Optimization failed for {size} mm")

    best_bars = int(round(sum(v.solution_value() for v in x)))
    solver.Add(total_bars == best_bars)
    surplus = sum(
        sum(pats[i][j] * x[i] for i in range(len(pats))) - demand[j]
        for j in range(len(lengths))
    )
    solver.Minimize(surplus)
    solver.SetTimeLimit(30000)
    second_status = solver.Solve()
    if second_status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        status = second_status

    result = []
    for i, pat in enumerate(pats):
        count = int(round(x[i].solution_value()))
        if count:
            cuts = [lengths[j] for j, n in enumerate(pat) for _ in range(n)]
            used = sum(cuts) + max(0, len(cuts) - 1) * kerf
            result.append({"size": size, "stock": stock, "cuts": cuts, "waste": round(stock-used, 6), "count": count})
    return result, "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE"

@app.get("/")
def health():
    return {"ok": True, "service": "Adil Rebar Optimizer API", "optimizer": "minimum-bars-v2"}


@app.post("/optimize")
def optimize(body: OptimizeRequest):
    grouped = defaultdict(lambda: defaultdict(int))
    for p in body.pieces:
        if p.length > body.stock_length:
            raise HTTPException(400, f"Cut {p.length} m is longer than stock")
        grouped[p.size][p.length] += p.quantity
    all_patterns, statuses = [], []
    for size, req in sorted(grouped.items()):
        rows, status = solve_size(size, req, body.stock_length, body.kerf_mm / 1000)
        all_patterns.extend(rows)
        statuses.append(status)
    bars = sum(p["count"] for p in all_patterns)
    waste = sum(p["waste"] * p["count"] for p in all_patterns)
    purchased = bars * body.stock_length
    return {
        "engine": "Google OR-Tools",
        "status": "OPTIMAL" if all(s == "OPTIMAL" for s in statuses) else "FEASIBLE",
        "bars": bars,
        "waste": round(waste, 6),
        "utilization": round(100 * (purchased - waste) / purchased, 3) if purchased else 0,
        "patterns": all_patterns,
    }
