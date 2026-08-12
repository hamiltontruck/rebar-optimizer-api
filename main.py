from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from hashlib import sha256
import json
from math import gcd
from threading import Lock
import time
from typing import Callable, List, Optional
from uuid import uuid4

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


app = FastAPI(title="Adil Rebar Optimizer API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cutting-optimizer-pro.adilabdu52.chatgpt.site"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


JobProgress = Optional[Callable[[str], None]]
JOB_TTL_SECONDS = 2 * 60 * 60
job_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rebar-optimizer")
job_lock = Lock()
jobs = {}


def _set_job(job_id, **values):
    with job_lock:
        if job_id in jobs:
            jobs[job_id].update(values)
            jobs[job_id]["updated_at"] = time.time()


def _cleanup_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS
    with job_lock:
        expired = [job_id for job_id, job in jobs.items() if job["updated_at"] < cutoff]
        for job_id in expired:
            jobs.pop(job_id, None)


def _request_key(body: OptimizeRequest):
    canonical = json.dumps(body.model_dump(), sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def solve_size(size, requirements, stock, kerf, progress: JobProgress = None):
    lengths = sorted(requirements, reverse=True)
    demand = [requirements[x] for x in lengths]
    scale = 1000
    capacity_mm = int(round(stock * scale))
    kerf_mm = int(round(kerf * scale))
    weights_mm = [int(round((length + kerf) * scale)) for length in lengths]
    priced_capacity_mm = capacity_mm + kerf_mm

    # Most rebar schedules use centimetre precision. Dividing by the common
    # unit keeps the pricing knapsack exact while making large schedules much
    # faster (for example, 12,000 states becomes 1,200 states at 10 mm precision).
    common_unit = reduce(gcd, [priced_capacity_mm, *weights_mm]) or 1
    priced_capacity = priced_capacity_mm // common_unit
    weights = [weight // common_unit for weight in weights_mm]

    # Start with homogeneous patterns so every demand is covered.
    pats = []
    for j, weight in enumerate(weights):
        pat = [0] * len(lengths)
        pat[j] = max(1, priced_capacity // weight)
        pats.append(tuple(pat))

    if progress:
        progress(f"{size:g} mm: generating efficient cutting patterns")

    # Gilmore-Gomory column generation. A bounded pricing phase prevents a
    # pathological schedule from monopolising the free Render instance.
    pricing_started = time.monotonic()
    max_columns = max(120, min(350, len(lengths) * 4))
    for _ in range(max_columns):
        if time.monotonic() - pricing_started > 120:
            break

        lp = pywraplp.Solver.CreateSolver("GLOP")
        variables = [lp.NumVar(0, lp.infinity(), f"p{i}") for i in range(len(pats))]
        constraints = []
        for j, qty in enumerate(demand):
            constraints.append(lp.Add(sum(pats[i][j] * variables[i] for i in range(len(pats))) >= qty))
        lp.Minimize(sum(variables))
        if lp.Solve() != pywraplp.Solver.OPTIMAL:
            break
        dual = [constraint.dual_value() for constraint in constraints]

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

        used = max(range(priced_capacity + 1), key=lambda candidate: best[candidate])
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

    if progress:
        progress(f"{size:g} mm: solving the minimum-bar plan")

    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC")
    if not solver:
        raise HTTPException(500, "No integer solver is available")
    x = [solver.IntVar(0, solver.infinity(), f"p{i}") for i in range(len(pats))]
    for j, qty in enumerate(demand):
        solver.Add(sum(pats[i][j] * x[i] for i in range(len(pats))) >= qty)
    total_bars = sum(x)
    solver.Minimize(total_bars)
    solver.SetTimeLimit(90000)
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise HTTPException(500, f"Optimization failed for {size:g} mm")

    best_bars = int(round(sum(variable.solution_value() for variable in x)))
    solver.Add(total_bars == best_bars)
    surplus = sum(
        sum(pats[i][j] * x[i] for i in range(len(pats))) - demand[j]
        for j in range(len(lengths))
    )
    solver.Minimize(surplus)
    solver.SetTimeLimit(20000)
    second_status = solver.Solve()
    if second_status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        status = second_status

    result = []
    for i, pat in enumerate(pats):
        count = int(round(x[i].solution_value()))
        if count:
            cuts = [lengths[j] for j, amount in enumerate(pat) for _ in range(amount)]
            used = sum(cuts) + max(0, len(cuts) - 1) * kerf
            result.append(
                {
                    "size": size,
                    "stock": stock,
                    "cuts": cuts,
                    "waste": round(stock - used, 6),
                    "count": count,
                }
            )
    return result, "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE"


def compute_optimization(body: OptimizeRequest, progress: JobProgress = None):
    grouped = defaultdict(lambda: defaultdict(int))
    for piece in body.pieces:
        if piece.length > body.stock_length:
            raise HTTPException(400, f"Cut {piece.length:g} m is longer than stock")
        # Normalise spreadsheet float representations and aggregate duplicate rows.
        grouped[round(piece.size, 6)][round(piece.length, 6)] += piece.quantity

    if not grouped:
        raise HTTPException(400, "At least one valid piece is required")

    all_patterns, statuses = [], []
    total_sizes = len(grouped)
    for index, (size, requirements) in enumerate(sorted(grouped.items()), start=1):
        if progress:
            progress(f"Size {index}/{total_sizes} — {size:g} mm")
        rows, status = solve_size(
            size,
            requirements,
            body.stock_length,
            body.kerf_mm / 1000,
            progress,
        )
        all_patterns.extend(rows)
        statuses.append(status)

    bars = sum(pattern["count"] for pattern in all_patterns)
    waste = sum(pattern["waste"] * pattern["count"] for pattern in all_patterns)
    purchased = bars * body.stock_length
    return {
        "engine": "Google OR-Tools",
        "status": "OPTIMAL" if all(status == "OPTIMAL" for status in statuses) else "FEASIBLE",
        "bars": bars,
        "waste": round(waste, 6),
        "utilization": round(100 * (purchased - waste) / purchased, 3) if purchased else 0,
        "patterns": all_patterns,
    }


def _run_job(job_id, payload):
    try:
        _set_job(job_id, state="running", message="Python OR-Tools is preparing the schedule")
        body = OptimizeRequest.model_validate(payload)
        result = compute_optimization(
            body,
            lambda message: _set_job(job_id, state="running", message=message),
        )
        _set_job(job_id, state="completed", message="Optimization complete", result=result)
    except HTTPException as exc:
        _set_job(job_id, state="failed", message=str(exc.detail))
    except Exception as exc:
        _set_job(job_id, state="failed", message=f"Optimization failed: {exc}")


@app.get("/")
def health():
    return {
        "ok": True,
        "service": "Adil Rebar Optimizer API",
        "optimizer": "column-generation-v4-jobs",
    }


@app.post("/optimize")
def optimize(body: OptimizeRequest):
    return compute_optimization(body)


@app.post("/optimize/jobs", status_code=202)
def start_optimize_job(body: OptimizeRequest):
    _cleanup_jobs()
    key = _request_key(body)
    with job_lock:
        for job_id, job in jobs.items():
            if job["key"] == key and job["state"] in {"queued", "running", "completed"}:
                return {"job_id": job_id, "state": job["state"], "message": job["message"]}
        job_id = uuid4().hex
        jobs[job_id] = {
            "key": key,
            "state": "queued",
            "message": "Queued for Python OR-Tools",
            "result": None,
            "updated_at": time.time(),
        }
    job_executor.submit(_run_job, job_id, body.model_dump())
    return {"job_id": job_id, "state": "queued", "message": "Queued for Python OR-Tools"}


@app.get("/optimize/jobs/{job_id}")
def get_optimize_job(job_id: str):
    _cleanup_jobs()
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Optimization job not found or expired")
        response = {
            "job_id": job_id,
            "state": job["state"],
            "message": job["message"],
        }
        if job["state"] == "completed":
            response["result"] = job["result"]
        return response
