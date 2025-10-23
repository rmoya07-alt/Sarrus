# -*- coding: utf-8 -*-
"""
DESARROLLO DEL SOFTWARE:
Esta implementación fue desarrollada por el autor con asistencia de 
Claude (Anthropic), un modelo de lenguaje de IA, utilizado como 
herramienta de programación asistida. Todas las especificaciones 
algorítmicas y verificaciones matemáticas fueron realizadas por 
el autor

Determinante – Polilíneas / Rectas (v12 - Modificado)

Cambios principales:
• "Polilíneas" renombrado a "Polilíneas rotados"
• Gráfica actualizada para mostrar rotación de monomios mediante 
  desplazamiento horizontal (ventana deslizante)

Basado en la v4 original (mantiene su flujo para n ≤ 5) y ampliado con:

• Tamaños: 2..9.
• Para n ≥ 6: SIN gráficas. En su lugar, generación de **orbitales con aportes**
  (polilíneas y rectas) y exportación a **Excel (.xlsx, si xlsxwriter disponible; si no, .csv)**
  y **PDF**. La matriz siempre se muestra ARRIBA.
• La generación de matrices conserva los modos de v4: manual, aleatoria con semilla
  (botón "Rellenar ejemplo") e importación CSV.

Notas de rendimiento:
- La enumeración de TODAS las órbitas crece como (n−1)!; para n=9 son 40,320 bases y
  362,880 monomios por modo. Por eso:
  – Para n ≥ 6, las **gráficas se desactivan**.
  – El **Top‑k global** de v4 se desactiva cuando n ≥ 6 (se indicará en resultados).
  – La exportación "TODOS los orbitales → PDF" genera un **sumario** (una línea por base).
    Para ver cada monomio con su aporte, use Excel/CSV o exporte el **orbital actual** a PDF.

Requisitos: numpy, matplotlib, tkinter. Opcional: xlsxwriter (para .xlsx).
"""

from __future__ import annotations
import itertools, os, math, sys, json, csv, io, datetime, platform, textwrap, time
from typing import List, Optional, Tuple, Iterable

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont

# ===== Excel opcional =====
try:
    import xlsxwriter  # type: ignore
    XLSX_OK = True
except Exception:
    XLSX_OK = False


# =========================
# Tiempos (añadido)
# =========================
def _fmt_ms(seconds: float) -> str:
    ms = seconds * 1000.0
    if ms < 1.0:
        return "< 1 ms"
    if ms < 1000.0:
        return f"{ms:.0f} ms"
    return f"{ms/1000.0:.1f} s"

def _timeit(func, *args, **kwargs):
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    dt = time.perf_counter() - t0
    return res, dt

# =========================
# Utilidades combinatorias
# =========================

def perm_sign(cols_per_row: List[int]) -> int:
    inv = 0
    for i in range(len(cols_per_row)):
        for j in range(i+1, len(cols_per_row)):
            if cols_per_row[i] > cols_per_row[j]:
                inv += 1
    return 1 if inv % 2 == 0 else -1

def rotate(lst: List[int], r: int) -> List[int]:
    if not lst: return lst
    r %= len(lst)
    if r == 0: return lst[:]
    return lst[-r:] + lst[:-r]  # rotación a la derecha

def inverse_perm(cols_per_row: List[int]) -> List[int]:
    n = len(cols_per_row)
    inv = [0]*n
    for row, col in enumerate(cols_per_row, start=1):
        inv[col-1] = row
    return inv

def companion_orbital(base: List[int]) -> List[int]:
    if len(base) <= 1: return base[:]
    return [base[0]] + list(reversed(base[1:]))

def all_bases(n: int) -> List[List[int]]:
    bases = []
    for c2 in range(2, n+1):
        resto = [k for k in range(2, n+1) if k != c2]
        for tail in itertools.permutations(resto):
            bases.append([1, c2] + list(tail))
    return bases

def all_bases_iter(n: int) -> Iterable[List[int]]:
    """Versión iteradora (no materializa la lista completa)."""
    for c2 in range(2, n+1):
        resto = [k for k in range(2, n+1) if k != c2]
        for tail in itertools.permutations(resto):
            yield [1, c2] + list(tail)

# =========================
# Determinantes (como v4)
# =========================

def det_gauss(A: np.ndarray) -> float:
    A = A.copy().astype(float)
    n = A.shape[0]
    det_sign = 1.0
    for k in range(n):
        p = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[p, k]) < 1e-15:
            return 0.0
        if p != k:
            A[[k, p], :] = A[[p, k], :]
            det_sign *= -1.0
        for i in range(k+1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
    return det_sign * float(np.prod(np.diag(A)))

def lu_factor(A: np.ndarray):
    A = A.copy().astype(float)
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros_like(A)
    U = A.copy()
    detP_sign = 1.0
    for k in range(n):
        p = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[p, k]) < 1e-15:
            return P, L, U, 0.0
        if p != k:
            U[[k, p], :] = U[[p, k], :]
            P[[k, p], :] = P[[p, k], :]
            if k > 0:
                L[[k, p], :k] = L[[p, k], :k]
            detP_sign *= -1.0
        L[k, k] = 1.0
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    return P, L, U, detP_sign

def det_lu(A: np.ndarray) -> float:
    P, L, U, detP_sign = lu_factor(A)
    if detP_sign == 0.0:
        return 0.0
    return detP_sign * float(np.prod(np.diag(U)))

def det_bareiss_exact_or_none(A: np.ndarray) -> Optional[int]:
    n = A.shape[0]
    A_int = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            x = float(A[i, j])
            xr = round(x)
            if abs(x - xr) <= 1e-12:
                A_int[i][j] = int(xr)
            else:
                return None
    M = [row[:] for row in A_int]
    sign = 1
    prev = 1
    for k in range(n-1):
        p = k
        while p < n and M[p][k] == 0:
            p += 1
        if p == n:
            return 0
        if p != k:
            M[k], M[p] = M[p], M[k]
            sign *= -1
        pivot = M[k][k]
        for i in range(k+1, n):
            for j in range(k+1, n):
                M[i][j] = (M[i][j]*pivot - M[i][k]*M[k][j]) // prev
            M[i][k] = 0
        prev = pivot if pivot != 0 else 1
    return sign * M[n-1][n-1]

# =========================
# Determinante por Polilíneas (Leibniz)
# =========================
def det_polilineas(A: np.ndarray) -> float:
    n = A.shape[0]
    total = 0.0
    # perm_sign espera permutaciones 1..n según el resto del código
    for cols in itertools.permutations(range(1, n+1)):
        sgn = perm_sign(list(cols))
        prod = 1.0
        for i, j in enumerate(cols, start=1):
            prod *= A[i-1, j-1]
        total += sgn * prod
    return float(total)

# =========================
# Aportes (polilíneas/rectas)
# =========================

def orbital_contributions(A: np.ndarray, base: List[int]):
    n = A.shape[0]
    base_sgn = perm_sign(base)
    out = []
    total = 0.0
    for r in range(n):
        cols = rotate(base, r)
        val = 1.0
        for row, col in enumerate(cols, start=1):
            val *= A[row-1, col-1]
        s = base_sgn if (n % 2 == 1) else base_sgn * ((-1)**r)
        tot = s * val
        total += tot
        out.append({"rotation": r, "perm": cols, "sign": int(s), "value": float(val), "contribution": float(tot)})
    return out, total


def orbital_contributions_increment(A: np.ndarray, base: List[int]):
    n = A.shape[0]
    base_sgn = perm_sign(base)
    out = []
    total = 0.0
    for r in range(n):
        cols = [((v - 1 + r) % n) + 1 for v in base]
        val = 1.0
        for row, col in enumerate(cols, start=1):
            val *= A[row-1, col-1]
        s = base_sgn if (n % 2 == 1) else base_sgn * ((-1)**r)
        tot = s * val
        total += tot
        out.append({"rotation": r, "perm": cols, "sign": int(s), "value": float(val), "contribution": float(tot)})
    return out, total

def all_monomials_with_contributions(A: np.ndarray):
    n = A.shape[0]
    items = []
    for b_idx, base in enumerate(all_bases(n), start=1):
        base_sgn = perm_sign(base)
        for r in range(n):
            cols = rotate(base, r)
            val = 1.0
            for row, col in enumerate(cols, start=1):
                val *= A[row-1, col-1]
            s = base_sgn if (n % 2 == 1) else base_sgn * ((-1)**r)
            items.append({
                "base_index": b_idx,
                "base": base,
                "rotation": r,
                "perm": cols,
                "sign": int(s),
                "value": float(val),
                "contribution": float(s*val),
                "abs_contribution": float(abs(s*val)),
            })
    return items

# =========================
# Figuras (solo n ≤ 5)
# =========================

def fig_polilineas_rotados_orbital(n: int, base: List[int]) -> plt.Figure:
    """
    POLILÍNEAS ROTADOS (rotación cíclica de la lista):
    Para cada rotación r, aplicamos rotate(base, r) que rota la lista completa.
    El patrón visual se mantiene pero se desplaza horizontalmente (ventana deslizante).
    
    Ejemplo: base=[1,2,3,4]
    r=0: [1,2,3,4]
    r=1: [4,1,2,3] (rotación a la derecha)
    r=2: [3,4,1,2]
    r=3: [2,3,4,1]
    """
    fsz = 11 if n <= 3 else (10 if n == 4 else 9)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0.5, n + 0.5)
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    ax.set_xticklabels([str(j) for j in range(1, n + 1)])
    ax.set_yticklabels([str(i) for i in range(1, n + 1)])
    ax.grid(True)
    ax.invert_yaxis()
    
    # Mostrar elementos aij en la matriz (fondo gris claro)
    for i in range(1, n+1):
        for j in range(1, n+1):
            ax.text(j, i, f"a{i}{j}", ha="center", va="center", 
                   fontsize=fsz, color='lightgray', alpha=0.6)
    
    base_sgn = perm_sign(base)
    
    # Colores diferentes para cada rotación
    colors = plt.cm.rainbow(np.linspace(0, 1, n))
    
    # Dibujar cada rotación del monomio (ROTACIÓN DE LA LISTA)
    for r in range(n):
        cols = rotate(base, r)  # Rotación cíclica de la lista completa
        
        # Calcular posiciones: columna j → fila cols[j-1]
        xs = list(range(1, n+1))
        ys = cols  # Las filas seleccionadas en cada columna
        
        # Dibujar la polilínea conectada
        ax.plot(xs, ys, color=colors[r], linewidth=2.5, marker='o', 
                markersize=9, label=f'r={r}: {cols}', alpha=0.85)
        
        # Etiquetar los elementos seleccionados
        for j, row in enumerate(cols, start=1):
            ax.text(j, row-0.3, f"a{row}{j}", ha="center", va="top", 
                   fontsize=fsz-1, color=colors[r], weight='bold')
    
    ax.set_title(f"Polilíneas rotados (rotación cíclica) – base {base}\n"
                 f"Signo base: {'+' if base_sgn>0 else '−'}  |  "
                 f"Patrón se desplaza horizontalmente")
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95, 
             title='Rotaciones')
    fig.tight_layout()
    return fig


def fig_polilineas_incremento_orbital(n: int, base: List[int]) -> plt.Figure:
    """
    POLILÍNEAS INCREMENTO (incremento de filas módulo n):
    Para cada rotación r, incrementamos cada elemento: cols[j] = ((base[j]-1 + r) % n) + 1
    Todas las polilíneas tienen la MISMA FORMA (mismo perfil), pero desplazadas verticalmente.
    
    Ejemplo: base=[1,2,3,4]
    r=0: [1,2,3,4]
    r=1: [2,3,4,1] (cada elemento +1 mod 4)
    r=2: [3,4,1,2] (cada elemento +2 mod 4)
    r=3: [4,1,2,3] (cada elemento +3 mod 4)
    """
    fsz = 11 if n <= 3 else (10 if n == 4 else 9)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0.5, n + 0.5)
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    ax.set_xticklabels([str(j) for j in range(1, n + 1)])
    ax.set_yticklabels([str(i) for i in range(1, n + 1)])
    ax.grid(True)
    ax.invert_yaxis()
    
    # Mostrar elementos aij en la matriz (fondo gris claro)
    for i in range(1, n+1):
        for j in range(1, n+1):
            ax.text(j, i, f"a{i}{j}", ha="center", va="center", 
                   fontsize=fsz, color='lightgray', alpha=0.6)
    
    base_sgn = perm_sign(base)
    
    # Colores diferentes para cada incremento
    colors = plt.cm.rainbow(np.linspace(0, 1, n))
    
    # Dibujar cada incremento (INCREMENTO DE FILAS)
    for r in range(n):
        # Incremento: cada elemento se incrementa r posiciones módulo n
        inc = [((v - 1 + r) % n) + 1 for v in base]
        
        # Calcular posiciones: columna j → fila inc[j-1]
        xs = list(range(1, n+1))
        ys = inc
        
        # Dibujar la polilínea conectada
        ax.plot(xs, ys, color=colors[r], linewidth=2.5, marker='o', 
                markersize=9, label=f'r={r}: {inc}', alpha=0.85)
        
        # Etiquetar los elementos seleccionados
        for j, row in enumerate(ys, start=1):
            ax.text(j, row-0.3, f"a{row}{j}", ha="center", va="top", 
                   fontsize=fsz-1, color=colors[r], weight='bold')
    
    ax.set_title(f"Polilíneas incremento (incremento módulo n) – base {base}\n"
                 f"Signo base: {'+' if base_sgn>0 else '−'}  |  "
                 f"Mismo perfil, desplazamiento vertical")
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95,
             title='Incrementos')
    fig.tight_layout()
    return fig

from matplotlib import gridspec

def draw_grid_extended_perm(ax, n: int, base: List[int]):
    cols = 2*n
    ax.set_xlim(0.5, cols + 0.5); ax.set_ylim(0.5, n + 0.5)
    ax.set_xticks(range(1, cols + 1)); ax.set_yticks(range(1, n + 1))
    ax.set_xticklabels([str(j) for j in range(1, cols + 1)])
    ax.set_yticklabels([str(i) for i in range(1, n + 1)])
    ax.grid(True); ax.invert_yaxis()
    fsz = 10 if n <= 4 else 9
    for i in range(1, n+1):
        for c in range(1, cols+1):
            j = base[(c-1) % n]
            ax.text(c, i, f"a{i}{j}", ha="center", va="center", fontsize=fsz)

def fig_rectas_orbital(A: np.ndarray, base: List[int], show_companion: bool=True) -> plt.Figure:
    n = A.shape[0]
    comp = companion_orbital(base)
    base_sgn = perm_sign(base); comp_sgn = perm_sign(comp)
    fig = plt.figure(figsize=(10.4, 7.4))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3.2, 1.4], hspace=0.18, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    ax_left = fig.add_subplot(gs[1, 0]); ax_right = fig.add_subplot(gs[1, 1])
    draw_grid_extended_perm(ax, n, base)
    ax.set_title(f"Rectas paralelas – {n}×{n}  (2n rectas: cada monomio una vez)")

    def trace_lines(n, slope):
        starts = range(1, n+1); lines = []
        for c0 in starts:
            xs=[]; ys=[]; r = 1 if slope>0 else n; c = c0
            for _ in range(n):
                xs.append(c); ys.append(r); r = r + 1 if slope>0 else r - 1; c += 1
            lines.append((xs, ys))
        return lines

    for xs, ys in trace_lines(n, slope=+1):
        ax.plot(xs, ys); ax.plot(xs, ys, marker="o", linestyle="None")
    for xs, ys in trace_lines(n, slope=-1):
        ax.plot(xs, ys, linestyle="--"); ax.plot(xs, ys, marker="o", linestyle="None")

    ax_left.axis('off')
    left_lines = [f"Orbital base {base} (signo base {'+' if base_sgn>0 else '−'})"]
    total_left = 0.0
    for r in range(n):
        cols = rotate(base, r)
        val = 1.0
        for row, col in enumerate(cols, start=1):
            val *= A[row-1, col-1]
        s = base_sgn if (n % 2 == 1) else base_sgn * ((-1)**r)
        total_left += s * val
        left_lines.append(f"  {'+' if s>0 else '−'} {cols}  ⇒  {val:.8f}")
    left_lines.append(f"Suma orbital base = {total_left:.8f}")
    ax_left.text(0.01, 0.98, "\n".join(left_lines), ha="left", va="top",
                 fontsize=10, family="monospace", transform=ax_left.transAxes)

    ax_right.axis('off')
    if show_companion:
        right_lines = [f"Orbital companion {comp} (signo base {'+' if comp_sgn>0 else '−'})"]
        total_right = 0.0
        for r in range(n):
            cols = rotate(comp, r)
            val = 1.0
            for row, col in enumerate(cols, start=1):
                val *= A[row-1, col-1]
            s = comp_sgn if (n % 2 == 1) else comp_sgn * ((-1)**r)
            total_right += s * val
            right_lines.append(f"  {'+' if s>0 else '−'} {cols}  ⇒  {val:.8f}")
        right_lines.append(f"Suma orbital companion = {total_right:.8f}")
        ax_right.text(0.01, 0.98, "\n".join(right_lines), ha="left", va="top",
                      fontsize=10, family="monospace", transform=ax_right.transAxes)
    else:
        ax_right.text(0.5, 0.5, "Companion oculto", ha="center", va="center")

    fig.tight_layout(); return fig

# =========================
# Tkinter GUI
# =========================

# =========================
# Rectas totales en A_ext (W = 3n - 2)
# =========================

def _sgn_perm(perm):
    inv = 0
    for i in range(len(perm)):
        for j in range(i+1, len(perm)):
            if perm[i] > perm[j]: inv += 1
    return -1 if (inv % 2) else 1

def _product_monomial(A, perm):
    n = A.shape[0]; val = 1.0
    for i in range(1, n+1):
        val *= A[i-1, perm[i-1]-1]
    return val


def _draw_Aext(ax, n, W):
    ax.set_xlim(0.5, W + 0.5)
    ax.set_ylim(0.5, n + 0.5)
    ax.set_xticks(range(1, W+1))
    ax.set_yticks(range(1, n+1))
    ax.grid(True, linestyle=':', linewidth=0.6)
    ax.invert_yaxis()
    left = n - 1
    ax.add_patch(plt.Rectangle((left + 0.5, 0.5), n, n, fill=False, edgecolor='black', linewidth=1.4))


def _label_Aext(ax, n, W):
    left = n - 1
    for i in range(1, n+1):
        for c in range(1, W+1):
            if c <= left:
                j = c
            elif c <= left + n:
                j = c - left
            else:
                j = c - (left + n)
            ax.text(c, i, f"a{i}{j}", ha="center", va="center", fontsize=9)

def _offsets_for_slope(n, slope):
    W = 3*n - 2
    if slope == +1: return list(range(0, W - n + 1))      # 0..2n-2
    else:           return list(range(n+1, W + 2))        # (n+1)..(3n-1)

def _perm_from_offset(n, d, slope):
    perm = []
    if slope == +1:
        for i in range(1, n+1):
            perm.append(((i + d - 1) % n) + 1)
    else:
        for i in range(1, n+1):
            perm.append(((-i + d - 1) % n) + 1)
    return perm


# === Arschon helpers: cosets S_n / D_n and right-extension geometry ===
import itertools

def _arschon_dihedral(n: int):
    """D_n: n rotations (+1 slope) and n reflection+rotation (-1 slope)."""
    H = []
    for r in range(n):  # rotations
        H.append([((i + r - 1) % n) + 1 for i in range(1, n+1)])
    for r in range(n):  # reflection + rotation
        H.append([(((-i) + r - 1) % n) + 1 for i in range(1, n+1)])
    return H  # size 2n

def _arschon_comp(q, p):
    """Composition q∘p in list notation 1..n (apply p then q)."""
    return [ q[p[i]-1] for i in range(len(p)) ]

def _arschon_row_reps(n: int):
    """Representatives of left cosets S_n / D_n (size n! / (2n))."""
    H = _arschon_dihedral(n)
    covered = set()
    reps = []
    for tau in itertools.permutations(range(1, n+1), n):
        if tuple(tau) in covered:
            continue
        reps.append(list(tau))
        for chi in H:
            covered.add(tuple(_arschon_comp(chi, list(tau))))
    return reps

def _arschon_recta_coords(n: int, slope: int, r: int):
    """
    Right-extension only (width 2n-1). Return coordinates of one diagonal.
    slope=+1: x = y + r, r=0..n-1
    slope=-1: x = (n+1+r) - y, r=0..n-1
    y = 1..n, x = 1..(2n-1)
    """
    xs, ys = [], []
    if slope == +1:
        for y in range(1, n+1):
            xs.append(y + r); ys.append(y)
    else:
        S = n + 1 + r
        for y in range(1, n+1):
            xs.append(S - y); ys.append(y)
    return xs, ys

def _recta_coords(n, d, slope):
    xs, ys = [], []
    if slope == +1:
        for i in range(1, n+1): xs.append(i + d); ys.append(i)
    else:
        for i in range(1, n+1): xs.append(-i + d); ys.append(i)
    return xs, ys



def fig_rectas_totales_Aext(A, slope_mode="Todas", show_labels=True):
    """
    Rectas totales (Arschon): for each representative τ ∈ S_n/D_n,
    permute rows by τ and draw the 2n diagonals (slopes +1 and -1)
    on the right-extended matrix (width = 2n-1). This covers all n! monomials
    exactly once across all tiles (cosets).
    """
    n = A.shape[0]
    reps = _arschon_row_reps(n)      # n!/(2n) tiles
    G   = len(reps)
    T   = 2*n - 1                    # right-extension only
    W   = G * T

    fig, ax = plt.subplots(1, 1, figsize=(max(6, 0.60*W), max(3.5, 0.9*n)))
    ax.set_xlim(0.5, W + 0.5)
    ax.set_ylim(0.5, n + 0.5)
    ax.set_xticks(range(1, W+1))
    ax.set_yticks(range(1, n+1))
    ax.grid(True, linestyle=':', linewidth=0.6)
    ax.invert_yaxis()

    # draw each tile (τ)
    for t, tau in enumerate(reps):
        start = t * T
        # central n×n block inside the tile is columns 1..n
        ax.add_patch(plt.Rectangle((start + 0.5, 0.5), n, n,
                                   fill=False, edgecolor='black', linewidth=1.4))
        if show_labels:
            # labels a_{τ(i), j} across the 2n-1 columns (j = ((c-1) mod n)+1)
            for i in range(1, n+1):
                for c in range(1, T+1):
                    j = ((c - 1) % n) + 1
                    ax.text(start + c, i, f"a{tau[i-1]}{j}", ha="center", va="center", fontsize=9)

    # choose slopes to plot
    slopes = []
    if slope_mode in ("+1", "Todas"):  slopes += [(+1, r) for r in range(n)]
    if slope_mode in ("-1", "Todas"):  slopes += [(-1, r) for r in range(n)]

    # plot 2n diagonals per tile
    for t, _tau in enumerate(reps):
        start = t * T
        for slope, r in slopes:
            xs, ys = _arschon_recta_coords(n, slope, r)
            xs = [start + x for x in xs]
            ax.plot(xs, ys, linewidth=2.0, marker='o', alpha=0.9)

    ax.set_title(f"Rectas totales (Arschon, extensión derecha) – {slope_mode}")
    fig.tight_layout()
    return fig

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Determinante – Polilíneas y Rectas (v12)")
        self.geometry("1380x880")

        # Estado
        self.n = tk.IntVar(value=3)
        
        # Trace para reconstruir la grilla cuando cambia n
        self.n.trace_add("write", lambda *a: self._rebuild_matrix_grid())
        self.preview_kind = tk.StringVar(value="Rectas")  # Rectas o Polilíneas rotados
        self.orbital_idx = tk.IntVar(value=1)
        self.outdir = tk.StringVar(value=os.path.abspath("."))
        self.show_companion = tk.BooleanVar(value=True)
        self.export_format = tk.StringVar(value="SVG")  # SVG o PDF (solo n<=5)
        self.seed_str = tk.StringVar(value="")  # Semilla opcional para ejemplo
        self.topk = tk.IntVar(value=10)  # Top-k global (solo n<=5)

        # Fuente monoespaciada para resultados + tamaño base (robusto para Tk 8.6/3.13)
        try:
            # En algunos entornos (Windows/Python 3.13) es necesario pasar root=self
            self.result_font = tkfont.nametofont("TkFixedFont", root=self)  # type: ignore[arg-type]
        except Exception:
            # Fallback: crear una fuente fija manualmente
            default_family = "Consolas" if sys.platform.startswith("win") else "Courier New"
            try:
                self.result_font = tkfont.Font(root=self, family=default_family, size=11)
            except Exception:
                # Último recurso: usar cualquier TkFixedFont (sin root) y seguimos
                self.result_font = tkfont.nametofont("TkFixedFont")

        try:
            init_sz = int(self.result_font.cget("size"))
        except Exception:
            init_sz = 11
        self._result_font_size = max(11, init_sz)
        try:
            self.result_font.configure(size=self._result_font_size)
        except Exception:
            pass

        # Matriz (grid dinámico)
        self.matrix_entries: List[List[tk.Entry]] = []

        self._build_ui()
        self._rebuild_matrix_grid()
        self._load_example()
        self.bind_events()

    # ----- Layout builders -----

    def _build_ui(self):
        # Frame contenedor principal para el panel izquierdo con scroll
        left_container = ttk.Frame(self)
        left_container.pack(side=tk.LEFT, fill=tk.Y)
        
        # Canvas para permitir scroll
        canvas = tk.Canvas(left_container, width=280, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        
        # Frame scrollable que contendrá todos los controles
        left = ttk.Frame(canvas, padding=10)
        
        # Configurar el canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Crear ventana en el canvas
        canvas_frame = canvas.create_window((0, 0), window=left, anchor="nw")
        
        # Función para actualizar el scrollregion
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        left.bind("<Configure>", configure_scroll_region)
        
        # Bind para scroll con rueda del mouse
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)  # Windows
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux
        
        # ===== CONTROLES DEL PANEL IZQUIERDO =====
        
        ttk.Label(left, text="Tamaño n (2..9):").pack(anchor="w")
        # Spinbox robusto: sin wrap, paso=1, validación 2..9 y sin callback recursivo
        def _valid_n(P):
            if P == "":
                return True
            try:
                v = int(P)
            except ValueError:
                return False
            return 2 <= v <= 9
        vcmd = (self.register(_valid_n), "%P")
        spn = ttk.Spinbox(left, from_=2, to=9, increment=1, wrap=False,
                          textvariable=self.n, width=5,
                          validate="all", validatecommand=vcmd)
        # Al confirmar con Enter o salir del campo, clamp y (si aplica) refrescar vista
        spn.bind("<Return>", lambda e: self.n.set(max(2, min(9, int(spn.get() or 3)))))
        spn.bind("<FocusOut>", lambda e: self.n.set(max(2, min(9, int(spn.get() or 3)))))
        spn.pack(anchor="w", pady=(0,8))

        ttk.Label(left, text="Vista n≤5:").pack(anchor="w")
        cmb = ttk.Combobox(left, values=["Rectas", "Polilíneas rotados", "Polilíneas incremento", "Rectas totales"], state="readonly",
                     textvariable=self.preview_kind)
        cmb.pack(anchor="w", pady=(0,6))
        cmb.bind("<<ComboboxSelected>>", lambda e: self.on_calculate())
        # Pendiente para "Rectas totales"
        ttk.Label(left, text="Pendiente (solo 'Rectas totales')").pack(anchor="w")
        self.slope_total = tk.StringVar(value="Todas")
        ttk.Combobox(left, textvariable=self.slope_total, values=["+1", "-1", "Todas"],
                     state="readonly", width=8).pack(anchor="w", pady=(0,6))


        ttk.Checkbutton(left, text="Mostrar companion (Rectas)", variable=self.show_companion, command=self.on_calculate).pack(anchor="w")

        # Seed
        seed_row = ttk.Frame(left); seed_row.pack(fill="x", pady=(8,0))
        ttk.Label(seed_row, text="Semilla (opcional):").pack(side=tk.LEFT)
        ttk.Entry(seed_row, textvariable=self.seed_str, width=12).pack(side=tk.LEFT, padx=(6,0))

        # top-k
        topk_row = ttk.Frame(left); topk_row.pack(fill="x", pady=(6,0))
        ttk.Label(topk_row, text="Top‑k global (n≤5):").pack(side=tk.LEFT)
        ttk.Spinbox(topk_row, from_=1, to=200, textvariable=self.topk, width=6).pack(side=tk.LEFT, padx=(6,0))

        ttk.Separator(left).pack(fill="x", pady=8)

        ttk.Button(left, text="Importar matriz (CSV)", command=self.on_import_csv).pack(fill="x")
        ttk.Button(left, text="Exportar matriz (CSV)", command=self.on_export_matrix_csv).pack(fill="x", pady=(4,0))
        ttk.Button(left, text="Rellenar ejemplo", command=self._load_example).pack(fill="x", pady=(4,0))
        ttk.Button(left, text="Limpiar matriz", command=self._clear_matrix).pack(fill="x", pady=(4,0))

        ttk.Separator(left).pack(fill="x", pady=8)

        ttk.Label(left, text="Orbital a previsualizar:").pack(anchor="w")
        self.orbital_spn = ttk.Spinbox(left, from_=1, to=1, textvariable=self.orbital_idx, width=7, command=self.on_calculate)
        self.orbital_spn.pack(anchor="w", pady=(0,6))

        self.orbital_spn.bind("<Return>", lambda e: self.on_calculate())
        self.orbital_spn.bind("<FocusOut>", lambda e: self.on_calculate())
        ttk.Button(left, text="Calcular", command=self.on_calculate).pack(fill="x")
        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Separator(left).pack(fill="x", pady=8)

        # Export / Clipboard / Session
        ttk.Label(left, text="Carpeta de salida:").pack(anchor="w")
        out_row = ttk.Frame(left); out_row.pack(fill="x")
        ttk.Entry(out_row, textvariable=self.outdir, width=24).pack(side=tk.LEFT, fill="x", expand=True)
        ttk.Button(out_row, text="Elegir…", command=self._choose_outdir).pack(side=tk.LEFT, padx=(6,0))

        exp_row = ttk.Frame(left); exp_row.pack(fill="x", pady=(6,0))
        ttk.Label(exp_row, text="Formato fig (n≤5):").pack(side=tk.LEFT)
        ttk.Combobox(exp_row, values=["SVG", "PDF"], width=6, state="readonly",
                     textvariable=self.export_format).pack(side=tk.LEFT, padx=(6,0))

        ttk.Button(left, text="Exportar figura (orbital actual)", command=self.on_export_current_figure).pack(fill="x", pady=(6,0))
        ttk.Button(left, text="Exportar CSV monomios (orbital)", command=self.on_export_csv_orbital).pack(fill="x", pady=(6,0))
        ttk.Button(left, text="Reporte PDF (matriz+figs)", command=self.on_export_report_pdf).pack(fill="x", pady=(6,0))
        ttk.Button(left, text="Exportar LaTeX (orbital)", command=self.on_export_tex_orbital).pack(fill="x", pady=(6,0))
        ttk.Button(left, text="Copiar resultados", command=self.on_copy_results).pack(fill="x", pady=(6,0))

        ttk.Separator(left).pack(fill="x", pady=8)

        # NUEVO: Exportación para n ≥ 6 (sin gráficas)
        ttk.Label(left, text="n ≥ 6 – Exportar orbitales:", foreground="#004080").pack(anchor="w")
        ttk.Button(left, text="Todos → Excel (.xlsx/.csv)", command=self.on_export_all_orbitals_excel).pack(fill="x", pady=(4,0))
        ttk.Button(left, text="Todos → PDF (sumario)", command=self.on_export_all_orbitals_pdf_summary).pack(fill="x", pady=(4,0))
        ttk.Button(left, text="Orbital ACTUAL → PDF (con aportes)", command=self.on_export_current_orbital_pdf_full).pack(fill="x", pady=(4,0))

        ttk.Separator(left).pack(fill="x", pady=8)
        
        # RESULTADOS / DETERMINANTES – botón que abre ventana emergente
        ttk.Button(left, text="Resultados / Determinantes…", command=self.open_results_window).pack(fill="x")

        # Sesión
        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Button(left, text="Guardar sesión", command=self.on_save_session).pack(fill="x")
        ttk.Button(left, text="Cargar sesión", command=self.on_load_session).pack(fill="x", pady=(6,0))

        # Ayuda
        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Button(left, text="Ayuda / Atajos de teclado", command=self.show_help).pack(fill="x")

        # Text oculto para mantener compatibilidad con 'Copiar resultados'
        self._hidden_results_container = ttk.Frame(left)  # No visible (no pack)
        self.result_text = tk.Text(self._hidden_results_container, width=64, height=22, font=self.result_font,
                                   wrap="none", undo=True, background="#F7FBFF")

        # Center: matrix grid (siempre arriba)
        center = ttk.Frame(self, padding=10)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(center, text="Matriz A (n×n) – siempre arriba").pack(anchor="w")
        self.matrix_frame = ttk.Frame(center)
        self.matrix_frame.pack(anchor="w", pady=(4,10))

        # Right: área de previsualización o mensaje n≥6
        self.right = ttk.Frame(center)
        self.right.pack(fill="both", expand=True)
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.no_graph_label = ttk.Label(self.right, text="")
        self.no_graph_label.pack_forget()

    def open_results_window(self):
        """Abre una ventana emergente con los resultados/determinantes actuales."""
        try:
            txt = self.result_text.get("1.0", "end-1c")
        except Exception:
            txt = ""
        # Reusar ventana si ya existe
        if hasattr(self, "_res_win") and self._res_win and self._res_win.winfo_exists():
            try:
                self._res_text.delete("1.0", "end")
                self._res_text.insert("1.0", txt)
                self._res_win.lift()
                return
            except Exception:
                pass
        win = tk.Toplevel(self)
        win.title("Resultados / Determinantes")
        win.geometry("900x600")
        frm = ttk.Frame(win)
        frm.pack(fill="both", expand=True)
        txtw = tk.Text(frm, wrap="none", font=self.result_font, background="#F7FBFF", undo=True)
        yscroll = ttk.Scrollbar(frm, orient="vertical", command=txtw.yview)
        xscroll = ttk.Scrollbar(frm, orient="horizontal", command=txtw.xview)
        txtw.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        txtw.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(0, weight=1)
        txtw.insert("1.0", txt)
        self._res_win = win
        self._res_text = txtw

    def _results_zoom_in(self):
        self._adjust_result_font(+1)
    
    def _results_zoom_out(self):
        self._adjust_result_font(-1)
    
    def _adjust_result_font(self, delta: int):
        try:
            self._result_font_size = max(8, int(self._result_font_size) + delta)
        except Exception:
            self._result_font_size = 12
        try:
            self.result_font.configure(size=self._result_font_size)
        except Exception:
            pass
    
    def bind_events(self):
        self.bind("<Left>", lambda e: self._bump_orbital(-1))
        self.bind("<Right>", lambda e: self._bump_orbital(+1))
        self.bind("<p>", lambda e: self._toggle_preview())
        self.bind("<P>", lambda e: self._toggle_preview())
        self.bind("<c>", lambda e: self.on_copy_results())
        self.bind("<C>", lambda e: self.on_copy_results())
    
    def _toggle_preview(self):
        self.preview_kind.set("Polilíneas rotados" if self.preview_kind.get()=="Rectas" else "Rectas")
        self.on_calculate()

    def _bump_orbital(self, delta: int):
        n = self.n.get(); m = math.factorial(n-1)
        new = self.orbital_idx.get() + delta
        if new < 1: new = m
        if new > m: new = 1
        self.orbital_idx.set(new)
        self.on_calculate()

    def _rebuild_matrix_grid(self):
        for w in self.matrix_frame.winfo_children():
            w.destroy()
        self.matrix_entries = []
        n = self.n.get()
        for i in range(n):
            row_entries = []
            for j in range(n):
                e = ttk.Entry(self.matrix_frame, width=7, justify="center")
                e.grid(row=i, column=j, padx=2, pady=2)
                row_entries.append(e)
            self.matrix_entries.append(row_entries)
        self._update_orbital_spin_limit()

    def _update_orbital_spin_limit(self):
        n = self.n.get()
        m = math.factorial(n-1)
        self.orbital_spn.config(to=m)
        if self.orbital_idx.get() > m:
            self.orbital_idx.set(m)

    def _load_example(self):
        
        n = int(self.n.get()) if str(self.n.get()).isdigit() else 3
        if n < 2: n = 2
        if n > 9: n = 9
        # RNG con semilla opcional
        seed_text = (self.seed_str.get() or "").strip() if hasattr(self, "seed_str") else ""
        if seed_text == "":
            rng = np.random.default_rng()
        else:
            try:
                seed_val = int(seed_text)
                rng = np.random.default_rng(seed_val)
            except Exception:
                messagebox.showerror("Semilla inválida", "La semilla debe ser un entero.")
                return
        # Generar A y reconstruir grilla
        A = rng.integers(-9, 10, size=(n, n))
        try:
            self._rebuild_matrix_grid()
        except Exception:
            pass
        # Volcar A a las entradas
        for i in range(n):
            for j in range(n):
                try:
                    e = self.matrix_entries[i][j]
                    e.delete(0, "end")
                    e.insert(0, str(int(A[i, j])))
                except Exception:
                    pass
        # Si n<=5 y existe on_calculate, refrescar
        try:
            if n <= 5:
                self.on_calculate()
        except Exception:
            pass


    def _clear_matrix(self):
        for row in self.matrix_entries:
            for e in row:
                e.delete(0, tk.END)

    def _on_n_change(self):
        self._rebuild_matrix_grid()
        self._load_example()
        self._refresh_right_area()

    def _choose_outdir(self):
        path = filedialog.askdirectory(initialdir=self.outdir.get() or ".")
        if path:
            self.outdir.set(path)

    # ----- Helpers -----

    def read_matrix(self) -> np.ndarray:
        n = self.n.get()
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                s = self.matrix_entries[i][j].get().strip()
                if not s:
                    raise ValueError("Hay celdas vacías en la matriz.")
                A[i, j] = float(s)
        return A

    def _set_preview_figure(self, fig: plt.Figure):
        if self.canvas is not None:
            try:
                self.canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self.canvas = None
        for w in self.right.winfo_children():
            w.destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _show_no_graph_message(self, n: int):
        for w in self.right.winfo_children():
            w.destroy()
        msg = (f"n = {n} ≥ 6 → Gráficas deshabilitadas.\n\n"
               "Use los botones de 'Exportar orbitales' para generar\n"
               "Excel/CSV con los monomios y sus aportes, o un PDF sumario.\n"
               "También puede exportar el ORBITAL ACTUAL a PDF con los aportes completos.")
        self.no_graph_label = ttk.Label(self.right, text=msg, justify="left")
        self.no_graph_label.pack(anchor="nw", padx=8, pady=8)

    def _refresh_right_area(self):
        n = self.n.get()
        if n <= 5:
            # placeholder de figura en blanco hasta que se calcule
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            ax.axis('off'); ax.text(0.5,0.5,"Listo para previsualizar (n≤5)", ha='center', va='center')
            self._set_preview_figure(fig)
        else:
            self._show_no_graph_message(n)

    def _diagnostics(self, A: np.ndarray) -> Tuple[str, Optional[float], Optional[int]]:
        try:
            cond = float(np.linalg.cond(A))
        except Exception:
            cond = None
        try:
            rk = int(np.linalg.matrix_rank(A))
        except Exception:
            rk = None
        notes = []
        if cond is not None:
            if not np.isfinite(cond):
                notes.append("⚠️ cond(A) = ∞ (singular)")
            elif cond > 1e12:
                notes.append(f"⚠️ cond(A) ≈ {cond:.3e} (casi singular)")
            else:
                notes.append(f"cond(A) ≈ {cond:.3e}")
        if rk is not None:
            n = A.shape[0]
            notes.append(f"rango(A) = {rk} / {n}")
            if rk < n:
                notes.append("⚠️ matriz singular o mal condicionada")
        block = "\n".join(notes) if notes else "–"
        return block, cond, rk

    def _result_block(self, A, base_idx, base):
        n = A.shape[0]
        g, _tg = _timeit(det_gauss, A)
        l, _tl = _timeit(det_lu, A)
        p, _tp = _timeit(det_polilineas, A)
        r, _tr = _timeit(self._det_orbit_expansion, A)
        bareiss = det_bareiss_exact_or_none(A)
        try:
            _t0 = time.perf_counter()
            det_np = float(np.linalg.det(A))
            _tnp = time.perf_counter() - _t0
            diff = abs(det_np - r)
            ok = diff <= 1e-8 * max(1.0, abs(r))
            chk = "✅" if ok else "⚠️"
        except Exception:
            det_np = None; chk = "–"; _tnp = 0.0
        diag_block, cond, rk = self._diagnostics(A)

        buf = io.StringIO()
        print("=== Determinantes ===", file=buf)
        print(f"Gauss:                 {g:.12f}  [{_fmt_ms(_tg)}]", file=buf)
        print(f"LU:                    {l:.12f}  [{_fmt_ms(_tl)}]", file=buf)
        print(f"Polilíneas rotados:    {p:.12f}  [{_fmt_ms(_tp)}]", file=buf)
        print(f"Rectas paralelas:      {r:.12f}  [{_fmt_ms(_tr)}]  (≡ determinante)", file=buf)
        if det_np is not None:
            print(f"numpy.linalg.det:      {det_np:.12f}  {chk}  [{_fmt_ms(_tnp)}]", file=buf)
        print("", file=buf)
        if bareiss is not None:
            print(f"Exacto (Bareiss):      {bareiss}  [{_fmt_ms(0.0)}]", file=buf)
            print("", file=buf)
        else:
            print("Exacto (Bareiss):      – (requiere entradas enteras)", file=buf)
            print("", file=buf)
        print("=== Diagnóstico ===", file=buf)
        print(diag_block, file=buf)
        print("", file=buf)

        bases = all_bases(n) if n <= 7 else None  # evitar materializar demasiadas bases
        print(f"Órbitas totales: {(math.factorial(n-1))} – mostrando orbital #{base_idx} base {base}", file=buf)


        # Aportes del orbital actual
        kind = self.preview_kind.get().strip()
        if kind == "Rectas":
            print("\nAportes por monomio – Rectas (base y companion)", file=buf)
            # Base
            contribs_b, tot_b = orbital_contributions(A, base)
            sgn0 = '+' if perm_sign(base)>0 else '−'
            print(f"Orbital base {base} (signo base {sgn0})", file=buf)
            for it in contribs_b:
                s = '+' if it["sign"]>0 else '−'
                print(f"  {s} {it['perm']}  ⇒  {it['value']:.12f}  (contrib: {it['contribution']:.12f})", file=buf)
            print(f"Suma orbital base = {tot_b:.12f}", file=buf)
            # Companion
            comp = companion_orbital(base)
            contribs_c, tot_c = orbital_contributions(A, comp)
            sgnc = '+' if perm_sign(comp)>0 else '−'
            print(f"\nOrbital companion {comp} (signo base {sgnc})", file=buf)
            for it in contribs_c:
                s = '+' if it["sign"]>0 else '−'
                print(f"  {s} {it['perm']}  ⇒  {it['value']:.12f}  (contrib: {it['contribution']:.12f})", file=buf)
            print(f"Suma orbital companion = {tot_c:.12f}", file=buf)

        elif kind == "Rectas totales":
            print("\nAportes por monomio – Rectas totales (n!)", file=buf)
            pend = getattr(self, "slope_total", tk.StringVar(value="Todas")).get()
            items = all_monomials_with_contributions(A)
            if pend == "Todas":
                items.sort(key=lambda d: -d["sign"])
            tot = 0.0
            for it in items:
                s = '+' if it["sign"]>0 else '−'
                tot += it["contribution"]
                print(f"  {s} {it['perm']}  ⇒  {it['value']:.12f}  (contrib: {it['contribution']:.12f})", file=buf)
            print(f"Suma n! = {tot:.12f}", file=buf)

        else:
            # Polilíneas rotados: solo el orbital analizado
            if kind == "Polilíneas incremento":
                contribs, tot = orbital_contributions_increment(A, base)
                tipo_str = "Polilíneas incremento"
            else:
                contribs, tot = orbital_contributions(A, base)
                tipo_str = "Polilíneas rotados"
            print(f"\nAportes por monomio – {tipo_str} (orbital analizado)", file=buf)
            sgn0 = '+' if perm_sign(base)>0 else '−'
            print(f"Orbital base {base} (signo base {sgn0})", file=buf)
            for it in contribs:
                s = '+' if it["sign"]>0 else '−'
                print(f"  {s} {it['perm']}  ⇒  {it['value']:.12f}  (contrib: {it['contribution']:.12f})", file=buf)
            print(f"Suma orbital = {tot:.12f}", file=buf)
        # Top‑k global (solo n≤5)
        if n <= 5:
            try:
                items = all_monomials_with_contributions(A)
                items.sort(key=lambda d: d["abs_contribution"], reverse=True)
                k = max(1, min(int(self.topk.get()), len(items)))
                cum = 0.0; total_abs = sum(x["abs_contribution"] for x in items) or 1.0
                print("\nTop-{} monomios por |contribución| (global)".format(k), file=buf)
                for idx in range(k):
                    it = items[idx]; cum += it["abs_contribution"]
                    s = '+' if it["sign"]>0 else '−'
                    frac = 100.0 * cum / total_abs
                    print(f"{idx+1:2d}. {s} {it['perm']}  base#{it['base_index']} rot{it['rotation']}"
                          f"  ⇒ val={it['value']:.12f}, contrib={it['contribution']:.12f}  |cum|≈{frac:5.1f}%", file=buf)
            except Exception as ex:
                print("\n[Top‑k] Error al calcular: {}".format(ex), file=buf)
        else:
            print("\nTop‑k global: desactivado para n ≥ 6 (tamaño factorial).", file=buf)

        return buf.getvalue()

    def _det_orbit_expansion(self, A: np.ndarray) -> float:
        """Determinante por expansión de orbitales (solo n≤5)."""
        n = A.shape[0]
        total = 0.0
        for base in all_bases(n):
            base_sgn = perm_sign(base)
            for r in range(n):
                cols = rotate(base, r)
                val = 1.0
                for row, col in enumerate(cols, start=1):
                    val *= A[row-1, col-1]
                s = base_sgn if (n % 2 == 1) else base_sgn * ((-1)**r)
                total += s * val
        return float(total)

    # ----- Actions -----

    def on_calculate(self):
        try:
            A = self.read_matrix()
        except Exception as e:
            messagebox.showerror("Error de entrada", str(e))
            return
        n = A.shape[0]
        m = math.factorial(n-1)
        idx = max(1, min(self.orbital_idx.get(), m))
        # Obtener base sin materializar todas cuando n grande
        base = None
        if n <= 7:
            bases = all_bases(n)
            base = bases[idx-1]
        else:
            for k, b in enumerate(all_bases_iter(n), start=1):
                if k == idx:
                    base = b; break
        if base is None:
            messagebox.showerror("Error", "No se pudo determinar la base seleccionada.")
            return

        block = self._result_block(A, idx, base)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, block)

        # Right area
        if n <= 5:
            mode = self.preview_kind.get()
            if mode == "Rectas totales":
                pend = getattr(self, "slope_total", tk.StringVar(value="+1")).get()
                fig = fig_rectas_totales_Aext(A, slope_mode=pend, show_labels=True)
            elif mode == "Rectas":
                fig = fig_rectas_orbital(A, base, show_companion=self.show_companion.get())
            elif mode == "Polilíneas incremento":
                fig = fig_polilineas_incremento_orbital(n, base)
            else:  # "Polilíneas rotados"
                fig = fig_polilineas_rotados_orbital(n, base)
            self._set_preview_figure(fig)
        else:
            self._show_no_graph_message(n)

    def on_copy_results(self):
        txt = self.result_text.get("1.0", tk.END)
        self.clipboard_clear(); self.clipboard_append(txt)
        self.update(); messagebox.showinfo("Copiado", "Resultados copiados al portapapeles.")

    def on_export_current_figure(self):
        try:
            A = self.read_matrix()
        except Exception as e:
            messagebox.showerror("Error de entrada", str(e)); return
        n = A.shape[0]
        if n > 5:
            messagebox.showinfo("Sin gráficas", "Las figuras están deshabilitadas para n ≥ 6.")
            return
        idx = self.orbital_idx.get()
        base = all_bases(n)[idx-1]
        mode = self.preview_kind.get()
        if mode == "Rectas totales":
            pend = getattr(self, "slope_total", tk.StringVar(value="+1")).get()
            fig = fig_rectas_totales_Aext(A, slope_mode=pend, show_labels=True)
        elif mode == "Rectas":
            fig = fig_rectas_orbital(A, base, show_companion=self.show_companion.get())
        elif mode == "Polilíneas incremento":
            fig = fig_polilineas_incremento_orbital(n, base)
        else:  # "Polilíneas rotados"
            fig = fig_polilineas_rotados_orbital(n, base)
        ext = ".svg" if self.export_format.get()=="SVG" else ".pdf"
        folder = self.outdir.get() or "."; os.makedirs(folder, exist_ok=True)
        fname = f"{self.preview_kind.get().lower().replace(' ', '_')}_orbital_{idx}_base_{''.join(map(str,base))}{ext}"
        path = os.path.join(folder, fname)
        fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
        messagebox.showinfo("Exportado", f"Figura guardada en:\n{os.path.abspath(path)}")

    def on_export_csv_orbital(self):
        try:
            A = self.read_matrix()
        except Exception as e:
            messagebox.showerror("Error de entrada", str(e)); return
        n = A.shape[0]
        idx = self.orbital_idx.get()
        # Base
        if n <= 7:
            base = all_bases(n)[idx-1]
        else:
            base = None
            for k, b in enumerate(all_bases_iter(n), start=1):
                if k == idx:
                    base = b; break
            if base is None:
                messagebox.showerror("Error", "No se pudo determinar la base seleccionada."); return
        contribs, total = orbital_contributions(A, base)
        folder = self.outdir.get() or "."; os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"orbital_{idx}_base_{''.join(map(str,base))}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rotation", "perm", "sign", "value", "contribution"])
            for c in contribs:
                w.writerow([c["rotation"], str(c["perm"]), c["sign"], f"{c['value']:.12f}", f"{c['contribution']:.12f}"])
            w.writerow([]); w.writerow(["sum_orbital", "", "", "", f"{total:.12f}"])
        messagebox.showinfo("Exportado", f"CSV guardado en:\n{os.path.abspath(path)}")

    def on_export_report_pdf(self):
        try:
            A = self.read_matrix()
        except Exception as e:
            messagebox.showerror("Error de entrada", str(e)); return
        n = A.shape[0]
        if n > 5:
            messagebox.showinfo("Nota", "El reporte multipágina con figuras está pensado para n ≤ 5.")
        idx = self.orbital_idx.get(); base = all_bases(n)[idx-1] if n<=7 else next(itertools.islice(all_bases_iter(n), idx-1, None))
        folder = self.outdir.get() or "."; os.makedirs(folder, exist_ok=True)
        nowtag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(folder, f"reporte_{n}x{n}_orbital_{idx}_{nowtag}.pdf")
        with PdfPages(path) as pdf:
            # Página 1: matriz + determinantes + aportes del orbital (texto)
            fig1 = plt.figure(figsize=(8.5, 11))
            fig1.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.05)
            fig1.suptitle(f"Reporte {n}×{n} – Orbital #{idx} base {base}", fontsize=14)
            ax1 = fig1.add_subplot(2,1,1); ax1.axis('off')
            # matriz
            text_mat = "\n".join("  ".join(f"{A[i,j]:.6g}" for j in range(n)) for i in range(n))
            ax1.text(0.02, 0.95, "Matriz A:", fontsize=12, va="top", family="monospace")
            ax1.text(0.02, 0.90, text_mat, fontsize=12, va="top", family="monospace")
            # resultados
            block = self._result_block(A, idx, base)
            ax2 = fig1.add_subplot(2,1,2); ax2.axis('off')
            ax2.text(0.02, 0.98, block, fontsize=10, va="top", family="monospace")
            pdf.savefig(fig1); plt.close(fig1)

            if n <= 5:
                # Página 2 y 3: figuras
                figR = fig_rectas_orbital(A, base, show_companion=self.show_companion.get())
                pdf.savefig(figR); plt.close(figR)
                figP = fig_polilineas_rotados_orbital(n, base)
                pdf.savefig(figP); plt.close(figP)
        messagebox.showinfo("Exportado", f"Reporte PDF creado en:\n{os.path.abspath(path)}")

    def on_export_tex_orbital(self):
        try:
            A = self.read_matrix()
        except Exception as e:
            messagebox.showerror("Error de entrada", str(e)); return
        n = A.shape[0]
        idx = self.orbital_idx.get()
        if n <= 7:
            base = all_bases(n)[idx-1]
        else:
            base = next(itertools.islice(all_bases_iter(n), idx-1, None))
        contribs, total = orbital_contributions(A, base)
        folder = self.outdir.get() or "."; os.makedirs(folder, exist_ok=True)

        # Guardar figuras solo cuando n≤5
        fR = f"rectas_orbital_{idx}_base_{''.join(map(str,base))}.pdf"
        fP = f"polilineas_rotados_orbital_{idx}_base_{''.join(map(str,base))}.pdf"
        if n <= 5:
            figR = fig_rectas_orbital(A, base, show_companion=self.show_companion.get())
            figP = fig_polilineas_rotados_orbital(n, base)
            plt.figure(figR.number); plt.savefig(os.path.join(folder, fR), dpi=300, bbox_inches="tight"); plt.close(figR)
            plt.figure(figP.number); plt.savefig(os.path.join(folder, fP), dpi=300, bbox_inches="tight"); plt.close(figP)

        bareiss = det_bareiss_exact_or_none(A)

        def tex_escape(s: str) -> str:
            return s.replace('_','\\_').replace('%','\\%')

        try:
            _t0 = time.perf_counter()
            det_np = float(np.linalg.det(A))
            _tnp = time.perf_counter() - _t0
        except Exception:
            det_np = None

        header = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb, booktabs, geometry, graphicx}
\geometry{margin=1in}
\begin{document}
"""
        title = f"\\section*{{Orbital {idx} base {tex_escape(str(base))} – Matriz {n}\\times{n}}}\n"
        mat_lines = ""
        for i in range(n):
            row = " & ".join(f"{A[i,j]:.6g}" for j in range(n))
            mat_lines += row + r" \\" + "\n"
        mat_tex = "\\[\nA = \\begin{bmatrix}\n" + mat_lines + "\\end{bmatrix}\n\\]\n"
        det_lines = "\\begin{tabular}{@{}ll@{}}\\toprule\n"
        det_lines += f"Gauss: & {det_gauss(A):.12f}\\\\\n"
        det_lines += f"LU: & {det_lu(A):.12f}\\\\\n"
        if n <= 5:
            d_orb = self._det_orbit_expansion(A)
            det_lines += f"Polilíneas rotados/Rectas: & {d_orb:.12f}\\\\\n"
        if det_bareiss_exact_or_none(A) is not None:
            det_lines += f"Exacto (Bareiss): & {bareiss}\\\\\n"
        if det_np is not None:
            det_lines += f"$\\det(A)$ (NumPy): & {det_np:.12f}\\\\\n"
        det_lines += "\\bottomrule\\end{tabular}\n\n"

        tab = "\\begin{tabular}{@{}r l r r r@{}}\\toprule\n"
        tab += "rot & $\\sigma$ & signo & monomio & contrib.\\\\\\midrule\n"
        for it in contribs:
            sgn = "+" if it["sign"]>0 else "-"
            tab += f"{it['rotation']} & {tex_escape(str(it['perm']))} & {sgn} & {it['value']:.12f} & {it['contribution']:.12f}\\\\\n"
        tab += "\\midrule\n"
        tab += f"\\multicolumn{{4}}{{r}}{{Suma orbital}} & {total:.12f}\\\\\n"
        tab += "\\bottomrule\\end{tabular}\n\n"

        figs = ""
        if n <= 5:
            figs = ("\\begin{figure}[h]\n\\centering\n"
                    f"\\includegraphics[width=.92\\textwidth]{{{tex_escape(fR)}}}\n"
                    "\\caption{Rectas paralelas – orbital actual}\n\\end{figure}\n\n"
                    "\\begin{figure}[h]\n\\centering\n"
                    f"\\includegraphics[width=.7\\textwidth]{{{tex_escape(fP)}}}\n"
                    "\\caption{Polilíneas rotados – orbital actual}\n\\end{figure}\n\n")

        footer = "\\end{document}\n"
        tex_path = os.path.join(folder, f"orbital_{idx}_base_{''.join(map(str,base))}.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(header + title + mat_tex + det_lines + tab + figs + footer)
        messagebox.showinfo("Exportado", f".tex guardado en:\n{os.path.abspath(tex_path)}")


    # ===== NUEVO: Exportar TODOS los orbitales (n≥6) =====
    def on_export_all_orbitals_excel(self):
        try:
            A = self.read_matrix()
        except Exception as e:
            messagebox.showerror("Error de entrada", str(e)); return
        n = A.shape[0]
        folder = self.outdir.get() or "."; os.makedirs(folder, exist_ok=True)
        base_fname = f"orbitales_{n}x{n}_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if XLSX_OK:
            path = os.path.join(folder, base_fname + ".xlsx")
            wb = xlsxwriter.Workbook(path)
            ws = wb.add_worksheet("orbitales")
            header = ["base_index","base","rotation","perm","sign","value","contribution"]
            for j,h in enumerate(header): ws.write(0, j, h)
            row = 1
            # Escribimos por streaming
            for b_idx, base in enumerate(all_bases_iter(n), start=1):
                contribs, total = orbital_contributions(A, base)
                for c in contribs:
                    ws.write_row(row, 0, [b_idx, str(base), c["rotation"], str(c["perm"]), c["sign"], c["value"], c["contribution"]])
                    row += 1
                # Línea de suma por base
                ws.write_row(row, 0, [b_idx, str(base), "sum", "", "", "", total]); row += 1
            wb.close()
            messagebox.showinfo("Exportado", f"Excel creado en:\n{os.path.abspath(path)}")
        else:
            path = os.path.join(folder, base_fname + ".csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["base_index","base","rotation","perm","sign","value","contribution"])
                for b_idx, base in enumerate(all_bases_iter(n), start=1):
                    contribs, total = orbital_contributions(A, base)
                    for c in contribs:
                        w.writerow([b_idx, str(base), c["rotation"], str(c["perm"]), c["sign"], f"{c['value']:.12f}", f"{c['contribution']:.12f}"])
                    w.writerow([b_idx, str(base), "sum", "", "", "", f"{total:.12f}"])
            messagebox.showinfo("Exportado", f"CSV creado en:\n{os.path.abspath(path)}\n(Excel no disponible; instale 'xlsxwriter' para .xlsx)")

    def on_export_all_orbitals_pdf_summary(self):
        try:
            A = self.read_matrix()
        except Exception as e:
            messagebox.showerror("Error de entrada", str(e)); return
        n = A.shape[0]
        folder = self.outdir.get() or "."; os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"orbitales_{n}x{n}_sumario.pdf")
        with PdfPages(path) as pdf:
            # Portada
            fig0 = plt.figure(figsize=(8.5, 11))
            ax0 = fig0.add_subplot(111); ax0.axis('off')
            title = f"Sumario de órbitas – n={n} (sin monomios)"
            ax0.text(0.5, 0.92, title, ha='center', va='center', fontsize=16)
            text_mat = "\n".join("  ".join(f"{A[i,j]:.6g}" for j in range(n)) for i in range(n))
            ax0.text(0.06, 0.85, "Matriz A:", fontsize=12, family="monospace")
            ax0.text(0.06, 0.80, text_mat, fontsize=11, family="monospace")
            pdf.savefig(fig0); plt.close(fig0)

            # Sumario: varias bases por página
            lines_per_page = 46
            buf = []
            for b_idx, base in enumerate(all_bases_iter(n), start=1):
                _, total = orbital_contributions(A, base)
                buf.append(f"{b_idx:>6}  base {base}  →  suma = {total:.12f}")
                if len(buf) >= lines_per_page:
                    fig = plt.figure(figsize=(8.5, 11)); ax = fig.add_subplot(111); ax.axis('off')
                    ax.text(0.06, 0.98, "\n".join(buf), va='top', family='monospace', fontsize=9)
                    pdf.savefig(fig); plt.close(fig); buf = []
            if buf:
                fig = plt.figure(figsize=(8.5, 11)); ax = fig.add_subplot(111); ax.axis('off')
                ax.text(0.06, 0.98, "\n".join(buf), va='top', family='monospace', fontsize=9)
                pdf.savefig(fig); plt.close(fig)
        messagebox.showinfo("Exportado", f"PDF (sumario) creado en:\n{os.path.abspath(path)}\nPara ver aportes de cada monomio use la exportación a Excel/CSV o el PDF del orbital actual.")

    def on_export_current_orbital_pdf_full(self):
        try:
            A = self.read_matrix()
        except Exception as e:
            messagebox.showerror("Error de entrada", str(e)); return
        n = A.shape[0]
        idx = self.orbital_idx.get()
        base = all_bases(n)[idx-1] if n<=7 else next(itertools.islice(all_bases_iter(n), idx-1, None))
        contribs, total = orbital_contributions(A, base)
        folder = self.outdir.get() or "."; os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"orbital_{idx}_base_{''.join(map(str,base))}_full.pdf")
        with PdfPages(path) as pdf:
            fig = plt.figure(figsize=(8.5, 11)); fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.06)
            ax = fig.add_subplot(111); ax.axis('off')
            head = [f"Orbital #{idx} base {base} – n={n}", "", "Matriz A:"]
            mat = "\n".join("  ".join(f"{A[i,j]:.6g}" for j in range(n)) for i in range(n))
            lines = head + [mat, "", "Aportes (monomios):"]
            for it in contribs:
                s = '+' if it["sign"]>0 else '−'
                lines.append(f"  {s} {it['perm']}  ⇒ val={it['value']:.12f}  contrib={it['contribution']:.12f}")
            lines.append(""); lines.append(f"Suma orbital = {total:.12f}")
            ax.text(0.06, 0.98, "\n".join(lines), va='top', family='monospace', fontsize=9)
            pdf.savefig(fig); plt.close(fig)
        messagebox.showinfo("Exportado", f"PDF (orbital actual) creado en:\n{os.path.abspath(path)}")

    # ----- Import/Export matriz & sesión -----

    def on_import_csv(self):
        fp = filedialog.askopenfilename(title="Importar matriz CSV", filetypes=[("CSV","*.csv"),("Todos","*.*")])
        if not fp: return
        try:
            data = []
            with open(fp, "r", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if not row: continue
                    data.append([float(x) for x in row])
            A = np.array(data, dtype=float)
            if A.shape[0] != A.shape[1] or not (2 <= A.shape[0] <= 9):
                raise ValueError("El CSV debe ser una matriz cuadrada de tamaño 2..9.")
        except Exception as e:
            messagebox.showerror("Error al importar", str(e)); return
        self.n.set(A.shape[0]); self._on_n_change()
        n = A.shape[0]
        for i in range(n):
            for j in range(n):
                self.matrix_entries[i][j].delete(0, tk.END)
                self.matrix_entries[i][j].insert(0, str(A[i, j]))
        messagebox.showinfo("Importado", f"Matriz {A.shape[0]}×{A.shape[1]} cargada.")

    def on_export_matrix_csv(self):
        try:
            A = self.read_matrix()
        except Exception as e:
            messagebox.showerror("Error de entrada", str(e)); return
        fp = filedialog.asksaveasfilename(title="Exportar matriz CSV", defaultextension=".csv",
                                          filetypes=[("CSV","*.csv")])
        if not fp: return
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for i in range(A.shape[0]):
                w.writerow([A[i,j] for j in range(A.shape[1])])
        messagebox.showinfo("Exportado", f"Matriz guardada en:\n{os.path.abspath(fp)}")

    def on_save_session(self):
        try:
            A = self.read_matrix()
        except Exception as e:
            messagebox.showerror("Error de entrada", str(e)); return
        session = {
            "n": int(self.n.get()),
            "preview_kind": self.preview_kind.get(),
            "orbital_idx": int(self.orbital_idx.get()),
            "show_companion": bool(self.show_companion.get()),
            "A": A.tolist(),
            "seed": self.seed_str.get(),
            "topk": int(self.topk.get()),
            "versions": {
                "python": sys.version.split()[0],
                "numpy": getattr(np, "__version__", "?"),
                "matplotlib": getattr(matplotlib, "__version__", "?"),
                "platform": platform.platform(),
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }
        fp = filedialog.asksaveasfilename(title="Guardar sesión", defaultextension=".json",
                                          filetypes=[("JSON","*.json")])
        if not fp: return
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)
        messagebox.showinfo("Guardado", f"Sesión guardada en:\n{os.path.abspath(fp)}")

    def on_load_session(self):
        fp = filedialog.askopenfilename(title="Cargar sesión", filetypes=[("JSON","*.json"),("Todos","*.*")])
        if not fp: return
        try:
            with open(fp, "r", encoding="utf-8") as f:
                s = json.load(f)
            A = np.array(s["A"], dtype=float)
            n = A.shape[0]
            if n<2 or n>9 or n!=s.get("n", n):
                raise ValueError("Sesión inválida (tamaño n fuera de 2..9).")
            self.n.set(n); self._on_n_change()
            for i in range(n):
                for j in range(n):
                    self.matrix_entries[i][j].delete(0, tk.END)
                    self.matrix_entries[i][j].insert(0, str(A[i, j]))
            self.preview_kind.set(s.get("preview_kind","Rectas"))
            self.orbital_idx.set(int(s.get("orbital_idx",1)))
            self.show_companion.set(bool(s.get("show_companion", True)))
            self.seed_str.set(s.get("seed",""))
            self.topk.set(int(s.get("topk", 10)))
            messagebox.showinfo("Cargado", "Sesión restaurada.")
        except Exception as e:
            messagebox.showerror("Error al cargar", str(e))

    def show_help(self):
        """Muestra ventana de ayuda con atajos de teclado e instrucciones."""
        help_text = """AYUDA - DETERMINANTE POR POLILÍNEAS Y RECTAS

╔══════════════════════════════════════════════╗
ATAJOS DE TECLADO
╚══════════════════════════════════════════════╝
← / →     Navegar entre orbitales (anterior/siguiente)
P         Alternar entre vista Rectas/Polilíneas rotados
C         Copiar resultados al portapapeles

╔══════════════════════════════════════════════╗
USO BÁSICO
╚══════════════════════════════════════════════╝
1. Seleccione el tamaño n (2-9)
2. Ingrese valores en la matriz o use "Rellenar ejemplo"
3. Presione "Calcular" para ver resultados
4. Use "Resultados / Determinantes..." para ver detalles

╔══════════════════════════════════════════════╗
TAMAÑOS DE MATRIZ
╚══════════════════════════════════════════════╝
• n ≤ 5: Gráficas disponibles (Rectas/Polilíneas rotados)
• n ≥ 6: Gráficas deshabilitadas por rendimiento
         Use exportación a Excel/CSV/PDF

╔══════════════════════════════════════════════╗
EXPORTACIÓN
╚══════════════════════════════════════════════╝
• Figuras: SVG o PDF (solo n≤5)
• Datos: CSV (orbital actual)
• Reporte: PDF completo con matriz y gráficas
• Orbitales: Excel/CSV con todos los monomios (n≥6)

╔══════════════════════════════════════════════╗
SEMILLA
╚══════════════════════════════════════════════╝
Use el campo "Semilla" para generar matrices
aleatorias reproducibles (ej: 123, 456, etc.)

╔══════════════════════════════════════════════╗
VERSIÓN
╚══════════════════════════════════════════════╝
Sarrus GUI v12 - Extensión de Regla de Sarrus
Método de Polilíneas rotados y Rectas Paralelas
"""
        win = tk.Toplevel(self)
        win.title("Ayuda")
        win.geometry("600x650")
        
        frm = ttk.Frame(win, padding=10)
        frm.pack(fill="both", expand=True)
        
        txt = tk.Text(frm, wrap="word", font=("Courier New", 10), padx=10, pady=10)
        scroll = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=scroll.set)
        
        txt.pack(side=tk.LEFT, fill="both", expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        txt.insert("1.0", help_text)
        txt.configure(state="disabled")
        
        btn_frame = ttk.Frame(win, padding=10)
        btn_frame.pack(fill="x")
        ttk.Button(btn_frame, text="Cerrar", command=win.destroy).pack()


# =========================
# Main
# =========================

if __name__ == "__main__":
    app = App()
    app.mainloop()