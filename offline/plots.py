"""Графики траектории (matplotlib, без экрана): X/Y/Z от кадра, вид сверху, ошибка по кадрам, число камер."""
from __future__ import annotations

from pathlib import Path

import numpy as np

C_EST = "#2a78d6"  # наша оценка — синий
C_TRUTH = "#eb6834"  # эталон — оранжевый
C_GRID = "#d9d8d3"
MODE_COLORS = {"tri": "#1baf7a", "init": "#1baf7a", "mono": "#eda100", "mono_size_prior": "#e87ba4", "predict": "#9a9a95", "none": "#ffffff"}


def _style(ax):
    ax.grid(True, color=C_GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _mode_band(ax, modes):
    """Полоса режима под графиком: зелёный — >=2 камер, жёлтый — одна, серый — прогноз."""
    if modes is None:
        return
    k0 = 0
    for k in range(1, len(modes) + 1):
        if k == len(modes) or modes[k] != modes[k0]:
            ax.axvspan(k0 - 0.5, k - 0.5, ymin=0, ymax=0.03, color=MODE_COLORS.get(modes[k0], "#cccccc"), lw=0)
            k0 = k


def make_plots(out_dir: Path, est_ue: np.ndarray, truth_ue: np.ndarray | None, modes=None, n_cams=None,
               sigma_m=None, title: str = "") -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    paths = []
    E = np.asarray(est_ue) / 100.0
    T = None if truth_ue is None else np.asarray(truth_ue)[: len(E)] / 100.0
    fr = np.arange(len(E))

    # 1. X/Y/Z от кадра
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for i, (ax, lab) in enumerate(zip(axs, ("X", "Y", "Z"))):
        if T is not None:
            ax.plot(fr[: len(T)], T[:, i], color=C_TRUTH, lw=2, label="эталон")
        ax.plot(fr, E[:, i], color=C_EST, lw=1.5, label="оценка")
        ax.set_ylabel(f"{lab}, м (Unreal)")
        _style(ax)
        _mode_band(ax, modes)
    axs[0].legend(loc="best", frameon=False)
    axs[-1].set_xlabel("кадр   (полоса внизу: зелёный — ≥2 камер, жёлтый — 1 камера, серый — прогноз)")
    fig.suptitle(f"Координаты по кадрам {title}")
    fig.tight_layout()
    p = out_dir / "plot_xyz.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # 2. Вид сверху
    fig, ax = plt.subplots(figsize=(9, 9))
    if T is not None:
        ax.plot(T[:, 0], T[:, 1], color=C_TRUTH, lw=2.5, label="эталон")
    ax.plot(E[:, 0], E[:, 1], color=C_EST, lw=1.5, label="оценка")
    if modes is not None:
        m = np.asarray(modes)
        for md, mk in (("mono", "o"), ("mono_size_prior", "o"), ("predict", "x")):
            s = m == md
            if s.any():
                ax.scatter(E[s, 0], E[s, 1], s=6, marker=mk, color=MODE_COLORS[md], label=f"режим {md}", zorder=3)
    ok = np.isfinite(E[:, 0])
    if ok.any():
        k = np.where(ok)[0]
        for j in range(k[0], k[-1] + 1, 100):
            if ok[j]:
                ax.annotate(str(j), E[j, :2], fontsize=8, color="#555")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X, м (Unreal)")
    ax.set_ylabel("Y, м (Unreal)")
    ax.invert_yaxis()  # Unreal — левая система: при X вправо ось Y направлена «вниз» на виде сверху
    _style(ax)
    ax.legend(loc="best", frameon=False)
    ax.set_title(f"Вид сверху {title}")
    fig.tight_layout()
    p = out_dir / "plot_top.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    # 3. Ошибка по кадрам + число камер
    if T is not None:
        err = np.linalg.norm(E[: len(T)] - T, axis=1)
        fig, axs = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax = axs[0]
        ax.plot(fr[: len(err)], err, color=C_EST, lw=1.2, label="ошибка 3D")
        if sigma_m is not None:
            ax.plot(fr, np.minimum(np.asarray(sigma_m, dtype=float), 1e3), color="#9a9a95", lw=1, ls="--", label="σ фильтра")
        ax.set_ylabel("м")
        fin = err[np.isfinite(err)]
        if len(fin):
            ax.set_ylim(0, max(5.0, float(np.percentile(fin, 99)) * 1.2))
        _style(ax)
        _mode_band(ax, modes)
        ax.legend(loc="upper left", frameon=False)
        ax.set_title(f"Ошибка 3D по кадрам {title}")
        if n_cams is not None:
            axs[1].step(fr, n_cams, where="mid", color="#555", lw=1)
            axs[1].set_ylabel("камер")
            axs[1].set_yticks(range(0, int(max(n_cams) if len(n_cams) else 1) + 1))
            _style(axs[1])
        axs[1].set_xlabel("кадр")
        fig.tight_layout()
        p = out_dir / "plot_error.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    return paths
