"""
Shared style module for all PAID-FD TMC figures.

Usage:
    from _tmc_style import apply_tmc_style, TMC_COLORS, TMC_MARKERS
    apply_tmc_style()
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    ax.plot(x, y, color=TMC_COLORS['paid_fd'])
"""

import matplotlib.pyplot as plt

# ── Figure size constants ────────────────────────────────────────────────────
SINGLE_COL      = (3.5, 2.5)    # IEEE single column
DOUBLE_COL      = (7.0, 2.5)    # IEEE double column wide
DOUBLE_COL_TALL = (7.0, 5.0)    # for 2×2 layout


# ── Color palette ────────────────────────────────────────────────────────────
TMC_COLORS = {
    # PAID-FD main
    'paid_fd':       '#1f77b4',   # main blue — PAID-FD identity

    # Same-pipeline baselines: Fair Fixed-ε series (light→dark blue gradient)
    'fair_eps_1':    '#6baed6',   # lighter
    'fair_eps_3':    '#4292c6',   # medium
    'fair_eps_5':    '#2171b5',   # darker

    # Different-pipeline LDP baselines (red/orange family — visually distinct)
    'old_fixed':     '#d62728',   # brick red — Old Fixed-ε
    'csra':          '#7f7f7f',   # grey — broken baseline

    # No-privacy baselines (green/purple family)
    'fedgmkd':       '#9467bd',   # purple
    'fedavg':        '#2ca02c',   # green
    'fedmd':         '#8c564b',   # brown

    # Dimensional colours
    'cost_red':      '#d62728',   # cost / privacy-exposure dimension
    'payment_green': '#2ca02c',   # payment dimension
    'neutral':       '#525252',   # neutral grey

    # Gradient blues for γ sweep (light → dark, 4 levels)
    'gradient_blue': ['#c6dbef', '#6baed6', '#2171b5', '#08306b'],
}


# ── Marker styles ────────────────────────────────────────────────────────────
TMC_MARKERS = {
    'paid_fd':   's',   # square
    'fair_eps':  'o',   # circle
    'old_fixed': 'x',   # x
    'csra':      'v',   # triangle down
    'fedgmkd':   'D',   # diamond
    'fedavg':    '+',   # plus
    'fedmd':     '*',   # star
}


# ── rcParams ─────────────────────────────────────────────────────────────────
def apply_tmc_style():
    """Apply unified TMC rcParams to matplotlib."""
    plt.rcParams.update({
        # Font
        'font.family':            'sans-serif',
        'font.sans-serif':        ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':              8,
        'axes.labelsize':         9,
        'axes.titlesize':         9,
        'xtick.labelsize':        7,
        'ytick.labelsize':        7,
        'legend.fontsize':        7,

        # Save / display
        'figure.dpi':             150,
        'savefig.dpi':            300,
        'savefig.bbox':           'tight',
        'savefig.format':         'pdf',
        'pdf.fonttype':           42,   # TrueType — avoids LaTeX font issues

        # Grid
        'axes.grid':              True,
        'axes.grid.axis':         'y',
        'grid.alpha':             0.3,
        'grid.linestyle':         '-',
        'grid.color':             '#CCCCCC',

        # Spines
        'axes.spines.top':        False,
        'axes.spines.right':      False,
    })


# ── Subplot label helper ─────────────────────────────────────────────────────
def subplot_label(ax, text, loc='upper-left'):
    """
    Place a bold panel label (e.g. "(a)") inside the axes.

    Parameters
    ----------
    ax   : matplotlib Axes
    text : str  — e.g. "(a)"
    loc  : str  — only 'upper-left' supported for now
    """
    x, y = 0.05, 0.95
    ax.text(x, y, text,
            transform=ax.transAxes,
            fontsize=9, fontweight='bold',
            va='top', ha='left')
