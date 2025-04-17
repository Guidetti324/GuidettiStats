# --------- START OF CODE BLOCK ------------
###############################################################################
#  PSYC‑250  –  Statistical Tables Explorer  (Streamlit, 12 × 4‑inch figures)
#  ---------------------------------------------------------------------------
#  Seven complete tabs:
#      1) t‑Distribution              5) Mann‑Whitney U
#      2) z‑Distribution              6) Wilcoxon Signed‑Rank T
#      3) F‑Distribution              7) Binomial
#      4) Chi‑Square
#
#  Features in every tab
#  ----------------------
#  • 12 × 4 Matplotlib plot (no user‑resizing)
#  • Animated step‑table: "Show Steps" button highlights
#      Row → Column → Intersection sequentially (~0.9s each).
#  • Complete APA‑7 interpretation block under each table.
#  • Scrolling containers for tables to prevent layout overflow.
#
#  This file is self‑contained. Copy it directly to app.py then run:
#      streamlit run app.py
#
#  Corrected Version: 2025-04-17
###############################################################################

import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import re # Added for style_row helper

plt.switch_backend("Agg")                      # head‑less backend

# ───────────────────────────────  helpers  ──────────────────────────────────

def place_label(ax, placed, x, y, txt, *, color="blue"):
    """Place text, pushing up slightly if needed based on existing labels."""
    # Simple vertical offset to avoid overlap
    offset = 0.02 * len(placed) * (ax.get_ylim()[1] - ax.get_ylim()[0]) # Scale offset by plot height
    # Ensure y + offset doesn't exceed plot limits significantly
    plot_top = ax.get_ylim()[1]
    final_y = min(y + offset, plot_top * 0.98) # Cap label position

    ax.text(x, final_y, txt, color=color,
            ha="center", va="bottom", fontsize=8, clip_on=True,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7)) # Add background box
    placed.append((x, final_y))

# ---------------------------------------------------------------------------

def style_cell(html: str, cid: str, *, color: str = "darkorchid", px: int = 3) -> str:
    """Give one <td id="cid"> a coloured border and background."""
    # Use regex to safely replace the style attribute or add one if missing
    pattern = f'(<td id="{cid}")( style="[^"]*")?'
    replacement = f'\\1 style="border:{px}px solid {color}; background-color: #e6e6fa; font-weight: bold;"' # Lavender background
    html, count = re.subn(pattern, replacement, html, count=1)
    # if count == 0: print(f"Warning: Cell ID {cid} not found for styling.") # Debugging
    return html

# ---------------------------------------------------------------------------

def style_row(html: str, row_prefix: str, *, color: str = "tomato", px: int = 2) -> str:
    """Highlight an entire row by styling its cells with background and borders."""
    # Regex to find <td> elements with IDs starting with the prefix
    pattern = re.compile(f'(<td id="({row_prefix}_\\d+)")( style="[^"]*")?')

    def add_row_style(match):
        tag_start = match.group(1) # e.g., '<td id="t_55_1"'
        existing_style = match.group(3) if match.group(3) else ' style=""'
        # Construct new style attribute, preserving existing if needed (though overwrite likely intended)
        new_style = f' style="border-top:{px}px solid {color}; border-bottom:{px}px solid {color}; background-color: #fff0f0;"' # Light red background
        return tag_start + new_style

    html = pattern.sub(add_row_style, html)

    # Specifically style the row header cell (index 0) with left border as well
    header_id = f"{row_prefix}_0"
    header_pattern = re.compile(f'(<td id="{header_id}")( style="[^"]*")?')
    def add_header_style(match):
        tag_start = match.group(1)
        new_style = f' style="border-top:{px}px solid {color}; border-bottom:{px}px solid {color}; border-left:{px}px solid {color}; background-color: #fff0f0; font-weight: bold;"'
        return tag_start + new_style
    html = header_pattern.sub(add_header_style, html, count=1)

    return html

# ---------------------------------------------------------------------------

def style_col(html: str, col_idx: int, rows: list, prefix: str, *, color: str = "cornflowerblue", px: int = 2) -> str:
    """Highlight an entire column by styling its cells with background and borders."""
    for r in rows:
        # Format row value correctly for ID generation (handle floats for z, ints otherwise)
        row_val_str = f"{r:.1f}" if prefix == "z" else str(int(r)) if isinstance(r, (int, float)) and r == int(r) else str(r)
        cid = f"{prefix}_{row_val_str}_{col_idx}"
        # Use regex to find the specific id attribute and replace/add its style
        pattern = f'(<td id="{cid}")( style="[^"]*")?'
        replacement = f'\\1 style="border-left:{px}px solid {color}; border-right:{px}px solid {color}; background-color: #f0f8ff;"' # AliceBlue background

        html, count = re.subn(pattern, replacement, html, count=1)
        # if count == 0: print(f"Warning: Cell ID {cid} not found for column styling.") # Debugging

    # Attempt to style the column header - requires knowing the table structure (complex)
    # Finding the correct <th> based only on col_idx is difficult. Skipping for robustness.

    return html

# ---------------------------------------------------------------------------

def wrap_table(css: str, table: str) -> str:
    # Ensure the input 'table' string already contains <table>...</table> tags
    return f"<style>{css}</style>{table}"

# ---------------------------------------------------------------------------

def container(html: str, *, height: int = 460) -> str:
    """Scrollable wrapper — does not steal the scroll wheel elsewhere."""
    # Add margin-bottom to avoid cutting off borders
    return f'<div style="overflow:auto; max-height:{height}px; border: 1px solid #eee; margin-bottom: 10px; padding: 5px;">{html}</div>' # Added border and padding

# ---------------------------------------------------------------------------

def animate(build_html, frames: int, *, key: str, height: int = 460,
            delay: float = 0.9): # Increased delay slightly
    """
    Display an HTML table animation with sequential highlighting.
      * build_html(step:int) -> html string (should include <table> tags)
      * frames: number of steps (should be 3 for Row -> Col -> Cell)
    """
    # Store the generated HTML outside the button check if needed later
    # html_cache = {}

    if st.button("Show Steps", key=key):
        holder = st.empty()
        final_html = "" # Store final state
        for s in range(frames):
            try:
                html_content = build_html(s)
                # html_cache[s] = html_content # Cache if needed
                # Wrap the generated HTML table in the scrollable container
                styled_html_with_container = container(html_content, height=height)
                holder.markdown(styled_html_with_container, unsafe_allow_html=True)
                if s == frames - 1:
                    final_html = html_content # Keep the last generated HTML
            except Exception as e:
                holder.error(f"Error during animation step {s}: {e}")
                break # Stop animation on error
            time.sleep(delay)
        # Keep the final frame visible if animation completed
        if final_html:
             styled_html_with_container = container(final_html, height=height)
             holder.markdown(styled_html_with_container, unsafe_allow_html=True)
        # st.success("Animation complete!") # Optionally show success message
    # else:
        # Optionally display the initial state (step -1 or 0) if not animating?
        # initial_html = build_html(-1) # Assuming build_html handles step -1 for initial state
        # styled_html_with_container = container(initial_html, height=height)
        # st.markdown(styled_html_with_container, unsafe_allow_html=True)
        # pass # Do nothing if button not pressed

# ---------------------------------------------------------------------------

CSS_BASE = (
    "table {border-collapse: collapse; margin: 5px 0; width: auto; border: 1px solid #ddd;}" # Less margin, auto width, table border
    "th, td {border: 1px solid #ccc; height: 26px; padding: 3px 6px; text-align: center;" # Smaller height/padding
    "font-family: monospace; font-size: 0.8rem; min-width: 45px; box-sizing: border-box;}" # Smaller font/min-width, border-box
    "th {background: #f0f0f0; font-weight: bold; position: sticky; top: 0; z-index: 1;}" # Sticky header
    "td:first-child {font-weight: bold; background: #f8f8f8; position: sticky; left: 0; z-index: 0;}" # Sticky first column
    "thead th { border-bottom: 2px solid #ccc; }" # Heavier border under headers
    "tbody td:first-child { border-right: 2px solid #ccc; }" # Heavier border after row header
    "tr:hover td {background-color: #f5f5f5 !important;}" # Row hover effect (use !important to override potential inline styles)
    "caption { caption-side: top; text-align: left; padding: 4px; font-size: 0.85rem; font-weight: bold; color: #333;}" # Add caption style
)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 • t‑Distribution
# ════════════════════════════════════════════════════════════════════════════

# ....................................... plot
def plot_t(t_calc, df, alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    try:
        t_calc = float(t_calc) # Ensure float
        df = int(df)
        alpha = float(alpha)
    except (ValueError, TypeError):
        ax.text(0.5, 0.5, "Invalid input for plot.", ha='center', va='center', fontsize=10, color='red')
        return fig


    if df <= 0: # Handle invalid df
        ax.text(0.5, 0.5, "df must be > 0", ha='center', va='center', fontsize=10, color='red')
        return fig

    max_lim = max(4, abs(t_calc) * 1.2, stats.t.ppf(1 - 0.001, df) * 1.1)

    xs = np.linspace(-max_lim, max_lim, 500)
    ys = stats.t.pdf(xs, df)

    ax.plot(xs, ys, "k", linewidth=1)
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.35, label="Fail to Reject $H_0$")

    labels = [] # To manage label placement

    if tail == "one-tailed":
        crit = stats.t.ppf(1 - alpha, df)
        ax.fill_between(xs[xs >= crit], ys[xs >= crit],
                        color="red", alpha=0.40, label="Reject $H_0$")
        ax.axvline(crit, color="green", ls="--", linewidth=1)
        place_label(ax, labels, crit, stats.t.pdf(crit, df),
                    f"$t_{{crit}} = {crit:.3f}$", color="green")
        # Add alpha area label
        ax.text(crit + 0.1*max_lim, ax.get_ylim()[1]*0.1, f"$\\alpha = {alpha:.3f}$",
                color="red", ha="left", va="bottom", fontsize=9)
    else: # two-tailed
        crit = stats.t.ppf(1 - alpha/2, df)
        ax.fill_between(xs[xs >= crit], ys[xs >= crit], color="red", alpha=0.40)
        ax.fill_between(xs[xs <= -crit], ys[xs <= -crit], color="red", alpha=0.40,
                        label="Reject $H_0$")
        ax.axvline( crit, color="green", ls="--", linewidth=1)
        ax.axvline(-crit, color="green", ls="--", linewidth=1)
        place_label(ax, labels,  crit, stats.t.pdf( crit, df),
                    f"$+t_{{crit}} = {crit:.3f}$", color="green")
        place_label(ax, labels, -crit, stats.t.pdf(-crit, df),
                    f"$-t_{{crit}} = {crit:.3f}$", color="green")
        # Add alpha area labels
        ax.text(crit + 0.1*max_lim, ax.get_ylim()[1]*0.1, f"$\\alpha/2 = {alpha/2:.3f}$",
                color="red", ha="left", va="bottom", fontsize=9)
        ax.text(-crit - 0.1*max_lim, ax.get_ylim()[1]*0.1, f"$\\alpha/2 = {alpha/2:.3f}$",
                color="red", ha="right", va="bottom", fontsize=9)

    ax.axvline(t_calc, color="blue", ls="-", linewidth=1.5) # Solid line for calculated
    place_label(ax, labels, t_calc, stats.t.pdf(t_calc, df),
                f"$t_{{calc}} = {t_calc:.3f}$", color="blue")

    ax.set_xlabel("$t$ value")
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=9)
    ax.set_title(f"$t$-Distribution ($df={df}$), {tail}", fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.set_xlim(-max_lim, max_lim)
    fig.tight_layout()
    return fig

# ....................................... animated table builder
def build_t_html(df: int, alpha: float, tail: str, step: int) -> str:
    try:
        df = int(df)
        alpha = float(alpha)
        if df < 1: raise ValueError("df must be >= 1")
    except (ValueError, TypeError):
        return "<table><caption>Invalid Input</caption><tr><td>Invalid input for t-table generation.</td></tr></table>"


    row_range = list(range(max(1, df - 5), df + 6))
    # Common alpha levels for t-tables
    alphas_one = [0.10, 0.05, 0.025, 0.01, 0.005]
    alphas_two = [0.20, 0.10, 0.05, 0.02, 0.01]

    # Find the column index for the given alpha and tail
    target_alpha = alpha
    col_idx = -1
    col_alpha_list = []
    warning_message = None
    if tail == "one-tailed":
        col_alpha_list = alphas_one
        effective_alpha_for_ppf = target_alpha
        offset = 0
    else: # two-tailed
        col_alpha_list = alphas_two
        effective_alpha_for_ppf = target_alpha / 2 # Use alpha/2 for ppf lookup
        offset = len(alphas_one) # Offset index by number of one-tailed columns

    # Find closest alpha if exact not present
    if target_alpha not in col_alpha_list:
        closest_alpha = min(col_alpha_list, key=lambda x:abs(x-alpha))
        warning_message = f"Alpha {alpha:.4f} not standard for {tail} t-table, using closest: {closest_alpha:.3f}"
        target_alpha = closest_alpha
        # Recalculate effective alpha for ppf if two-tailed
        if tail == "two-tailed": effective_alpha_for_ppf = target_alpha / 2

    col_idx = col_alpha_list.index(target_alpha) + 1 + offset # +1 for df column

    # Find row index within display range
    try:
        row_idx_display = row_range.index(df)
    except ValueError:
         return f"<table><caption>Error</caption><tr><td>df={df} not in display range {row_range}.</td></tr></table>"


    # Build Header
    header = '<thead>' # Use thead for sticky headers
    header += '<tr><th rowspan="2" style="position: sticky; left: 0; z-index: 2;">df</th>' # Make df header sticky too
    header += f'<th colspan="{len(alphas_one)}">One-Tailed $\\alpha$</th>'
    header += f'<th colspan="{len(alphas_two)}">Two-Tailed $\\alpha$</th></tr>'
    header += '<tr>' + "".join(f"<th>{a:.3f}</th>" for a in alphas_one)
    header += "".join(f"<th>{a:.3f}</th>" for a in alphas_two) + '</tr></thead>'

    # Build Body
    body = "<tbody>" # Use tbody
    all_cols = alphas_one + alphas_two
    for r in row_range:
        row_html = f'<tr><td id="t_{r}_0">{r}</td>' # Row header ID includes value
        # One-tailed values
        for i, a in enumerate(alphas_one):
            try: crit_val = stats.t.ppf(1 - a, r)
            except ValueError: crit_val = np.nan # Handle potential errors for extreme low df/alpha
            row_html += f'<td id="t_{r}_{i+1}">{crit_val:.3f}</td>'
        # Two-tailed values
        for i, a in enumerate(alphas_two):
            try: crit_val = stats.t.ppf(1 - a/2, r) # Use a/2 for lookup
            except ValueError: crit_val = np.nan
            row_html += f'<td id="t_{r}_{len(alphas_one) + i + 1}">{crit_val:.3f}</td>'
        body += row_html + "</tr>"
    body += "</tbody>"

    # Caption
    caption = f"<caption>t-Distribution Critical Values (df={df}, {tail}, \\(\\alpha={alpha:.3f}\\))</caption>"

    # Base table structure
    base_html_table = f"<table>{caption}{header}{body}</table>"

    # Apply sequential highlighting
    row_prefix = f"t_{df}"
    target_cell_id = f"t_{df}_{col_idx}"

    # Create fresh table for each step to avoid style accumulation
    step_html_table = f"<table>{caption}{header}{body}</table>" # Include caption

    if step == 0: # Highlight Row
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=2)
    elif step == 1: # Highlight Column
        step_html_table = style_col(step_html_table, col_idx, row_range, prefix="t", color="cornflowerblue", px=2)
    elif step == 2: # Highlight Cell
        # Apply row and column styles first for context, then cell
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=1) # Lighter border
        step_html_table = style_col(step_html_table, col_idx, row_range, prefix="t", color="cornflowerblue", px=1) # Lighter border
        step_html_table = style_cell(step_html_table, target_cell_id, color="darkorchid", px=3) # Strong border

    # Add warning if alpha was adjusted
    warning_html = f'<p style="color: orange; font-size: 0.8em;">⚠️ {warning_message}</p>' if warning_message else ""

    # Return the styled table wrapped in CSS for this step
    return warning_html + wrap_table(CSS_BASE, step_html_table)


# ....................................... table wrapper
def t_table(df: int, alpha: float, tail: str):
    animate(lambda s: build_t_html(df, alpha, tail, s),
            frames=3, key=f"t_anim_{df}_{alpha}_{tail}")

# ....................................... APA narrative
def t_apa(t_val: float, df: int, alpha: float, tail: str):
    try:
        t_val = float(t_val)
        df = int(df)
        alpha = float(alpha)
        if df < 1: raise ValueError("df must be >= 1")
        if not (0 < alpha < 1): raise ValueError("alpha must be between 0 and 1")
    except (ValueError, TypeError):
        st.error("Invalid input values for APA interpretation.")
        return

    p_crit = alpha # The critical p-value is alpha

    # Calculate p-value and critical value based on tail type
    if tail == "one-tailed":
        # Calculate p-value for the OBSERVED direction:
        if t_val >= 0: # Assumes upper tail test
            p_calc = stats.t.sf(t_val, df)
            crit = stats.t.ppf(1 - alpha, df)
            reject_stat = t_val > crit
            comparison_stat = f"{t_val:.3f} > {crit:.3f}" if reject_stat else f"{t_val:.3f} \\le {crit:.3f}$"
            crit_str = f"$t_{{crit}}({df}) = {crit:.3f}$ (upper tail)"
        else: # t_val < 0, assumes lower tail test
            p_calc = stats.t.cdf(t_val, df)
            crit = stats.t.ppf(alpha, df) # Lower critical value
            reject_stat = t_val < crit
            comparison_stat = f"{t_val:.3f} < {crit:.3f}" if reject_stat else f"{t_val:.3f} \\ge {crit:.3f}$"
            crit_str = f"$t_{{crit}}({df}) = {crit:.3f}$ (lower tail)"

        reject_p = p_calc < alpha
        comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"

    else: # two-tailed
        p_calc = stats.t.sf(abs(t_val), df) * 2
        crit = stats.t.ppf(1 - alpha / 2, df)
        reject_stat = abs(t_val) > crit
        reject_p = p_calc < alpha
        comparison_stat = f"$|{t_val:.3f}| > {crit:.3f}$" if reject_stat else f"$|{t_val:.3f}| \\le {crit:.3f}$"
        comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"
        crit_str = f"$t_{{crit}}({df}) = \\pm{crit:.3f}$"


    decision = "rejected" if reject_stat else "failed to reject"
    # Use p-value comparison for final text
    decision_final = "rejected" if reject_p else "failed to reject"

    st.markdown(f"""
    ##### APA-7 Interpretation
    * **Calculated statistic:** $t({df}) = {t_val:.3f}$
    * **Calculated *p*-value:** $p = {p_calc:.3f}$ {'(one-tailed)' if tail == 'one-tailed' else '(two-tailed)'}
    * **Critical value(s) ({tail}, $\\alpha={alpha:.3f}$):** {crit_str}
    * **Critical *p*-value (alpha):** $p_{{crit}} = {alpha:.3f}$

    * **Decision based on statistic vs. critical value:** $H_0$ is **{decision}** because {comparison_stat}.
    * **Decision based on calculated *p*-value vs. alpha:** $H_0$ is **{decision_final}** because {comparison_p}.

    * **APA-7 Sentence:** A {tail} t-test revealed that the effect was {'statistically significant' if reject_p else 'not statistically significant'}, $t({df}) = {t_val:.3f}$, $p = {p_calc:.3f}$. The null hypothesis was **{decision_final}** at the $\\alpha = {alpha:.3f}$ level.
    """, unsafe_allow_html=True) # Allow LaTeX rendering

# ....................................... tab assembly
def tab_t():
    st.subheader("Tab 1 • $t$-Distribution")
    c1, c2 = st.columns(2)
    with c1:
        # Using text input for flexibility, converting later
        t_val_str = st.text_input("$t$ statistic ($t_{calc}$)", value="2.87", key="t_val_str")
        df_str = st.text_input("Degrees of freedom ($df$)", value="55", key="t_df_str")
    with c2:
        alpha_str = st.text_input("Significance level ($\\alpha$)", value="0.05", key="t_alpha_str")
        tail = st.radio("Tail(s)", ["one-tailed", "two-tailed"], index=1, key="t_tail", horizontal=True)

    # Validate inputs
    t_val, df, alpha = None, None, None
    valid_inputs = False
    try:
        t_val = float(t_val_str)
        df = int(df_str)
        alpha = float(alpha_str)
        if not (0 < alpha < 1): raise ValueError("Alpha must be between 0 and 1")
        if df < 1: raise ValueError("df must be at least 1")
        valid_inputs = True
    except ValueError as e:
        st.error(f"⚠️ Invalid input: {e}. Please enter valid numbers.")
        # Show placeholder plot on error
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "Invalid input. Cannot draw plot.", ha='center', va='center', color='red')
        ax.set_title("$t$-Distribution")
        st.pyplot(fig)

    if valid_inputs:
        st.pyplot(plot_t(t_val, df, alpha, tail))

        with st.expander("Show $t$-table lookup steps & Interpretation"):
            t_table(df, alpha, tail)
            # Add APA interpretation inside the expander, after the table animation logic
            st.markdown("---") # Separator
            t_apa(t_val, df, alpha, tail)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 • z‑Distribution
# ════════════════════════════════════════════════════════════════════════════

# ....................................... plot
def plot_z(z_calc, alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    try:
        z_calc = float(z_calc)
        alpha = float(alpha)
    except (ValueError, TypeError):
        ax.text(0.5, 0.5, "Invalid input for plot.", ha='center', va='center', fontsize=10, color='red')
        return fig

    max_lim = max(4, abs(z_calc) * 1.2)
    xs = np.linspace(-max_lim, max_lim, 500)
    ys = stats.norm.pdf(xs)

    ax.plot(xs, ys, "k", linewidth=1)
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.35, label="Fail to Reject $H_0$")

    labels = [] # To manage label placement

    if tail == "one-tailed":
        # Assuming upper tail for positive z, lower for negative z for plotting crit val
        if z_calc >= 0:
            crit = stats.norm.ppf(1 - alpha)
            ax.fill_between(xs[xs >= crit], ys[xs >= crit],
                            color="red", alpha=0.40, label="Reject $H_0$")
            ax.axvline(crit, color="green", ls="--", linewidth=1)
            place_label(ax, labels, crit, stats.norm.pdf(crit),
                        f"$z_{{crit}} = {crit:.3f}$", color="green")
            ax.text(crit + 0.1*max_lim, ax.get_ylim()[1]*0.1, f"$\\alpha = {alpha:.3f}$",
                    color="red", ha="left", va="bottom", fontsize=9)
        else: # z_calc < 0
            crit = stats.norm.ppf(alpha)
            ax.fill_between(xs[xs <= crit], ys[xs <= crit],
                            color="red", alpha=0.40, label="Reject $H_0$")
            ax.axvline(crit, color="green", ls="--", linewidth=1)
            place_label(ax, labels, crit, stats.norm.pdf(crit),
                        f"$z_{{crit}} = {crit:.3f}$", color="green")
            ax.text(crit - 0.1*max_lim, ax.get_ylim()[1]*0.1, f"$\\alpha = {alpha:.3f}$",
                    color="red", ha="right", va="bottom", fontsize=9)

    else: # two-tailed
        crit = stats.norm.ppf(1 - alpha/2)
        ax.fill_between(xs[xs >= crit], ys[xs >= crit], color="red", alpha=0.40)
        ax.fill_between(xs[xs <= -crit], ys[xs <= -crit], color="red", alpha=0.40,
                        label="Reject $H_0$")
        ax.axvline( crit, color="green", ls="--", linewidth=1)
        ax.axvline(-crit, color="green", ls="--", linewidth=1)
        place_label(ax, labels,  crit, stats.norm.pdf( crit),
                    f"$+z_{{crit}} = {crit:.3f}$", color="green")
        place_label(ax, labels, -crit, stats.norm.pdf(-crit),
                    f"$-z_{{crit}} = {crit:.3f}$", color="green")
        ax.text(crit + 0.1*max_lim, ax.get_ylim()[1]*0.1, f"$\\alpha/2 = {alpha/2:.3f}$",
                color="red", ha="left", va="bottom", fontsize=9)
        ax.text(-crit - 0.1*max_lim, ax.get_ylim()[1]*0.1, f"$\\alpha/2 = {alpha/2:.3f}$",
                color="red", ha="right", va="bottom", fontsize=9)

    ax.axvline(z_calc, color="blue", ls="-", linewidth=1.5)
    place_label(ax, labels, z_calc, stats.norm.pdf(z_calc),
                f"$z_{{calc}} = {z_calc:.3f}$", color="blue")

    ax.set_xlabel("$z$ score")
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=9)
    ax.set_title(f"Standard Normal ($z$) Distribution, {tail}", fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.set_xlim(-max_lim, max_lim)
    fig.tight_layout()
    return fig

# ....................................... animated table builder (z-table: Area to the LEFT)
def build_z_html(z: float, step: int) -> str:
    try:
        z = float(z)
    except (ValueError, TypeError):
        return "<table><caption>Invalid Input</caption><tr><td>Invalid z-value for table.</td></tr></table>"

    # Clamp z to typical table range for lookup
    z_lookup = np.clip(z, -3.499, 3.499) # Use small epsilon to handle edges like 3.49
    target_row_val = np.floor(z_lookup * 10) / 10 # e.g., -1.96 -> -2.0, 1.64 -> 1.6, 3.49 -> 3.4
    target_row_val = np.round(target_row_val, 1) # Ensure correct precision

    target_col_offset = np.round(z_lookup - target_row_val, 2) # e.g., -1.96 - (-2.0) = 0.04
    # Ensure column offset is within the expected range [0.00, 0.09]
    target_col_offset = np.clip(target_col_offset, 0.00, 0.09)
    target_col_offset = np.round(target_col_offset, 2)


    # Define the full range of rows and columns for a standard z-table
    all_rows = np.round(np.arange(-3.4, 3.5, 0.1), 1)
    all_cols_offsets = np.round(np.arange(0.00, 0.10, 0.01), 2)

    # Find the index of the target row
    warning_message = None
    try:
        row_idx_arr = np.where(np.isclose(all_rows, target_row_val))[0]
        if len(row_idx_arr) == 0: raise IndexError
        row_idx = row_idx_arr[0]
    except IndexError:
        row_idx = np.argmin(np.abs(all_rows - target_row_val))
        target_row_val = all_rows[row_idx]
        warning_message = f"Adjusted target row for z-table lookup to: {target_row_val:.1f}"

    # Select ±10 rows around the target row index
    start_idx = max(0, row_idx - 10)
    end_idx = min(len(all_rows), row_idx + 11) # +11 because slice excludes end
    display_rows = all_rows[start_idx:end_idx]

    # Find the column index based on the offset
    try:
        col_idx_arr = np.where(np.isclose(all_cols_offsets, target_col_offset))[0]
        if len(col_idx_arr) == 0: raise IndexError
        col_idx = col_idx_arr[0] + 1 # +1 for the z-score column header offset
    except IndexError:
        col_idx = np.argmin(np.abs(all_cols_offsets - target_col_offset)) + 1
        target_col_offset = all_cols_offsets[col_idx - 1] # Adjust target if needed
        warning_message = (warning_message + "; " if warning_message else "") + f"Adjusted target column offset to: {target_col_offset:.2f}"

    # Build Header
    header = '<thead><tr><th style="position: sticky; left: 0; z-index: 2;">z</th>'
    header += "".join(f"<th>{c:.2f}</th>" for c in all_cols_offsets) + '</tr></thead>'

    # Build Body
    body = "<tbody>"
    for r in display_rows:
        row_html = f'<tr><td id="z_{r:.1f}_0">{r:.1f}</td>' # Row header ID includes value
        for i, c_offset in enumerate(all_cols_offsets):
            cell_val = stats.norm.cdf(r + c_offset)
            row_html += f'<td id="z_{r:.1f}_{i+1}">{cell_val:.4f}</td>' # Cell ID includes row value and col index
        body += row_html + "</tr>"
    body += "</tbody>"

    # Caption
    caption = f"<caption>Standard Normal (z) Table - Area to the Left (Lookup for z={z:.2f})</caption>"

    # Base table structure
    base_html_table = f"<table>{caption}{header}{body}</table>"

    # Apply sequential highlighting
    row_prefix = f"z_{target_row_val:.1f}" # Prefix includes the numeric value, formatted
    target_cell_id = f"z_{target_row_val:.1f}_{col_idx}"

    # Create fresh table for each step
    step_html_table = f"<table>{caption}{header}{body}</table>" # Include caption

    if step == 0: # Highlight Row
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=2)
    elif step == 1: # Highlight Column
        step_html_table = style_col(step_html_table, col_idx, display_rows, prefix="z", color="cornflowerblue", px=2)
    elif step == 2: # Highlight Cell
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=1)
        step_html_table = style_col(step_html_table, col_idx, display_rows, prefix="z", color="cornflowerblue", px=1)
        step_html_table = style_cell(step_html_table, target_cell_id, color="darkorchid", px=3)

    # Add warning if adjusted
    warning_html = f'<p style="color: orange; font-size: 0.8em;">⚠️ {warning_message}</p>' if warning_message else ""

    return warning_html + wrap_table(CSS_BASE, step_html_table)


# ....................................... table wrapper
def z_table(z_val: float):
    # Note: Alpha and tail are not needed for standard z-table lookup itself, only for interpretation
    animate(lambda s: build_z_html(z_val, s),
            frames=3, key=f"z_anim_{z_val}")

# ....................................... APA narrative
def z_apa(z_val: float, alpha: float, tail: str):
    try:
        z_val = float(z_val)
        alpha = float(alpha)
        if not (0 < alpha < 1): raise ValueError("alpha must be between 0 and 1")
    except (ValueError, TypeError):
        st.error("Invalid input values for APA interpretation.")
        return

    p_crit = alpha

    # Calculate p-value and critical value based on tail type
    if tail == "one-tailed":
        # Determine direction based on z_val and calculate p-value accordingly
        if z_val >= 0: # Assumes upper tail H1 (z > 0)
            p_calc = stats.norm.sf(z_val)
            crit = stats.norm.ppf(1 - alpha)
            reject_stat = z_val > crit
            comparison_stat = f"{z_val:.3f} > {crit:.3f}" if reject_stat else f"{z_val:.3f} \\le {crit:.3f}$"
            crit_str = f"$z_{{crit}} = {crit:.3f}$ (upper tail)"
        else: # z_val < 0, assumes lower tail H1 (z < 0)
            p_calc = stats.norm.cdf(z_val)
            crit = stats.norm.ppf(alpha)
            reject_stat = z_val < crit
            comparison_stat = f"{z_val:.3f} < {crit:.3f}" if reject_stat else f"{z_val:.3f} \\ge {crit:.3f}$"
            crit_str = f"$z_{{crit}} = {crit:.3f}$ (lower tail)"

        reject_p = p_calc < alpha
        comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"
    else: # two-tailed
        p_calc = stats.norm.sf(abs(z_val)) * 2
        crit = stats.norm.ppf(1 - alpha / 2)
        reject_stat = abs(z_val) > crit
        reject_p = p_calc < alpha
        comparison_stat = f"$|{z_val:.3f}| > {crit:.3f}$" if reject_stat else f"$|{z_val:.3f}| \\le {crit:.3f}$"
        comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"
        crit_str = f"$z_{{crit}} = \\pm{crit:.3f}$"

    decision = "rejected" if reject_stat else "failed to reject"
    decision_final = "rejected" if reject_p else "failed to reject"

    st.markdown(f"""
    ##### APA-7 Interpretation
    * **Calculated statistic:** $z = {z_val:.3f}$
    * **Calculated *p*-value:** $p = {p_calc:.3f}$ {'(one-tailed)' if tail == 'one-tailed' else '(two-tailed)'}
    * **Critical value(s) ({tail}, $\\alpha={alpha:.3f}$):** {crit_str}
    * **Critical *p*-value (alpha):** $p_{{crit}} = {alpha:.3f}$

    * **Decision based on statistic vs. critical value:** $H_0$ is **{decision}** because {comparison_stat}.
    * **Decision based on calculated *p*-value vs. alpha:** $H_0$ is **{decision_final}** because {comparison_p}.

    * **APA-7 Sentence:** A {tail} z-test indicated that the result was {'statistically significant' if reject_p else 'not statistically significant'}, $z = {z_val:.3f}$, $p = {p_calc:.3f}$. The null hypothesis was **{decision_final}** at the $\\alpha = {alpha:.3f}$ level.
    """, unsafe_allow_html=True)

# ....................................... tab assembly
def tab_z():
    st.subheader("Tab 2 • $z$-Distribution (Standard Normal)")
    c1, c2 = st.columns(2)
    with c1:
        z_val_str = st.text_input("$z$ statistic ($z_{calc}$)", value="1.96", key="z_val_str")
    with c2:
        alpha_str = st.text_input("Significance level ($\\alpha$)", value="0.05", key="z_alpha_str")
        tail = st.radio("Tail(s)", ["one-tailed", "two-tailed"], index=1, key="z_tail", horizontal=True)

    # Validate inputs
    z_val, alpha = None, None
    valid_inputs = False
    try:
        z_val = float(z_val_str)
        alpha = float(alpha_str)
        if not (0 < alpha < 1): raise ValueError("Alpha must be between 0 and 1")
        valid_inputs = True
    except ValueError as e:
        st.error(f"⚠️ Invalid input: {e}. Please enter valid numbers.")
        # Placeholder plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "Invalid input. Cannot draw plot.", ha='center', va='center', color='red')
        ax.set_title("Standard Normal ($z$) Distribution")
        st.pyplot(fig)


    if valid_inputs:
        st.pyplot(plot_z(z_val, alpha, tail))

        with st.expander("Show $z$-table lookup steps & Interpretation (Area to the Left)"):
            z_table(z_val)
            st.markdown("---") # Separator
            z_apa(z_val, alpha, tail)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 • F‑Distribution
# ════════════════════════════════════════════════════════════════════════════

# ....................................... plot
def plot_f(f_calc, df1, df2, alpha):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    try:
        f_calc = float(f_calc)
        df1 = int(df1)
        df2 = int(df2)
        alpha = float(alpha)
        if df1 < 1 or df2 < 1: raise ValueError("df must be >= 1")
        if f_calc < 0: raise ValueError("F cannot be negative")
    except (ValueError, TypeError):
        ax.text(0.5, 0.5, "Invalid input for plot.", ha='center', va='center', fontsize=10, color='red')
        return fig

    crit = stats.f.ppf(1 - alpha, df1, df2)
    # Adjust x-axis limit dynamically
    plot_max = max(crit * 1.5, f_calc * 1.2, stats.f.ppf(0.999, df1, df2))
    plot_max = max(plot_max, 5) # Ensure at least range up to 5
    xs = np.linspace(0.001, plot_max, 500) # Start slightly > 0 for pdf
    ys = stats.f.pdf(xs, df1, df2)

    ax.plot(xs, ys, "k", linewidth=1)
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.35, label="Fail to Reject $H_0$")

    ax.fill_between(xs[xs >= crit], ys[xs >= crit],
                    color="red", alpha=0.40, label="Reject $H_0$")

    labels = []
    ax.axvline(crit, color="green", ls="--", linewidth=1)
    place_label(ax, labels, crit, stats.f.pdf(crit, df1, df2),
                f"$F_{{crit}} = {crit:.3f}$", color="green")

    ax.axvline(f_calc, color="blue", ls="-", linewidth=1.5)
    # Adjust y-position for label if pdf is near zero
    y_pos_calc = stats.f.pdf(f_calc, df1, df2)
    if y_pos_calc < ax.get_ylim()[1] * 0.01: y_pos_calc = ax.get_ylim()[1] * 0.05
    place_label(ax, labels, f_calc, y_pos_calc,
                f"$F_{{calc}} = {f_calc:.3f}$", color="blue")

    ax.text(crit + 0.1*(plot_max-crit), ax.get_ylim()[1]*0.1, f"$\\alpha = {alpha:.3f}$",
                color="red", ha="left", va="bottom", fontsize=9)

    ax.set_xlabel("$F$ value")
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=9)
    ax.set_title(f"$F$-Distribution ($df_1={df1}, df_2={df2}$)", fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=plot_max)
    fig.tight_layout()
    return fig

# ....................................... animated table builder
def build_f_html(df1: int, df2: int, alpha: float, step: int) -> str:
    try:
        df1 = int(df1)
        df2 = int(df2)
        alpha = float(alpha)
        if df1 < 1 or df2 < 1: raise ValueError("df must be >= 1")
    except (ValueError, TypeError):
        return "<table><caption>Invalid Input</caption><tr><td>Invalid input for F-table generation.</td></tr></table>"

    # Define ranges around the target df values
    row_range = list(range(max(1, df1 - 5), df1 + 6)) # Rows are df1
    col_range = list(range(max(1, df2 - 5), df2 + 6)) # Columns are df2

    # Find indices for highlighting (relative to displayed rows/cols)
    try: row_val_idx = row_range.index(df1)
    except ValueError: return f"<table><caption>Error</caption><tr><td>df1={df1} not in display range {row_range}.</td></tr></table>"
    try: col_val_idx = col_range.index(df2)
    except ValueError: return f"<table><caption>Error</caption><tr><td>df2={df2} not in display range {col_range}.</td></tr></table>"

    # The actual column index in the HTML table includes the row header column
    col_idx_html = col_val_idx + 1

    # Build Header
    header = f'<thead><tr><th style="position: sticky; left: 0; z-index: 2;">$df_1 \\setminus df_2$</th>'
    header += "".join(f"<th>{c}</th>" for c in col_range) + '</tr></thead>'

    # Build Body
    body = "<tbody>"
    for r in row_range:
        row_html = f'<tr><td id="f_{r}_0">{r}</td>' # Row header ID includes df1 value
        for i, c in enumerate(col_range):
            try: crit_val = stats.f.ppf(1 - alpha, r, c)
            except ValueError: crit_val = np.nan
            row_html += f'<td id="f_{r}_{i+1}">{crit_val:.3f}</td>' # Cell ID includes df1 and col index (1-based)
        body += row_html + "</tr>"
    body += "</tbody>"

    # Caption
    caption = f"<caption>F-Distribution Critical Values (Upper Tail, df1={df1}, df2={df2}, \\(\\alpha={alpha:.3f}\\))</caption>"

    # Base table structure
    base_html_table = f"<table>{caption}{header}{body}</table>"

    # Apply sequential highlighting
    row_prefix = f"f_{df1}" # Row prefix uses df1 value
    target_cell_id = f"f_{df1}_{col_idx_html}"

    # Create fresh table for each step
    step_html_table = f"<table>{caption}{header}{body}</table>" # Include caption

    if step == 0: # Highlight Row (df1)
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=2)
    elif step == 1: # Highlight Column (df2)
        step_html_table = style_col(step_html_table, col_idx_html, row_range, prefix="f", color="cornflowerblue", px=2)
    elif step == 2: # Highlight Cell
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=1)
        step_html_table = style_col(step_html_table, col_idx_html, row_range, prefix="f", color="cornflowerblue", px=1)
        step_html_table = style_cell(step_html_table, target_cell_id, color="darkorchid", px=3)

    return wrap_table(CSS_BASE, step_html_table)


# ....................................... table wrapper
def f_table(df1: int, df2: int, alpha: float):
    animate(lambda s: build_f_html(df1, df2, alpha, s),
            frames=3, key=f"f_anim_{df1}_{df2}_{alpha}")

# ....................................... APA narrative
def f_apa(f_val: float, df1: int, df2: int, alpha: float):
    try:
        f_val = float(f_val)
        df1 = int(df1)
        df2 = int(df2)
        alpha = float(alpha)
        if df1 < 1 or df2 < 1: raise ValueError("df must be >= 1")
        if not (0 < alpha < 1): raise ValueError("alpha must be between 0 and 1")
        if f_val < 0: raise ValueError("F cannot be negative")
    except (ValueError, TypeError):
        st.error("Invalid input values for APA interpretation.")
        return

    p_crit = alpha

    # F-test is typically one-tailed (upper tail)
    p_calc = stats.f.sf(f_val, df1, df2)
    crit = stats.f.ppf(1 - alpha, df1, df2)
    reject_stat = f_val > crit
    reject_p = p_calc < alpha
    comparison_stat = f"{f_val:.3f} > {crit:.3f}" if reject_stat else f"{f_val:.3f} \\le {crit:.3f}$"
    comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"
    crit_str = f"$F_{{crit}}({df1}, {df2}) = {crit:.3f}$"

    decision = "rejected" if reject_stat else "failed to reject"
    decision_final = "rejected" if reject_p else "failed to reject"

    st.markdown(f"""
    ##### APA-7 Interpretation
    * **Calculated statistic:** $F({df1}, {df2}) = {f_val:.3f}$
    * **Calculated *p*-value:** $p = {p_calc:.3f}$
    * **Critical value ($\\alpha={alpha:.3f}$):** {crit_str}
    * **Critical *p*-value (alpha):** $p_{{crit}} = {alpha:.3f}$

    * **Decision based on statistic vs. critical value:** $H_0$ is **{decision}** because {comparison_stat}.
    * **Decision based on calculated *p*-value vs. alpha:** $H_0$ is **{decision_final}** because {comparison_p}.

    * **APA-7 Sentence:** The results indicated that the finding was {'statistically significant' if reject_p else 'not statistically significant'}, $F({df1}, {df2}) = {f_val:.3f}$, $p = {p_calc:.3f}$. The null hypothesis was **{decision_final}** at the $\\alpha = {alpha:.3f}$ significance level.
    """, unsafe_allow_html=True)

# ....................................... tab assembly
def tab_f():
    st.subheader("Tab 3 • $F$-Distribution")
    c1, c2 = st.columns(2)
    with c1:
        f_val_str = st.text_input("$F$ statistic ($F_{calc}$)", value="4.32", key="f_val_str")
        df1_str = st.text_input("$df_1$ (numerator)", value="5", key="f_df1_str")
    with c2:
        df2_str = st.text_input("$df_2$ (denominator)", value="20", key="f_df2_str")
        alpha_str = st.text_input("Significance level ($\\alpha$)", value="0.05", key="f_alpha_str")

    # Validate inputs
    f_val, df1, df2, alpha = None, None, None, None
    valid_inputs = False
    try:
        f_val = float(f_val_str)
        df1 = int(df1_str)
        df2 = int(df2_str)
        alpha = float(alpha_str)
        if not (0 < alpha < 1): raise ValueError("Alpha must be between 0 and 1")
        if df1 < 1 or df2 < 1: raise ValueError("Degrees of freedom must be at least 1")
        if f_val < 0: raise ValueError("F cannot be negative")
        valid_inputs = True
    except ValueError as e:
        st.error(f"⚠️ Invalid input: {e}. Please enter valid numbers.")
        # Placeholder
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "Invalid input. Cannot draw plot.", ha='center', va='center', color='red')
        ax.set_title("$F$-Distribution")
        st.pyplot(fig)


    if valid_inputs:
        st.pyplot(plot_f(f_val, df1, df2, alpha))

        with st.expander("Show $F$-table lookup steps & Interpretation"):
            f_table(df1, df2, alpha)
            st.markdown("---") # Separator
            f_apa(f_val, df1, df2, alpha)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 • Chi‑Square (χ²)
# ════════════════════════════════════════════════════════════════════════════

# ....................................... plot
def plot_chi(chi_calc, df, alpha):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    try:
        chi_calc = float(chi_calc)
        df = int(df)
        alpha = float(alpha)
        if df < 1: raise ValueError("df must be >= 1")
        if chi_calc < 0: raise ValueError("Chi-square cannot be negative")
    except (ValueError, TypeError):
        ax.text(0.5, 0.5, "Invalid input for plot.", ha='center', va='center', fontsize=10, color='red')
        return fig


    crit = stats.chi2.ppf(1 - alpha, df)
    # Adjust x-axis limit dynamically
    plot_max = max(crit * 1.5, chi_calc * 1.2, stats.chi2.ppf(0.999, df))
    plot_max = max(plot_max, 5) # Ensure range at least to 5
    xs = np.linspace(0.001, plot_max, 500) # Start slightly > 0
    ys = stats.chi2.pdf(xs, df)

    ax.plot(xs, ys, "k", linewidth=1)
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.35, label="Fail to Reject $H_0$")

    ax.fill_between(xs[xs >= crit], ys[xs >= crit],
                    color="red", alpha=0.40, label="Reject $H_0$")

    labels = []
    ax.axvline(crit, color="green", ls="--", linewidth=1)
    place_label(ax, labels, crit, stats.chi2.pdf(crit, df),
                f"$\\chi^2_{{crit}} = {crit:.3f}$", color="green")

    ax.axvline(chi_calc, color="blue", ls="-", linewidth=1.5)
    y_pos_calc = stats.chi2.pdf(chi_calc, df)
    if y_pos_calc < ax.get_ylim()[1] * 0.01: y_pos_calc = ax.get_ylim()[1] * 0.05
    place_label(ax, labels, chi_calc, y_pos_calc,
                f"$\\chi^2_{{calc}} = {chi_calc:.3f}$", color="blue")

    ax.text(crit + 0.1*(plot_max-crit), ax.get_ylim()[1]*0.1, f"$\\alpha = {alpha:.3f}$",
                color="red", ha="left", va="bottom", fontsize=9)

    ax.set_xlabel("$\\chi^2$ value")
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=9)
    ax.set_title(f"$\\chi^2$-Distribution ($df={df}$)", fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=plot_max)
    fig.tight_layout()
    return fig

# ....................................... animated table builder
def build_chi_html(df: int, alpha: float, step: int) -> str:
    try:
        df = int(df)
        alpha = float(alpha)
        if df < 1: raise ValueError("df must be >= 1")
    except (ValueError, TypeError):
        return "<table><caption>Invalid Input</caption><tr><td>Invalid input for Chi-square table generation.</td></tr></table>"

    # Standard alpha levels for upper tail critical values
    upper_tail_alphas = [0.10, 0.05, 0.025, 0.01, 0.005] # Common choices
    warning_message = None

    # Find the column index for the given alpha
    target_alpha = alpha # This alpha represents the area in the upper tail
    col_idx = -1
    if target_alpha in upper_tail_alphas:
         col_idx = upper_tail_alphas.index(target_alpha) + 1 # +1 for df column
    else:
        # Find closest standard alpha if exact not present
        closest_alpha = min(upper_tail_alphas, key=lambda x:abs(x-alpha))
        col_idx = upper_tail_alphas.index(closest_alpha) + 1
        warning_message = f"Alpha {alpha:.4f} not standard for Chi-Square table, using closest upper tail value: {closest_alpha:.3f}"
        target_alpha = closest_alpha # Use the closest alpha for lookup

    # Define df range around the target df
    row_range = list(range(max(1, df - 5), df + 6))

    # Find row index
    try:
        row_idx_display = row_range.index(df)
    except ValueError:
         return f"<table><caption>Error</caption><tr><td>df={df} not in display range {row_range}.</td></tr></table>"


    # Build Header
    header = f'<thead><tr><th style="position: sticky; left: 0; z-index: 2;">df \\ $\\alpha$</th>'
    header += "".join(f"<th>{a:.3f}</th>" for a in upper_tail_alphas) + '</tr></thead>'

    # Build Body
    body = "<tbody>"
    for r in row_range:
        row_html = f'<tr><td id="chi_{r}_0">{r}</td>'
        for i, a in enumerate(upper_tail_alphas):
            try: crit_val = stats.chi2.ppf(1 - a, r)
            except ValueError: crit_val = np.nan
            row_html += f'<td id="chi_{r}_{i+1}">{crit_val:.3f}</td>'
        body += row_html + "</tr>"
    body += "</tbody>"

    # Caption
    caption = f"<caption>Chi-Square Critical Values (Upper Tail, df={df}, \\(\\alpha={alpha:.3f}\\))</caption>"

    # Base table structure
    base_html_table = f"<table>{caption}{header}{body}</table>"

    # Apply sequential highlighting
    row_prefix = f"chi_{df}"
    target_cell_id = f"chi_{df}_{col_idx}" # Uses the index in the displayed list

    # Create fresh table for each step
    step_html_table = f"<table>{caption}{header}{body}</table>" # Include caption

    if step == 0: # Highlight Row
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=2)
    elif step == 1: # Highlight Column
        step_html_table = style_col(step_html_table, col_idx, row_range, prefix="chi", color="cornflowerblue", px=2)
    elif step == 2: # Highlight Cell
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=1)
        step_html_table = style_col(step_html_table, col_idx, row_range, prefix="chi", color="cornflowerblue", px=1)
        step_html_table = style_cell(step_html_table, target_cell_id, color="darkorchid", px=3)

    # Add warning if alpha was adjusted
    warning_html = f'<p style="color: orange; font-size: 0.8em;">⚠️ {warning_message}</p>' if warning_message else ""

    return warning_html + wrap_table(CSS_BASE, step_html_table)

# ....................................... table wrapper
def chi_table(df: int, alpha: float):
    animate(lambda s: build_chi_html(df, alpha, s),
            frames=3, key=f"chi_anim_{df}_{alpha}")

# ....................................... APA narrative
def chi_apa(chi_val: float, df: int, alpha: float):
    try:
        chi_val = float(chi_val)
        df = int(df)
        alpha = float(alpha)
        if df < 1: raise ValueError("df must be >= 1")
        if not (0 < alpha < 1): raise ValueError("alpha must be between 0 and 1")
        if chi_val < 0: raise ValueError("Chi-square cannot be negative")
    except (ValueError, TypeError):
        st.error("Invalid input values for APA interpretation.")
        return

    p_crit = alpha

    # Chi-square test is typically one-tailed (upper tail)
    p_calc = stats.chi2.sf(chi_val, df)
    crit = stats.chi2.ppf(1 - alpha, df)
    reject_stat = chi_val > crit
    reject_p = p_calc < alpha
    comparison_stat = f"{chi_val:.3f} > {crit:.3f}" if reject_stat else f"{chi_val:.3f} \\le {crit:.3f}$"
    comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"
    crit_str = f"$\\chi^2_{{crit}}({df}) = {crit:.3f}$"

    decision = "rejected" if reject_stat else "failed to reject"
    decision_final = "rejected" if reject_p else "failed to reject"

    st.markdown(f"""
    ##### APA-7 Interpretation
    * **Calculated statistic:** $\\chi^2({df}) = {chi_val:.3f}$
    * **Calculated *p*-value:** $p = {p_calc:.3f}$
    * **Critical value ($\\alpha={alpha:.3f}$):** {crit_str}
    * **Critical *p*-value (alpha):** $p_{{crit}} = {alpha:.3f}$

    * **Decision based on statistic vs. critical value:** $H_0$ is **{decision}** because {comparison_stat}.
    * **Decision based on calculated *p*-value vs. alpha:** $H_0$ is **{decision_final}** because {comparison_p}.

    * **APA-7 Sentence:** The Chi-square test indicated that the result was {'statistically significant' if reject_p else 'not statistically significant'}, $\\chi^2({df}) = {chi_val:.3f}$, $p = {p_calc:.3f}$. The null hypothesis was **{decision_final}** at the $\\alpha = {alpha:.3f}$ significance level.
    """, unsafe_allow_html=True)

# ....................................... tab assembly
def tab_chi():
    st.subheader("Tab 4 • Chi‑Square ($\\chi^2$) Distribution")
    c1, c2 = st.columns(2)
    with c1:
        chi_val_str = st.text_input("$\\chi^2$ statistic ($\\chi^2_{calc}$)", value="7.88", key="chi_val_str")
        df_str = st.text_input("Degrees of freedom ($df$)", value="3", key="chi_df_str")
    with c2:
        # Using a select box for common alphas
        alpha = st.selectbox("Significance level ($\\alpha$)", [0.10, 0.05, 0.025, 0.01, 0.005],
                             index=1, key="chi_alpha", format_func=lambda x: f"{x:.3f}") # Default 0.05

    # Validate inputs
    chi_val, df = None, None
    valid_inputs = False
    try:
        chi_val = float(chi_val_str)
        df = int(df_str)
        # alpha is already float from selectbox
        if df < 1: raise ValueError("Degrees of freedom must be at least 1")
        if chi_val < 0: raise ValueError("Chi-square cannot be negative")
        valid_inputs = True
    except ValueError as e:
        st.error(f"⚠️ Invalid input: {e}. Please enter valid numbers.")
        # Placeholder
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "Invalid input. Cannot draw plot.", ha='center', va='center', color='red')
        ax.set_title("$\\chi^2$-Distribution")
        st.pyplot(fig)

    if valid_inputs:
        st.pyplot(plot_chi(chi_val, df, alpha))

        with st.expander("Show $\\chi^2$-table lookup steps & Interpretation"):
            chi_table(df, alpha)
            st.markdown("---") # Separator
            chi_apa(chi_val, df, alpha)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 • Mann‑Whitney U (using Normal Approximation)
# ════════════════════════════════════════════════════════════════════════════
# Note: Exact tables are complex. This implementation uses the normal approximation
# which is generally suitable for n1, n2 > ~10 or 20. The 'table' shown will
# reflect critical values based on this approximation.

# ....................................... critical U (normal approximation)
def u_crit_approx(n1: int, n2: int, alpha: float, tail: str) -> float:
    """Calculate critical U value using normal approximation WITH continuity correction."""
    try:
        n1, n2 = int(n1), int(n2)
        alpha = float(alpha)
        if n1 <= 0 or n2 <= 0: return np.nan # Avoid division by zero
        if not (0 < alpha < 1): return np.nan
    except (ValueError, TypeError):
        return np.nan


    mu_U = n1 * n2 / 2
    # Add epsilon to prevent division by zero if n1+n2+1 = 0 (shouldn't happen)
    sigma_U_sq = n1 * n2 * (n1 + n2 + 1) / 12
    if sigma_U_sq <= 1e-9: return mu_U # Handle case of zero variance
    sigma_U = np.sqrt(sigma_U_sq)

    if tail == "one-tailed":
        # Assumes H1 predicts SMALL U (use alpha level directly for lower tail z)
        z_crit = stats.norm.ppf(alpha)
        u_critical = mu_U + z_crit * sigma_U
    else: # two-tailed
        z_crit = stats.norm.ppf(alpha / 2) # For lower tail
        # Symmetric: lower critical U and upper critical U = n1*n2 - lower_crit
        u_critical = mu_U + z_crit * sigma_U # This gives the lower critical value

    # Apply continuity correction (adjust critical value 0.5 away from the mean)
    # For lower tail critical value, subtract 0.5
    u_critical_corrected = u_critical - 0.5

    # Return the corrected critical value (often floor is taken for lookup tables)
    return u_critical_corrected

# ....................................... plot (using normal approximation)
def plot_u(u_calc, n1, n2, alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    try:
        u_calc = float(u_calc) # U stat is typically integer, but allow float input
        n1 = int(n1)
        n2 = int(n2)
        alpha = float(alpha)
        if n1 < 1 or n2 < 1: raise ValueError("n1/n2 must be >= 1")
    except (ValueError, TypeError):
        ax.text(0.5, 0.5, "Invalid input for plot.", ha='center', va='center', fontsize=10, color='red')
        return fig

    mu_U = n1 * n2 / 2
    sigma_U_sq = n1 * n2 * (n1 + n2 + 1) / 12
    if sigma_U_sq <= 1e-9: # If sigma is effectively zero
         ax.text(0.5, 0.5, "Cannot plot: Standard deviation of U is zero.", ha='center', va='center', fontsize=10, color='orange')
         return fig
    sigma_U = np.sqrt(sigma_U_sq)

    # Determine plot range
    plot_max_hi = mu_U + 4*sigma_U
    plot_max_lo = max(0, mu_U - 4*sigma_U) # U cannot be negative
    # Ensure plot range is reasonable if sigma is very small
    if plot_max_hi - plot_max_lo < 1:
        plot_max_lo = max(0, mu_U - 2)
        plot_max_hi = mu_U + 2
    xs = np.linspace(plot_max_lo, plot_max_hi, 500)
    ys = stats.norm.pdf(xs, mu_U, sigma_U)

    ax.plot(xs, ys, "k", linewidth=1)
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.35, label="Fail to Reject $H_0$")

    labels = []

    # Get critical value (lower bound) using approximation WITH continuity correction
    crit_lo_approx = u_crit_approx(n1, n2, alpha, tail)
    # Calculate approx upper critical bound based on symmetry around mean
    crit_hi_approx = n1*n2 - crit_lo_approx # This uses the corrected lower bound

    if tail == "one-tailed":
        # Mann-Whitney U is significant if U is small (typically)
        ax.fill_between(xs[xs <= crit_lo_approx], ys[xs <= crit_lo_approx],
                        color="red", alpha=0.40, label="Reject $H_0$")
        ax.axvline(crit_lo_approx, color="green", ls="--", linewidth=1)
        place_label(ax, labels, crit_lo_approx, stats.norm.pdf(crit_lo_approx, mu_U, sigma_U),
                    f"$U_{{crit}} \\approx {crit_lo_approx:.3f}$", color="green")
        ax.text(crit_lo_approx - sigma_U*0.5, ax.get_ylim()[1]*0.1, f"$\\alpha = {alpha:.3f}$",
                color="red", ha="right", va="bottom", fontsize=9)
    else: # two-tailed
        ax.fill_between(xs[xs <= crit_lo_approx], ys[xs <= crit_lo_approx], color="red", alpha=0.40)
        ax.fill_between(xs[xs >= crit_hi_approx], ys[xs >= crit_hi_approx], color="red", alpha=0.40,
                        label="Reject $H_0$")
        ax.axvline(crit_lo_approx, color="green", ls="--", linewidth=1)
        ax.axvline(crit_hi_approx, color="green", ls="--", linewidth=1)
        place_label(ax, labels, crit_lo_approx, stats.norm.pdf(crit_lo_approx, mu_U, sigma_U),
                    f"$U_{{crit,lo}} \\approx {crit_lo_approx:.3f}$", color="green")
        place_label(ax, labels, crit_hi_approx, stats.norm.pdf(crit_hi_approx, mu_U, sigma_U),
                    f"$U_{{crit,hi}} \\approx {crit_hi_approx:.3f}$", color="green")
        ax.text(crit_lo_approx - sigma_U*0.5, ax.get_ylim()[1]*0.1, f"$\\alpha/2$",
                color="red", ha="right", va="bottom", fontsize=9)
        ax.text(crit_hi_approx + sigma_U*0.5, ax.get_ylim()[1]*0.1, f"$\\alpha/2$",
                color="red", ha="left", va="bottom", fontsize=9)

    ax.axvline(u_calc, color="blue", ls="-", linewidth=1.5)
    y_pos_calc = stats.norm.pdf(u_calc, mu_U, sigma_U)
    place_label(ax, labels, u_calc, y_pos_calc,
                f"$U_{{calc}} = {u_calc:.0f}$", color="blue") # U is integer

    ax.set_xlabel("$U$ statistic value")
    ax.set_ylabel("Approx. Probability Density")
    ax.legend(fontsize=9)
    ax.set_title(f"Mann-Whitney $U$ Distribution (Normal Approx., $n_1={n1}, n_2={n2}$), {tail}", fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=plot_max_lo, right=plot_max_hi)
    fig.tight_layout()
    return fig

# ....................................... animated table builder (using normal approx)
def build_u_html(n1: int, n2: int, alpha: float, tail: str, step: int) -> str:
    try:
        n1 = int(n1)
        n2 = int(n2)
        alpha = float(alpha)
        if n1 < 1 or n2 < 1: raise ValueError("n1/n2 must be >= 1")
    except (ValueError, TypeError):
        return "<table><caption>Invalid Input</caption><tr><td>Invalid input for U-table generation.</td></tr></table>"

    # Define ranges around the target n values
    row_range = list(range(max(1, n1 - 5), n1 + 6)) # Rows are n1
    col_range = list(range(max(1, n2 - 5), n2 + 6)) # Columns are n2

    # Find indices for highlighting
    try: row_val_idx = row_range.index(n1)
    except ValueError: return f"<table><caption>Error</caption><tr><td>n1={n1} not in display range.</td></tr></table>"
    try: col_val_idx = col_range.index(n2)
    except ValueError: return f"<table><caption>Error</caption><tr><td>n2={n2} not in display range.</td></tr></table>"
    col_idx_html = col_val_idx + 1 # HTML index

    # Build Header
    header = f'<thead><tr><th style="position: sticky; left: 0; z-index: 2;">$n_1 \\setminus n_2$</th>'
    header += "".join(f"<th>{c}</th>" for c in col_range) + '</tr></thead>'

    # Build Body
    body = "<tbody>"
    for r in row_range:
        row_html = f'<tr><td id="u_{r}_0">{r}</td>' # Row header ID includes n1 value
        for i, c in enumerate(col_range):
            # Calculate approximate critical U value (lower tail, corrected)
            crit_val = u_crit_approx(r, c, alpha, tail)
            # Display as integer, often floor is used in tables for lower bound
            display_val = int(np.floor(crit_val)) if not np.isnan(crit_val) else "N/A"
            row_html += f'<td id="u_{r}_{i+1}">{display_val}</td>' # Cell ID includes n1 and col index
        body += row_html + "</tr>"
    body += "</tbody>"

    # Caption
    caption = f"<caption>Approx. Lower Critical Values for Mann-Whitney $U$ (\\(\\alpha={alpha:.3f}\\), {tail})</caption>"

    # Base table structure
    base_html_table = f"<table>{caption}{header}{body}</table>"

    # Apply sequential highlighting
    row_prefix = f"u_{n1}" # Row prefix uses n1 value
    target_cell_id = f"u_{n1}_{col_idx_html}"

    # Create fresh table for each step
    step_html_table = f"<table>{caption}{header}{body}</table>" # Include caption

    if step == 0: # Highlight Row (n1)
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=2)
    elif step == 1: # Highlight Column (n2)
        step_html_table = style_col(step_html_table, col_idx_html, row_range, prefix="u", color="cornflowerblue", px=2)
    elif step == 2: # Highlight Cell
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=1)
        step_html_table = style_col(step_html_table, col_idx_html, row_range, prefix="u", color="cornflowerblue", px=1)
        step_html_table = style_cell(step_html_table, target_cell_id, color="darkorchid", px=3)

    return wrap_table(CSS_BASE, step_html_table)


# ....................................... table wrapper
def u_table(n1: int, n2: int, alpha: float, tail: str):
    animate(lambda s: build_u_html(n1, n2, alpha, tail, s),
            frames=3, key=f"u_anim_{n1}_{n2}_{alpha}_{tail}")

# ....................................... APA narrative (using normal approx)
def u_apa(u_val: int, n1: int, n2: int, alpha: float, tail: str):
    try:
        u_val = int(u_val) # U should be an integer
        n1 = int(n1)
        n2 = int(n2)
        alpha = float(alpha)
        if n1 < 1 or n2 < 1: raise ValueError("n1/n2 must be >= 1")
        if not (0 < alpha < 1): raise ValueError("alpha must be between 0 and 1")
        max_U = n1*n2
        if not (0 <= u_val <= max_U): raise ValueError(f"U must be between 0 and {max_U}")

    except (ValueError, TypeError):
        st.error("Invalid input values for APA interpretation.")
        return

    p_crit = alpha

    # Calculate z and p directly using normal approx with continuity correction.
    mu_U = n1 * n2 / 2
    sigma_U_sq = n1 * n2 * (n1 + n2 + 1) / 12

    if sigma_U_sq <= 1e-9:
        st.warning("Cannot calculate z-score or p-value: Variance of U is zero.", icon="⚠️")
        p_calc = np.nan
        z_calc = np.nan
    else:
        sigma_U = np.sqrt(sigma_U_sq)
        # Apply continuity correction (adjust U by 0.5 towards the mean)
        if abs(u_val - mu_U) < 1e-9: # Handle U exactly at the mean
             z_calc = 0
        elif u_val < mu_U:
            z_calc = (u_val + 0.5 - mu_U) / sigma_U
        else: # u_val > mu_U:
            z_calc = (u_val - 0.5 - mu_U) / sigma_U

        # Calculate p-value based on z_calc and tail
        if tail == "one-tailed":
            # Assumes H1 predicts SMALL U (use CDF for left tail p)
            p_calc = stats.norm.cdf(z_calc)
        else: # two-tailed
            p_calc = 2 * stats.norm.sf(abs(z_calc)) # 2 * area in the tail away from mean


    # Get critical value using approximation (lower tail, corrected)
    crit_lo = u_crit_approx(n1, n2, alpha, tail) # Already corrected
    crit_hi = n1 * n2 - crit_lo # Approx upper bound based on symmetry of corrected lower bound

    # Determine rejection based on statistic vs critical value
    reject_stat = False
    if tail == "one-tailed":
        # Reject if U_calc is less than or equal to lower critical value
        reject_stat = u_val <= crit_lo + 1e-9 # Add tolerance for float comparison
        comparison_stat = f"{u_val} \\le {crit_lo:.3f}" if reject_stat else f"{u_val} > {crit_lo:.3f}$"
        crit_str = f"$U_{{crit}} \\approx {crit_lo:.3f}$ (lower tail, corrected)"
    else: # two-tailed
        # Reject if U_calc is <= lower_crit or >= upper_crit
        reject_stat = (u_val <= crit_lo + 1e-9) or (u_val >= crit_hi - 1e-9) # Add tolerance
        comparison_stat = f"({u_val} \\le {crit_lo:.3f}) \\lor ({u_val} \\ge {crit_hi:.3f})" if reject_stat else f"({crit_lo:.3f} < {u_val} < {crit_hi:.3f})$"
        crit_str = f"$U_{{crit}} \\approx {crit_lo:.3f}$ (lower) or ${crit_hi:.3f}$ (upper, corrected)"

    # Determine rejection based on p-value
    reject_p = p_calc < alpha if not np.isnan(p_calc) else False
    comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"
    if np.isnan(p_calc): comparison_p = "N/A (zero variance)"

    decision = "rejected" if reject_stat else "failed to reject"
    decision_final = "rejected" if reject_p else "failed to reject"

    st.markdown(f"""
    ##### APA-7 Interpretation (using Normal Approximation)
    * **Calculated statistic:** $U = {u_val}$ (Assuming this is $U_1$)
    * **Sample sizes:** $n_1 = {n1}, n_2 = {n2}$
    * **Approximate $z$-statistic (corrected):** $z \\approx {z_calc:.3f}$
    * **Calculated *p*-value (approx.):** $p \\approx {p_calc:.3f}$ {'(one-tailed)' if tail == 'one-tailed' else '(two-tailed)'}
    * **Approx. Critical value(s) ({tail}, $\\alpha={alpha:.3f}$):** {crit_str}
    * **Critical *p*-value (alpha):** $p_{{crit}} = {alpha:.3f}$

    * **Decision based on statistic vs. critical value:** $H_0$ is **{decision}** because {comparison_stat}.
    * **Decision based on calculated *p*-value vs. alpha:** $H_0$ is **{decision_final}** because {comparison_p}.

    * **APA-7 Sentence:** A Mann-Whitney U test indicated that the result was {'statistically significant' if reject_p else 'not statistically significant'}, $U = {u_val}$, $z \\approx {z_calc:.3f}$, $p \\approx {p_calc:.3f}$. The null hypothesis was **{decision_final}** using a {tail} test at the $\\alpha = {alpha:.3f}$ level. (Note: Reporting U, z, and p based on approximation is common when sample sizes justify it).
    """, unsafe_allow_html=True)

# ....................................... tab assembly
def tab_u():
    st.subheader("Tab 5 • Mann‑Whitney $U$ Test (Normal Approximation)")
    st.caption("Note: Uses normal approximation with continuity correction. Accuracy improves for larger $n_1, n_2$ (e.g., > 10-20). The table shows approximate critical values based on this method.")

    c1, c2 = st.columns(2)
    with c1:
        u_val_str = st.text_input("$U$ statistic ($U_{calc}$)", value="23", key="u_val_str")
        n1_str = st.text_input("$n_1$ (sample size 1)", value="10", key="u_n1_str")
    with c2:
        n2_str = st.text_input("$n_2$ (sample size 2)", value="12", key="u_n2_str")
        alpha_str = st.text_input("Significance level ($\\alpha$)", value="0.05", key="u_alpha_str")
        tail = st.radio("Tail(s)", ["one-tailed", "two-tailed"], index=1, key="u_tail", horizontal=True)

    # Validate inputs
    u_val, n1, n2, alpha = None, None, None, None
    valid_inputs = False
    try:
        u_val = int(u_val_str)
        n1 = int(n1_str)
        n2 = int(n2_str)
        alpha = float(alpha_str)
        if not (0 < alpha < 1): raise ValueError("Alpha must be between 0 and 1")
        if n1 < 1 or n2 < 1: raise ValueError("Sample sizes (n1, n2) must be at least 1")
        max_U = n1*n2
        if not (0 <= u_val <= max_U): raise ValueError(f"U must be between 0 and {max_U}")
        valid_inputs = True
    except ValueError as e:
        st.error(f"⚠️ Invalid input: {e}. Please enter valid numbers.")
        # Placeholder
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "Invalid input. Cannot draw plot.", ha='center', va='center', color='red')
        ax.set_title("Mann-Whitney $U$ Distribution (Approx.)")
        st.pyplot(fig)

    if valid_inputs:
        st.pyplot(plot_u(u_val, n1, n2, alpha, tail))

        with st.expander("Show Approx. $U$-table lookup steps & Interpretation"):
            u_table(n1, n2, alpha, tail)
            st.markdown("---") # Separator
            u_apa(u_val, n1, n2, alpha, tail)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 • Wilcoxon Signed‑Rank T (using Normal Approximation)
# ════════════════════════════════════════════════════════════════════════════
# Note: Similar to Mann-Whitney U, exact tables are complex. Uses normal approximation,
# suitable for N (non-zero pairs) > ~20. Table reflects approx critical values.

# ....................................... critical T (normal approximation)
def w_crit_approx(n: int, alpha: float, tail: str) -> float:
    """Calculate critical T value using normal approximation WITH continuity correction."""
    try:
        n = int(n)
        alpha = float(alpha)
        if n <= 0: return np.nan
        if not (0 < alpha < 1): return np.nan
    except (ValueError, TypeError):
        return np.nan

    mu_T = n * (n + 1) / 4
    sigma_T_sq = n * (n + 1) * (2 * n + 1) / 24
    if sigma_T_sq <= 1e-9: return mu_T # Handle zero variance
    sigma_T = np.sqrt(sigma_T_sq)

    if tail == "one-tailed":
        # Typically testing if T is significantly SMALL
        z_crit = stats.norm.ppf(alpha)
        t_critical = mu_T + z_crit * sigma_T
    else: # two-tailed
        z_crit = stats.norm.ppf(alpha / 2) # For lower tail
        # Symmetric: lower critical T, upper critical T = n(n+1)/2 - lower_crit
        t_critical = mu_T + z_crit * sigma_T # Lower critical value

    # Apply continuity correction (adjust critical value 0.5 away from the mean)
    # For lower tail critical value, subtract 0.5
    t_critical_corrected = t_critical - 0.5

    return t_critical_corrected

# ....................................... plot (using normal approximation)
def plot_w(t_calc, n, alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    try:
        t_calc = float(t_calc) # T stat is typically integer
        n = int(n)
        alpha = float(alpha)
        if n < 1: raise ValueError("N must be >= 1")
    except (ValueError, TypeError):
        ax.text(0.5, 0.5, "Invalid input for plot.", ha='center', va='center', fontsize=10, color='red')
        return fig

    mu_T = n * (n + 1) / 4
    sigma_T_sq = n * (n + 1) * (2 * n + 1) / 24
    if sigma_T_sq <= 1e-9:
         ax.text(0.5, 0.5, "Cannot plot: Standard deviation of T is zero.", ha='center', va='center', fontsize=10, color='orange')
         return fig
    sigma_T = np.sqrt(sigma_T_sq)

    # Determine plot range
    plot_max_hi = mu_T + 4 * sigma_T
    plot_max_lo = max(0, mu_T - 4 * sigma_T) # T cannot be negative
    if plot_max_hi - plot_max_lo < 1: # Ensure reasonable range
        plot_max_lo = max(0, mu_T - 2)
        plot_max_hi = mu_T + 2
    xs = np.linspace(plot_max_lo, plot_max_hi, 500)
    ys = stats.norm.pdf(xs, mu_T, sigma_T)

    ax.plot(xs, ys, "k", linewidth=1)
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.35, label="Fail to Reject $H_0$")

    labels = []

    # Get critical value (lower bound) using approximation WITH continuity correction
    crit_lo_approx = w_crit_approx(n, alpha, tail)
    # Calculate symmetric upper critical value based on corrected lower bound
    max_T = n * (n + 1) / 2
    crit_hi_approx = max_T - crit_lo_approx # Approx upper bound

    if tail == "one-tailed":
        # Wilcoxon T is significant if T is small (sum of ranks of pos/neg diffs)
        ax.fill_between(xs[xs <= crit_lo_approx], ys[xs <= crit_lo_approx],
                        color="red", alpha=0.40, label="Reject $H_0$")
        ax.axvline(crit_lo_approx, color="green", ls="--", linewidth=1)
        place_label(ax, labels, crit_lo_approx, stats.norm.pdf(crit_lo_approx, mu_T, sigma_T),
                    f"$T_{{crit}} \\approx {crit_lo_approx:.3f}$", color="green")
        ax.text(crit_lo_approx - sigma_T*0.5, ax.get_ylim()[1]*0.1, f"$\\alpha = {alpha:.3f}$",
                color="red", ha="right", va="bottom", fontsize=9)
    else: # two-tailed
        ax.fill_between(xs[xs <= crit_lo_approx], ys[xs <= crit_lo_approx], color="red", alpha=0.40)
        ax.fill_between(xs[xs >= crit_hi_approx], ys[xs >= crit_hi_approx], color="red", alpha=0.40,
                        label="Reject $H_0$")
        ax.axvline(crit_lo_approx, color="green", ls="--", linewidth=1)
        ax.axvline(crit_hi_approx, color="green", ls="--", linewidth=1)
        place_label(ax, labels, crit_lo_approx, stats.norm.pdf(crit_lo_approx, mu_T, sigma_T),
                    f"$T_{{crit,lo}} \\approx {crit_lo_approx:.3f}$", color="green")
        place_label(ax, labels, crit_hi_approx, stats.norm.pdf(crit_hi_approx, mu_T, sigma_T),
                    f"$T_{{crit,hi}} \\approx {crit_hi_approx:.3f}$", color="green")
        ax.text(crit_lo_approx - sigma_T*0.5, ax.get_ylim()[1]*0.1, f"$\\alpha/2$",
                color="red", ha="right", va="bottom", fontsize=9)
        ax.text(crit_hi_approx + sigma_T*0.5, ax.get_ylim()[1]*0.1, f"$\\alpha/2$",
                color="red", ha="left", va="bottom", fontsize=9)

    ax.axvline(t_calc, color="blue", ls="-", linewidth=1.5)
    y_pos_calc = stats.norm.pdf(t_calc, mu_T, sigma_T)
    place_label(ax, labels, t_calc, y_pos_calc,
                f"$T_{{calc}} = {t_calc:.0f}$", color="blue") # T is integer

    ax.set_xlabel("$T$ statistic value")
    ax.set_ylabel("Approx. Probability Density")
    ax.legend(fontsize=9)
    ax.set_title(f"Wilcoxon Signed-Rank $T$ Dist. (Normal Approx., $N={n}$), {tail}", fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=plot_max_lo, right=plot_max_hi)
    fig.tight_layout()
    return fig


# ....................................... animated table builder (using normal approx)
def build_w_html(n: int, alpha: float, tail: str, step: int) -> str:
    try:
        n = int(n)
        alpha = float(alpha)
        if n < 1: raise ValueError("N must be >= 1")
    except (ValueError, TypeError):
        return "<table><caption>Invalid Input</caption><tr><td>Invalid input for T-table generation.</td></tr></table>"


    # Define N range around the target N
    row_range = list(range(max(5, n - 5), n + 6)) # Rows are N (min N usually ~5-8 for tables)
    warning_message = None

    # Define common alpha levels for columns based on tail type
    if tail == "one-tailed":
        standard_alphas = [0.05, 0.025, 0.01, 0.005]
    else: # two-tailed
        standard_alphas = [0.10, 0.05, 0.02, 0.01]

    # Find column index based on alpha and tail
    target_alpha = alpha
    col_idx = -1
    if target_alpha in standard_alphas:
        col_idx = standard_alphas.index(target_alpha) + 1 # +1 for N column
    else:
        closest_alpha = min(standard_alphas, key=lambda x: abs(x - alpha))
        col_idx = standard_alphas.index(closest_alpha) + 1
        warning_message = f"Alpha {alpha:.4f} not standard for {tail} Wilcoxon table, using closest: {closest_alpha:.3f}"
        target_alpha = closest_alpha # Use closest for lookup

    # Find row index
    try: row_val_idx = row_range.index(n)
    except ValueError: return f"<table><caption>Error</caption><tr><td>N={n} not in display range.</td></tr></table>"


    # Build Header
    header = f'<thead><tr><th style="position: sticky; left: 0; z-index: 2;">N \\ $\\alpha$ ({tail})</th>'
    header += "".join(f"<th>{a:.3f}</th>" for a in standard_alphas) + '</tr></thead>'

    # Build Body
    body = "<tbody>"
    for r_n in row_range:
        row_html = f'<tr><td id="w_{r_n}_0">{r_n}</td>' # Row header ID includes N value
        for i, a_col in enumerate(standard_alphas):
            # Calculate approximate critical T value (lower tail, corrected)
            crit_val = w_crit_approx(r_n, a_col, tail)
            # Display as integer (floor is common for lower bound)
            display_val = int(np.floor(crit_val)) if not np.isnan(crit_val) else "N/A"
            row_html += f'<td id="w_{r_n}_{i+1}">{display_val}</td>' # Cell ID includes N and col index
        body += row_html + "</tr>"
    body += "</tbody>"

    # Caption
    caption = f"<caption>Approx. Lower Critical Values for Wilcoxon T (N={n}, \\(\\alpha={alpha:.3f}\\), {tail})</caption>"

    # Base table structure
    base_html_table = f"<table>{caption}{header}{body}</table>"

    # Apply sequential highlighting
    row_prefix = f"w_{n}" # Row prefix uses N value
    target_cell_id = f"w_{n}_{col_idx}"

    # Create fresh table for each step
    step_html_table = f"<table>{caption}{header}{body}</table>" # Include caption

    if step == 0: # Highlight Row (N)
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=2)
    elif step == 1: # Highlight Column (alpha)
        step_html_table = style_col(step_html_table, col_idx, row_range, prefix="w", color="cornflowerblue", px=2)
    elif step == 2: # Highlight Cell
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=1)
        step_html_table = style_col(step_html_table, col_idx, row_range, prefix="w", color="cornflowerblue", px=1)
        step_html_table = style_cell(step_html_table, target_cell_id, color="darkorchid", px=3)

    # Add warning if alpha was adjusted
    warning_html = f'<p style="color: orange; font-size: 0.8em;">⚠️ {warning_message}</p>' if warning_message else ""

    return warning_html + wrap_table(CSS_BASE, step_html_table)


# ....................................... table wrapper
def w_table(n: int, alpha: float, tail: str):
    animate(lambda s: build_w_html(n, alpha, tail, s),
            frames=3, key=f"w_anim_{n}_{alpha}_{tail}")

# ....................................... APA narrative (using normal approx)
def w_apa(t_val: int, n: int, alpha: float, tail: str):
    try:
        t_val = int(t_val) # T should be integer
        n = int(n)
        alpha = float(alpha)
        if n < 1: raise ValueError("N must be >= 1")
        if not (0 < alpha < 1): raise ValueError("alpha must be between 0 and 1")
        max_T = n * (n + 1) / 2
        if not (0 <= t_val <= max_T): raise ValueError(f"T must be between 0 and {max_T:.0f}")

    except (ValueError, TypeError):
        st.error("Invalid input values for APA interpretation.")
        return

    p_crit = alpha

    # Use normal approximation to calculate z and p-value
    mu_T = n * (n + 1) / 4
    sigma_T_sq = n * (n + 1) * (2 * n + 1) / 24

    if sigma_T_sq <= 1e-9:
        st.warning("Cannot calculate z-score or p-value: Variance of T is zero.", icon="⚠️")
        p_calc = np.nan
        z_calc = np.nan
    else:
        sigma_T = np.sqrt(sigma_T_sq)
        # Calculate z-score for observed T with continuity correction
        if abs(t_val - mu_T) < 1e-9:
            z_calc = 0
        elif t_val < mu_T:
            z_calc = (t_val + 0.5 - mu_T) / sigma_T
        else: # t_val > mu_T
            z_calc = (t_val - 0.5 - mu_T) / sigma_T

        # Calculate p-value based on z_calc and tail
        if tail == "one-tailed":
            # Assumes H1 predicts SMALL T (use CDF for left tail p)
            p_calc = stats.norm.cdf(z_calc)
        else: # two-tailed
            p_calc = 2 * stats.norm.sf(abs(z_calc)) # 2 * area in tail


    # Get critical value using approximation (lower tail, corrected)
    crit_lo = w_crit_approx(n, alpha, tail) # Already corrected
    max_T = n * (n + 1) / 2
    crit_hi = max_T - crit_lo # Approx upper bound based on symmetry

    # Determine rejection based on statistic vs critical value
    reject_stat = False
    if tail == "one-tailed":
        reject_stat = t_val <= crit_lo + 1e-9 # Reject if T is small enough
        comparison_stat = f"{t_val} \\le {crit_lo:.3f}" if reject_stat else f"{t_val} > {crit_lo:.3f}$"
        crit_str = f"$T_{{crit}} \\approx {crit_lo:.3f}$ (lower tail, corrected)"
    else: # two-tailed
        # Reject if T <= lower_crit or >= upper_crit
        reject_stat = (t_val <= crit_lo + 1e-9) or (t_val >= crit_hi - 1e-9)
        comparison_stat = f"({t_val} \\le {crit_lo:.3f}) \\lor ({t_val} \\ge {crit_hi:.3f})" if reject_stat else f"({crit_lo:.3f} < {t_val} < {crit_hi:.3f})$"
        crit_str = f"$T_{{crit}} \\approx {crit_lo:.3f}$ (lower) or ${crit_hi:.3f}$ (upper, corrected)"


    # Determine rejection based on p-value
    reject_p = p_calc < alpha if not np.isnan(p_calc) else False
    comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"
    if np.isnan(p_calc): comparison_p = "N/A (zero variance)"


    decision = "rejected" if reject_stat else "failed to reject"
    decision_final = "rejected" if reject_p else "failed to reject"


    st.markdown(f"""
    ##### APA-7 Interpretation (using Normal Approximation)
    * **Calculated statistic:** $T = {t_val}$
    * **Number of pairs (non-zero diffs):** $N = {n}$
    * **Approximate $z$-statistic (corrected):** $z \\approx {z_calc:.3f}$
    * **Calculated *p*-value (approx.):** $p \\approx {p_calc:.3f}$ {'(one-tailed)' if tail == 'one-tailed' else '(two-tailed)'}
    * **Approx. Critical value(s) ({tail}, $\\alpha={alpha:.3f}$):** {crit_str}
    * **Critical *p*-value (alpha):** $p_{{crit}} = {alpha:.3f}$

    * **Decision based on statistic vs. critical value:** $H_0$ is **{decision}** because {comparison_stat}.
    * **Decision based on calculated *p*-value vs. alpha:** $H_0$ is **{decision_final}** because {comparison_p}.

    * **APA-7 Sentence:** A Wilcoxon signed-rank test indicated that the result was {'statistically significant' if reject_p else 'not statistically significant'}, $T({n}) = {t_val}$, $z \\approx {z_calc:.3f}$, $p \\approx {p_calc:.3f}$. The null hypothesis was **{decision_final}** using a {tail} test at the $\\alpha = {alpha:.3f}$ level. (Note: Reporting T with N, z, and p based on approximation is common).
    """, unsafe_allow_html=True)


# ....................................... tab assembly
def tab_w():
    st.subheader("Tab 6 • Wilcoxon Signed‑Rank $T$ Test (Normal Approx.)")
    st.caption("Note: Uses normal approximation with continuity correction. Accuracy improves for larger $N$ (e.g., > 20). The table shows approximate critical values based on this method.")

    c1, c2 = st.columns(2)
    with c1:
        t_val_str = st.text_input("$T$ statistic ($T_{calc}$)", value="15", key="w_val_str")
        n_str = st.text_input("$N$ (non‑zero differences)", value="12", key="w_n_str")
    with c2:
        alpha_str = st.text_input("Significance level ($\\alpha$)", value="0.05", key="w_alpha_str")
        tail = st.radio("Tail(s)", ["one-tailed", "two-tailed"], index=1, key="w_tail", horizontal=True)

    # Validate inputs
    t_val, n, alpha = None, None, None
    valid_inputs = False
    try:
        t_val = int(t_val_str)
        n = int(n_str)
        alpha = float(alpha_str)
        max_T = n * (n + 1) / 2
        if not (0 < alpha < 1): raise ValueError("Alpha must be between 0 and 1")
        if n < 1 : raise ValueError("N must be at least 1")
        # Allow T=0
        if t_val < 0 or t_val > max_T: raise ValueError(f"T must be between 0 and {max_T:.0f}")
        valid_inputs = True
    except ValueError as e:
        st.error(f"⚠️ Invalid input: {e}. Please enter valid numbers.")
        # Placeholder
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "Invalid input. Cannot draw plot.", ha='center', va='center', color='red')
        ax.set_title("Wilcoxon Signed-Rank $T$ Distribution (Approx.)")
        st.pyplot(fig)


    if valid_inputs:
        st.pyplot(plot_w(t_val, n, alpha, tail))

        with st.expander("Show Approx. $T$-table lookup steps & Interpretation"):
            w_table(n, alpha, tail)
            st.markdown("---") # Separator
            w_apa(t_val, n, alpha, tail)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 7 • Binomial Distribution
# ════════════════════════════════════════════════════════════════════════════

# ....................................... critical bounds (exact)
def critical_binom(n: int, p_h0: float, alpha: float, tail: str):
    """Return (lower k, upper k) rejection bounds for EXACT binomial test."""
    try:
        n = int(n)
        p_h0 = float(p_h0)
        alpha = float(alpha)
        if n < 1: raise ValueError("n must be >= 1")
        if not (0 <= p_h0 <= 1): raise ValueError("p H0 must be 0-1")
        if not (0 < alpha < 1): raise ValueError("alpha must be 0-1")
    except (ValueError, TypeError):
        return -1, n+1, np.nan # Return invalid bounds on bad input

    k_values = np.arange(n + 1)
    # CDF P(X <= k)
    cdf_vals = stats.binom.cdf(k_values, n, p_h0)
    # P(X >= k) = P(X > k-1) = sf(k-1)
    sf_ge_k = np.array([stats.binom.sf(k - 1, n, p_h0) for k in k_values])

    k_crit_lo = -1 # Default: no lower rejection region (k cannot be < 0)
    k_crit_hi = n + 1 # Default: no upper rejection region (k cannot be > n)

    # Define tolerance for floating point comparisons
    tol = 1e-9

    if tail == "one-tailed-lower": # Reject if k is too small: P(X <= k) <= alpha
        possible_k = k_values[cdf_vals <= alpha + tol]
        if len(possible_k) > 0:
            k_crit_lo = possible_k[-1] # The largest k satisfying the condition
    elif tail == "one-tailed-upper": # Reject if k is too large: P(X >= k) <= alpha
        possible_k = k_values[sf_ge_k <= alpha + tol]
        if len(possible_k) > 0:
            k_crit_hi = possible_k[0] # The smallest k satisfying the condition
    else: # two-tailed
        # Find lower k such that P(X <= k_lo) <= alpha/2
        possible_k_lo = k_values[cdf_vals <= alpha / 2 + tol]
        if len(possible_k_lo) > 0:
            k_crit_lo = possible_k_lo[-1]
        # Find upper k such that P(X >= k_hi) <= alpha/2
        possible_k_hi = k_values[sf_ge_k <= alpha / 2 + tol]
        if len(possible_k_hi) > 0:
            k_crit_hi = possible_k_hi[0]

    # Calculate actual alpha achieved by these critical values
    actual_alpha = 0.0
    if k_crit_lo >= 0:
        actual_alpha += cdf_vals[k_crit_lo]
    if k_crit_hi <= n:
        # For two-tailed, only add upper tail probability if it doesn't overlap lower
        if tail != "two-tailed" or (k_crit_hi > k_crit_lo):
             actual_alpha += sf_ge_k[k_crit_hi]

    # Recalculate two-tailed actual alpha precisely
    if tail == "two-tailed":
        alpha_lo = cdf_vals[k_crit_lo] if k_crit_lo >= 0 else 0
        alpha_hi = sf_ge_k[k_crit_hi] if k_crit_hi <= n else 0
        actual_alpha = alpha_lo + alpha_hi

    return k_crit_lo, k_crit_hi, actual_alpha # Return bounds k and actual overall alpha

# ....................................... plot
def plot_binom(k_calc, n, p_h0, alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    try:
        k_calc = int(k_calc)
        n = int(n)
        p_h0 = float(p_h0)
        alpha = float(alpha)
        if n < 1: raise ValueError("n must be >= 1")
        if not (0 <= p_h0 <= 1): raise ValueError("p H0 must be 0-1")
        if not (0 <= k_calc <= n): raise ValueError("k must be 0-n")
    except (ValueError, TypeError):
        ax.text(0.5, 0.5, "Invalid input for plot.", ha='center', va='center', fontsize=10, color='red')
        return fig

    k_values = np.arange(n + 1)
    pmf = stats.binom.pmf(k_values, n, p_h0)

    # Bar plot for PMF
    ax.bar(k_values, pmf, color="lightgrey", alpha=0.6, label="Fail to Reject $H_0$ (Prob Mass)")

    labels = []
    crit_lo, crit_hi, actual_alpha = critical_binom(n, p_h0, alpha, tail)

    # Highlight rejection regions
    reject_color = "red"
    reject_label_added = False
    for k in k_values:
        is_reject = False
        # Determine if k falls into the rejection region based on critical values
        if tail == "one-tailed-lower" and k <= crit_lo: is_reject = True
        elif tail == "one-tailed-upper" and k >= crit_hi: is_reject = True
        elif tail == "two-tailed" and (k <= crit_lo or k >= crit_hi): is_reject = True

        if is_reject:
            label = "Reject $H_0$" if not reject_label_added else ""
            ax.bar(k, pmf[k], color=reject_color, alpha=0.7, label=label)
            reject_label_added = True

    # Add lines for critical values (if they exist within 0..n)
    line_y_max = ax.get_ylim()[1] * 0.95 # Position lines slightly below top

    if tail == "one-tailed-lower" and crit_lo >= 0:
        ax.axvline(crit_lo + 0.5, color="green", ls="--", linewidth=1, ymax=0.9) # Place between bars
        place_label(ax, labels, crit_lo + 0.5, line_y_max, f"$k \\le {crit_lo}$", color="green")
        ax.text(crit_lo + 0.5, ax.get_ylim()[1]*0.05, f"Actual $\\alpha \\approx {actual_alpha:.4f}$",
                color="red", ha="center", va="bottom", fontsize=8)
    elif tail == "one-tailed-upper" and crit_hi <= n:
        ax.axvline(crit_hi - 0.5, color="green", ls="--", linewidth=1, ymax=0.9)
        place_label(ax, labels, crit_hi - 0.5, line_y_max, f"$k \\ge {crit_hi}$", color="green")
        ax.text(crit_hi - 0.5, ax.get_ylim()[1]*0.05, f"Actual $\\alpha \\approx {actual_alpha:.4f}$",
                color="red", ha="center", va="bottom", fontsize=8)
    elif tail == "two-tailed":
        if crit_lo >= 0:
             ax.axvline(crit_lo + 0.5, color="green", ls="--", linewidth=1, ymax=0.9)
             place_label(ax, labels, crit_lo + 0.5, line_y_max, f"$k \\le {crit_lo}$", color="green")
        if crit_hi <= n:
             ax.axvline(crit_hi - 0.5, color="green", ls="--", linewidth=1, ymax=0.9)
             place_label(ax, labels, crit_hi - 0.5, line_y_max, f"$k \\ge {crit_hi}$", color="green")
        if crit_lo >= 0 or crit_hi <= n: # Add alpha text if region exists
            mid_point = n/2 # Place text near middle
            ax.text(mid_point, ax.get_ylim()[1]*0.05, f"Actual $\\alpha \\approx {actual_alpha:.4f}$",
                    color="red", ha="center", va="bottom", fontsize=8)


    # Add line for calculated k
    ax.axvline(k_calc, color="blue", ls="-", linewidth=1.5, ymax=0.85)
    # Adjust y-pos for calculated k label if pmf is very low
    y_pos_k_calc = pmf[k_calc] if k_calc < len(pmf) else 0
    if y_pos_k_calc < ax.get_ylim()[1] * 0.02: y_pos_k_calc = ax.get_ylim()[1] * 0.05
    place_label(ax, labels, k_calc, y_pos_k_calc, f"$k_{{calc}} = {k_calc}$", color="blue")


    ax.set_xlabel("Number of Successes ($k$)")
    ax.set_ylabel("Probability Mass $P(X=k)$")
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title(f"Binomial Distribution ($n={n}, p_{{H0}}={p_h0:.2f}$), {tail}", fontsize=12)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_ylim(bottom=0)
    # Ensure all k values are shown as ticks if n is not too large
    if n <= 30: ax.set_xticks(k_values)
    else: ax.set_xticks(np.linspace(0, n, 11, dtype=int)) # Show fewer ticks for large n
    # ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig

# ....................................... animated table builder (Binomial CDF/PMF)
def build_binom_html(k_calc: int, n: int, p_h0: float, step: int) -> str:
    try:
        n = int(n)
        p_h0 = float(p_h0)
        k_calc = int(k_calc) # The observed number of successes to highlight
        if not (0 <= k_calc <= n): raise ValueError("k_calc out of range")
    except (ValueError, TypeError):
         return "<table><caption>Invalid Input</caption><tr><td>Invalid input for Binomial table.</td></tr></table>"


    # Define k range (show all possible outcomes 0 to n)
    row_range = list(range(n + 1)) # Rows are k values

    # Find index for highlighting k_calc
    row_idx_html = k_calc # k_calc directly corresponds to the row index in this table

    # Build Header (showing k, PMF, CDF Lower, CDF Upper)
    header = (f'<thead><tr><th style="position: sticky; left: 0; z-index: 2;">$k$</th><th>PMF $P(X=k)$</th>'
              f'<th>CDF $P(X \\le k)$</th><th>Upper CDF $P(X \\ge k)$</th></tr></thead>')

    # Build Body
    body = "<tbody>"
    for r_k in row_range:
        try:
            pmf_val = stats.binom.pmf(r_k, n, p_h0)
            cdf_lo_val = stats.binom.cdf(r_k, n, p_h0)
            cdf_hi_val = stats.binom.sf(r_k - 1, n, p_h0) # P(X >= k) = 1 - P(X <= k-1)
        except ValueError: # Handle potential issues with extreme p_h0
            pmf_val, cdf_lo_val, cdf_hi_val = np.nan, np.nan, np.nan

        # Cell IDs: binom_kValue_colIndex (0=k, 1=PMF, 2=CDF_lo, 3=CDF_hi)
        row_html = f'<tr><td id="binom_{r_k}_0">{r_k}</td>'
        row_html += f'<td id="binom_{r_k}_1">{pmf_val:.4f}</td>'
        row_html += f'<td id="binom_{r_k}_2">{cdf_lo_val:.4f}</td>'
        row_html += f'<td id="binom_{r_k}_3">{cdf_hi_val:.4f}</td>'
        body += row_html + "</tr>"
    body += "</tbody>"

    # Caption
    caption = f"<caption>Binomial Probabilities (n={n}, p<sub>H0</sub>={p_h0:.2f}, k<sub>calc</sub>={k_calc})</caption>"


    # Base table structure
    base_html_table = f"<table>{caption}{header}{body}</table>"

    # Apply sequential highlighting to the row corresponding to k_calc
    # Step 0: Highlight row
    # Step 1: Highlight PMF cell
    # Step 2: Highlight appropriate CDF cell (e.g., lower CDF for p-value less)
    row_prefix = f"binom_{k_calc}"
    target_cell_pmf_id = f"binom_{k_calc}_1"
    target_cell_cdf_lo_id = f"binom_{k_calc}_2"
    # target_cell_cdf_hi_id = f"binom_{k_calc}_3" # Could highlight this too/instead

    # Create fresh table for each step
    step_html_table = f"<table>{caption}{header}{body}</table>" # Include caption

    if step == 0: # Highlight Row for k_calc
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=2)
    elif step == 1: # Highlight PMF cell for k_calc
        step_html_table = style_col(step_html_table, 1, row_range, prefix="binom", color="cornflowerblue", px=2) # Column index 1 is PMF
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=1) # Keep row highlight subtle
        step_html_table = style_cell(step_html_table, target_cell_pmf_id, color="darkorchid", px=3) # Highlight PMF cell
    elif step == 2: # Highlight Lower CDF cell for k_calc
        step_html_table = style_col(step_html_table, 2, row_range, prefix="binom", color="cornflowerblue", px=2) # Column index 2 is CDF_lo
        step_html_table = style_row(step_html_table, row_prefix, color="tomato", px=1) # Keep row highlight subtle
        step_html_table = style_cell(step_html_table, target_cell_cdf_lo_id, color="darkorchid", px=3) # Highlight CDF cell


    return wrap_table(CSS_BASE, step_html_table)

# ....................................... table wrapper
def binom_table(k_calc: int, n: int, p_h0: float):
    animate(lambda s: build_binom_html(k_calc, n, p_h0, s),
            frames=3, key=f"binom_anim_{k_calc}_{n}_{p_h0}")

# ....................................... APA narrative (Exact Binomial Test)
def binom_apa(k_calc: int, n: int, p_h0: float, alpha: float, tail: str):
    try:
        k_calc = int(k_calc)
        n = int(n)
        p_h0 = float(p_h0)
        alpha = float(alpha)
        if n < 1: raise ValueError("n must be >= 1")
        if not (0 <= p_h0 <= 1): raise ValueError("p H0 must be 0-1")
        if not (0 < alpha < 1): raise ValueError("alpha must be 0-1")
        if not (0 <= k_calc <= n): raise ValueError("k must be 0-n")
    except (ValueError, TypeError):
        st.error("Invalid input values for APA interpretation.")
        return

    p_crit_nominal = alpha # The desired alpha level
    tol = 1e-9 # Tolerance for float comparisons

    # Perform exact binomial test to get p-value using CDF/SF
    if tail == "one-tailed-lower":
        p_calc = stats.binom.cdf(k_calc, n, p_h0)
        compare_str = "less"
        alt_hyp = f"$p < {p_h0}$"
    elif tail == "one-tailed-upper":
        p_calc = stats.binom.sf(k_calc - 1, n, p_h0) # P(X >= k)
        compare_str = "greater"
        alt_hyp = f"$p > {p_h0}$"
    else: # two-tailed (sum of probabilities as or more extreme than observed)
        k_values = np.arange(n + 1)
        pmf_vals = stats.binom.pmf(k_values, n, p_h0)
        observed_pmf = stats.binom.pmf(k_calc, n, p_h0)
        p_calc = np.sum(pmf_vals[pmf_vals <= observed_pmf + tol])
        compare_str = "two-sided"
        alt_hyp = f"$p \\neq {p_h0}$"


    # Get critical values and actual alpha for the rejection region
    crit_lo, crit_hi, actual_alpha = critical_binom(n, p_h0, alpha, tail)

    # Determine rejection based on critical values
    reject_stat = False
    if tail == "one-tailed-lower" and k_calc <= crit_lo: reject_stat = True
    elif tail == "one-tailed-upper" and k_calc >= crit_hi: reject_stat = True
    elif tail == "two-tailed" and (k_calc <= crit_lo or k_calc >= crit_hi): reject_stat = True

    # Determine rejection based on p-value
    reject_p = p_calc <= alpha + tol # Use tolerance for discrete p-value

    # Format critical value string for report
    if tail == "one-tailed-lower": crit_region_str = f"$k \\le {crit_lo}$" if crit_lo >=0 else "No rejection region"
    elif tail == "one-tailed-upper": crit_region_str = f"$k \\ge {crit_hi}$" if crit_hi <= n else "No rejection region"
    else:
        parts = []
        if crit_lo >= 0: parts.append(f"$k \\le {crit_lo}$")
        if crit_hi <= n: parts.append(f"$k \\ge {crit_hi}$")
        crit_region_str = " or ".join(parts) if parts else "No rejection region"

    # Format comparison string based on critical region
    comparison_stat = f"{k_calc} is in rejection region ({crit_region_str})" if reject_stat else f"{k_calc} is not in rejection region ({crit_region_str})"
    comparison_p = f"{p_calc:.4f} \\le {p_crit_nominal:.3f}" if reject_p else f"{p_calc:.4f} > {p_crit_nominal:.3f}$"

    decision = "rejected" if reject_stat else "failed to reject"
    decision_final = "rejected" if reject_p else "failed to reject"

    # Note: Decisions based on critical value vs p-value should align for exact tests.

    st.markdown(f"""
    ##### APA-7 Interpretation (Exact Binomial Test)
    * **Observed successes:** $k = {k_calc}$
    * **Number of trials:** $n = {n}$
    * **Null hypothesis probability:** $p_{{H0}} = {p_h0}$
    * **Alternative hypothesis:** {alt_hyp} ({compare_str})
    * **Calculated *p*-value (exact):** $p = {p_calc:.4f}$
    * **Rejection Region ({tail}, nominal $\\alpha={alpha:.3f}$):** {crit_region_str}
    * **(Actual Type I error rate for region: $\\alpha_{{actual}} \\approx {actual_alpha:.4f}$)**
    * **Nominal Critical *p*-value (alpha):** $\\alpha = {alpha:.3f}$

    * **Decision based on statistic vs. critical region:** $H_0$ is **{decision}** because {comparison_stat}.
    * **Decision based on calculated *p*-value vs. alpha:** $H_0$ is **{decision_final}** because {comparison_p}.

    * **APA-7 Sentence:** An exact binomial test was conducted to examine whether the proportion of successes differed from ${p_h0:.2f}$. The results indicated that the observed proportion ($k={k_calc}, n={n}$) was {'statistically significant' if reject_p else 'not statistically significant'}, $p = {p_calc:.4f}$ ({compare_str}). The null hypothesis was **{decision_final}** at the nominal $\\alpha = {alpha:.3f}$ level.
    """, unsafe_allow_html=True)

# ....................................... tab assembly
def tab_binom():
    st.subheader("Tab 7 • Binomial Distribution & Exact Test")
    st.caption("Shows the Binomial PMF, CDF table, and performs an exact Binomial test.")

    c1, c2 = st.columns(2)
    with c1:
        k_calc_str = st.text_input("Observed successes ($k_{calc}$)", value="8", key="binom_k_str")
        n_str = st.text_input("Number of trials ($n$)", value="10", key="binom_n_str")
    with c2:
        p_h0_str = st.text_input("Null hypothesis probability ($p_{H0}$)", value="0.5", key="binom_p0_str")
        alpha_str = st.text_input("Significance level ($\\alpha$)", value="0.05", key="binom_alpha_str")
        tail = st.radio("Tail(s) / Alternative",
                        options=["two-tailed", "one-tailed-lower", "one-tailed-upper"],
                        format_func=lambda x: {"two-tailed": "Two-sided (p ≠ p₀)",
                                               "one-tailed-lower": "Lower-tailed (p < p₀)",
                                               "one-tailed-upper": "Upper-tailed (p > p₀)"}[x],
                        index=0, key="binom_tail", horizontal=True)


    # Validate inputs
    k_calc, n, p_h0, alpha = None, None, None, None
    valid_inputs = False
    try:
        k_calc = int(k_calc_str)
        n = int(n_str)
        p_h0 = float(p_h0_str)
        alpha = float(alpha_str)
        if not (0 < alpha < 1): raise ValueError("Alpha must be between 0 and 1")
        if not (0 <= p_h0 <= 1): raise ValueError("p H0 must be between 0 and 1")
        if n < 1 : raise ValueError("n must be at least 1")
        if not (0 <= k_calc <= n): raise ValueError(f"k must be between 0 and n={n}")
        valid_inputs = True
    except ValueError as e:
        st.error(f"⚠️ Invalid input: {e}. Please enter valid numbers.")
        # Placeholder
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "Invalid input. Cannot draw plot.", ha='center', va='center', color='red')
        ax.set_title("Binomial Distribution")
        st.pyplot(fig)


    if valid_inputs:
        st.pyplot(plot_binom(k_calc, n, p_h0, alpha, tail))

        with st.expander("Show Binomial probability table animation & Interpretation"):
            binom_table(k_calc, n, p_h0) # Table just shows probs for k_calc
            st.markdown("---")
            # Add APA interpretation based on the test inside the expander
            binom_apa(k_calc, n, p_h0, alpha, tail)


# ════════════════════════════════════════════════════════════════════════════
#  Main App Structure
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(layout="wide", page_title="Statistical Tables Explorer")
st.title("📊 Statistical Tables Explorer")
st.markdown("Interactive plots and animated table lookups for common statistical distributions.")

tab_names = [
    "1: t-Dist",
    "2: z-Dist",
    "3: F-Dist",
    "4: Chi-Sq (χ²)",
    "5: Mann-Whitney U",
    "6: Wilcoxon T",
    "7: Binomial"
]
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_names)

with tab1:
    tab_t()

with tab2:
    tab_z()

with tab3:
    tab_f()

with tab4:
    tab_chi()

with tab5:
    tab_u()

with tab6:
    tab_w()

with tab7:
    tab_binom()

st.markdown("---")
st.caption("App by [Your Name/Institution, Year]. Features: 12x4 plots, animated tables (Row→Col→Cell), APA-7 interpretation, scrolling tables.")
# --------- END OF CODE BLOCK ------------
