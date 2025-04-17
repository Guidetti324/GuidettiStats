import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

plt.switch_backend("Agg")                      # head‑less backend

# ───────────────────────────────  helpers  ──────────────────────────────────

def place_label(ax, placed, x, y, txt, *, color="blue"):
    """Place text, pushing right/up if colliding with previous labels."""
    # Simple vertical offset to avoid overlap, might need refinement
    offset = 0.02 * len(placed)
    ax.text(x, y + offset, txt, color=color,
            ha="center", va="bottom", fontsize=8, clip_on=True)
    placed.append((x, y + offset))

# ---------------------------------------------------------------------------

def style_cell(html: str, cid: str, *, color: str = "red", px: int = 2) -> str:
    """Give one <td id="cid"> a coloured border."""
    return html.replace(
        f'id="{cid}"',
        f'id="{cid}" style="border:{px}px solid {color}; background-color: #ffff99;"', 1)

# ---------------------------------------------------------------------------

def style_row(html: str, row_prefix: str, *, color: str = "red", px: int = 2) -> str:
    """Highlight an entire row by styling its cells."""
    import re
    # Find all cell IDs starting with the row prefix
    cell_ids = re.findall(f'id="({row_prefix}_[^"]+)"', html)
    for cid in cell_ids:
        html = html.replace(
            f'id="{cid}"',
            f'id="{cid}" style="border-top:{px}px solid {color}; border-bottom:{px}px solid {color}; background-color: #ffe0e0;"', 1)
    # Style the row header cell specifically
    header_id = f"{row_prefix}_0"
    html = html.replace(
            f'id="{header_id}"',
            f'id="{header_id}" style="border:{px}px solid {color}; background-color: #ffe0e0;"', 1)

    return html

# ---------------------------------------------------------------------------

def style_col(html: str, col_idx: int, rows: list, prefix: str, *, color: str = "red", px: int = 2) -> str:
    """Highlight an entire column by styling its cells."""
    for r in rows:
        row_val = f"{r:.1f}" if prefix == "z" else str(r) # Handle float formatting for z-table
        cid = f"{prefix}_{row_val}_{col_idx}"
        html = html.replace(
            f'id="{cid}"',
            f'id="{cid}" style="border-left:{px}px solid {color}; border-right:{px}px solid {color}; background-color: #e0e0ff;"', 1)
    # Style the column header cell - requires knowing the header structure, which varies
    # This part is complex due to varying header structures and might be omitted for simplicity,
    # or requires passing header info to this function. For now, skipping header styling.
    return html

# ---------------------------------------------------------------------------

def wrap_table(css: str, table: str) -> str:
    return f"<style>{css}</style><table>{table}</table>"

# ---------------------------------------------------------------------------

def container(html: str, *, height: int = 460) -> str:
    """Scrollable wrapper — does not steal the scroll wheel elsewhere."""
    # Add margin-bottom to avoid cutting off borders
    return f'<div style="overflow:auto; max-height:{height}px; margin-bottom: 5px;">{html}</div>'

# ---------------------------------------------------------------------------

def animate(build_html, frames: int, *, key: str, height: int = 460,
            delay: float = 0.9): # Increased delay slightly
    """
    Display an HTML table animation with sequential highlighting.
      * build_html(step:int) -> html string
      * frames: number of steps (should be 3 for Row -> Col -> Cell)
    """
    if st.button("Show Steps", key=key):
        holder = st.empty()
        for s in range(frames):
            html_content = build_html(s)
            # Wrap the generated HTML table in the scrollable container
            styled_html = container(html_content, height=height)
            holder.markdown(styled_html, unsafe_allow_html=True)
            time.sleep(delay)
        # Optionally keep the final frame visible or clear it
        # holder.empty() # Uncomment to clear after animation
        st.success("Animation complete!")

# ---------------------------------------------------------------------------

CSS_BASE = (
    "table{border-collapse:collapse; margin: 10px 0;}"  # Added margin
    "th,td{border:1px solid #ccc; height:28px; padding: 4px; text-align:center;" # Adjusted padding/height
    "font-family:monospace; font-size:0.85rem; min-width: 50px;}" # Monospace font, min-width
    "th{background:#f0f0f0; font-weight: bold;}" # Bolder header
    "td:first-child {font-weight: bold; background:#f8f8f8;}" # Row header style
)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 • t‑Distribution
# ════════════════════════════════════════════════════════════════════════════

# ....................................... plot
def plot_t(t_calc, df, alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    t_calc = float(t_calc) # Ensure float
    df = int(df)
    alpha = float(alpha)

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
    df = int(df)
    alpha = float(alpha)

    row_range = list(range(max(1, df - 5), df + 6))
    # Common alpha levels for t-tables
    alphas_one = [0.10, 0.05, 0.025, 0.01, 0.005]
    alphas_two = [0.20, 0.10, 0.05, 0.02, 0.01]

    # Find the column index for the given alpha and tail
    target_alpha = alpha
    col_idx = -1
    if tail == "one-tailed":
        if target_alpha in alphas_one:
            col_idx = alphas_one.index(target_alpha) + 1 # +1 for df column
        else:
             # Find closest alpha if exact not present (or handle error)
            target_alpha = min(alphas_one, key=lambda x:abs(x-alpha))
            col_idx = alphas_one.index(target_alpha) + 1
            st.warning(f"Alpha {alpha} not standard for one-tailed, using closest: {target_alpha}", icon="⚠️")
    else: # two-tailed
         if target_alpha in alphas_two:
            col_idx = alphas_two.index(target_alpha) + len(alphas_one) + 1 # Offset by one-tailed cols
         else:
            target_alpha = min(alphas_two, key=lambda x:abs(x-alpha))
            col_idx = alphas_two.index(target_alpha) + len(alphas_one) + 1
            st.warning(f"Alpha {alpha} not standard for two-tailed, using closest: {target_alpha}", icon="⚠️")

    # Build Header
    header_row = '<tr><th rowspan="2">df</th>'
    header_row += f'<th colspan="{len(alphas_one)}">One-Tailed $\\alpha$</th>'
    header_row += f'<th colspan="{len(alphas_two)}">Two-Tailed $\\alpha$</th></tr>'
    header_row += '<tr>' + "".join(f"<th>{a:.3f}</th>" for a in alphas_one)
    header_row += "".join(f"<th>{a:.3f}</th>" for a in alphas_two) + '</tr>'

    # Build Body
    body = ""
    all_cols = alphas_one + alphas_two
    for r in row_range:
        row_html = f'<tr><td id="t_{r}_0">{r}</td>'
        # One-tailed values
        for i, a in enumerate(alphas_one):
            crit_val = stats.t.ppf(1 - a, r)
            row_html += f'<td id="t_{r}_{i+1}">{crit_val:.3f}</td>'
        # Two-tailed values
        for i, a in enumerate(alphas_two):
            crit_val = stats.t.ppf(1 - a/2, r)
            row_html += f'<td id="t_{r}_{len(alphas_one) + i + 1}">{crit_val:.3f}</td>'
        body += row_html + "</tr>"

    html = wrap_table(CSS_BASE, f"{header_row}{body}")

    # Apply sequential highlighting
    row_prefix = f"t_{df}"
    target_cell_id = f"t_{df}_{col_idx}"

    if step == 0: # Highlight Row
        html = style_row(html, row_prefix, color="tomato", px=2)
    elif step == 1: # Highlight Column
        # Need the actual numeric values of the rows being displayed
        html = style_col(html, col_idx, row_range, prefix="t", color="cornflowerblue", px=2)
    elif step == 2: # Highlight Cell
        html = style_cell(html, target_cell_id, color="darkorchid", px=3)

    return html

# ....................................... table wrapper
def t_table(df: int, alpha: float, tail: str):
    animate(lambda s: build_t_html(df, alpha, tail, s),
            frames=3, key=f"t_anim_{df}_{alpha}_{tail}")

# ....................................... APA narrative
def t_apa(t_val: float, df: int, alpha: float, tail: str):
    t_val = float(t_val)
    df = int(df)
    alpha = float(alpha)
    p_crit = alpha # The critical p-value is alpha

    if tail == "one-tailed":
        p_calc = stats.t.sf(t_val, df) # Assumes upper tail test for one-tailed
        crit = stats.t.ppf(1 - alpha, df)
        reject_stat = t_val > crit
        reject_p = p_calc < alpha
        comparison_stat = f"{t_val:.3f} > {crit:.3f}" if reject_stat else f"{t_val:.3f} <= {crit:.3f}"
        comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} >= {p_crit:.3f}"
        crit_str = f"$t_{{crit}}({df}) = {crit:.3f}$"
    else: # two-tailed
        p_calc = stats.t.sf(abs(t_val), df) * 2
        crit = stats.t.ppf(1 - alpha / 2, df)
        reject_stat = abs(t_val) > crit
        reject_p = p_calc < alpha
        comparison_stat = f"$|{t_val:.3f}| > {crit:.3f}$" if reject_stat else f"$|{t_val:.3f}| \\le {crit:.3f}$"
        comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"
        crit_str = f"$t_{{crit}}({df}) = \\pm{crit:.3f}$"


    decision = "rejected" if reject_stat else "failed to reject"
    decision_p = "rejected" if reject_p else "failed to reject" # Should always match

    st.markdown(f"""
    **APA-7 Interpretation**

    * **Calculated statistic:** $t({df}) = {t_val:.3f}$
    * **Calculated *p*-value:** $p = {p_calc:.3f}$
    * **Critical value ({tail}, $\\alpha={alpha:.3f}$):** {crit_str}
    * **Critical *p*-value:** $p_{{crit}} = {alpha:.3f}$

    * **Decision based on statistic:** $H_0$ is **{decision}** because {comparison_stat}.
    * **Decision based on *p*-value:** $H_0$ is **{decision_p}** because {comparison_p}.

    * **APA-7 Sentence:** The analysis revealed a significant effect, $t({df}) = {t_val:.3f}$, $p = {p_calc:.3f}$, using a {tail} test at $\\alpha = {alpha:.3f}$. The null hypothesis was **{decision}**.
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
    try:
        t_val = float(t_val_str)
        df = int(df_str)
        alpha = float(alpha_str)
        if not (0 < alpha < 1): raise ValueError("Alpha must be between 0 and 1")
        if df < 1: raise ValueError("df must be at least 1")
        valid_inputs = True
    except ValueError as e:
        st.error(f"Invalid input: {e}. Please enter valid numbers.")
        valid_inputs = False

    if valid_inputs:
        st.pyplot(plot_t(t_val, df, alpha, tail))

        with st.expander("Show $t$-table lookup steps"):
            t_table(df, alpha, tail)
            t_apa(t_val, df, alpha, tail)
    else:
        st.warning("Please correct the inputs above to proceed.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 • z‑Distribution
# ════════════════════════════════════════════════════════════════════════════

# ....................................... plot
def plot_z(z_calc, alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    z_calc = float(z_calc)
    alpha = float(alpha)

    max_lim = max(4, abs(z_calc) * 1.2)
    xs = np.linspace(-max_lim, max_lim, 500)
    ys = stats.norm.pdf(xs)

    ax.plot(xs, ys, "k", linewidth=1)
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.35, label="Fail to Reject $H_0$")

    labels = [] # To manage label placement

    if tail == "one-tailed":
        crit = stats.norm.ppf(1 - alpha) # Assumes upper tail
        ax.fill_between(xs[xs >= crit], ys[xs >= crit],
                        color="red", alpha=0.40, label="Reject $H_0$")
        ax.axvline(crit, color="green", ls="--", linewidth=1)
        place_label(ax, labels, crit, stats.norm.pdf(crit),
                    f"$z_{{crit}} = {crit:.3f}$", color="green")
        ax.text(crit + 0.1*max_lim, ax.get_ylim()[1]*0.1, f"$\\alpha = {alpha:.3f}$",
                color="red", ha="left", va="bottom", fontsize=9)
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
    z = float(z)
    # Clamp z to typical table range, but allow slightly beyond for display
    z_lookup = np.clip(z, -3.49, 3.49)
    target_row_val = np.floor(z_lookup * 10) / 10
    target_col_val = round(z_lookup - target_row_val, 2)

    # Define the full range of rows and columns for a standard z-table
    all_rows = np.round(np.arange(-3.4, 3.5, 0.1), 1)
    all_cols = np.round(np.arange(0.00, 0.10, 0.01), 2)

    # Find the index of the target row
    try:
        row_idx = np.where(np.isclose(all_rows, target_row_val))[0][0]
    except IndexError:
        # Handle case where target_row_val might be slightly off due to float issues
        row_idx = np.argmin(np.abs(all_rows - target_row_val))
        target_row_val = all_rows[row_idx] # Use the actual value from the array
        st.warning(f"Adjusted target row for z-table lookup to: {target_row_val:.1f}", icon="ℹ️")


    # Select ±10 rows around the target row index
    start_idx = max(0, row_idx - 10)
    end_idx = min(len(all_rows), row_idx + 11) # +11 because slice excludes end
    display_rows = all_rows[start_idx:end_idx]

    # Find the column index
    try:
        col_idx = np.where(np.isclose(all_cols, target_col_val))[0][0] + 1 # +1 for the z-score column
    except IndexError:
        col_idx = np.argmin(np.abs(all_cols - target_col_val)) + 1
        target_col_val = all_cols[col_idx - 1] # Adjust target if needed
        st.warning(f"Adjusted target column for z-table lookup to: {target_col_val:.2f}", icon="ℹ️")


    # Build Header
    header = '<tr><th>z</th>' + "".join(f"<th>{c:.2f}</th>" for c in all_cols) + '</tr>'

    # Build Body
    body = ""
    for r in display_rows:
        row_html = f'<tr><td id="z_{r:.1f}_0">{r:.1f}</td>' # Row header ID includes value
        for i, c in enumerate(all_cols):
            # Calculate cumulative probability (area to the left)
            cell_val = stats.norm.cdf(r + c)
            row_html += f'<td id="z_{r:.1f}_{i+1}">{cell_val:.4f}</td>' # Cell ID includes row value and col index
        body += row_html + "</tr>"

    html = wrap_table(CSS_BASE, f"{header}{body}")

    # Apply sequential highlighting
    row_prefix = f"z_{target_row_val:.1f}" # Prefix includes the numeric value
    target_cell_id = f"z_{target_row_val:.1f}_{col_idx}"

    if step == 0: # Highlight Row
        html = style_row(html, row_prefix, color="tomato", px=2)
    elif step == 1: # Highlight Column
        # Pass the actual row values being displayed
        html = style_col(html, col_idx, display_rows, prefix="z", color="cornflowerblue", px=2)
    elif step == 2: # Highlight Cell
        html = style_cell(html, target_cell_id, color="darkorchid", px=3)

    return html

# ....................................... table wrapper
def z_table(z_val: float):
    # Note: Alpha and tail are not needed for standard z-table lookup itself, only for interpretation
    animate(lambda s: build_z_html(z_val, s),
            frames=3, key=f"z_anim_{z_val}")

# ....................................... APA narrative
def z_apa(z_val: float, alpha: float, tail: str):
    z_val = float(z_val)
    alpha = float(alpha)
    p_crit = alpha

    if tail == "one-tailed":
        p_calc = stats.norm.sf(z_val) # Assumes upper tail test
        crit = stats.norm.ppf(1 - alpha)
        reject_stat = z_val > crit
        reject_p = p_calc < alpha
        comparison_stat = f"{z_val:.3f} > {crit:.3f}" if reject_stat else f"{z_val:.3f} \\le {crit:.3f}$"
        comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"
        crit_str = f"$z_{{crit}} = {crit:.3f}$"
    else: # two-tailed
        p_calc = stats.norm.sf(abs(z_val)) * 2
        crit = stats.norm.ppf(1 - alpha / 2)
        reject_stat = abs(z_val) > crit
        reject_p = p_calc < alpha
        comparison_stat = f"$|{z_val:.3f}| > {crit:.3f}$" if reject_stat else f"$|{z_val:.3f}| \\le {crit:.3f}$"
        comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"
        crit_str = f"$z_{{crit}} = \\pm{crit:.3f}$"

    decision = "rejected" if reject_stat else "failed to reject"
    decision_p = "rejected" if reject_p else "failed to reject" # Should always match

    st.markdown(f"""
    **APA-7 Interpretation**

    * **Calculated statistic:** $z = {z_val:.3f}$
    * **Calculated *p*-value:** $p = {p_calc:.3f}$
    * **Critical value ({tail}, $\\alpha={alpha:.3f}$):** {crit_str}
    * **Critical *p*-value:** $p_{{crit}} = {alpha:.3f}$

    * **Decision based on statistic:** $H_0$ is **{decision}** because {comparison_stat}.
    * **Decision based on *p*-value:** $H_0$ is **{decision_p}** because {comparison_p}.

    * **APA-7 Sentence:** The analysis yielded a significant result, $z = {z_val:.3f}$, $p = {p_calc:.3f}$, using a {tail} test at $\\alpha = {alpha:.3f}$. The null hypothesis was **{decision}**.
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
    try:
        z_val = float(z_val_str)
        alpha = float(alpha_str)
        if not (0 < alpha < 1): raise ValueError("Alpha must be between 0 and 1")
        valid_inputs = True
    except ValueError as e:
        st.error(f"Invalid input: {e}. Please enter valid numbers.")
        valid_inputs = False

    if valid_inputs:
        st.pyplot(plot_z(z_val, alpha, tail))

        with st.expander("Show $z$-table lookup steps (Area to the Left)"):
            # Pass only z_val to the table function
            z_table(z_val)
            # Pass z_val, alpha, tail to the interpretation function
            z_apa(z_val, alpha, tail)
    else:
        st.warning("Please correct the inputs above to proceed.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 • F‑Distribution
# ════════════════════════════════════════════════════════════════════════════

# ....................................... plot
def plot_f(f_calc, df1, df2, alpha):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    f_calc = float(f_calc)
    df1 = int(df1)
    df2 = int(df2)
    alpha = float(alpha)

    crit = stats.f.ppf(1 - alpha, df1, df2)
    max_lim = max(crit * 1.5, f_calc * 1.2, stats.f.ppf(0.999, df1, df2)) # Adjust x-axis limit
    xs = np.linspace(0.001, max_lim, 500) # Start slightly > 0 for pdf
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
    if y_pos_calc < ax.get_ylim()[1] * 0.01:
        y_pos_calc = ax.get_ylim()[1] * 0.05 # Place higher if density is low
    place_label(ax, labels, f_calc, y_pos_calc,
                f"$F_{{calc}} = {f_calc:.3f}$", color="blue")

    ax.text(crit + 0.1*(max_lim-crit), ax.get_ylim()[1]*0.1, f"$\\alpha = {alpha:.3f}$",
                color="red", ha="left", va="bottom", fontsize=9)

    ax.set_xlabel("$F$ value")
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=9)
    ax.set_title(f"$F$-Distribution ($df_1={df1}, df_2={df2}$)", fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=max_lim)
    fig.tight_layout()
    return fig

# ....................................... animated table builder
def build_f_html(df1: int, df2: int, alpha: float, step: int) -> str:
    df1 = int(df1)
    df2 = int(df2)
    alpha = float(alpha)

    # Define ranges around the target df values
    row_range = list(range(max(1, df1 - 5), df1 + 6)) # Rows are df1
    col_range = list(range(max(1, df2 - 5), df2 + 6)) # Columns are df2

    # Find indices for highlighting
    try:
        row_idx = row_range.index(df1) # Relative index within displayed rows
    except ValueError:
        st.error(f"df1={df1} not found in generated row range {row_range}. Check logic.")
        return "Error generating table."
    try:
        col_idx = col_range.index(df2) + 1 # +1 because first col is df1 label
    except ValueError:
        st.error(f"df2={df2} not found in generated column range {col_range}. Check logic.")
        return "Error generating table."

    # Build Header
    header = f'<tr><th>$df_1 \\setminus df_2$</th>' + "".join(f"<th>{c}</th>" for c in col_range) + '</tr>'

    # Build Body
    body = ""
    for r in row_range:
        row_html = f'<tr><td id="f_{r}_0">{r}</td>' # Row header ID includes df1 value
        for i, c in enumerate(col_range):
            # Calculate critical F value for this combination
            crit_val = stats.f.ppf(1 - alpha, r, c)
            row_html += f'<td id="f_{r}_{i+1}">{crit_val:.3f}</td>' # Cell ID includes df1 and col index
        body += row_html + "</tr>"

    html = wrap_table(CSS_BASE, f"{header}{body}")

    # Apply sequential highlighting
    row_prefix = f"f_{df1}" # Row prefix uses df1 value
    target_cell_id = f"f_{df1}_{col_idx}"

    if step == 0: # Highlight Row (df1)
        html = style_row(html, row_prefix, color="tomato", px=2)
    elif step == 1: # Highlight Column (df2)
        html = style_col(html, col_idx, row_range, prefix="f", color="cornflowerblue", px=2)
    elif step == 2: # Highlight Cell
        html = style_cell(html, target_cell_id, color="darkorchid", px=3)

    # Add a title above the table indicating the alpha level
    title = f"<h4>Critical Values for $F$-Distribution at $\\alpha = {alpha:.3f}$</h4>"
    return title + html


# ....................................... table wrapper
def f_table(df1: int, df2: int, alpha: float):
    animate(lambda s: build_f_html(df1, df2, alpha, s),
            frames=3, key=f"f_anim_{df1}_{df2}_{alpha}")

# ....................................... APA narrative
def f_apa(f_val: float, df1: int, df2: int, alpha: float):
    f_val = float(f_val)
    df1 = int(df1)
    df2 = int(df2)
    alpha = float(alpha)
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
    decision_p = "rejected" if reject_p else "failed to reject" # Should always match

    st.markdown(f"""
    **APA-7 Interpretation**

    * **Calculated statistic:** $F({df1}, {df2}) = {f_val:.3f}$
    * **Calculated *p*-value:** $p = {p_calc:.3f}$
    * **Critical value ($\\alpha={alpha:.3f}$):** {crit_str}
    * **Critical *p*-value:** $p_{{crit}} = {alpha:.3f}$

    * **Decision based on statistic:** $H_0$ is **{decision}** because {comparison_stat}.
    * **Decision based on *p*-value:** $H_0$ is **{decision_p}** because {comparison_p}.

    * **APA-7 Sentence:** The results indicated a significant finding, $F({df1}, {df2}) = {f_val:.3f}$, $p = {p_calc:.3f}$, at the $\\alpha = {alpha:.3f}$ significance level. The null hypothesis was **{decision}**.
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
    try:
        f_val = float(f_val_str)
        df1 = int(df1_str)
        df2 = int(df2_str)
        alpha = float(alpha_str)
        if not (0 < alpha < 1): raise ValueError("Alpha must be between 0 and 1")
        if df1 < 1 or df2 < 1: raise ValueError("Degrees of freedom must be at least 1")
        valid_inputs = True
    except ValueError as e:
        st.error(f"Invalid input: {e}. Please enter valid numbers.")
        valid_inputs = False

    if valid_inputs:
        st.pyplot(plot_f(f_val, df1, df2, alpha))

        with st.expander("Show $F$-table lookup steps"):
            f_table(df1, df2, alpha)
            f_apa(f_val, df1, df2, alpha)
    else:
        st.warning("Please correct the inputs above to proceed.")

# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 • Chi‑Square (χ²)
# ════════════════════════════════════════════════════════════════════════════

# ....................................... plot
def plot_chi(chi_calc, df, alpha):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    chi_calc = float(chi_calc)
    df = int(df)
    alpha = float(alpha)

    crit = stats.chi2.ppf(1 - alpha, df)
    max_lim = max(crit * 1.5, chi_calc * 1.2, stats.chi2.ppf(0.999, df)) # Adjust x-axis limit
    xs = np.linspace(0.001, max_lim, 500) # Start slightly > 0
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

    ax.text(crit + 0.1*(max_lim-crit), ax.get_ylim()[1]*0.1, f"$\\alpha = {alpha:.3f}$",
                color="red", ha="left", va="bottom", fontsize=9)

    ax.set_xlabel("$\\chi^2$ value")
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=9)
    ax.set_title(f"$\\chi^2$-Distribution ($df={df}$)", fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=max_lim)
    fig.tight_layout()
    return fig

# ....................................... animated table builder
def build_chi_html(df: int, alpha: float, step: int) -> str:
    df = int(df)
    alpha = float(alpha)

    # Standard alpha levels often found in Chi-square tables
    standard_alphas = [0.995, 0.99, 0.975, 0.95, 0.90, 0.10, 0.05, 0.025, 0.01, 0.005]
    # We are interested in the upper tail critical values, so usually alphas like 0.10, 0.05, 0.01 etc.
    upper_tail_alphas = [a for a in standard_alphas if a <= 0.10]

    # Find the column index for the given alpha
    target_alpha = alpha # This alpha represents the area in the upper tail
    col_idx = -1
    if target_alpha in upper_tail_alphas:
         # Find the index within the *subset* of upper_tail_alphas
         col_idx_in_subset = upper_tail_alphas.index(target_alpha)
         # Find the corresponding index in the *full* standard_alphas list
         col_idx = standard_alphas.index(target_alpha) + 1 # +1 for df column
    else:
        # Find closest alpha if exact not present
        closest_alpha = min(upper_tail_alphas, key=lambda x:abs(x-alpha))
        col_idx_in_subset = upper_tail_alphas.index(closest_alpha)
        col_idx = standard_alphas.index(closest_alpha) + 1
        st.warning(f"Alpha {alpha} not standard for Chi-Square table, using closest upper tail value: {closest_alpha}", icon="⚠️")
        target_alpha = closest_alpha # Use the closest alpha for lookup


    # Define df range around the target df
    row_range = list(range(max(1, df - 5), df + 6))

    # Build Header
    header = '<tr><th>df \\ $\\alpha$</th>' + "".join(f"<th>{a:.3f}</th>" for a in standard_alphas) + '</tr>'

    # Build Body
    body = ""
    for r in row_range:
        row_html = f'<tr><td id="chi_{r}_0">{r}</td>'
        for i, a in enumerate(standard_alphas):
            # Calculate critical Chi-square value (area 1-a to the left)
            crit_val = stats.chi2.ppf(1 - a, r)
            row_html += f'<td id="chi_{r}_{i+1}">{crit_val:.3f}</td>'
        body += row_html + "</tr>"

    html = wrap_table(CSS_BASE, f"{header}{body}")

    # Apply sequential highlighting
    row_prefix = f"chi_{df}"
    target_cell_id = f"chi_{df}_{col_idx}" # Uses the index in the full list

    if step == 0: # Highlight Row
        html = style_row(html, row_prefix, color="tomato", px=2)
    elif step == 1: # Highlight Column
        html = style_col(html, col_idx, row_range, prefix="chi", color="cornflowerblue", px=2)
    elif step == 2: # Highlight Cell
        html = style_cell(html, target_cell_id, color="darkorchid", px=3)

    return html

# ....................................... table wrapper
def chi_table(df: int, alpha: float):
    animate(lambda s: build_chi_html(df, alpha, s),
            frames=3, key=f"chi_anim_{df}_{alpha}")

# ....................................... APA narrative
def chi_apa(chi_val: float, df: int, alpha: float):
    chi_val = float(chi_val)
    df = int(df)
    alpha = float(alpha)
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
    decision_p = "rejected" if reject_p else "failed to reject" # Should always match

    st.markdown(f"""
    **APA-7 Interpretation**

    * **Calculated statistic:** $\\chi^2({df}) = {chi_val:.3f}$
    * **Calculated *p*-value:** $p = {p_calc:.3f}$
    * **Critical value ($\\alpha={alpha:.3f}$):** {crit_str}
    * **Critical *p*-value:** $p_{{crit}} = {alpha:.3f}$

    * **Decision based on statistic:** $H_0$ is **{decision}** because {comparison_stat}.
    * **Decision based on *p*-value:** $H_0$ is **{decision_p}** because {comparison_p}.

    * **APA-7 Sentence:** The Chi-square test indicated a significant result, $\\chi^2({df}) = {chi_val:.3f}$, $p = {p_calc:.3f}$, using a significance level of $\\alpha = {alpha:.3f}$. The null hypothesis was **{decision}**.
    """, unsafe_allow_html=True)

# ....................................... tab assembly
def tab_chi():
    st.subheader("Tab 4 • Chi‑Square ($\\chi^2$) Distribution")
    c1, c2 = st.columns(2)
    with c1:
        chi_val_str = st.text_input("$\\chi^2$ statistic ($\\chi^2_{calc}$)", value="7.88", key="chi_val_str")
        df_str = st.text_input("Degrees of freedom ($df$)", value="3", key="chi_df_str")
    with c2:
        # Use a select box for common alphas, but allow custom input?
        # For simplicity, sticking to select box as in original code
        alpha = st.selectbox("Significance level ($\\alpha$)", [0.10, 0.05, 0.025, 0.01, 0.005],
                             index=1, key="chi_alpha") # Default 0.05

    # Validate inputs
    try:
        chi_val = float(chi_val_str)
        df = int(df_str)
        # alpha is already float from selectbox
        if df < 1: raise ValueError("Degrees of freedom must be at least 1")
        valid_inputs = True
    except ValueError as e:
        st.error(f"Invalid input: {e}. Please enter valid numbers.")
        valid_inputs = False

    if valid_inputs:
        st.pyplot(plot_chi(chi_val, df, alpha))

        with st.expander("Show $\\chi^2$-table lookup steps"):
            chi_table(df, alpha)
            chi_apa(chi_val, df, alpha)
    else:
        st.warning("Please correct the inputs above to proceed.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 • Mann‑Whitney U (using Normal Approximation)
# ════════════════════════════════════════════════════════════════════════════
# Note: Exact tables are complex. This implementation uses the normal approximation
# which is generally suitable for n1, n2 > ~10 or 20. The 'table' shown will
# reflect critical values based on this approximation.

# ....................................... critical U (normal approximation)
def u_crit_approx(n1: int, n2: int, alpha: float, tail: str) -> float:
    """Calculate critical U value using normal approximation."""
    n1, n2 = int(n1), int(n2)
    alpha = float(alpha)
    if n1 <= 0 or n2 <= 0: return np.nan # Avoid division by zero

    mu_U = n1 * n2 / 2
    # Add epsilon to prevent division by zero if n1+n2+1 = 0 (shouldn't happen)
    sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12 + 1e-9)

    if sigma_U == 0: return mu_U # Handle case of zero variance

    if tail == "one-tailed":
        # Typically testing if U is significantly SMALL
        z_crit = stats.norm.ppf(alpha)
        u_critical = mu_U + z_crit * sigma_U
    else: # two-tailed
        z_crit = stats.norm.ppf(alpha / 2) # For lower tail
        # Symmetric, so lower critical U and upper critical U = n1*n2 - lower_crit
        u_critical = mu_U + z_crit * sigma_U

    # Return the critical value (often floor is taken for lookup tables)
    # Here we return the float value for plotting/comparison flexibility
    return u_critical

# ....................................... plot (using normal approximation)
def plot_u(u_calc, n1, n2, alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    u_calc = float(u_calc)
    n1 = int(n1)
    n2 = int(n2)
    alpha = float(alpha)

    if n1 <= 0 or n2 <= 0:
        ax.text(0.5, 0.5, "n1 and n2 must be positive", ha='center', va='center')
        return fig

    mu_U = n1 * n2 / 2
    sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12 + 1e-9) # Epsilon added

    if sigma_U <= 1e-9: # If sigma is effectively zero
         ax.text(0.5, 0.5, "Cannot plot: Standard deviation of U is zero.", ha='center', va='center')
         return fig

    max_lim_hi = mu_U + 4*sigma_U
    max_lim_lo = max(0, mu_U - 4*sigma_U) # U cannot be negative
    xs = np.linspace(max_lim_lo, max_lim_hi, 500)
    ys = stats.norm.pdf(xs, mu_U, sigma_U)

    ax.plot(xs, ys, "k", linewidth=1)
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.35, label="Fail to Reject $H_0$")

    labels = []

    crit_lo = u_crit_approx(n1, n2, alpha, tail) # This is the lower bound for rejection
    crit_hi = n1 * n2 - crit_lo # Upper bound for two-tailed

    if tail == "one-tailed":
        # Mann-Whitney U is significant if U is small (or large, depending on H1)
        # We assume H1 implies small U (e.g., group 1 < group 2)
        ax.fill_between(xs[xs <= crit_lo], ys[xs <= crit_lo],
                        color="red", alpha=0.40, label="Reject $H_0$")
        ax.axvline(crit_lo, color="green", ls="--", linewidth=1)
        place_label(ax, labels, crit_lo, stats.norm.pdf(crit_lo, mu_U, sigma_U),
                    f"$U_{{crit}} = {crit_lo:.3f}$", color="green")
        ax.text(crit_lo - sigma_U*0.5, ax.get_ylim()[1]*0.1, f"$\\alpha = {alpha:.3f}$",
                color="red", ha="right", va="bottom", fontsize=9)
    else: # two-tailed
        ax.fill_between(xs[xs <= crit_lo], ys[xs <= crit_lo], color="red", alpha=0.40)
        ax.fill_between(xs[xs >= crit_hi], ys[xs >= crit_hi], color="red", alpha=0.40,
                        label="Reject $H_0$")
        ax.axvline(crit_lo, color="green", ls="--", linewidth=1)
        ax.axvline(crit_hi, color="green", ls="--", linewidth=1)
        place_label(ax, labels, crit_lo, stats.norm.pdf(crit_lo, mu_U, sigma_U),
                    f"$U_{{crit,lo}} = {crit_lo:.3f}$", color="green")
        place_label(ax, labels, crit_hi, stats.norm.pdf(crit_hi, mu_U, sigma_U),
                    f"$U_{{crit,hi}} = {crit_hi:.3f}$", color="green")
        ax.text(crit_lo - sigma_U*0.5, ax.get_ylim()[1]*0.1, f"$\\alpha/2 = {alpha/2:.3f}$",
                color="red", ha="right", va="bottom", fontsize=9)
        ax.text(crit_hi + sigma_U*0.5, ax.get_ylim()[1]*0.1, f"$\\alpha/2 = {alpha/2:.3f}$",
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
    ax.set_xlim(left=max_lim_lo, right=max_lim_hi)
    fig.tight_layout()
    return fig

# ....................................... animated table builder (using normal approx)
def build_u_html(n1: int, n2: int, alpha: float, tail: str, step: int) -> str:
    n1 = int(n1)
    n2 = int(n2)
    alpha = float(alpha)

    # Define ranges around the target n values
    # Mann-Whitney tables often require n1 <= n2, but we'll display around the inputs
    row_range = list(range(max(1, n1 - 5), n1 + 6)) # Rows are n1
    col_range = list(range(max(1, n2 - 5), n2 + 6)) # Columns are n2

    # Find indices for highlighting
    try:
        row_idx = row_range.index(n1) # Relative index within displayed rows
    except ValueError:
        st.error(f"n1={n1} not found in generated row range {row_range}.")
        return "Error generating table."
    try:
        col_idx = col_range.index(n2) + 1 # +1 because first col is n1 label
    except ValueError:
        st.error(f"n2={n2} not found in generated column range {col_range}.")
        return "Error generating table."

    # Build Header
    header = f'<tr><th>$n_1 \\setminus n_2$</th>' + "".join(f"<th>{c}</th>" for c in col_range) + '</tr>'

    # Build Body
    body = ""
    for r in row_range:
        row_html = f'<tr><td id="u_{r}_0">{r}</td>' # Row header ID includes n1 value
        for i, c in enumerate(col_range):
            # Calculate approximate critical U value (lower tail)
            crit_val = u_crit_approx(r, c, alpha, tail)
            # Display as integer, often floor is used in tables
            display_val = int(np.floor(crit_val)) if not np.isnan(crit_val) else "N/A"
            row_html += f'<td id="u_{r}_{i+1}">{display_val}</td>' # Cell ID includes n1 and col index
        body += row_html + "</tr>"

    html = wrap_table(CSS_BASE, f"{header}{body}")

    # Apply sequential highlighting
    row_prefix = f"u_{n1}" # Row prefix uses n1 value
    target_cell_id = f"u_{n1}_{col_idx}"

    if step == 0: # Highlight Row (n1)
        html = style_row(html, row_prefix, color="tomato", px=2)
    elif step == 1: # Highlight Column (n2)
        html = style_col(html, col_idx, row_range, prefix="u", color="cornflowerblue", px=2)
    elif step == 2: # Highlight Cell
        html = style_cell(html, target_cell_id, color="darkorchid", px=3)

    # Add a title above the table
    tail_desc = "Lower Critical" if tail == "one-tailed" else "Lower/Upper Critical (Approx.)"
    title = f"<h4>Approximate {tail_desc} Values for Mann-Whitney $U$ at $\\alpha = {alpha:.3f}$ ({tail})</h4>"
    return title + html


# ....................................... table wrapper
def u_table(n1: int, n2: int, alpha: float, tail: str):
    animate(lambda s: build_u_html(n1, n2, alpha, tail, s),
            frames=3, key=f"u_anim_{n1}_{n2}_{alpha}_{tail}")

# ....................................... APA narrative (using normal approx)
def u_apa(u_val: int, n1: int, n2: int, alpha: float, tail: str):
    u_val = int(u_val)
    n1 = int(n1)
    n2 = int(n2)
    alpha = float(alpha)
    p_crit = alpha

    mu_U = n1 * n2 / 2
    sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12 + 1e-9)

    # Calculate z-score for the observed U
    # Apply continuity correction (adjust U by 0.5 towards the mean)
    if u_val < mu_U:
        z_calc = (u_val + 0.5 - mu_U) / sigma_U
    elif u_val > mu_U:
        z_calc = (u_val - 0.5 - mu_U) / sigma_U
    else:
        z_calc = 0

    crit_lo = u_crit_approx(n1, n2, alpha, tail) # Lower critical U
    crit_hi = n1 * n2 - crit_lo                  # Upper critical U

    if tail == "one-tailed":
        # Assumes H1 predicts small U (e.g., group 1 ranks lower than group 2)
        p_calc = stats.norm.cdf(z_calc) # Area to the left of z_calc
        reject_stat = u_val <= crit_lo
        reject_p = p_calc < alpha
        comparison_stat = f"{u_val} \\le {crit_lo:.3f}" if reject_stat else f"{u_val} > {crit_lo:.3f}$"
        comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"
        crit_str = f"$U_{{crit}} \\approx {crit_lo:.3f}$ (lower tail)"
    else: # two-tailed
        p_calc = 2 * stats.norm.sf(abs(z_calc)) # 2 * area in the tail away from mean
        reject_stat = (u_val <= crit_lo) or (u_val >= crit_hi)
        reject_p = p_calc < alpha
        comparison_stat = f"({u_val} \\le {crit_lo:.3f}) \\lor ({u_val} \\ge {crit_hi:.3f})" if reject_stat else f"({crit_lo:.3f} < {u_val} < {crit_hi:.3f})$"
        comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"
        crit_str = f"$U_{{crit}} \\approx {crit_lo:.3f}$ (lower) or ${crit_hi:.3f}$ (upper)"

    decision = "rejected" if reject_stat else "failed to reject"
    decision_p = "rejected" if reject_p else "failed to reject" # Should always match

    st.markdown(f"""
    **APA-7 Interpretation (using Normal Approximation)**

    * **Calculated statistic:** $U = {u_val}$
    * **Approximate $z$-statistic:** $z \\approx {z_calc:.3f}$
    * **Calculated *p*-value (approx.):** $p \\approx {p_calc:.3f}$
    * **Critical value(s) ({tail}, $\\alpha={alpha:.3f}$):** {crit_str}
    * **Critical *p*-value:** $p_{{crit}} = {alpha:.3f}$

    * **Decision based on statistic:** $H_0$ is **{decision}** because {comparison_stat}.
    * **Decision based on *p*-value:** $H_0$ is **{decision_p}** because {comparison_p}.

    * **APA-7 Sentence:** A Mann-Whitney U test indicated a significant difference, $U = {u_val}$, $z \\approx {z_calc:.3f}$, $p \\approx {p_calc:.3f}$, using a {tail} test at $\\alpha = {alpha:.3f}$. The null hypothesis was **{decision}**. (Note: Report U, z, and p based on approximation).
    """, unsafe_allow_html=True)

# ....................................... tab assembly
def tab_u():
    st.subheader("Tab 5 • Mann‑Whitney $U$ Test (Normal Approximation)")
    st.caption("Note: Uses normal approximation. Accuracy improves for larger $n_1, n_2$ (e.g., > 10-20). The table shows approximate critical values based on this method.")

    c1, c2 = st.columns(2)
    with c1:
        u_val_str = st.text_input("$U$ statistic ($U_{calc}$)", value="23", key="u_val_str")
        n1_str = st.text_input("$n_1$ (sample size 1)", value="10", key="u_n1_str")
    with c2:
        n2_str = st.text_input("$n_2$ (sample size 2)", value="12", key="u_n2_str")
        alpha_str = st.text_input("Significance level ($\\alpha$)", value="0.05", key="u_alpha_str")
        tail = st.radio("Tail(s)", ["one-tailed", "two-tailed"], index=1, key="u_tail", horizontal=True)

    # Validate inputs
    try:
        u_val = int(u_val_str)
        n1 = int(n1_str)
        n2 = int(n2_str)
        alpha = float(alpha_str)
        if not (0 < alpha < 1): raise ValueError("Alpha must be between 0 and 1")
        if n1 < 1 or n2 < 1: raise ValueError("Sample sizes (n1, n2) must be at least 1")
        if u_val < 0 or u_val > n1*n2: raise ValueError(f"U must be between 0 and {n1*n2}")
        valid_inputs = True
    except ValueError as e:
        st.error(f"Invalid input: {e}. Please enter valid numbers.")
        valid_inputs = False

    if valid_inputs:
        st.pyplot(plot_u(u_val, n1, n2, alpha, tail))

        with st.expander("Show Approx. $U$-table lookup steps"):
            u_table(n1, n2, alpha, tail)
            u_apa(u_val, n1, n2, alpha, tail)
    else:
        st.warning("Please correct the inputs above to proceed.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 • Wilcoxon Signed‑Rank T (using Normal Approximation)
# ════════════════════════════════════════════════════════════════════════════
# Note: Similar to Mann-Whitney U, exact tables are complex. Uses normal approximation,
# suitable for N (non-zero pairs) > ~20. Table reflects approx critical values.

# ....................................... critical T (normal approximation)
def w_crit_approx(n: int, alpha: float, tail: str) -> float:
    """Calculate critical T value using normal approximation."""
    n = int(n)
    alpha = float(alpha)
    if n <= 0: return np.nan

    mu_T = n * (n + 1) / 4
    # Add epsilon to prevent division by zero
    sigma_T = np.sqrt(n * (n + 1) * (2 * n + 1) / 24 + 1e-9)

    if sigma_T == 0: return mu_T

    if tail == "one-tailed":
        # Typically testing if T is significantly SMALL
        z_crit = stats.norm.ppf(alpha)
        t_critical = mu_T + z_crit * sigma_T
    else: # two-tailed
        z_crit = stats.norm.ppf(alpha / 2) # For lower tail
        # Symmetric: lower critical T, upper critical T = n(n+1)/2 - lower_crit
        t_critical = mu_T + z_crit * sigma_T

    return t_critical

# ....................................... plot (using normal approximation)
def plot_w(t_calc, n, alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    t_calc = float(t_calc) # T can be float if using approximation, but usually integer
    n = int(n)
    alpha = float(alpha)

    if n <= 0:
        ax.text(0.5, 0.5, "N must be positive", ha='center', va='center')
        return fig

    mu_T = n * (n + 1) / 4
    sigma_T = np.sqrt(n * (n + 1) * (2 * n + 1) / 24 + 1e-9) # Epsilon added

    if sigma_T <= 1e-9:
         ax.text(0.5, 0.5, "Cannot plot: Standard deviation of T is zero.", ha='center', va='center')
         return fig

    max_lim_hi = mu_T + 4 * sigma_T
    max_lim_lo = max(0, mu_T - 4 * sigma_T) # T cannot be negative
    xs = np.linspace(max_lim_lo, max_lim_hi, 500)
    ys = stats.norm.pdf(xs, mu_T, sigma_T)

    ax.plot(xs, ys, "k", linewidth=1)
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.35, label="Fail to Reject $H_0$")

    labels = []

    crit_lo = w_crit_approx(n, alpha, tail) # Lower bound critical T
    max_T = n * (n + 1) / 2
    crit_hi = max_T - crit_lo              # Upper bound critical T

    if tail == "one-tailed":
        # Wilcoxon T is significant if T is small (sum of ranks of positive/negative diffs)
        ax.fill_between(xs[xs <= crit_lo], ys[xs <= crit_lo],
                        color="red", alpha=0.40, label="Reject $H_0$")
        ax.axvline(crit_lo, color="green", ls="--", linewidth=1)
        place_label(ax, labels, crit_lo, stats.norm.pdf(crit_lo, mu_T, sigma_T),
                    f"$T_{{crit}} \\approx {crit_lo:.3f}$", color="green")
        ax.text(crit_lo - sigma_T*0.5, ax.get_ylim()[1]*0.1, f"$\\alpha = {alpha:.3f}$",
                color="red", ha="right", va="bottom", fontsize=9)
    else: # two-tailed
        ax.fill_between(xs[xs <= crit_lo], ys[xs <= crit_lo], color="red", alpha=0.40)
        ax.fill_between(xs[xs >= crit_hi], ys[xs >= crit_hi], color="red", alpha=0.40,
                        label="Reject $H_0$")
        ax.axvline(crit_lo, color="green", ls="--", linewidth=1)
        ax.axvline(crit_hi, color="green", ls="--", linewidth=1)
        place_label(ax, labels, crit_lo, stats.norm.pdf(crit_lo, mu_T, sigma_T),
                    f"$T_{{crit,lo}} \\approx {crit_lo:.3f}$", color="green")
        place_label(ax, labels, crit_hi, stats.norm.pdf(crit_hi, mu_T, sigma_T),
                    f"$T_{{crit,hi}} \\approx {crit_hi:.3f}$", color="green")
        ax.text(crit_lo - sigma_T*0.5, ax.get_ylim()[1]*0.1, f"$\\alpha/2 = {alpha/2:.3f}$",
                color="red", ha="right", va="bottom", fontsize=9)
        ax.text(crit_hi + sigma_T*0.5, ax.get_ylim()[1]*0.1, f"$\\alpha/2 = {alpha/2:.3f}$",
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
    ax.set_xlim(left=max_lim_lo, right=max_lim_hi)
    fig.tight_layout()
    return fig


# ....................................... animated table builder (using normal approx)
def build_w_html(n: int, alpha: float, tail: str, step: int) -> str:
    n = int(n)
    alpha = float(alpha)

    # Define N range around the target N
    row_range = list(range(max(5, n - 5), n + 6)) # Rows are N (min N usually ~5-8 for tables)

    # Define common alpha levels for columns
    standard_alphas_one = [0.05, 0.025, 0.01, 0.005]
    standard_alphas_two = [0.10, 0.05, 0.02, 0.01]

    # Find column index based on alpha and tail
    target_alpha = alpha
    col_idx = -1
    alpha_headers = []

    if tail == "one-tailed":
        alpha_headers = standard_alphas_one
        if target_alpha in alpha_headers:
            col_idx = alpha_headers.index(target_alpha) + 1
        else:
            closest_alpha = min(alpha_headers, key=lambda x: abs(x - alpha))
            col_idx = alpha_headers.index(closest_alpha) + 1
            st.warning(f"Alpha {alpha} not standard for one-tailed Wilcoxon, using closest: {closest_alpha}", icon="⚠️")
            target_alpha = closest_alpha
    else: # two-tailed
        alpha_headers = standard_alphas_two
        if target_alpha in alpha_headers:
            col_idx = alpha_headers.index(target_alpha) + 1
        else:
            closest_alpha = min(alpha_headers, key=lambda x: abs(x - alpha))
            col_idx = alpha_headers.index(closest_alpha) + 1
            st.warning(f"Alpha {alpha} not standard for two-tailed Wilcoxon, using closest: {closest_alpha}", icon="⚠️")
            target_alpha = closest_alpha

    # Build Header
    header = f'<tr><th>N \\ $\\alpha$ ({tail})</th>' + "".join(f"<th>{a:.3f}</th>" for a in alpha_headers) + '</tr>'

    # Build Body
    body = ""
    for r_n in row_range:
        row_html = f'<tr><td id="w_{r_n}_0">{r_n}</td>' # Row header ID includes N value
        for i, a_col in enumerate(alpha_headers):
            # Calculate approximate critical T value (lower tail) for this N and alpha
            crit_val = w_crit_approx(r_n, a_col, tail)
            # Display as integer (floor is common)
            display_val = int(np.floor(crit_val)) if not np.isnan(crit_val) else "N/A"
            row_html += f'<td id="w_{r_n}_{i+1}">{display_val}</td>' # Cell ID includes N and col index
        body += row_html + "</tr>"

    html = wrap_table(CSS_BASE, f"{header}{body}")

    # Apply sequential highlighting
    row_prefix = f"w_{n}" # Row prefix uses N value
    target_cell_id = f"w_{n}_{col_idx}"

    if step == 0: # Highlight Row (N)
        html = style_row(html, row_prefix, color="tomato", px=2)
    elif step == 1: # Highlight Column (alpha)
        html = style_col(html, col_idx, row_range, prefix="w", color="cornflowerblue", px=2)
    elif step == 2: # Highlight Cell
        html = style_cell(html, target_cell_id, color="darkorchid", px=3)

    # Add a title above the table
    tail_desc = "Lower Critical" if tail == "one-tailed" else "Lower/Upper Critical (Approx.)"
    title = f"<h4>Approximate {tail_desc} Values for Wilcoxon $T$ at Various $\\alpha$ ({tail})</h4>"
    return title + html


# ....................................... table wrapper
def w_table(n: int, alpha: float, tail: str):
    animate(lambda s: build_w_html(n, alpha, tail, s),
            frames=3, key=f"w_anim_{n}_{alpha}_{tail}")

# ....................................... APA narrative (using normal approx)
def w_apa(t_val: int, n: int, alpha: float, tail: str):
    t_val = int(t_val)
    n = int(n)
    alpha = float(alpha)
    p_crit = alpha

    mu_T = n * (n + 1) / 4
    sigma_T = np.sqrt(n * (n + 1) * (2 * n + 1) / 24 + 1e-9)

    # Calculate z-score for the observed T with continuity correction
    if t_val < mu_T:
        z_calc = (t_val + 0.5 - mu_T) / sigma_T
    elif t_val > mu_T:
        z_calc = (t_val - 0.5 - mu_T) / sigma_T
    else:
        z_calc = 0

    crit_lo = w_crit_approx(n, alpha, tail) # Lower critical T
    max_T = n * (n + 1) / 2
    crit_hi = max_T - crit_lo              # Upper critical T

    if tail == "one-tailed":
        # Assumes H1 predicts small T
        p_calc = stats.norm.cdf(z_calc) # Area to the left
        reject_stat = t_val <= crit_lo
        reject_p = p_calc < alpha
        comparison_stat = f"{t_val} \\le {crit_lo:.3f}" if reject_stat else f"{t_val} > {crit_lo:.3f}$"
        comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"
        crit_str = f"$T_{{crit}} \\approx {crit_lo:.3f}$ (lower tail)"
    else: # two-tailed
        p_calc = 2 * stats.norm.sf(abs(z_calc)) # 2 * area in tail
        reject_stat = (t_val <= crit_lo) or (t_val >= crit_hi)
        reject_p = p_calc < alpha
        comparison_stat = f"({t_val} \\le {crit_lo:.3f}) \\lor ({t_val} \\ge {crit_hi:.3f})" if reject_stat else f"({crit_lo:.3f} < {t_val} < {crit_hi:.3f})$"
        comparison_p = f"{p_calc:.3f} < {p_crit:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit:.3f}$"
        crit_str = f"$T_{{crit}} \\approx {crit_lo:.3f}$ (lower) or ${crit_hi:.3f}$ (upper)"

    decision = "rejected" if reject_stat else "failed to reject"
    decision_p = "rejected" if reject_p else "failed to reject" # Should always match

    st.markdown(f"""
    **APA-7 Interpretation (using Normal Approximation)**

    * **Calculated statistic:** $T = {t_val}$
    * **Number of pairs (non-zero diffs):** $N = {n}$
    * **Approximate $z$-statistic:** $z \\approx {z_calc:.3f}$
    * **Calculated *p*-value (approx.):** $p \\approx {p_calc:.3f}$
    * **Critical value(s) ({tail}, $\\alpha={alpha:.3f}$):** {crit_str}
    * **Critical *p*-value:** $p_{{crit}} = {alpha:.3f}$

    * **Decision based on statistic:** $H_0$ is **{decision}** because {comparison_stat}.
    * **Decision based on *p*-value:** $H_0$ is **{decision_p}** because {comparison_p}.

    * **APA-7 Sentence:** A Wilcoxon signed-rank test indicated a significant result, $T = {t_val}$, $z \\approx {z_calc:.3f}$, $p \\approx {p_calc:.3f}$, based on $N={n}$ pairs, using a {tail} test at $\\alpha = {alpha:.3f}$. The null hypothesis was **{decision}**. (Note: Report T, N, z, and p based on approximation).
    """, unsafe_allow_html=True)


# ....................................... tab assembly
def tab_w():
    st.subheader("Tab 6 • Wilcoxon Signed‑Rank $T$ Test (Normal Approx.)")
    st.caption("Note: Uses normal approximation. Accuracy improves for larger $N$ (e.g., > 20). The table shows approximate critical values based on this method.")

    c1, c2 = st.columns(2)
    with c1:
        t_val_str = st.text_input("$T$ statistic ($T_{calc}$)", value="15", key="w_val_str")
        n_str = st.text_input("$N$ (non‑zero differences)", value="12", key="w_n_str")
    with c2:
        alpha_str = st.text_input("Significance level ($\\alpha$)", value="0.05", key="w_alpha_str")
        tail = st.radio("Tail(s)", ["one-tailed", "two-tailed"], index=1, key="w_tail", horizontal=True)

    # Validate inputs
    try:
        t_val = int(t_val_str)
        n = int(n_str)
        alpha = float(alpha_str)
        max_T = n * (n + 1) / 2
        if not (0 < alpha < 1): raise ValueError("Alpha must be between 0 and 1")
        if n < 1 : raise ValueError("N must be at least 1")
        if t_val < 0 or t_val > max_T: raise ValueError(f"T must be between 0 and {max_T:.0f}")
        valid_inputs = True
    except ValueError as e:
        st.error(f"Invalid input: {e}. Please enter valid numbers.")
        valid_inputs = False

    if valid_inputs:
        st.pyplot(plot_w(t_val, n, alpha, tail))

        with st.expander("Show Approx. $T$-table lookup steps"):
            w_table(n, alpha, tail)
            w_apa(t_val, n, alpha, tail)
    else:
        st.warning("Please correct the inputs above to proceed.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 7 • Binomial Distribution
# ════════════════════════════════════════════════════════════════════════════

# ....................................... critical bounds (exact)
def critical_binom(n: int, p_h0: float, alpha: float, tail: str):
    """Return (lower, upper) rejection bounds for EXACT binomial test."""
    n = int(n)
    p_h0 = float(p_h0)
    alpha = float(alpha)

    k_values = np.arange(n + 1)
    pmf = stats.binom.pmf(k_values, n, p_h0)
    cdf = stats.binom.cdf(k_values, n, p_h0)

    if tail == "one-tailed-lower": # Reject if k is too small
        try:
            # Find largest k such that P(X <= k) <= alpha
            k_crit_lo = k_values[cdf <= alpha][-1]
        except IndexError:
            k_crit_lo = -1 # No rejection region if even k=0 is too likely
        k_crit_hi = n + 1 # Effectively infinity for lower-tailed test
        actual_alpha = cdf[k_crit_lo] if k_crit_lo >=0 else 0
    elif tail == "one-tailed-upper": # Reject if k is too large
        try:
            # Find smallest k such that P(X >= k) <= alpha
            # P(X >= k) = 1 - P(X <= k-1)
            sf = 1 - stats.binom.cdf(k_values - 1, n, p_h0)
            k_crit_hi = k_values[sf <= alpha][0]
        except IndexError:
            k_crit_hi = n + 1 # No rejection region if even k=n is too likely
        k_crit_lo = -1 # Effectively -infinity for upper-tailed test
        actual_alpha = sf[k_crit_hi] if k_crit_hi <= n else 0
    else: # two-tailed (symmetric probability method)
        # Find lower k such that P(X <= k_lo) <= alpha/2
        try: k_crit_lo = k_values[cdf <= alpha / 2][-1]
        except IndexError: k_crit_lo = -1
        # Find upper k such that P(X >= k_hi) <= alpha/2
        try:
            sf = 1 - stats.binom.cdf(k_values - 1, n, p_h0)
            k_crit_hi = k_values[sf <= alpha / 2][0]
        except IndexError: k_crit_hi = n + 1
        # Calculate actual alpha achieved
        alpha_lo = cdf[k_crit_lo] if k_crit_lo >= 0 else 0
        alpha_hi = sf[k_crit_hi] if k_crit_hi <= n else 0
        actual_alpha = alpha_lo + alpha_hi

    return k_crit_lo, k_crit_hi, actual_alpha # Return bounds and actual alpha

# ....................................... plot
def plot_binom(k_calc, n, p_h0, alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    k_calc = int(k_calc)
    n = int(n)
    p_h0 = float(p_h0)
    alpha = float(alpha)

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
        if tail == "one-tailed-lower" and k <= crit_lo: is_reject = True
        elif tail == "one-tailed-upper" and k >= crit_hi: is_reject = True
        elif tail == "two-tailed" and (k <= crit_lo or k >= crit_hi): is_reject = True

        if is_reject:
            label = "Reject $H_0$" if not reject_label_added else ""
            ax.bar(k, pmf[k], color=reject_color, alpha=0.7, label=label)
            reject_label_added = True

    # Add lines for critical values (if they exist within 0..n)
    if tail == "one-tailed-lower" and crit_lo >= 0:
        ax.axvline(crit_lo + 0.5, color="green", ls="--", linewidth=1) # Place between bars
        place_label(ax, labels, crit_lo + 0.5, max(pmf)*0.8, f"$k_{{crit}} = {crit_lo}$", color="green")
        ax.text(crit_lo - 1, max(pmf)*0.1, f"Actual $\\alpha \\le {alpha:.3f}$",
                color="red", ha="right", va="bottom", fontsize=9)
    elif tail == "one-tailed-upper" and crit_hi <= n:
        ax.axvline(crit_hi - 0.5, color="green", ls="--", linewidth=1)
        place_label(ax, labels, crit_hi - 0.5, max(pmf)*0.8, f"$k_{{crit}} = {crit_hi}$", color="green")
        ax.text(crit_hi + 1, max(pmf)*0.1, f"Actual $\\alpha \\le {alpha:.3f}$",
                color="red", ha="left", va="bottom", fontsize=9)
    elif tail == "two-tailed":
        if crit_lo >= 0:
             ax.axvline(crit_lo + 0.5, color="green", ls="--", linewidth=1)
             place_label(ax, labels, crit_lo + 0.5, max(pmf)*0.8, f"$k_{{crit,lo}} = {crit_lo}$", color="green")
             ax.text(crit_lo - 1, max(pmf)*0.1, f"$\\alpha/2$", color="red", ha="right", va="bottom", fontsize=9)
        if crit_hi <= n:
             ax.axvline(crit_hi - 0.5, color="green", ls="--", linewidth=1)
             place_label(ax, labels, crit_hi - 0.5, max(pmf)*0.8, f"$k_{{crit,hi}} = {crit_hi}$", color="green")
             ax.text(crit_hi + 1, max(pmf)*0.1, f"$\\alpha/2$", color="red", ha="left", va="bottom", fontsize=9)

    # Add line for calculated k
    ax.axvline(k_calc, color="blue", ls="-", linewidth=1.5)
    place_label(ax, labels, k_calc, pmf[k_calc], f"$k_{{calc}} = {k_calc}$", color="blue")

    ax.set_xlabel("Number of Successes ($k$)")
    ax.set_ylabel("Probability Mass $P(X=k)$")
    ax.legend(fontsize=9)
    ax.set_title(f"Binomial Distribution ($n={n}, p_{{H0}}={p_h0}$), {tail}", fontsize=12)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.set_xticks(k_values) # Show ticks for all possible k values
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig

# ....................................... animated table builder (Binomial CDF/PMF)
def build_binom_html(k_calc: int, n: int, p_h0: float, step: int) -> str:
    n = int(n)
    p_h0 = float(p_h0)
    k_calc = int(k_calc) # The observed number of successes to highlight

    # Define k range (show all possible outcomes 0 to n)
    row_range = list(range(n + 1)) # Rows are k values

    # Find indices for highlighting
    try:
        row_idx = row_range.index(k_calc) # Relative index within displayed rows
    except ValueError:
        st.error(f"k_calc={k_calc} is out of range for n={n}.")
        return "Error generating table."

    # Build Header (showing k, PMF, CDF Lower, CDF Upper)
    header = (f'<tr><th>$k$</th><th>PMF $P(X=k)$</th>'
              f'<th>CDF $P(X \\le k)$</th><th>Upper CDF $P(X \\ge k)$</th></tr>')

    # Build Body
    body = ""
    for r_k in row_range:
        pmf_val = stats.binom.pmf(r_k, n, p_h0)
        cdf_lo_val = stats.binom.cdf(r_k, n, p_h0)
        cdf_hi_val = stats.binom.sf(r_k - 1, n, p_h0) # P(X >= k) = 1 - P(X <= k-1)

        # Cell IDs: binom_kValue_colIndex (0=k, 1=PMF, 2=CDF_lo, 3=CDF_hi)
        row_html = f'<tr><td id="binom_{r_k}_0">{r_k}</td>'
        row_html += f'<td id="binom_{r_k}_1">{pmf_val:.4f}</td>'
        row_html += f'<td id="binom_{r_k}_2">{cdf_lo_val:.4f}</td>'
        row_html += f'<td id="binom_{r_k}_3">{cdf_hi_val:.4f}</td>'
        body += row_html + "</tr>"

    html = wrap_table(CSS_BASE, f"{header}{body}")

    # Apply sequential highlighting to the row corresponding to k_calc
    # Step 0: Highlight row
    # Step 1: Highlight PMF cell
    # Step 2: Highlight appropriate CDF cell (depends on interest, e.g., lower CDF)
    row_prefix = f"binom_{k_calc}"
    target_cell_pmf_id = f"binom_{k_calc}_1"
    target_cell_cdf_lo_id = f"binom_{k_calc}_2"
    # target_cell_cdf_hi_id = f"binom_{k_calc}_3" # Could highlight this too/instead

    if step == 0: # Highlight Row for k_calc
        html = style_row(html, row_prefix, color="tomato", px=2)
    elif step == 1: # Highlight PMF cell for k_calc
        html = style_cell(html, target_cell_pmf_id, color="cornflowerblue", px=3)
    elif step == 2: # Highlight Lower CDF cell for k_calc
        html = style_cell(html, target_cell_cdf_lo_id, color="darkorchid", px=3)

    # Add a title
    title = f"<h4>Binomial Probabilities for $n={n}, p_{{H0}}={p_h0}$</h4>"
    return title + html

# ....................................... table wrapper
def binom_table(k_calc: int, n: int, p_h0: float):
    animate(lambda s: build_binom_html(k_calc, n, p_h0, s),
            frames=3, key=f"binom_anim_{k_calc}_{n}_{p_h0}")

# ....................................... APA narrative (Exact Binomial Test)
def binom_apa(k_calc: int, n: int, p_h0: float, alpha: float, tail: str):
    k_calc = int(k_calc)
    n = int(n)
    p_h0 = float(p_h0)
    alpha = float(alpha)
    p_crit_nominal = alpha # The desired alpha level

    # Perform exact binomial test to get p-value
    if tail == "one-tailed-lower":
        p_calc = stats.binom_test(k_calc, n, p_h0, alternative='less')
        compare_str = "less"
    elif tail == "one-tailed-upper":
        p_calc = stats.binom_test(k_calc, n, p_h0, alternative='greater')
        compare_str = "greater"
    else: # two-tailed
        p_calc = stats.binom_test(k_calc, n, p_h0, alternative='two-sided')
        compare_str = "two-sided"

    # Get critical values and actual alpha
    crit_lo, crit_hi, actual_alpha = critical_binom(n, p_h0, alpha, tail)

    # Determine rejection based on critical values
    reject_stat = False
    if tail == "one-tailed-lower" and k_calc <= crit_lo: reject_stat = True
    elif tail == "one-tailed-upper" and k_calc >= crit_hi: reject_stat = True
    elif tail == "two-tailed" and (k_calc <= crit_lo or k_calc >= crit_hi): reject_stat = True

    # Determine rejection based on p-value
    reject_p = p_calc < alpha # Compare calculated p against nominal alpha

    # Format critical value string
    if tail == "one-tailed-lower": crit_str = f"$k \\le {crit_lo}$" if crit_lo >=0 else "None"
    elif tail == "one-tailed-upper": crit_str = f"$k \\ge {crit_hi}$" if crit_hi <= n else "None"
    else: crit_str = f"$k \\le {crit_lo}$ or $k \\ge {crit_hi}$"

    # Format comparison string
    comparison_stat = f"{k_calc} in rejection region ({crit_str})" if reject_stat else f"{k_calc} not in rejection region ({crit_str})"
    comparison_p = f"{p_calc:.3f} < {p_crit_nominal:.3f}" if reject_p else f"{p_calc:.3f} \\ge {p_crit_nominal:.3f}$"

    decision = "rejected" if reject_stat else "failed to reject"
    decision_p = "rejected" if reject_p else "failed to reject"

    # Note: Decisions based on critical value vs p-value might differ slightly
    # for discrete distributions depending on how p-value/critical value are defined.
    # Using p < alpha is standard.

    st.markdown(f"""
    **APA-7 Interpretation (Exact Binomial Test)**

    * **Observed successes:** $k = {k_calc}$
    * **Number of trials:** $n = {n}$
    * **Null hypothesis probability:** $p_{{H0}} = {p_h0}$
    * **Calculated *p*-value (exact, {compare_str}):** $p = {p_calc:.3f}$
    * **Critical value(s) / Rejection Region ({tail}, nominal $\\alpha={alpha:.3f}$):** {crit_str}
    * **(Actual $\\alpha \\approx {actual_alpha:.4f}$ for critical region)**
    * **Critical *p*-value (Nominal):** $p_{{crit}} = {alpha:.3f}$

    * **Decision based on statistic:** $H_0$ is **{decision}** because {comparison_stat}.
    * **Decision based on *p*-value:** $H_0$ is **{decision_p}** because {comparison_p}.

    * **APA-7 Sentence:** An exact binomial test indicated that the observed number of successes ($k={k_calc}, n={n}$) was significantly different from the expected probability under the null hypothesis ($p_{{H0}}={p_h0}$), $p = {p_calc:.3f}$, using a {tail} test at nominal $\\alpha = {alpha:.3f}$. The null hypothesis was **{decision_p}**.
    """, unsafe_allow_html=True)

# ....................................... tab assembly
def tab_binom():
    st.subheader("Tab 7 • Binomial Distribution & Test")
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
                        index=0, key="binom_tail")


    # Validate inputs
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
        st.error(f"Invalid input: {e}. Please enter valid numbers.")
        valid_inputs = False

    if valid_inputs:
        st.pyplot(plot_binom(k_calc, n, p_h0, alpha, tail))

        with st.expander("Show Binomial probability table"):
            binom_table(k_calc, n, p_h0) # Table just shows probs for k_calc

        # Separate expander for the APA interpretation based on the test
        with st.expander("Show APA Interpretation (Exact Binomial Test)"):
            binom_apa(k_calc, n, p_h0, alpha, tail)
    else:
        st.warning("Please correct the inputs above to proceed.")


# ════════════════════════════════════════════════════════════════════════════
#  Main App Structure
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(layout="wide")
st.title("PSYC-250 Statistical Tables Explorer")
st.markdown("Interactive plots and animated table lookups for common statistical distributions.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1: t-Distribution",
    "2: z-Distribution",
    "3: F-Distribution",
    "4: Chi-Square (χ²)",
    "5: Mann-Whitney U",
    "6: Wilcoxon T",
    "7: Binomial"
])

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
st.caption("App updated based on requirements. Plots are 12x4 inches. Table animations highlight Row -> Column -> Cell sequentially. z-table shows ±10 rows around target. APA-7 blocks included below tables. Scrolling enabled via containers.")
```
