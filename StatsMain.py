###############################################################################
#  PSYC‑250  ‑‑  Statistical Tables Explorer  (Streamlit, 12 × 4 figures)
#  ---------------------------------------------------------------------------
#  Tabs:
#     1) t‑Distribution              5) Mann‑Whitney U
#     2) z‑Distribution              6) Wilcoxon Signed‑Rank T
#     3) F‑Distribution              7) Binomial
#     4) Chi‑Square
#
#  Features (all tabs):
#     • 12 × 4 Matplotlib plot
#     • Step‑wise HTML lookup table
#     • Detailed APA‑7 narrative (calculated / critical stat & p, decisions)
#
#  v2025‑04‑17‑z‑table‑10row
###############################################################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.switch_backend("Agg")     # head‑less backend

# ──────────────────────────────────  helpers  ───────────────────────────────
def place_label(ax, placed, x, y, txt, *, color="blue"):
    dx = dy = 0.0
    for xx, yy in placed:
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05:
            dx += 0.06; dy += 0.04
    ax.text(x + dx, y + dy, txt, color=color,
            ha="left", va="bottom", fontsize=8, clip_on=True)
    placed.append((x + dx, y + dy))

def style_cell(html, cell_id, *, color="red", px=2):
    return html.replace(f'id="{cell_id}"',
                        f'id="{cell_id}" style="border:{px}px solid {color};"', 1)

def iframe(html, *, height=460):
    st.markdown(f'<div style="overflow:auto;max-height:{height}px;">{html}</div>',
                unsafe_allow_html=True)

def next_button(step_key):
    if st.button("Next Step", key=f"{step_key}__btn"):
        st.session_state[step_key] += 1

def wrap(css, body):
    return f"<style>{css}</style><table>{body}</table>"

CSS = ("table{border-collapse:collapse}"
       "th,td{border:1px solid #000;height:30px;text-align:center;"
       "font-family:sans-serif}")

# ─────────────────────────────  TAB 1 • t‑Distribution  ─────────────────────
def plot_t(t_calc, df, alpha, tail):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    xs = np.linspace(-4,4,400)
    ys = stats.t.pdf(xs, df)
    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H₀")
    labels=[]
    if tail.startswith("one"):
        crit=stats.t.ppf(1-alpha, df)
        ax.fill_between(xs[xs>=crit],ys[xs>=crit],color="red",alpha=0.30,
                        label="Reject H₀")
        ax.axvline(crit,color="green",ls="--")
        place_label(ax,labels,crit,stats.t.pdf(crit,df)+.02,
                    f"t₍crit₎={crit:.2f}",color="green")
    else:
        crit=stats.t.ppf(1-alpha/2, df)
        ax.fill_between(xs[xs>= crit],ys[xs>= crit],color="red",alpha=0.30)
        ax.fill_between(xs[xs<=-crit],ys[xs<=-crit],color="red",alpha=0.30,
                        label="Reject H₀")
        ax.axvline( crit,color="green",ls="--")
        ax.axvline(-crit,color="green",ls="--")
        place_label(ax,labels, crit,stats.t.pdf( crit,df)+.02,
                    f"+t₍crit₎={crit:.2f}",color="green")
        place_label(ax,labels,-crit,stats.t.pdf(-crit,df)+.02,
                    f"–t₍crit₎={crit:.2f}",color="green")
    ax.axvline(t_calc,color="blue",ls="--")
    place_label(ax,labels,t_calc,stats.t.pdf(t_calc,df)+.02,
                f"t₍calc₎={t_calc:.2f}",color="blue")
    ax.set_xlabel("t"); ax.set_ylabel("Density"); ax.legend()
    ax.set_title("t‑Distribution"); fig.tight_layout(); return fig

def t_table(df, alpha, tail, key):
    step_key=f"{key}_step"; step=st.session_state.setdefault(step_key,-1)
    rows=list(range(max(1,df-5),df+6))
    heads=[("one",0.10),("one",0.05),("one",0.01),("one",0.001),
           ("two",0.10),("two",0.05),("two",0.01),("two",0.001)]
    head="".join(f"<th>{m}_{a}</th>" for m,a in heads)
    body=""
    for r in rows:
        body+=f'<tr><td id="t_{r}_0">{r}</td>'
        for i,(m,a) in enumerate(heads,start=1):
            body+=f'<td id="t_{r}_{i}">{stats.t.ppf(1-a/(1 if m=="one" else 2),r):.2f}</td>'
        body+="</tr>"
    html=wrap(f"{CSS}th{{background:#fafafa}}",
              f"<tr><th>df</th>{head}</tr>{body}")
    mode="one" if tail.startswith("one") else "two"
    col_idx=[i for i,(m,a) in enumerate(heads,start=1)
             if m==mode and np.isclose(a,alpha)][0]
    if step>=0:
        for i in range(len(heads)+1):
            html=style_cell(html,f"t_{df}_{i}")
    if step>=1:
        for r in rows:
            html=style_cell(html,f"t_{r}_{col_idx}")
    if step>=2:
        html=style_cell(html,f"t_{df}_{col_idx}",color="blue",px=3)
    iframe(html)
    msgs=["Highlight df row","Highlight α/tail column","Intersection → t₍crit₎"]
    if step<0: st.write("Click **Next Step** to begin.")
    elif step>=len(msgs): st.write("All steps complete!")
    else: st.write(f"**Step {step+1}**: {msgs[step]}")
    next_button(step_key)

    t_val=st.session_state["t_val"]
    if tail.startswith("one"):
        p_calc=stats.t.sf(abs(t_val),df); crit=stats.t.ppf(1-alpha,df)
        p_crit=alpha; reject=t_val>crit
    else:
        p_calc=stats.t.sf(abs(t_val),df)*2; crit=stats.t.ppf(1-alpha/2,df)
        p_crit=alpha; reject=abs(t_val)>crit
    decision="rejected" if reject else "failed to reject"
    st.markdown(
        f"**Interpretation details**  \n"
        f"*Calculated statistic:* t({df}) = {t_val:.2f}, *p* = {p_calc:.3f}.  \n"
        f"*Critical statistic:* t₍crit₎ = {crit:.2f}, *p* = {p_crit:.3f}.  \n"
        f"Calculated t {'>' if reject else '≤'} critical t → H₀ {decision}.  \n"
        f"Calculated *p* {'<' if p_calc<p_crit else '≥'} α → H₀ {decision}.  \n"
        f"**APA:** *t*({df}) = {t_val:.2f}, *p* = {p_calc:.3f} ({tail}). "
        f"H₀ was **{decision}** at α = {alpha:.2f}."
    )

def tab_t():
    st.subheader("Tab 1 • t‑Distribution")
    c1,c2=st.columns(2)
    with c1:
        t_val=st.number_input("t statistic",value=2.87,key="t_val")
        df   =st.number_input("df",min_value=1,value=55,step=1,key="t_df")
    with c2:
        alpha=st.number_input("α",value=0.05,step=0.01,
                              min_value=0.0001,max_value=0.5,key="t_alpha")
        tail =st.radio("Tail",["one‑tailed","two‑tailed"],key="t_tail")
    if st.button("Update Plot",key="t_plot"):
        st.pyplot(plot_t(t_val,df,alpha,tail))
    with st.expander("Step‑by‑step t‑table"):
        t_table(df,alpha,tail,"t")

# ─────────────────────────────  TAB 2 • z‑Distribution  ─────────────────────
CSS_Z = CSS + "th{background:#fafafa}"    # reuse

def z_table(z_in, key, alpha, tail):
    """Show ±10 rows around the relevant z row; handles negatives."""
    step_key=f"{key}_step"; step=st.session_state.setdefault(step_key,-1)

    z = max(-3.49, min(3.49, z_in))
    row_base = np.floor(z * 10) / 10         # -1.3, 2.1, …
    col_part = round(z - row_base, 2)        # 0.00–0.09

    all_rows = np.round(np.arange(-3.4, 3.5, 0.1), 1)
    all_cols = np.round(np.arange(0, 0.1, 0.01), 2)

    # pick ±10 rows
    idx = np.where(all_rows == row_base)[0][0]
    start = max(0, idx - 10); end = min(len(all_rows)-1, idx + 10)
    rows = all_rows[start:end+1]

    if col_part not in all_cols:
        col_part = min(all_cols, key=lambda c: abs(c - col_part))

    head = "".join(f"<th>{c:.2f}</th>" for c in all_cols)
    body = ""
    for r in rows:
        body += f'<tr><td id="z_{r:.1f}_0">{r:.1f}</td>'
        for c in all_cols:
            body += f'<td id="z_{r:.1f}_{c:.2f}">{stats.norm.cdf(r+c):.4f}</td>'
        body += "</tr>"

    html = wrap(CSS_Z, f"<tr><th>z.x</th>{head}</tr>{body}")

    # highlights
    if step >= 0:
        for c in all_cols:
            html = style_cell(html, f"z_{row_base:.1f}_{c:.2f}")
        html = style_cell(html, f"z_{row_base:.1f}_0")
    if step >= 1:
        for r in rows:
            html = style_cell(html, f"z_{r:.1f}_{col_part:.2f}")
    if step >= 2:
        html = style_cell(html, f"z_{row_base:.1f}_{col_part:.2f}",
                          color="blue", px=3)

    iframe(html)

    msgs = ["Highlight row", "Highlight column", "Intersection → Φ(z)"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step+1}**: {msgs[step]}")
    next_button(step_key)

    # APA narrative
    z_val = st.session_state["z_val"]
    p_calc = stats.norm.sf(abs(z_val)) * (1 if tail.startswith("one") else 2)
    crit   = stats.norm.ppf(1 - alpha) if tail.startswith("one") else stats.norm.ppf(1 - alpha/2)
    p_crit = alpha
    reject = abs(z_val) > crit if tail.startswith("two") else z_val > crit
    decision = "rejected" if reject else "failed to reject"
    st.markdown(
        f"**Interpretation details**  \n"
        f"*Calculated statistic:* z = {z_val:.2f}, *p* = {p_calc:.3f}.  \n"
        f"*Critical statistic:* z₍crit₎ = {crit:.2f}, *p* = {p_crit:.3f}.  \n"
        f"Calculated z {'>' if reject else '≤'} critical z → H₀ {decision}.  \n"
        f"Calculated *p* {'<' if p_calc < p_crit else '≥'} α → H₀ {decision}.  \n"
        f"**APA:** *z* = {z_val:.2f}, *p* = {p_calc:.3f} ({tail}). "
        f"H₀ was **{decision}** at α = {alpha:.2f}."
    )

def plot_z(z_calc, alpha, tail):
    fig,ax=plt.subplots(figsize=(12,4),dpi=100)
    xs=np.linspace(-4,4,400); ys=stats.norm.pdf(xs)
    ax.plot(xs,ys,"k"); ax.fill_between(xs,ys,color="lightgrey",alpha=0.25,
                                        label="Fail to Reject H₀")
    labels=[]
    if tail.startswith("one"):
        crit=stats.norm.ppf(1-alpha)
        ax.fill_between(xs[xs>=crit],ys[xs>=crit],color="red",alpha=0.30,
                        label="Reject H₀")
        ax.axvline(crit,color="green",ls="--")
        place_label(ax,labels,crit,stats.norm.pdf(crit)+.02,
                    f"z₍crit₎={crit:.2f}",color="green")
    else:
        crit=stats.norm.ppf(1-alpha/2)
        ax.fill_between(xs[xs>=crit],ys[xs>=crit],color="red",alpha=0.30)
        ax.fill_between(xs[xs<=-crit],ys[xs<=-crit],color="red",alpha=0.30,
                        label="Reject H₀")
        ax.axvline( crit,color="green",ls="--")
        ax.axvline(-crit,color="green",ls="--")
        place_label(ax,labels, crit,stats.norm.pdf( crit)+.02,
                    f"+z₍crit₎={crit:.2f}",color="green")
        place_label(ax,labels,-crit,stats.norm.pdf(-crit)+.02,
                    f"–z₍crit₎={crit:.2f}",color="green")
    ax.axvline(z_calc,color="blue",ls="--")
    place_label(ax,labels,z_calc,stats.norm.pdf(z_calc)+.02,
                f"z₍calc₎={z_calc:.2f}",color="blue")
    ax.set_xlabel("z"); ax.set_ylabel("Density"); ax.legend()
    ax.set_title("z‑Distribution"); fig.tight_layout(); return fig

def tab_z():
    st.subheader("Tab 2 • z‑Distribution")
    c1,c2=st.columns(2)
    with c1:
        z_val=st.number_input("z statistic",value=1.64,key="z_val")
    with c2:
        alpha=st.number_input("α",value=0.05,step=0.01,
                              min_value=0.0001,max_value=0.5,key="z_alpha")
        tail =st.radio("Tail",["one‑tailed","two‑tailed"],key="z_tail")
    if st.button("Update Plot",key="z_plot"):
        st.pyplot(plot_z(z_val,alpha,tail))
    with st.expander("Step‑by‑step z‑table"):
        z_table(z_val,"z",alpha,tail)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 • F‑Distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_f(f_calc, df1, df2, alpha):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x_max = stats.f.ppf(0.995, df1, df2) * 1.1
    xs = np.linspace(0, x_max, 400)
    ys = stats.f.pdf(xs, df1, df2)
    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H₀")

    crit = stats.f.ppf(1 - alpha, df1, df2)
    ax.fill_between(xs[xs >= crit], ys[xs >= crit],
                    color="red", alpha=0.30, label="Reject H₀")
    ax.axvline(x=crit, color="green", linestyle="--")
    ax.axvline(x=f_calc, color="blue", linestyle="--")
    lbl = []
    place_label(ax, lbl, crit, stats.f.pdf(crit, df1, df2)+.02,
                f"F₍crit₎={crit:.2f}", color="green")
    place_label(ax, lbl, f_calc, stats.f.pdf(f_calc, df1, df2)+.02,
                f"F₍calc₎={f_calc:.2f}", color="blue")
    ax.set_xlabel("F")
    ax.set_ylabel("Density")
    ax.set_title(f"F‑Distribution (df₁ {df1}, df₂ {df2})")
    ax.legend()
    fig.tight_layout()
    return fig


def tab_f():
    st.subheader("Tab 3 • F‑Distribution")
    c1, c2 = st.columns(2)
    with c1:
        f_val = st.number_input("F statistic", value=4.32, key="f_val")
        df1   = st.number_input("df₁ (numerator)", min_value=1, value=5,
                                step=1, key="f_df1")
    with c2:
        df2   = st.number_input("df₂ (denominator)", min_value=1, value=20,
                                step=1, key="f_df2")
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="f_alpha")

    if st.button("Update Plot", key="f_plot"):
        st.pyplot(plot_f(f_val, df1, df2, alpha))

    with st.expander("Step‑by‑step F‑table"):
        f_table(df1, df2, alpha, "f", f_val)


def f_table(df1, df2, alpha, key, f_val):
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    row_vals = list(range(max(1, df1-5), df1+6))
    col_vals = list(range(max(1, df2-5), df2+6))

    head_html = "".join(f"<th>{c}</th>" for c in col_vals)
    body_html = ""
    for r in row_vals:
        body_html += f'<tr><td id="f_{r}_0">{r}</td>'
        for i, c in enumerate(col_vals, start=1):
            body_html += f'<td id="f_{r}_{i}">{stats.f.ppf(1 - alpha, r, c):.2f}</td>'
        body_html += "</tr>"

    css = ("table{border-collapse:collapse}"
           "th,td{border:1px solid #000;width:90px;height:30px;"
           "text-align:center;font-size:0.85rem}")
    html = wrap(css, f"<tr><th>df₁＼df₂</th>{head_html}</tr>{body_html}")

    col_idx = col_vals.index(df2) + 1

    if step >= 0:
        for i in range(len(col_vals)+1):
            html = style_cell(html, f"f_{df1}_{i}")
    if step >= 1:
        for r in row_vals:
            html = style_cell(html, f"f_{r}_{col_idx}")
    if step >= 2:
        html = style_cell(html, f"f_{df1}_{col_idx}", color="blue", px=3)

    iframe(html)

    msgs = ["Highlight df₁ row", "Highlight df₂ column", "Intersection → F₍crit₎"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step+1}**: {msgs[step]}")
    next_button(step_key)

    crit   = stats.f.ppf(1 - alpha, df1, df2)
    p_calc = stats.f.sf(f_val, df1, df2)
    p_crit = alpha
    reject = f_val > crit
    decision = "rejected" if reject else "failed to reject"
    st.markdown(
        f"**Interpretation details**  \n"
        f"*Calculated statistic:* F({df1}, {df2}) = {f_val:.2f}, *p* = {p_calc:.3f}.  \n"
        f"*Critical statistic:* F₍crit₎ = {crit:.2f}, *p* = {p_crit:.3f}.  \n"
        f"Calculated F {'exceeds' if reject else 'does not exceed'} critical F "
        f"→ H₀ {'' if reject else 'not '}rejected.  \n"
        f"Calculated *p* is {'below' if p_calc < p_crit else 'above'} α "
        f"→ H₀ {'' if p_calc < p_crit else 'not '}rejected.  \n"
        f"**APA‑style conclusion:** *F*({df1}, {df2}) = {f_val:.2f}, "
        f"*p* = {p_calc:.3f}. H₀ was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 • Chi‑Square
# ─────────────────────────────────────────────────────────────────────────────
def plot_chi(chi_calc, df, alpha):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    x_max = stats.chi2.ppf(0.995, df) * 1.1
    xs = np.linspace(0, x_max, 400)
    ys = stats.chi2.pdf(xs, df)
    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H₀")

    crit = stats.chi2.ppf(1 - alpha, df)
    ax.fill_between(xs[xs >= crit], ys[xs >= crit],
                    color="red", alpha=0.30, label="Reject H₀")
    ax.axvline(x=crit, color="green", linestyle="--")
    ax.axvline(x=chi_calc, color="blue", linestyle="--")
    lbl = []
    place_label(ax, lbl, crit, stats.chi2.pdf(crit, df)+.02,
                f"χ²₍crit₎={crit:.2f}", color="green")
    place_label(ax, lbl, chi_calc, stats.chi2.pdf(chi_calc, df)+.02,
                f"χ²₍calc₎={chi_calc:.2f}", color="blue")
    ax.set_xlabel("χ²")
    ax.set_ylabel("Density")
    ax.set_title(f"χ²‑Distribution (df = {df})")
    ax.legend()
    fig.tight_layout()
    return fig


def tab_chi():
    st.subheader("Tab 4 • Chi‑Square (χ²)")
    c1, c2 = st.columns(2)
    with c1:
        chi_val = st.number_input("χ² statistic", value=7.88, key="chi_val")
        df      = st.number_input("df", min_value=1, value=3,
                                  step=1, key="chi_df")
    with c2:
        alpha = st.selectbox("α", [0.10, 0.05, 0.01, 0.001],
                             index=1, key="chi_alpha")

    if st.button("Update Plot", key="chi_plot"):
        st.pyplot(plot_chi(chi_val, df, alpha))

    with st.expander("Step‑by‑step χ²‑table"):
        chi_table(df, alpha, "chi", chi_val)


def chi_table(df, alpha, key, chi_val):
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    row_vals = list(range(max(1, df-5), df+6))
    alphas   = [0.10, 0.05, 0.01, 0.001]

    head_html = "".join(f"<th>{a}</th>" for a in alphas)
    body_html = ""
    for r in row_vals:
        body_html += f'<tr><td id="chi_{r}_0">{r}</td>'
        for i, a in enumerate(alphas, start=1):
            body_html += f'<td id="chi_{r}_{i}">{stats.chi2.ppf(1 - a, r):.2f}</td>'
        body_html += "</tr>"

    css = ("table{border-collapse:collapse}"
           "th,td{border:1px solid #000;width:80px;height:30px;"
           "text-align:center;font-size:0.85rem}")
    html = wrap(css, f"<tr><th>df＼α</th>{head_html}</tr>{body_html}")

    col_idx = alphas.index(alpha) + 1

    if step >= 0:
        for i in range(len(alphas)+1):
            html = style_cell(html, f"chi_{df}_{i}")
    if step >= 1:
        for r in row_vals:
            html = style_cell(html, f"chi_{r}_{col_idx}")
    if step >= 2:
        html = style_cell(html, f"chi_{df}_{col_idx}", color="blue", px=3)

    iframe(html)

    msgs = ["Highlight df row", "Highlight α column", "Intersection → χ²₍crit₎"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step+1}**: {msgs[step]}")
    next_button(step_key)

    crit   = stats.chi2.ppf(1 - alpha, df)
    p_calc = stats.chi2.sf(chi_val, df)
    p_crit = alpha
    reject = chi_val > crit
    decision = "rejected" if reject else "failed to reject"
    st.markdown(
        f"**Interpretation details**  \n"
        f"*Calculated statistic:* χ²({df}) = {chi_val:.2f}, *p* = {p_calc:.3f}.  \n"
        f"*Critical statistic:* χ²₍crit₎ = {crit:.2f}, *p* = {p_crit:.3f}.  \n"
        f"Calculated χ² {'exceeds' if reject else 'does not exceed'} critical χ² "
        f"→ H₀ {'' if reject else 'not '}rejected.  \n"
        f"Calculated *p* is {'below' if p_calc < p_crit else 'above'} α "
        f"→ H₀ {'' if p_calc < p_crit else 'not '}rejected.  \n"
        f"**APA‑style conclusion:** χ²({df}) = {chi_val:.2f}, "
        f"*p* = {p_calc:.3f}. H₀ was **{decision}** at α = {alpha:.3f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 • Mann‑Whitney U
# ─────────────────────────────────────────────────────────────────────────────
def plot_u(u_calc, n1, n2, alpha, tail):
    μ = n1 * n2 / 2
    σ = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(μ - 4*σ, μ + 4*σ, 400)
    ys = stats.norm.pdf(xs, μ, σ)
    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H₀")

    if tail.startswith("one"):
        crit = int(np.floor(μ + stats.norm.ppf(alpha) * σ))
        ax.fill_between(xs[xs <= crit], ys[xs <= crit],
                        color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x=crit, color="green", linestyle="--")
    else:
        crit = int(np.floor(μ + stats.norm.ppf(alpha / 2) * σ))
        hi   = n1 * n2 - crit
        ax.fill_between(xs[xs <= crit], ys[xs <= crit], color="red", alpha=0.30)
        ax.fill_between(xs[xs >= hi],   ys[xs >= hi],   color="red", alpha=0.30,
                        label="Reject H₀")
        ax.axvline(x=crit, color="green", linestyle="--")
        ax.axvline(x=hi,   color="green", linestyle="--")

    ax.axvline(x=u_calc, color="blue", linestyle="--")
    lbl = []
    place_label(ax, lbl, u_calc, stats.norm.pdf(u_calc, μ, σ)+.02,
                f"U₍calc₎={u_calc}", color="blue")
    ax.set_xlabel("U")
    ax.set_ylabel("Approx. density (normal)")
    ax.set_title("Mann‑Whitney U")
    ax.legend()
    fig.tight_layout()
    return fig


def tab_u():
    st.subheader("Tab 5 • Mann–Whitney U")
    c1, c2 = st.columns(2)
    with c1:
        u_val = st.number_input("U statistic", value=23, key="u_val")
        n1    = st.number_input("n₁", min_value=2, value=10,
                                step=1, key="u_n1")
    with c2:
        n2    = st.number_input("n₂", min_value=2, value=12,
                                step=1, key="u_n2")
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="u_alpha")
        tail  = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="u_tail")

    if st.button("Update Plot", key="u_plot"):
        st.pyplot(plot_u(u_val, n1, n2, alpha, tail))

    with st.expander("Step‑by‑step U‑table"):
        u_table(n1, n2, alpha, tail, "u", u_val)


def u_table(n1, n2, alpha, tail, key, u_val):
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    row_vals = list(range(max(2, n1-5), n1+6))
    col_vals = list(range(max(2, n2-5), n2+6))

    head_html = "".join(f"<th>{c}</th>" for c in col_vals)
    body_html = ""
    for r in row_vals:
        body_html += f'<tr><td id="u_{r}_0">{r}</td>'
        for i, c in enumerate(col_vals, start=1):
            body_html += f'<td id="u_{r}_{i}">{int(np.floor(n1*n2/2 + stats.norm.ppf(alpha if tail.startswith("one") else alpha/2)*np.sqrt(n1*n2*(n1+n2+1)/12))):d}</td>'
        body_html += "</tr>"

    css = ("table{border-collapse:collapse}"
           "th,td{border:1px solid #000;width:90px;height:30px;"
           "text-align:center;font-size:0.8rem}")
    html = wrap(css, f"<tr><th>n₁＼n₂</th>{head_html}</tr>{body_html}")

    col_idx = col_vals.index(n2) + 1

    if step >= 0:
        for i in range(len(col_vals)+1):
            html = style_cell(html, f"u_{n1}_{i}")
    if step >= 1:
        for r in row_vals:
            html = style_cell(html, f"u_{r}_{col_idx}")
    if step >= 2:
        html = style_cell(html, f"u_{n1}_{col_idx}", color="blue", px=3)

    iframe(html)

    msgs = ["Highlight n₁ row", "Highlight n₂ column", "Intersection → U₍crit₎"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step+1}**: {msgs[step]}")
    next_button(step_key)

    μ = n1 * n2 / 2
    σ = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if tail.startswith("one"):
        p_calc = stats.norm.cdf((u_val - μ) / σ)
        crit   = int(np.floor(μ + stats.norm.ppf(alpha) * σ))
        p_crit = alpha
        reject = u_val <= crit
    else:
        p_calc = stats.norm.sf(abs(u_val - μ) / σ) * 2
        crit   = int(np.floor(μ + stats.norm.ppf(alpha/2) * σ))
        p_crit = alpha
        reject = (u_val <= crit) or (u_val >= n1 * n2 - crit)
    decision = "rejected" if reject else "failed to reject"
    tail_txt = "one‑tailed" if tail.startswith("one") else "two‑tailed"
    st.markdown(
        f"**Interpretation details**  \n"
        f"*Calculated statistic:* U = {u_val}, *p* = {p_calc:.3f}.  \n"
        f"*Critical statistic:* U₍crit₎ = {crit}, *p* = {p_crit:.3f}.  \n"
        f"Calculated U is {'beyond' if reject else 'not beyond'} critical U "
        f"→ H₀ {'' if reject else 'not '}rejected.  \n"
        f"Calculated *p* is {'below' if p_calc < p_crit else 'above'} α "
        f"→ H₀ {'' if p_calc < p_crit else 'not '}rejected.  \n"
        f"**APA‑style conclusion:** *U* = {u_val}, *p* = {p_calc:.3f} "
        f"({tail_txt}). H₀ was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 6 • Wilcoxon Signed‑Rank T
# ─────────────────────────────────────────────────────────────────────────────
def plot_w(t_calc, n, alpha, tail):
    μ = n * (n + 1) / 4
    σ = np.sqrt(n * (n + 1) * (2*n + 1) / 24)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(μ - 4*σ, μ + 4*σ, 400)
    ys = stats.norm.pdf(xs, μ, σ)
    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H₀")

    if tail.startswith("one"):
        crit = int(np.floor(μ + stats.norm.ppf(alpha) * σ))
        ax.fill_between(xs[xs <= crit], ys[xs <= crit],
                        color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x=crit, color="green", linestyle="--")
    else:
        crit = int(np.floor(μ + stats.norm.ppf(alpha / 2) * σ))
        hi   = n * (n + 1) / 2 - crit
        ax.fill_between(xs[xs <= crit], ys[xs <= crit], color="red", alpha=0.30)
        ax.fill_between(xs[xs >= hi],   ys[xs >= hi],   color="red", alpha=0.30,
                        label="Reject H₀")
        ax.axvline(x=crit, color="green", linestyle="--")
        ax.axvline(x=hi,   color="green", linestyle="--")

    ax.axvline(x=t_calc, color="blue", linestyle="--")
    lbl = []
    place_label(ax, lbl, t_calc, stats.norm.pdf(t_calc, μ, σ)+.02,
                f"T₍calc₎={t_calc}", color="blue")
    ax.set_xlabel("T")
    ax.set_ylabel("Approx. density (normal)")
    ax.set_title("Wilcoxon Signed‑Rank T")
    ax.legend()
    fig.tight_layout()
    return fig


def tab_w():
    st.subheader("Tab 6 • Wilcoxon Signed‑Rank T")
    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("T statistic", value=15, key="w_val")
        n     = st.number_input("N (non‑zero diffs)", min_value=5, value=12,
                                step=1, key="w_n")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="w_alpha")
        tail  = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="w_tail")

    if st.button("Update Plot", key="w_plot"):
        st.pyplot(plot_w(t_val, n, alpha, tail))

    with st.expander("Step‑by‑step T‑table"):
        w_table(n, alpha, tail, "w", t_val)


def w_table(n, alpha, tail, key, t_val):
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    row_vals = list(range(max(5, n-5), n+6))
    alphas   = [0.10, 0.05, 0.01, 0.001]

    head_html = "".join(f"<th>{a}</th>" for a in alphas)
    body_html = ""
    for r in row_vals:
        body_html += f'<tr><td id="w_{r}_0">{r}</td>'
        for i, a in enumerate(alphas, start=1):
            body_html += f'<td id="w_{r}_{i}">{int(np.floor(r*(r+1)/4 + stats.norm.ppf(a if tail.startswith("one") else a/2)*np.sqrt(r*(r+1)*(2*r+1)/24))):d}</td>'
        body_html += "</tr>"

    css = ("table{border-collapse:collapse}"
           "th,td{border:1px solid #000;width:80px;height:30px;"
           "text-align:center;font-size:0.8rem}")
    html = wrap(css, f"<tr><th>N＼α</th>{head_html}</tr>{body_html}")

    col_idx = alphas.index(alpha) + 1

    if step >= 0:
        for i in range(len(alphas)+1):
            html = style_cell(html, f"w_{n}_{i}")
    if step >= 1:
        for r in row_vals:
            html = style_cell(html, f"w_{r}_{col_idx}")
    if step >= 2:
        html = style_cell(html, f"w_{n}_{col_idx}", color="blue", px=3)

    iframe(html)

    msgs = ["Highlight N row", "Highlight α column", "Intersection → T₍crit₎"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step+1}**: {msgs[step]}")
    next_button(step_key)

    μ = n * (n + 1) / 4
    σ = np.sqrt(n * (n + 1) * (2*n + 1) / 24)
    if tail.startswith("one"):
        p_calc = stats.norm.cdf((t_val - μ) / σ)
        crit   = int(np.floor(μ + stats.norm.ppf(alpha) * σ))
        p_crit = alpha
        reject = t_val <= crit
    else:
        p_calc = stats.norm.sf(abs(t_val - μ) / σ) * 2
        crit   = int(np.floor(μ + stats.norm.ppf(alpha/2) * σ))
        p_crit = alpha
        reject = (t_val <= crit) or (t_val >= n * (n + 1) / 2 - crit)
    decision = "rejected" if reject else "failed to reject"
    tail_txt = "one‑tailed" if tail.startswith("one") else "two‑tailed"
    st.markdown(
        f"**Interpretation details**  \n"
        f"*Calculated statistic:* T = {t_val}, *p* = {p_calc:.3f}.  \n"
        f"*Critical statistic:* T₍crit₎ = {crit}, *p* = {p_crit:.3f}.  \n"
        f"Calculated T is {'beyond' if reject else 'not beyond'} critical T "
        f"→ H₀ {'' if reject else 'not '}rejected.  \n"
        f"Calculated *p* is {'below' if p_calc < p_crit else 'above'} α "
        f"→ H₀ {'' if p_calc < p_crit else 'not '}rejected.  \n"
        f"**APA‑style conclusion:** *T* = {t_val}, *p* = {p_calc:.3f} "
        f"({tail_txt}). H₀ was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 7 • Binomial
# ─────────────────────────────────────────────────────────────────────────────
def plot_binom(k, n, p):
    xs = np.arange(0, n+1)
    ys = stats.binom.pmf(xs, n, p)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    ax.bar(xs, ys, color="lightgrey")
    ax.bar(k, stats.binom.pmf(k, n, p), color="blue", label=f"k = {k}")
    ax.set_xlabel("k")
    ax.set_ylabel("P(X=k)")
    ax.set_title(f"Binomial (n = {n}, p = {p})")
    ax.legend()
    fig.tight_layout()
    return fig


def tab_binom():
    st.subheader("Tab 7 • Binomial")
    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input("n (trials)", min_value=1, value=20,
                            step=1, key="b_n")
        p = st.number_input("π (null proportion)", value=0.50,
                            step=0.01, min_value=0.01, max_value=0.99, key="b_p")
    with c2:
        k = st.number_input("k (successes)", min_value=0, value=12,
                            step=1, key="b_k")
        alpha = st.number_input("α (two‑tailed)", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="b_alpha")

    if st.button("Update Plot", key="b_plot"):
        st.pyplot(plot_binom(k, n, p))

    with st.expander("Quick table (k ±5)"):
        binom_table(k, n, p, "b", alpha)


def critical_binom(n, p, alpha):
    cum = 0
    k_low = None
    for k in range(n+1):
        cum += stats.binom.pmf(k, n, p)
        if cum >= alpha / 2:
            k_low = k
            break
    cum = 0
    k_hi = None
    for k in range(n, -1, -1):
        cum += stats.binom.pmf(k, n, p)
        if cum >= alpha / 2:
            k_hi = k
            break
    return k_low, k_hi


def binom_table(k, n, p, key, alpha):
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    k_vals = list(range(max(0, k-5), min(n, k+5)+1))

    head_html = "<th>P(X=k)</th><th>P(X≤k)</th><th>P(X≥k)</th>"
    body_html = ""
    for kv in k_vals:
        pmf = stats.binom.pmf(kv, n, p)
        cdf = stats.binom.cdf(kv, n, p)
        sf  = 1 - stats.binom.cdf(kv-1, n, p)
        body_html += (
            f'<tr><td id="b_{kv}_0">{kv}</td>'
            f'<td id="b_{kv}_1">{pmf:.4f}</td>'
            f'<td id="b_{kv}_2">{cdf:.4f}</td>'
            f'<td id="b_{kv}_3">{sf :.4f}</td></tr>'
        )

    css = ("table{border-collapse:collapse}"
           "th,td{border:1px solid #000;width:110px;height:30px;"
           "text-align:center;font-size:0.8rem}")
    html = wrap(css, f"<tr><th>k</th>{head_html}</tr>{body_html}")

    if step >= 0:
        for i in range(4):
            html = style_cell(html, f"b_{k}_{i}")
    if step >= 1:
        html = style_cell(html, f"b_{k}_1", color="blue", px=3)

    iframe(html)

    msgs = ["Highlight k row", "Highlight P(X=k)"]
    if step < 0:
        st.write("Click **Next Step** to begin.")
    elif step >= len(msgs):
        st.write("All steps complete!")
    else:
        st.write(f"**Step {step+1}**: {msgs[step]}")
    next_button(step_key)

    cdf    = stats.binom.cdf(k, n, p)
    p_calc = 2 * min(cdf, 1 - cdf + stats.binom.pmf(k, n, p))
    p_calc = min(p_calc, 1.0)
    k_lo, k_hi = critical_binom(n, p, alpha)
    p_crit = alpha
    reject = (k <= k_lo) or (k >= k_hi)
    decision = "rejected" if reject else "failed to reject"
    st.markdown(
        f"**Interpretation details**  \n"
        f"*Calculated statistic:* k = {k}, *p* = {p_calc:.3f}.  \n"
        f"*Critical region:* k ≤ {k_lo} or k ≥ {k_hi} (total *p* = {p_crit:.3f}).  \n"
        f"Observed k is {'inside' if reject else 'outside'} rejection region "
        f"→ H₀ {'' if reject else 'not '}rejected.  \n"
        f"*p* is {'below' if p_calc < p_crit else 'above'} α "
        f"→ H₀ {'' if p_calc < p_crit else 'not '}rejected.  \n"
        f"**APA‑style conclusion:** Exact binomial test, *p* = {p_calc:.3f}. "
        f"H₀ was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        "PSYC250 – Statistical Tables Explorer",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("PSYC250 – Statistical Tables Explorer (12 × 4 figures)")

    tabs = st.tabs([
        "t‑Dist", "z‑Dist", "F‑Dist", "Chi‑Square",
        "Mann–Whitney U", "Wilcoxon T", "Binomial"
    ])
    with tabs[0]: tab_t()
    with tabs[1]: tab_z()
    with tabs[2]: tab_f()
    with tabs[3]: tab_chi()
    with tabs[4]: tab_u()
    with tabs[5]: tab_w()
    with tabs[6]: tab_binom()


if __name__ == "__main__":
    main()
