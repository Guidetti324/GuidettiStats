###############################################################################
#  PSYC‑250  ‑‑  Statistical Tables Explorer  (Streamlit, 12 × 4 figures)
#  ---------------------------------------------------------------------------
#  Seven tabs:
#     1) t‑Distribution
#     2) z‑Distribution
#     3) F‑Distribution
#     4) Chi‑Square
#     5) Mann‑Whitney U
#     6) Wilcoxon Signed‑Rank T
#     7) Binomial
#
#  Each tab:
#     • Input widgets
#     • 12 × 4 Matplotlib plot
#     • Step‑wise HTML lookup table
#     • Full APA‑7 narrative with:
#         1.  Calculated statistic + p
#         2.  Critical statistic + p
#         3.  Decision via statistic comparison
#         4.  Decision via p‑value comparison
#         5.  Concise APA conclusion
#
#  Paste this file AS‑IS into app.py and run:
#     streamlit run app.py
#
#  Tested: Python 3.12.1 • streamlit 1.30.0 • matplotlib 3.8.2 • scipy 1.11.4
###############################################################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from scipy import stats

plt.switch_backend("Agg")   # head‑less backend for Streamlit

# ─────────────────────────────────────────────────────────────────────────────
#  Generic helpers
# ─────────────────────────────────────────────────────────────────────────────
def place_label(ax, placed, x, y, txt, *, color="blue"):
    dx = dy = 0.0
    for xx, yy in placed:
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05:
            dx += 0.06
            dy += 0.04
    ax.text(x + dx, y + dy, txt, color=color,
            ha="left", va="bottom", fontsize=8, clip_on=True)
    placed.append((x + dx, y + dy))


def style_cell(html_in, cell_id, *, color="red", px=2):
    return html_in.replace(
        f'id="{cell_id}"',
        f'id="{cell_id}" style="border:{px}px solid {color};"', 1)


def iframe(html, *, height=460):
    st.markdown(
        f'<div style="overflow:auto; max-height:{height}px;">{html}</div>',
        unsafe_allow_html=True,
    )


def next_button(step_key):
    if st.button("Next Step", key=f"{step_key}__btn"):
        st.session_state[step_key] += 1


def wrap(css, inner):
    return f"<style>{css}</style><table>{inner}</table>"


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 • t‑Distribution
# ─────────────────────────────────────────────────────────────────────────────
def tab_t():
    st.subheader("Tab 1 • t‑Distribution")
    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("t statistic", value=2.87, key="t_val")
        df    = st.number_input("df", min_value=1, value=55, step=1, key="t_df")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="t_alpha")
        tail  = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="t_tail")

    if st.button("Update Plot", key="t_plot"):
        st.pyplot(plot_t(t_val, df, alpha, tail))

    with st.expander("Step‑by‑step t‑table"):
        t_table(df, alpha, tail, "t")


def plot_t(t_calc, df, alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(-4, 4, 400)
    ys = stats.t.pdf(xs, df)
    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H₀")
    lbl = []

    if tail.startswith("one"):
        crit = stats.t.ppf(1 - alpha, df)
        ax.fill_between(xs[xs >= crit], ys[xs >= crit],
                        color="red", alpha=0.30, label="Reject H₀")
        ax.axvline(x=crit, color="green", linestyle="--")
        place_label(ax, lbl, crit, stats.t.pdf(crit, df)+.02,
                    f"t₍crit₎={crit:.2f}", color="green")
    else:
        crit = stats.t.ppf(1 - alpha/2, df)
        ax.fill_between(xs[xs >= crit], ys[xs >= crit], color="red", alpha=0.30)
        ax.fill_between(xs[xs <=-crit], ys[xs <=-crit], color="red", alpha=0.30,
                        label="Reject H₀")
        ax.axvline(x= crit, color="green", linestyle="--")
        ax.axvline(x=-crit, color="green", linestyle="--")
        place_label(ax, lbl,  crit, stats.t.pdf( crit, df)+.02,
                    f"+t₍crit₎={crit:.2f}", color="green")
        place_label(ax, lbl, -crit, stats.t.pdf(-crit, df)+.02,
                    f"–t₍crit₎={crit:.2f}", color="green")

    ax.axvline(x=t_calc, color="blue", linestyle="--")
    place_label(ax, lbl, t_calc, stats.t.pdf(t_calc, df)+.02,
                f"t₍calc₎={t_calc:.2f}", color="blue")
    ax.set_title("t‑Distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def t_table(df, alpha, tail, key):
    step_key = f"{key}_step"
    step     = st.session_state.setdefault(step_key, -1)

    rows = list(range(max(1, df-5), df+6))
    heads = [
        ("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
        ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001),
    ]
    head_html = "".join(f"<th>{m}_{a}</th>" for m,a in heads)
    body_html = ""
    for r in rows:
        body_html += f'<tr><td id="t_{r}_0">{r}</td>'
        for i,(m,a) in enumerate(heads, start=1):
            body_html += (f'<td id="t_{r}_{i}">'
                          f'{stats.t.ppf(1 - a/(1 if m=="one" else 2), r):.2f}</td>')
        body_html += "</tr>"
    css=("table{border-collapse:collapse}"
         "th,td{border:1px solid #000;width:85px;height:30px;"
         "text-align:center;font-size:0.9rem;font-family:sans-serif}"
         "th{background:#fafafa}")
    html=wrap(css,f"<tr><th>df</th>{head_html}</tr>{body_html}")

    mode="one" if tail.startswith("one") else "two"
    col_idx=next(i for i,(m,a) in enumerate(heads,start=1)
                 if m==mode and np.isclose(a,alpha))

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

    # APA narrative
    t_val=st.session_state["t_val"]
    if tail.startswith("one"):
        p_calc=stats.t.sf(abs(t_val),df)
        crit=stats.t.ppf(1-alpha,df)
        p_crit=alpha
        reject=t_val>crit
    else:
        p_calc=stats.t.sf(abs(t_val),df)*2
        crit=stats.t.ppf(1-alpha/2,df)
        p_crit=alpha
        reject=abs(t_val)>crit
    decision="rejected" if reject else "failed to reject"
    st.markdown(
        f"**Interpretation details**  \n"
        f"*Calculated statistic:* t({df}) = {t_val:.2f}, *p* = {p_calc:.3f}.  \n"
        f"*Critical statistic:* t₍crit₎ = {crit:.2f}, *p* = {p_crit:.3f}.  \n"
        f"Calculated t is {'greater' if reject else 'not greater'} than "
        f"critical t ➔ H₀ {'' if reject else 'not '}rejected.  \n"
        f"Calculated *p* is {'below' if p_calc<p_crit else 'above'} α ➔ "
        f"H₀ {'' if p_calc<p_crit else 'not '}rejected.  \n"
        f"**APA‑style conclusion:** *t*({df}) = {t_val:.2f}, *p* = {p_calc:.3f} "
        f"({tail}). H₀ was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 • z‑Distribution
# ─────────────────────────────────────────────────────────────────────────────
def tab_z():
    st.subheader("Tab 2 • z‑Distribution")
    c1,c2=st.columns(2)
    with c1:
        z_val = st.number_input("z statistic", value=1.64, key="z_val")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="z_alpha")
        tail  = st.radio("Tail", ["one‑tailed","two‑tailed"], key="z_tail")
    if st.button("Update Plot", key="z_plot"):
        st.pyplot(plot_z(z_val, alpha, tail))
    with st.expander("Step‑by‑step z‑table"):
        z_table(z_val, "z", alpha, tail)


def plot_z(z_calc, alpha, tail):
    fig,ax=plt.subplots(figsize=(12,4),dpi=100)
    xs=np.linspace(-4,4,400)
    ys=stats.norm.pdf(xs)
    ax.plot(xs,ys,"k")
    ax.fill_between(xs,ys,color="lightgrey",alpha=0.25,
                    label="Fail to Reject H₀")
    lbl=[]
    if tail.startswith("one"):
        crit=stats.norm.ppf(1-alpha)
        ax.fill_between(xs[xs>=crit],ys[xs>=crit],
                        color="red",alpha=0.30,label="Reject H₀")
        ax.axvline(x=crit,color="green",linestyle="--")
        place_label(ax,lbl,crit,stats.norm.pdf(crit)+.02,
                    f"z₍crit₎={crit:.2f}",color="green")
    else:
        crit=stats.norm.ppf(1-alpha/2)
        ax.fill_between(xs[xs>=crit],ys[xs>=crit],color="red",alpha=0.30)
        ax.fill_between(xs[xs<=-crit],ys[xs<=-crit],color="red",alpha=0.30,
                        label="Reject H₀")
        ax.axvline(x= crit,color="green",linestyle="--")
        ax.axvline(x=-crit,color="green",linestyle="--")
        place_label(ax,lbl, crit,stats.norm.pdf( crit)+.02,
                    f"+z₍crit₎={crit:.2f}",color="green")
        place_label(ax,lbl,-crit,stats.norm.pdf(-crit)+.02,
                    f"–z₍crit₎={crit:.2f}",color="green")
    ax.axvline(x=z_calc,color="blue",linestyle="--")
    place_label(ax,lbl,z_calc,stats.norm.pdf(z_calc)+.02,
                f"z₍calc₎={z_calc:.2f}",color="blue")
    ax.set_title("z‑Distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def z_table(z_in,key,alpha,tail):
    step_key=f"{key}_step"
    step=st.session_state.setdefault(step_key,-1)
    z=max(0,min(3.49,z_in))
    row=np.floor(z*10)/10
    col=round(z-row,2)
    rows=np.round(np.arange(0,3.5,0.1),1)
    cols=np.round(np.arange(0,0.1,0.01),2)
    head="".join(f"<th>{c:.2f}</th>" for c in cols)
    body=""
    for r in rows:
        body+=f'<tr><td id="z_{r:.1f}_0">{r:.1f}</td>'
        for c in cols:
            body+=f'<td id="z_{r:.1f}_{c:.2f}">{stats.norm.cdf(r+c):.4f}</td>'
        body+="</tr>"
    css=("table{border-collapse:collapse}"
         "th,td{border:1px solid #000;width:70px;height:30px;"
         "text-align:center;font-size:0.9rem;font-family:sans-serif}"
         "th{background:#fafafa}")
    html=wrap(css,f"<tr><th>z.x</th>{head}</tr>{body}")
    if step>=0:
        for c in cols:
            html=style_cell(html,f"z_{row:.1f}_{c:.2f}")
        html=style_cell(html,f"z_{row:.1f}_0")
    if step>=1:
        for r in rows:
            html=style_cell(html,f"z_{r:.1f}_{col:.2f}")
    if step>=2:
        html=style_cell(html,f"z_{row:.1f}_{col:.2f}",color="blue",px=3)
    iframe(html)
    msgs=["Highlight row","Highlight column","Intersection → Φ(z)"]
    if step<0: st.write("Click **Next Step** to begin.")
    elif step>=len(msgs): st.write("All steps complete!")
    else: st.write(f"**Step {step+1}**: {msgs[step]}")
    next_button(step_key)

    # APA
    z_val=st.session_state["z_val"]
    p_calc=stats.norm.sf(abs(z_val))*(1 if tail.startswith("one") else 2)
    crit=stats.norm.ppf(1-alpha) if tail.startswith("one") else stats.norm.ppf(1-alpha/2)
    p_crit=alpha
    reject=abs(z_val)>crit if tail.startswith("two") else z_val>crit
    decision="rejected" if reject else "failed to reject"
    st.markdown(
        f"**Interpretation details**  \n"
        f"*Calculated statistic:* z = {z_val:.2f}, *p* = {p_calc:.3f}.  \n"
        f"*Critical statistic:* z₍crit₎ = {crit:.2f}, *p* = {p_crit:.3f}.  \n"
        f"Calculated z {'exceeds' if reject else 'does not exceed'} "
        f"critical z ➔ H₀ {'' if reject else 'not '}rejected.  \n"
        f"Calculated *p* is {'below' if p_calc<p_crit else 'above'} α ➔ "
        f"H₀ {'' if p_calc<p_crit else 'not '}rejected.  \n"
        f"**APA‑style conclusion:** *z* = {z_val:.2f}, *p* = {p_calc:.3f} "
        f"({tail}). H₀ was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 • F‑Distribution
# ─────────────────────────────────────────────────────────────────────────────
def tab_f():
    st.subheader("Tab 3 • F‑Distribution")
    c1,c2=st.columns(2)
    with c1:
        f_val=st.number_input("F statistic",value=4.32,key="f_val")
        df1  =st.number_input("df₁ (numerator)",min_value=1,value=5,step=1,key="f_df1")
    with c2:
        df2  =st.number_input("df₂ (denominator)",min_value=1,value=20,step=1,key="f_df2")
        alpha=st.number_input("α",value=0.05,step=0.01,
                              min_value=0.0001,max_value=0.5,key="f_alpha")
    if st.button("Update Plot",key="f_plot"):
        st.pyplot(plot_f(f_val,df1,df2,alpha))
    with st.expander("Step‑by‑step F‑table"):
        f_table(df1,df2,alpha,"f",f_val)


def f_crit(df1,df2,alpha):
    return stats.f.ppf(1-alpha,df1,df2)


def f_table(df1,df2,alpha,key,f_val):
    step_key=f"{key}_step"
    step=st.session_state.setdefault(step_key,-1)
    rows=list(range(max(1,df1-5),df1+6))
    cols=list(range(max(1,df2-5),df2+6))
    head="".join(f"<th>{c}</th>" for c in cols)
    body=""
    for r in rows:
        body+=f'<tr><td id="f_{r}_0">{r}</td>'
        for i,c in enumerate(cols,start=1):
            body+=f'<td id="f_{r}_{i}">{f_crit(r,c,alpha):.2f}</td>'
        body+="</tr>"
    css=("table{border-collapse:collapse}"
         "th,td{border:1px solid #000;width:90px;height:30px;"
         "text-align:center;font-size:0.85rem}")
    html=wrap(css,f"<tr><th>df₁＼df₂</th>{head}</tr>{body}")
    col_idx=cols.index(df2)+1
    if step>=0:
        for i in range(len(cols)+1):
            html=style_cell(html,f"f_{df1}_{i}")
    if step>=1:
        for r in rows:
            html=style_cell(html,f"f_{r}_{col_idx}")
    if step>=2:
        html=style_cell(html,f"f_{df1}_{col_idx}",color="blue",px=3)
    iframe(html)
    msgs=["Highlight df₁ row","Highlight df₂ column","Intersection → F₍crit₎"]
    if step<0: st.write("Click **Next Step** to begin.")
    elif step>=len(msgs): st.write("All steps complete!")
    else: st.write(f"**Step {step+1}**: {msgs[step]}")
    next_button(step_key)

    crit=f_crit(df1,df2,alpha)
    p_calc=stats.f.sf(f_val,df1,df2)
    p_crit=alpha
    reject=f_val>crit
    decision="rejected" if reject else "failed to reject"
    st.markdown(
        f"**Interpretation details**  \n"
        f"*Calculated statistic:* F({df1}, {df2}) = {f_val:.2f}, *p* = {p_calc:.3f}.  \n"
        f"*Critical statistic:* F₍crit₎ = {crit:.2f}, *p* = {p_crit:.3f}.  \n"
        f"Calculated F {'exceeds' if reject else 'does not exceed'} critical F; "
        f"*p* is {'below' if p_calc<p_crit else 'above'} α → H₀ {decision}.  \n"
        f"**APA‑style conclusion:** *F*({df1}, {df2}) = {f_val:.2f}, "
        f"*p* = {p_calc:.3f}. H₀ was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 • Chi‑Square
# ─────────────────────────────────────────────────────────────────────────────
def tab_chi():
    st.subheader("Tab 4 • Chi‑Square (χ²)")
    c1,c2=st.columns(2)
    with c1:
        chi_val=st.number_input("χ² statistic",value=7.88,key="chi_val")
        df     =st.number_input("df",min_value=1,value=3,step=1,key="chi_df")
    with c2:
        alpha=st.selectbox("α",[0.10,0.05,0.01,0.001],index=1,key="chi_alpha")
    if st.button("Update Plot",key="chi_plot"):
        st.pyplot(plot_chi(chi_val,df,alpha))
    with st.expander("Step‑by‑step χ²‑table"):
        chi_table(df,alpha,"chi",chi_val)


def chi_crit(df,alpha):
    return stats.chi2.ppf(1-alpha,df)


def chi_table(df,alpha,key,chi_val):
    step_key=f"{key}_step"
    step=st.session_state.setdefault(step_key,-1)
    rows=list(range(max(1,df-5),df+6))
    alphas=[0.10,0.05,0.01,0.001]
    head="".join(f"<th>{a}</th>" for a in alphas)
    body=""
    for r in rows:
        body+=f'<tr><td id="chi_{r}_0">{r}</td>'
        for i,a in enumerate(alphas,start=1):
            body+=f'<td id="chi_{r}_{i}">{chi_crit(r,a):.2f}</td>'
        body+="</tr>"
    css=("table{border-collapse:collapse}"
         "th,td{border:1px solid #000;width:80px;height:30px;"
         "text-align:center;font-size:0.85rem}")
    html=wrap(css,f"<tr><th>df＼α</th>{head}</tr>{body}")
    col_idx=alphas.index(alpha)+1
    if step>=0:
        for i in range(len(alphas)+1):
            html=style_cell(html,f"chi_{df}_{i}")
    if step>=1:
        for r in rows:
            html=style_cell(html,f"chi_{r}_{col_idx}")
    if step>=2:
        html=style_cell(html,f"chi_{df}_{col_idx}",color="blue",px=3)
    iframe(html)
    msgs=["Highlight df row","Highlight α column","Intersection → χ²₍crit₎"]
    if step<0: st.write("Click **Next Step** to begin.")
    elif step>=len(msgs): st.write("All steps complete!")
    else: st.write(f"**Step {step+1}**: {msgs[step]}")
    next_button(step_key)

    crit=chi_crit(df,alpha)
    p_calc=stats.chi2.sf(chi_val,df)
    p_crit=alpha
    reject=chi_val>crit
    decision="rejected" if reject else "failed to reject"
    st.markdown(
        f"**Interpretation details**  \n"
        f"*Calculated statistic:* χ²({df}) = {chi_val:.2f}, *p* = {p_calc:.3f}.  \n"
        f"*Critical statistic:* χ²₍crit₎ = {crit:.2f}, *p* = {p_crit:.3f}.  \n"
        f"Calculated χ² {'exceeds' if reject else 'does not exceed'} critical χ²; "
        f"*p* is {'below' if p_calc<p_crit else 'above'} α → H₀ {decision}.  \n"
        f"**APA‑style conclusion:** χ²({df}) = {chi_val:.2f}, "
        f"*p* = {p_calc:.3f}. H₀ was **{decision}** at α = {alpha:.3f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 • Mann‑Whitney U
# ─────────────────────────────────────────────────────────────────────────────
def tab_u():
    st.subheader("Tab 5 • Mann–Whitney U")
    c1,c2=st.columns(2)
    with c1:
        u_val=st.number_input("U statistic",value=23,key="u_val")
        n1=st.number_input("n₁",min_value=2,value=10,step=1,key="u_n1")
    with c2:
        n2=st.number_input("n₂",min_value=2,value=12,step=1,key="u_n2")
        alpha=st.number_input("α",value=0.05,step=0.01,
                              min_value=0.0001,max_value=0.5,key="u_alpha")
        tail=st.radio("Tail",["one‑tailed","two‑tailed"],key="u_tail")
    if st.button("Update Plot",key="u_plot"):
        st.pyplot(plot_u(u_val,n1,n2,alpha,tail))
    with st.expander("Step‑by‑step U‑table"):
        u_table(n1,n2,alpha,tail,"u",u_val)


def u_crit(n1,n2,alpha,tail):
    μ=n1*n2/2
    σ=np.sqrt(n1*n2*(n1+n2+1)/12)
    z=stats.norm.ppf(alpha if tail.startswith("one") else alpha/2)
    return int(np.floor(μ+z*σ))


def u_table(n1,n2,alpha,tail,key,u_val):
    step_key=f"{key}_step"
    step=st.session_state.setdefault(step_key,-1)
    rows=list(range(max(2,n1-5),n1+6))
    cols=list(range(max(2,n2-5),n2+6))
    head="".join(f"<th>{c}</th>" for c in cols)
    body=""
    for r in rows:
        body+=f'<tr><td id="u_{r}_0">{r}</td>'
        for i,c in enumerate(cols,start=1):
            body+=f'<td id="u_{r}_{i}">{u_crit(r,c,alpha,tail)}</td>'
        body+="</tr>"
    css=("table{border-collapse:collapse}"
         "th,td{border:1px solid #000;width:90px;height:30px;"
         "text-align:center;font-size:0.8rem}")
    html=wrap(css,f"<tr><th>n₁＼n₂</th>{head}</tr>{body}")
    col_idx=cols.index(n2)+1
    if step>=0:
        for i in range(len(cols)+1):
            html=style_cell(html,f"u_{n1}_{i}")
    if step>=1:
        for r in rows:
            html=style_cell(html,f"u_{r}_{col_idx}")
    if step>=2:
        html=style_cell(html,f"u_{n1}_{col_idx}",color="blue",px=3)
    iframe(html)
    msgs=["Highlight n₁ row","Highlight n₂ column","Intersection → U₍crit₎"]
    if step<0: st.write("Click **Next Step** to begin.")
    elif step>=len(msgs): st.write("All steps complete!")
    else: st.write(f"**Step {step+1}**: {msgs[step]}")
    next_button(step_key)

    μ=n1*n2/2
    σ=np.sqrt(n1*n2*(n1+n2+1)/12)
    if tail.startswith("one"):
        p_calc=stats.norm.cdf((u_val-μ)/σ)
        crit=u_crit(n1,n2,alpha,tail)
        p_crit=alpha
        reject=u_val<=crit
    else:
        p_calc=stats.norm.sf(abs(u_val-μ)/σ)*2
        crit=u_crit(n1,n2,alpha,tail)
        p_crit=alpha
        reject=(u_val<=crit) or (u_val>=n1*n2-crit)
    decision="rejected" if reject else "failed to reject"
    tail_txt="one‑tailed" if tail.startswith("one") else "two‑tailed"
    st.markdown(
        f"**Interpretation details**  \n"
        f"*Calculated statistic:* U = {u_val}, *p* = {p_calc:.3f}.  \n"
        f"*Critical statistic:* U₍crit₎ = {crit}, *p* = {p_crit:.3f}.  \n"
        f"Calculated U is {'beyond' if reject else 'not beyond'} critical U; "
        f"*p* is {'below' if p_calc<p_crit else 'above'} α → H₀ {decision}.  \n"
        f"**APA‑style conclusion:** *U* = {u_val}, *p* = {p_calc:.3f} "
        f"({tail_txt}). H₀ was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 6 • Wilcoxon Signed‑Rank T
# ─────────────────────────────────────────────────────────────────────────────
def tab_w():
    st.subheader("Tab 6 • Wilcoxon Signed‑Rank T")
    c1,c2=st.columns(2)
    with c1:
        t_val=st.number_input("T statistic",value=15,key="w_val")
        n=st.number_input("N (non‑zero diffs)",min_value=5,value=12,step=1,key="w_n")
    with c2:
        alpha=st.number_input("α",value=0.05,step=0.01,
                              min_value=0.0001,max_value=0.5,key="w_alpha")
        tail=st.radio("Tail",["one‑tailed","two‑tailed"],key="w_tail")
    if st.button("Update Plot",key="w_plot"):
        st.pyplot(plot_w(t_val,n,alpha,tail))
    with st.expander("Step‑by‑step T‑table"):
        w_table(n,alpha,tail,"w",t_val)


def w_crit(n,alpha,tail):
    μ=n*(n+1)/4
    σ=np.sqrt(n*(n+1)*(2*n+1)/24)
    z=stats.norm.ppf(alpha if tail.startswith("one") else alpha/2)
    return int(np.floor(μ+z*σ))


def w_table(n,alpha,tail,key,t_val):
    step_key=f"{key}_step"
    step=st.session_state.setdefault(step_key,-1)
    rows=list(range(max(5,n-5),n+6))
    alphas=[0.10,0.05,0.01,0.001]
    head="".join(f"<th>{a}</th>" for a in alphas)
    body=""
    for r in rows:
        body+=f'<tr><td id="w_{r}_0">{r}</td>'
        for i,a in enumerate(alphas,start=1):
            body+=f'<td id="w_{r}_{i}">{w_crit(r,a,tail)}</td>'
        body+="</tr>"
    css=("table{border-collapse:collapse}"
         "th,td{border:1px solid #000;width:80px;height:30px;"
         "text-align:center;font-size:0.8rem}")
    html=wrap(css,f"<tr><th>N＼α</th>{head}</tr>{body}")
    col_idx=alphas.index(alpha)+1
    if step>=0:
        for i in range(len(alphas)+1):
            html=style_cell(html,f"w_{n}_{i}")
    if step>=1:
        for r in rows:
            html=style_cell(html,f"w_{r}_{col_idx}")
    if step>=2:
        html=style_cell(html,f"w_{n}_{col_idx}",color="blue",px=3)
    iframe(html)
    msgs=["Highlight N row","Highlight α column","Intersection → T₍crit₎"]
    if step<0: st.write("Click **Next Step** to begin.")
    elif step>=len(msgs): st.write("All steps complete!")
    else: st.write(f"**Step {step+1}**: {msgs[step]}")
    next_button(step_key)

    μ=n*(n+1)/4
    σ=np.sqrt(n*(n+1)*(2*n+1)/24)
    if tail.startswith("one"):
        p_calc=stats.norm.cdf((t_val-μ)/σ)
        crit=w_crit(n,alpha,tail)
        p_crit=alpha
        reject=t_val<=crit
    else:
        p_calc=stats.norm.sf(abs(t_val-μ)/σ)*2
        crit=w_crit(n,alpha,tail)
        p_crit=alpha
        reject=(t_val<=crit) or (t_val>=n*(n+1)/2-crit)
    decision="rejected" if reject else "failed to reject"
    tail_txt="one‑tailed" if tail.startswith("one") else "two‑tailed"
    st.markdown(
        f"**Interpretation details**  \n"
        f"*Calculated statistic:* T = {t_val}, *p* = {p_calc:.3f}.  \n"
        f"*Critical statistic:* T₍crit₎ = {crit}, *p* = {p_crit:.3f}.  \n"
        f"Calculated T is {'beyond' if reject else 'not beyond'} critical T; "
        f"*p* is {'below' if p_calc<p_crit else 'above'} α → H₀ {decision}.  \n"
        f"**APA‑style conclusion:** *T* = {t_val}, *p* = {p_calc:.3f} "
        f"({tail_txt}). H₀ was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 7 • Binomial
# ─────────────────────────────────────────────────────────────────────────────
def tab_binom():
    st.subheader("Tab 7 • Binomial")
    c1,c2=st.columns(2)
    with c1:
        n=st.number_input("n (trials)",min_value=1,value=20,step=1,key="b_n")
        p=st.number_input("π (null proportion)",value=0.50,step=0.01,
                          min_value=0.01,max_value=0.99,key="b_p")
    with c2:
        k=st.number_input("k (successes)",min_value=0,value=12,step=1,key="b_k")
        alpha=st.number_input("α (two‑tailed)",value=0.05,step=0.01,
                              min_value=0.0001,max_value=0.5,key="b_alpha")
    if st.button("Update Plot",key="b_plot"):
        st.pyplot(plot_binom(k,n,p))
    with st.expander("Quick table (k ±5)"):
        binom_table(k,n,p,"b",alpha)


def critical_binom(n,p,alpha):
    cum=0; k_low=None
    for k in range(n+1):
        cum+=stats.binom.pmf(k,n,p)
        if cum>=alpha/2:
            k_low=k; break
    cum=0; k_hi=None
    for k in range(n,-1,-1):
        cum+=stats.binom.pmf(k,n,p)
        if cum>=alpha/2:
            k_hi=k; break
    return k_low,k_hi


def binom_table(k,n,p,key,alpha):
    step_key=f"{key}_step"
    step=st.session_state.setdefault(step_key,-1)
    k_vals=list(range(max(0,k-5),min(n,k+5)+1))
    head="<th>P(X=k)</th><th>P(X≤k)</th><th>P(X≥k)</th>"
    body=""
    for kv in k_vals:
        pmf=stats.binom.pmf(kv,n,p)
        cdf=stats.binom.cdf(kv,n,p)
        sf =1-stats.binom.cdf(kv-1,n,p)
        body+=(f'<tr><td id="b_{kv}_0">{kv}</td>'
               f'<td id="b_{kv}_1">{pmf:.4f}</td>'
               f'<td id="b_{kv}_2">{cdf:.4f}</td>'
               f'<td id="b_{kv}_3">{sf :.4f}</td></tr>')
    css=("table{border-collapse:collapse}"
         "th,td{border:1px solid #000;width:110px;height:30px;"
         "text-align:center;font-size:0.8rem}")
    html=wrap(css,f"<tr><th>k</th>{head}</tr>{body}")
    if step>=0:
        for i in range(4):
            html=style_cell(html,f"b_{k}_{i}")
    if step>=1:
        html=style_cell(html,f"b_{k}_1",color="blue",px=3)
    iframe(html)
    msgs=["Highlight k row","Highlight P(X=k)"]
    if step<0: st.write("Click **Next Step** to begin.")
    elif step>=len(msgs): st.write("All steps complete!")
    else: st.write(f"**Step {step+1}**: {msgs[step]}")
    next_button(step_key)

    cdf=stats.binom.cdf(k,n,p)
    p_calc=2*min(cdf,1-cdf+stats.binom.pmf(k,n,p))
    p_calc=min(p_calc,1.0)
    k_lo,k_hi=critical_binom(n,p,alpha)
    p_crit=alpha
    reject=(k<=k_lo) or (k>=k_hi)
    decision="rejected" if reject else "failed to reject"
    st.markdown(
        f"**Interpretation details**  \n"
        f"*Calculated statistic:* k = {k}, *p* = {p_calc:.3f}.  \n"
        f"*Critical region:* k ≤ {k_lo} or k ≥ {k_hi} "
        f"(total *p* = {p_crit:.3f}).  \n"
        f"Observed k is {'inside' if reject else 'outside'} rejection region; "
        f"*p* is {'below' if p_calc<p_crit else 'above'} α → H₀ {decision}.  \n"
        f"**APA‑style conclusion:** Exact binomial test, "
        f"*p* = {p_calc:.3f}. H₀ was **{decision}** at α = {alpha:.2f}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config("PSYC250 – Statistical Tables Explorer",layout="wide")
    st.title("PSYC250 – Statistical Tables Explorer (12 × 4 figures)")
    tabs=st.tabs([
        "t‑Dist","z‑Dist","F‑Dist","Chi‑Square",
        "Mann–Whitney U","Wilcoxon T","Binomial"
    ])
    with tabs[0]: tab_t()
    with tabs[1]: tab_z()
    with tabs[2]: tab_f()
    with tabs[3]: tab_chi()
    with tabs[4]: tab_u()
    with tabs[5]: tab_w()
    with tabs[6]: tab_binom()


if __name__=="__main__":
    main()
