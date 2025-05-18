###############################################################################
#  PSYC-250 – Statistical Tables Explorer
#  ---------------------------------------------------------------------------
#  (Comments as per original)
###############################################################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.switch_backend("Agg")

###############################################################################
#                               COMMON SETUP
###############################################################################

def show_cumulative_note():
    st.info(
        "Note: The values in this table represent cumulative probabilities "
        "(i.e., the area under the curve to the left of a given value). For "
        "one-tailed tests, use the area directly. For two-tailed tests, you must "
        "double the area in the tail beyond your observed value (i.e., "
        "$p=2 \\times (1−P(Z \\le |z|))$). The same logic applies for t-distributions. The table "
        "itself does not change—only how you interpret it does."
    )

def place_label(ax, placed_list, x, y, txt, *, color="blue"):
    dx = dy = 0.0
    for (xx, yy) in placed_list:
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05:
            dx += 0.06
            dy += 0.04
    ax.text(x + dx, y + dy, txt, color=color,
            ha="left", va="bottom", fontsize=8, clip_on=True)
    placed_list.append((x + dx, y + dy))

def style_cell(html: str, cid: str, color: str = "red", px: int = 2) -> str:
    return html.replace(
        f'id="{cid}"',
        f'id="{cid}" style="border:{px}px solid {color};"',
        1
    )

def wrap_table(css: str, table_html: str) -> str:
    return f"<style>{css}</style><table>{table_html}</table>"

def container(html_content: str, *, height: int = 460) -> str:
    return f'<div style="overflow:auto; max-height:{height}px;">{html_content}</div>'

CSS_BASE = (
    "table{border-collapse:collapse}"
    "th,td{border:1px solid #000;height:30px;text-align:center;"
    "font-family:sans-serif;font-size:0.9rem}"
    "th{background:#fafafa}"
)

###############################################################################
#                             TAB 1: t-Distribution
###############################################################################

def plot_t(t_calc, df, input_alpha, tail):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    if df <= 0:
        ax.text(0.5, 0.5, "Degrees of freedom (df) must be positive.", ha='center', va='center')
        return fig
        
    xs = np.linspace(-4.5, 4.5, 400)
    try:
        ys = stats.t.pdf(xs, df)
    except Exception:
        ax.text(0.5, 0.5, f"Invalid df ({df}) for t-distribution PDF.", ha='center', va='center')
        return fig

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")
    placed_labels = []
    plot_crit_val = np.nan 
    try:
        if tail.startswith("one"):
            plot_crit_val_one_sided = stats.t.ppf(1 - input_alpha, df)
            if t_calc >= 0: actual_plot_crit = plot_crit_val_one_sided
            else: actual_plot_crit = -plot_crit_val_one_sided
            
            if t_calc >=0 : region = xs[xs >= actual_plot_crit]
            else: region = xs[xs <= actual_plot_crit]
            # Ensure region is not empty before trying to access ys with it
            if region.size > 0:
                 ax.fill_between(region, ys[np.isin(xs,region)], color="red", alpha=0.3, label="Reject H0")
            ax.axvline(actual_plot_crit, color="green", ls="--")
            pdf_val = stats.t.pdf(actual_plot_crit, df)
            if not np.isnan(pdf_val): place_label(ax, placed_labels, actual_plot_crit, pdf_val + 0.02, f"tcrit={actual_plot_crit:.2f}", color="green")
        else: 
            plot_crit_val = stats.t.ppf(1 - input_alpha / 2, df)
            ax.fill_between(xs[xs >= plot_crit_val], ys[xs >= plot_crit_val], color="red", alpha=0.3)
            ax.fill_between(xs[xs <= -plot_crit_val], ys[xs <= -plot_crit_val], color="red", alpha=0.3, label="Reject H0")
            ax.axvline(plot_crit_val, color="green", ls="--"); ax.axvline(-plot_crit_val, color="green", ls="--")
            pdf_val = stats.t.pdf(plot_crit_val, df)
            if not np.isnan(pdf_val):
                place_label(ax, placed_labels, plot_crit_val, pdf_val + 0.02, f"+tcrit={plot_crit_val:.2f}", color="green")
                place_label(ax, placed_labels, -plot_crit_val, pdf_val + 0.02, f"–tcrit={plot_crit_val:.2f}", color="green")
        
        pdf_at_t_calc = stats.t.pdf(t_calc, df)
        if not np.isnan(pdf_at_t_calc): place_label(ax, placed_labels, t_calc, pdf_at_t_calc + 0.02, f"tcalc={t_calc:.2f}", color="blue")
        ax.axvline(t_calc, color="blue", ls="--")
    except Exception as e: st.warning(f"Could not draw plot elements: {e}")
    ax.set_xlabel("t"); ax.set_ylabel("Density"); ax.legend(loc='upper right')
    ax.set_title(f"t-Distribution (df={df:.0f}, input $\\alpha$={input_alpha:.4f})"); fig.tight_layout()
    return fig

def build_t_html(df: int, user_input_alpha: float, tail: str) -> str:
    rows = list(range(max(1, df - 5), df + 6))
    heads = [
        ("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
        ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001)
    ]
    mode = "one" if tail.startswith("one") else "two"
    col_idx_to_highlight = -1; best_match_info_for_table = None; min_diff_for_table = float('inf')
    for i, (h_mode, h_alpha_val) in enumerate(heads, start=1):
        if h_mode == mode:
            if np.isclose(h_alpha_val, user_input_alpha):
                col_idx_to_highlight = i; best_match_info_for_table = {'alpha_val': h_alpha_val, 'is_exact': True, 'index': i}; break
            current_diff = abs(h_alpha_val - user_input_alpha)
            if current_diff < min_diff_for_table:
                min_diff_for_table = current_diff; best_match_info_for_table = {'index': i, 'alpha_val': h_alpha_val, 'is_exact': False}
            elif current_diff == min_diff_for_table:
                if best_match_info_for_table and h_alpha_val < best_match_info_for_table['alpha_val']:
                     best_match_info_for_table = {'index': i, 'alpha_val': h_alpha_val, 'is_exact': False}
    if col_idx_to_highlight == -1: 
        if best_match_info_for_table:
            col_idx_to_highlight = best_match_info_for_table['index']
            selected_alpha_for_table_highlight = best_match_info_for_table['alpha_val']
            if not best_match_info_for_table.get('is_exact', False): 
                 st.warning(
                    f"Entered $\\alpha$ ({user_input_alpha:.4f}) is not standard for {mode}-tailed. "
                    f"Highlighting closest table $\\alpha$: {selected_alpha_for_table_highlight:.4f}."
                )
        else:
            st.error(f"No columns for mode '{mode}'. Defaulting highlight to col 1."); col_idx_to_highlight = 1
            if heads: st.info(f"Defaulted highlight (col 1: $\\alpha$={heads[0][1]:.3f} for mode='{heads[0][0]}').")
    if not (1 <= col_idx_to_highlight <= len(heads)):
        st.warning(f"Invalid col index {col_idx_to_highlight}. Resetting to 1."); col_idx_to_highlight = 1
    head_html_parts = [f"<th>{m_h}<br>$\\alpha$={a_h:.3f}</th>" for m_h, a_h in heads]; head_html = "".join(head_html_parts)
    body_html = ""
    for r_val in rows:
        row_cells = f'<td id="t_{r_val}_0">{r_val}</td>'
        for i_cell, (m_cell, a_cell) in enumerate(heads, start=1):
            try: crit_val_cell = stats.t.ppf(1 - a_cell if m_cell == "one" else 1 - a_cell / 2, r_val); cell_text = f"{crit_val_cell:.3f}"
            except Exception: cell_text = "N/A"
            row_cells += f'<td id="t_{r_val}_{i_cell}">{cell_text}</td>'
        body_html += f"<tr>{row_cells}</tr>"
    table_code = f"<tr><th>df</th>{head_html}</tr>{body_html}"; html_output = wrap_table(CSS_BASE, table_code)
    if df in rows:
        for i_highlight in range(len(heads) + 1): html_output = style_cell(html_output, f"t_{df}_{i_highlight}")
    for rr_val in rows: html_output = style_cell(html_output, f"t_{rr_val}_{col_idx_to_highlight}")
    if df in rows: html_output = style_cell(html_output, f"t_{df}_{col_idx_to_highlight}", color="blue", px=3)
    return html_output

def t_table(df: int, user_input_alpha: float, tail: str):
    code = build_t_html(df, user_input_alpha, tail)
    st.markdown(container(code), unsafe_allow_html=True)

def t_apa(t_val: float, df: int, input_alpha: float, tail: str):
    p_calc_val, crit_val_apa, reject = np.nan, np.nan, False
    try:
        if tail.startswith("one"):
            p_calc_val = stats.t.sf(abs(t_val), df) 
            crit_val_apa = stats.t.ppf(1 - input_alpha, df) 
            if t_val >= 0: reject = t_val > crit_val_apa
            else: reject = t_val < -crit_val_apa 
        else:
            p_calc_val = stats.t.sf(abs(t_val), df) * 2
            crit_val_apa = stats.t.ppf(1 - input_alpha / 2, df)
            reject = abs(t_val) > crit_val_apa
    except Exception as e: st.warning(f"Could not calc t-APA details: {e}")
    decision = "rejected" if reject else "failed to reject"
    reason_stats = "because $t_{calc}$ was in the rejection region" if reject else "because $t_{calc}$ was not in the rejection region"
    reason_p = f"because $p \\approx {p_calc_val:.3f} < \\alpha$" if reject else f"because $p \\approx {p_calc_val:.3f} \\ge \\alpha$"
    cdf_val = np.nan
    try: cdf_val = stats.t.cdf(t_val, df)
    except Exception: pass # Changed from except:pass to except Exception: pass

    expl_parts = [f"For $t_{{calc}} = {t_val:.2f}$ with $df = {df}$ (using your input $\\alpha = {input_alpha:.4f}$):"]
    expl_parts.append(f"The cumulative probability $P(T \\le {t_val:.2f}) \\approx {cdf_val:.4f}$ (from t-distribution CDF).")
    if tail.startswith("one"):
        if t_val >= 0: expl_parts.append(f"For a **one-tailed** (right tail) test, the p-value is $1 - P(T \\le {t_val:.2f}) \\approx {1-cdf_val if not np.isnan(cdf_val) else np.nan:.4f}$.") # Handle nan for cdf_val
        else: expl_parts.append(f"For a **one-tailed** (left tail) test, the p-value is $P(T \\le {t_val:.2f}) \\approx {cdf_val:.4f}$.")
    else: expl_parts.append(f"For a **two-tailed** test, the p-value is $2 \\times P(T \\ge |{t_val:.2f}|) \\approx {2*min(cdf_val, 1-cdf_val if not np.isnan(cdf_val) else 0.5):.4f}$.")
    expl_parts.append(f"The calculated p-value (more directly) is $p \\approx {p_calc_val:.4f}$.")
    st.write("\n\n".join(expl_parts))
    st.markdown(
        "**APA interpretation**\n"
        f"Calculated statistic: *$t$({df}) = {t_val:.2f}, *$p$ $\\approx$ {p_calc_val:.3f}.\n"
        f"Critical statistic for your input $\\alpha={input_alpha:.4f}$ ({tail}): $t_{{crit}} \\approx {crit_val_apa:.2f}$.\n"
        f"Comparison of statistics ($t_{{calc}}$ vs $t_{{crit}}$ for your $\\alpha$) $\\rightarrow$ H₀ **{decision}** ({reason_stats}).\n"
        f"Comparison of *$p$*-values ($p$ vs your $\\alpha$) $\\rightarrow$ H₀ **{decision}** ({reason_p}).\n"
        f"**APA 7 report:** *$t$({df}) = {t_val:.2f}, *$p$ $\\approx$ {p_calc_val:.3f} ({tail}). The null hypothesis was **{decision}** at $\\alpha$={input_alpha:.2f}."
    )

def tab_t():
    st.subheader("Tab 1 • t-Distribution"); c1,c2=st.columns(2);
    with c1: t_val_in=st.number_input("t statistic",value=2.10,step=0.01,key="t_val_w"); df_in=st.number_input("df",min_value=1,value=10,step=1,key="t_df_w")
    with c2: alpha_in=st.number_input("Your $\\alpha$ (alpha level)",value=0.05,step=0.001,min_value=0.0001,max_value=0.99,format="%.4f",key="t_alpha_w"); tail_in=st.radio("Tail",["one-tailed","two-tailed"],key="t_tail_w",horizontal=True)
    if 't_show_results' not in st.session_state: st.session_state.t_show_results = False
    if st.button("Generate Plot, Table & APA for t-Distribution", key="t_generate_w"): st.session_state.t_show_results = True
    if st.session_state.t_show_results:
        try:
            fig_t = plot_t(float(t_val_in), int(df_in), float(alpha_in), tail_in)
            if fig_t: st.pyplot(fig_t)
        except Exception as e: st.error(f"Plot error: {e}")
        st.write(f"**t-table** (Column highlight for table's closest $\\alpha$ to your input $\\alpha$={alpha_in:.4f})")
        ct_t,ce_t=st.columns([3,2])
        with ct_t:
            try: t_table(int(df_in),float(alpha_in),tail_in); show_cumulative_note()
            except Exception as e: st.error(f"Table error: {e}"); st.exception(e)
        with ce_t:
            try: st.subheader("P-value Calculation Explanation"); t_apa(float(t_val_in),int(df_in),float(alpha_in),tail_in)
            except Exception as e: st.error(f"APA error: {e}")

# ----- Z-Distribution -----
def plot_z(z_calc, input_alpha, tail):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100); xs = np.linspace(-4,4,400); ys = stats.norm.pdf(xs)
    ax.plot(xs, ys, "k"); ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0"); placed_z = []
    try:
        if tail.startswith("one"):
            crit_z_plot = stats.norm.ppf(1 - input_alpha) if z_calc >=0 else stats.norm.ppf(input_alpha)
            region_z = xs[xs >= crit_z_plot] if z_calc >=0 else xs[xs <= crit_z_plot]
            if region_z.size > 0: ax.fill_between(region_z, ys[np.isin(xs,region_z)], color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_z_plot, color="green", ls="--"); place_label(ax,placed_z,crit_z_plot,stats.norm.pdf(crit_z_plot)+0.02,f"z₍crit₎={crit_z_plot:.2f}",color="green")
        else:
            crit_z_plot = stats.norm.ppf(1 - input_alpha/2)
            ax.fill_between(xs[xs>=crit_z_plot], ys[xs>=crit_z_plot],color="red",alpha=0.3); ax.fill_between(xs[xs<=-crit_z_plot],ys[xs<=-crit_z_plot],color="red",alpha=0.3,label="Reject H0")
            ax.axvline(crit_z_plot,color="green",ls="--"); ax.axvline(-crit_z_plot,color="green",ls="--")
            place_label(ax,placed_z,crit_z_plot,stats.norm.pdf(crit_z_plot)+0.02,f"+z₍crit₎={crit_z_plot:.2f}",color="green"); place_label(ax,placed_z,-crit_z_plot,stats.norm.pdf(-crit_z_plot)+0.02,f"–z₍crit₎={crit_z_plot:.2f}",color="green")
        ax.axvline(z_calc,color="blue",ls="--"); place_label(ax,placed_z,z_calc,stats.norm.pdf(z_calc)+0.02,f"z₍calc₎={z_calc:.2f}",color="blue")
    except Exception as e: st.warning(f"Plot error: {e}")
    ax.set_xlabel("z"); ax.set_ylabel("Density"); ax.legend(loc='upper right'); ax.set_title(f"z-Distribution (input $\\alpha$={input_alpha:.4f})"); fig.tight_layout(); return fig

def build_z_html(z_val: float) -> str:
    z_clip = np.clip(z_val, -3.499, 3.499); row_lbl = np.sign(z_clip)*np.floor(abs(z_clip)*10)/10; col_lbl = round(abs(z_clip)-abs(row_lbl),2)
    TRows,TCols = np.round(np.arange(-3.4,3.5,0.1),1),np.round(np.arange(0.00,0.10,0.01),2)
    row_lbl = min(TRows,key=lambda r:abs(r-row_lbl)); col_lbl = min(TCols,key=lambda c:abs(c-col_lbl))
    try: c_row_idx = list(TRows).index(row_lbl)
    except ValueError: c_row_idx = len(TRows)//2
    s_rows = TRows[max(0,c_row_idx-10):min(len(TRows),c_row_idx+11)]
    h_html="".join(f"<th>{c:.2f}</th>" for c in TCols); b_html=""
    for rr_h in s_rows:
        r_cells=f'<td id="z_{rr_h:.1f}_0">{rr_h:.1f}</td>'
        for cc_h in TCols: r_cells+=f'<td id="z_{rr_h:.1f}_{cc_h:.2f}">{stats.norm.cdf(rr_h+cc_h):.4f}</td>'
        b_html+=f"<tr>{r_cells}</tr>"
    t_code=f"<tr><th>z</th>{h_html}</tr>{b_html}"; html_o=wrap_table(CSS_BASE,t_code)
    if row_lbl in s_rows:
        for cc_hl in TCols: html_o=style_cell(html_o,f"z_{row_lbl:.1f}_{cc_hl:.2f}")
        html_o=style_cell(html_o,f"z_{row_lbl:.1f}_0")
        for rr_hl in s_rows: html_o=style_cell(html_o,f"z_{rr_hl:.1f}_{col_lbl:.2f}")
        html_o=style_cell(html_o,f"z_{row_lbl:.1f}_{col_lbl:.2f}",color="blue",px=3)
    return html_o

def z_table(z_val: float): code=build_z_html(z_val); st.markdown(container(code),unsafe_allow_html=True)

def z_apa(z_val: float, input_alpha: float, tail: str):
    p_c,crit_z,rej = np.nan,np.nan,False
    try:
        if tail.startswith("one"): 
            p_c=stats.norm.sf(abs(z_val))
            if z_val>=0: crit_z=stats.norm.ppf(1-input_alpha); rej=z_val>crit_z
            else: crit_z=stats.norm.ppf(input_alpha); rej=z_val<crit_z
        else: 
            p_c=stats.norm.sf(abs(z_val))*2; crit_z=stats.norm.ppf(1-input_alpha/2); rej=abs(z_val)>crit_z
    except Exception as e:st.warning(f"Z APA calc error: {e}")
    dec="rejected" if rej else "failed to reject"; rs=f"$|z_{{calc}}|$ ({abs(z_val):.2f}) {'exceeded' if rej else 'did not exceed'} $|z_{{crit}}|$ ({abs(crit_z):.2f})" ; rp=f"$p \\approx {p_c:.3f} {'<' if rej else '≥'} \\alpha$"
    
    cdf_z=np.nan # Initialize cdf_z
    try: 
        cdf_z=stats.norm.cdf(z_val)
    except Exception: # Changed from except:pass to except Exception: pass
        pass

    expl_z=[f"For $z_{{calc}} = {z_val:.2f}$ (using your input $\\alpha = {input_alpha:.4f}$):", f"Lookup $P(Z \\le {z_val:.2f}) \\approx {cdf_z:.4f}$ (from Z-table/CDF)."]
    if tail.startswith("one"):
        if z_val >=0: expl_z.append(f"For a **one-tailed** (right tail) test, $p = 1 - P(Z \\le {z_val:.2f}) \\approx {1-cdf_z if not np.isnan(cdf_z) else np.nan:.4f}$.") # Handle nan for cdf_z
        else: expl_z.append(f"For a **one-tailed** (left tail) test, $p = P(Z \\le {z_val:.2f}) \\approx {cdf_z:.4f}$.")
    else: expl_z.append(f"For a **two-tailed** test, $p = 2 \\times P(Z \\ge |{z_val:.2f}|) \\approx {2*min(cdf_z, 1-cdf_z if not np.isnan(cdf_z) else 0.5):.4f}$.")
    expl_z.append(f"Calculated $p \\approx {p_c:.4f}$.")
    st.write("\n\n".join(expl_z))
    st.markdown(f"**APA interpretation**\nCalculated statistic: *$z$*={z_val:.2f}, *$p$ $\\approx$ {p_c:.3f}.\nCritical statistic for your $\\alpha$={input_alpha:.4f} ({tail}): $z_{{crit}} \\approx {crit_z:.2f}.\nStatistic comp. $\\rightarrow$ H₀ **{dec}** ({rs}).\n*$p$* comp. $\\rightarrow$ H₀ **{dec}** ({rp}).\n**APA 7:** *$z$*={z_val:.2f}, *$p$ $\\approx$ {p_c:.3f} ({tail}). Null hypothesis was **{dec}** at $\\alpha$={input_alpha:.2f}.")

def tab_z():
    st.subheader("Tab 2 • z-Distribution"); c1,c2=st.columns(2)
    with c1: z_val_in = st.number_input("z statistic",value=1.64,step=0.01,key="z_val_w")
    with c2: alpha_in=st.number_input("Your $\\alpha$",value=0.05,step=0.001,min_value=0.0001,max_value=0.99,format="%.4f",key="z_alpha_w"); tail_in=st.radio("Tail",["one-tailed","two-tailed"],key="z_tail_w",horizontal=True)
    if 'z_show_results' not in st.session_state: st.session_state.z_show_results = False
    if st.button("Generate Plot, Table & APA for z-Distribution", key="z_generate_w"): st.session_state.z_show_results = True
    if st.session_state.z_show_results:
        try: fig_z=plot_z(float(z_val_in),float(alpha_in),tail_in);
             if fig_z:st.pyplot(fig_z)
        except Exception as e:st.error(f"Z-Plot error: {e}")
        st.write("**z-table** (highlighted based on z-statistic value)")
        ct_z,ce_z=st.columns([3,2])
        with ct_z:
            try: z_table(float(z_val_in));show_cumulative_note()
            except Exception as e:st.error(f"Z-Table error: {e}");st.exception(e)
        with ce_z:
            try: st.subheader("P-value Calculation Explanation");z_apa(float(z_val_in),float(alpha_in),tail_in)
            except Exception as e:st.error(f"Z-APA error: {e}")

# ----- F-Distribution -----
def plot_f(f_calc, df1, df2, input_alpha):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    if df1 <= 0 or df2 <= 0: ax.text(0.5,0.5, "df1, df2 must be > 0.", ha='center', va='center'); return fig
    try:
        crit_f_plot = stats.f.ppf(1 - input_alpha, df1, df2)
        ux = max(f_calc, crit_f_plot if not np.isnan(crit_f_plot) else f_calc, 5)*1.5 # Handle potential nan crit_f_plot
        ppf_ux = stats.f.ppf(0.999,df1,df2); ux = ppf_ux*1.1 if not np.isnan(ppf_ux) and ppf_ux < ux else ux
        xs=np.linspace(0,ux,400); ys=stats.f.pdf(xs,df1,df2); v=(~np.isnan(ys)&~np.isinf(ys)); xs,ys=xs[v],ys[v]
        if len(xs)<2: ax.text(0.5,0.5,"Cannot generate F-PDF.",ha='center',va='center'); return fig
        ax.plot(xs,ys,"k"); ax.fill_between(xs,ys,color="lightgrey",alpha=0.25,label="Fail to Reject H0")
        if not np.isnan(crit_f_plot): # Only plot if crit_f_plot is valid
            ax.fill_between(xs[xs>=crit_f_plot],ys[xs>=crit_f_plot],color="red",alpha=0.3,label="Reject H0")
            ax.axvline(crit_f_plot,color="green",ls="--")
            pdf_c=stats.f.pdf(crit_f_plot,df1,df2)
            if not np.isnan(pdf_c): place_label(ax,[],crit_f_plot,pdf_c+0.02,f"F₍crit₎={crit_f_plot:.2f}",color="green") # Empty list for placed_labels

        ax.axvline(f_calc,color="blue",ls="--")
        pdf_f=stats.f.pdf(f_calc,df1,df2)
        if not np.isnan(pdf_f): place_label(ax,[],f_calc,pdf_f+0.02,f"F₍calc₎={f_calc:.2f}",color="blue") # Empty list for placed_labels

    except Exception as e: st.warning(f"F-Plot elements error: {e}")
    ax.set_xlabel("F");ax.set_ylabel("Density");ax.legend(loc='upper right');ax.set_title(f"F-Dist (df1={df1:.0f},df2={df2:.0f}, input $\\alpha$={input_alpha:.4f})");fig.tight_layout();return fig

def build_f_table(df1: int, df2: int, input_alpha: float) -> str:
    rs=list(range(max(1,df1-2),df1+3+1)); cs=list(range(max(1,df2-2),df2+3+1)); ci=-1
    if df2 in cs: ci=cs.index(df2)+1
    hd="".join(f"<th>{c}</th>" for c in cs); bd=""
    for r in rs:
        rc=f'<td id="f_{r}_0">{r}</td>'
        for i,c_val in enumerate(cs,start=1):
            v=np.nan; 
            try: v=stats.f.ppf(1-input_alpha,r,c_val)
            except Exception: pass # Changed from except:pass to except Exception: pass
            rc+=f'<td id="f_{r}_{i}">{v:.2f if not np.isnan(v) else "N/A"}</td>'
        bd+=f"<tr>{rc}</tr>"
    cd=f"<tr><th>df1\\df2</th>{hd}</tr>{bd}"; ht=wrap_table(CSS_BASE,cd)
    if df1 in rs:
        for i in range(len(cs)+1): ht=style_cell(ht,f"f_{df1}_{i}")
    if ci!=-1 and 1 <= ci <= len(cs): # Added check for ci bounds
        for r_rr in rs: ht=style_cell(ht,f"f_{r_rr}_{ci}")
        if df1 in rs:ht=style_cell(ht,f"f_{df1}_{ci}",color="blue",px=3)
    return ht

def f_table(df1: int, df2: int, input_alpha: float): code=build_f_table(df1,df2,input_alpha); st.markdown(container(code,height=300),unsafe_allow_html=True)

def f_apa(f_val: float, df1: int, df2: int, input_alpha: float): # Original APA structure
    pc=np.nan; cr=np.nan; rj=False
    try:
        pc=stats.f.sf(f_val,df1,df2); cr=stats.f.ppf(1-input_alpha,df1,df2); rj=(f_val>cr)
    except Exception as e: st.warning(f"F-APA calc error: {e}")
    dec="rejected" if rj else "failed to reject"
    rs_f=("because F₍calc₎ exceeded F₍crit₎" if rj else "because F₍calc₎ did not exceed F₍crit₎") # Restored original
    rp_f=("because p < α" if rj else "because p ≥ α") # Restored original
    st.markdown(f"**APA interpretation**\nCalculated statistic: *F*({df1},{df2})={f_val:.2f},*p*={pc:.3f}.\nCritical statistic for your $\\alpha$={input_alpha:.4f}: F₍crit₎={cr:.2f}.\nStatistic comparison $\\rightarrow$ H0 **{dec}** ({rs_f}).\n*$p$* comparison $\\rightarrow$ H0 **{dec}** ({rp_f}).\n**APA 7 report:** *F*({df1},{df2})={f_val:.2f},*p*={pc:.3f}. The null hypothesis was **{dec}** at $\\alpha$={input_alpha:.2f}.")

def tab_f():
    st.subheader("Tab 3 • F-Distribution");c1,c2=st.columns(2)
    with c1: fv=st.number_input("F statistic",value=4.32,step=0.01,key="f_v_w");d1=st.number_input("df1 (numerator)",min_value=1,value=5,step=1,key="f_d1_w")
    with c2: d2=st.number_input("df2 (denominator)",min_value=1,value=20,step=1,key="f_d2_w");ai=st.number_input("Your $\\alpha$ (alpha level)",value=0.05,step=0.001,min_value=0.0001,max_value=0.99,format="%.4f",key="f_ai_w")
    if 'f_show_results' not in st.session_state: st.session_state.f_show_results = False
    if st.button("Generate Plot, Table & APA for F-Distribution",key="f_gen_w"): st.session_state.f_show_results = True
    if st.session_state.f_show_results:
        try: figf=plot_f(float(fv),int(d1),int(d2),float(ai));
             if figf:st.pyplot(figf)
        except Exception as e:st.error(f"F-Plot error: {e}")
        st.write("**F-table** (Values are F-crit for your input $\\alpha$. Always one-tailed.)")
        try: f_table(int(d1),int(d2),float(ai)); f_apa(float(fv),int(d1),int(d2),float(ai))
        except Exception as e:st.error(f"F-Table/APA error: {e}");st.exception(e)

# ----- Chi-Square -----
def plot_chi(chi_calc, df, input_alpha):
    fig,ax=plt.subplots(figsize=(12,4),dpi=100)
    if df<=0: ax.text(0.5,0.5,"df must be > 0.",ha='center',va='center'); return fig
    try:
        crit_c_plot=stats.chi2.ppf(1-input_alpha,df); ux_c=max(chi_calc,crit_c_plot if not np.isnan(crit_c_plot) else chi_calc,10)*1.5
        ppf_ux_c=stats.chi2.ppf(0.999,df); ux_c=ppf_ux_c*1.1 if not np.isnan(ppf_ux_c) and ppf_ux_c<ux_c else ux_c
        xs_c=np.linspace(0,ux_c,400); ys_c=stats.chi2.pdf(xs_c,df); v_c=(~np.isnan(ys_c)&~np.isinf(ys_c)); xs_c,ys_c=xs_c[v_c],ys_c[v_c]
        if len(xs_c)<2: ax.text(0.5,0.5,"Cannot generate Chi-PDF.",ha='center',va='center'); return fig
        ax.plot(xs_c,ys_c,"k"); ax.fill_between(xs_c,ys_c,color="lightgrey",alpha=0.25,label="Fail to Reject H0")
        if not np.isnan(crit_c_plot):
            ax.fill_between(xs_c[xs_c>=crit_c_plot],ys_c[xs_c>=crit_c_plot],color="red",alpha=0.3,label="Reject H0")
            ax.axvline(crit_c_plot,color="green",ls="--")
            pdf_cc=stats.chi2.pdf(crit_c_plot,df)
            if not np.isnan(pdf_cc):place_label(ax,[],crit_c_plot,pdf_cc+0.02,f"χ²₍crit₎={crit_c_plot:.2f}",color="green")
        ax.axvline(chi_calc,color="blue",ls="--")
        pdf_chc=stats.chi2.pdf(chi_calc,df)
        if not np.isnan(pdf_chc):place_label(ax,[],chi_calc,pdf_chc+0.02,f"χ²₍calc₎={chi_calc:.2f}",color="blue")
    except Exception as e: st.warning(f"Chi-Plot elements error: {e}")
    ax.set_xlabel("χ²");ax.set_ylabel("Density");ax.legend(loc='upper right');ax.set_title(f"χ²-Dist (df={df:.0f}, input $\\alpha$={input_alpha:.4f})");fig.tight_layout();return fig

def build_chi_table(df: int, alpha_for_highlight: float) -> str:
    rs_c=list(range(max(1,df-2),df+3+1)); std_as=[0.10,0.05,0.025,0.01,0.005,0.001]; ci_c=-1
    if alpha_for_highlight in std_as: ci_c=std_as.index(alpha_for_highlight)+1
    else: 
        md_c=float('inf')
        for i,sa in enumerate(std_as,start=1):
            if abs(sa-alpha_for_highlight)<md_c: md_c=abs(sa-alpha_for_highlight); ci_c=i
        if ci_c!=-1 and std_as: st.warning(f"Table $\\alpha$ {alpha_for_highlight:.4f} for highlight not exact, using closest: {std_as[ci_c-1]:.3f}.")
        elif not std_as: ci_c = 1 # Should not happen
        else: ci_c=1 
    hd_c="".join(f"<th>{a:.3f}</th>" for a in std_as);bd_c=""
    for r in rs_c:
        rc_c=f'<td id="chi_{r}_0">{r}</td>'
        for i,ac in enumerate(std_as,start=1):
            vc=np.nan; 
            try: vc=stats.chi2.ppf(1-ac,r)
            except Exception: pass # Changed from except:pass
            rc_c+=f'<td id="chi_{r}_{i}">{vc:.2f if not np.isnan(vc) else "N/A"}</td>'
        bd_c+=f"<tr>{rc_c}</tr>"
    cdc=f"<tr><th>df\\α</th>{hd_c}</tr>{bd_c}";htc=wrap_table(CSS_BASE,cdc)
    if df in rs_c:
        for i in range(len(std_as)+1): htc=style_cell(htc,f"chi_{df}_{i}")
    if ci_c!=-1 and 1 <= ci_c <= len(std_as): # Added bounds check for ci_c
        for rr in rs_c:htc=style_cell(htc,f"chi_{rr}_{ci_c}")
        if df in rs_c:htc=style_cell(htc,f"chi_{df}_{ci_c}",color="blue",px=3)
    return htc

def chi_table(df: int, alpha_for_highlight: float): code=build_chi_table(df,alpha_for_highlight); st.markdown(container(code,height=300),unsafe_allow_html=True)

def chi_apa(chi_val: float, df: int, input_alpha: float): # Original APA structure
    pc=np.nan; cr=np.nan; rj=False
    try:
        pc=stats.chi2.sf(chi_val,df); cr=stats.chi2.ppf(1-input_alpha,df); rj=(chi_val>cr)
    except Exception as e: st.warning(f"Chi-APA calc error: {e}")
    dec="rejected" if rj else "failed to reject"; rs_c="because χ²₍calc₎ exceeded χ²₍crit₎" if rj else "because χ²₍calc₎ did not exceed χ²₍crit₎"; rp_c="because p < α" if rj else "because p ≥ α" # Original reasons
    st.markdown(f"**APA interpretation**\nCalculated statistic: χ²({df})={chi_val:.2f},*p*={pc:.3f}.\nCritical statistic for your $\\alpha$={input_alpha:.4f}: χ²₍crit₎={cr:.2f}.\nStatistic comparison $\\rightarrow$ H0 **{dec}** ({rs_c}).\n*$p$* comparison $\\rightarrow$ H0 **{dec}** ({rp_c}).\n**APA 7 report:** χ²({df})={chi_val:.2f},*p*={pc:.3f}. The null hypothesis was **{dec}** at $\\alpha$={input_alpha:.2f}.")

def tab_chi():
    st.subheader("Tab 4 • Chi-Square Distribution");c1,c2=st.columns(2)
    with c1: cv=st.number_input("χ² statistic",value=7.88,step=0.01,key="c_v_w");dfi=st.number_input("df ",min_value=1,value=3,step=1,key="c_df_w") # Key was chi_val and chi_df
    with c2:
        sa_c=[0.10,0.05,0.025,0.01,0.005,0.001]; afth=st.selectbox("Table's $\\alpha$ for highlight",sa_c,index=1,key="c_as_w")
        aia=st.number_input("Your test's $\\alpha$",value=0.05,step=0.001,min_value=0.0001,max_value=0.99,format="%.4f",key="c_aia_w")
    if 'chi_show_results' not in st.session_state: st.session_state.chi_show_results = False
    if st.button("Generate Plot, Table & APA for Chi-Square",key="c_gen_w"): st.session_state.chi_show_results = True
    if st.session_state.chi_show_results:
        try: figc=plot_chi(float(cv),int(dfi),float(aia));
             if figc:st.pyplot(figc)
        except Exception as e:st.error(f"Chi-Plot error: {e}")
        st.write(f"**χ²-table** (Table columns show standard $\\alpha$'s. Column for selected table $\\alpha$={afth} is highlighted.)")
        try: chi_table(int(dfi),float(afth)); chi_apa(float(cv),int(dfi),float(aia))
        except Exception as e:st.error(f"Chi-Table/APA error: {e}");st.exception(e)

# ----- Mann-Whitney U -----
def plot_u(u_calc, n1, n2, input_alpha, tail):
    mu_u = n1*n2/2.0; sigma_u = np.sqrt(n1*n2*(n1+n2+1)/12.0)
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    if sigma_u <= 1e-9: ax.text(0.5, 0.5, "Cannot plot U: Variance is zero.", ha='center', va='center'); return fig
    
    plot_min_x = max(0, mu_u - 4.5 * sigma_u) 
    plot_max_x = min(n1 * n2, mu_u + 4.5 * sigma_u) 
    if plot_min_x >= plot_max_x : 
        plot_min_x = max(0, u_calc - (3 * sigma_u if sigma_u > 1e-9 else 1) -1) # Use sigma_u if valid
        plot_max_x = u_calc + (3 * sigma_u if sigma_u > 1e-9 else 1) + 1
        if plot_min_x >= plot_max_x: 
             plot_max_x = plot_min_x + 2 

    xs = np.linspace(plot_min_x, plot_max_x, 400); ys = stats.norm.pdf(xs, mu_u, sigma_u)
    valid_ys_u = ~np.isnan(ys) & ~np.isinf(ys); xs, ys = xs[valid_ys_u], ys[valid_ys_u]
    if len(xs) < 2: ax.text(0.5,0.5, "Cannot generate U-PDF.", ha='center', va='center'); return fig
    ax.plot(xs, ys, "k"); ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")
    placed_u_p = []; from math import floor, ceil # Original import location
    try:
        if tail.startswith("one"):
            if u_calc <= mu_u: 
                zc=stats.norm.ppf(input_alpha); cu=floor(mu_u+zc*sigma_u)
                if xs[xs<=cu].size > 0: ax.fill_between(xs[xs<=cu],ys[xs<=cu],color="red",alpha=0.3,label="Reject H0")
                ax.axvline(cu,color="green",ls="--")
                if not np.isnan(stats.norm.pdf(cu,mu_u,sigma_u)): place_label(ax,placed_u_p,cu,stats.norm.pdf(cu,mu_u,sigma_u)+0.005,f"Ucrit L≈{cu}",color="green")
            else: 
                zc=stats.norm.ppf(1-input_alpha); cu=ceil(mu_u+zc*sigma_u)
                if xs[xs>=cu].size > 0: ax.fill_between(xs[xs>=cu],ys[xs>=cu],color="red",alpha=0.3,label="Reject H0")
                ax.axvline(cu,color="green",ls="--")
                if not np.isnan(stats.norm.pdf(cu,mu_u,sigma_u)): place_label(ax,placed_u_p,cu,stats.norm.pdf(cu,mu_u,sigma_u)+0.005,f"Ucrit U≈{cu}",color="green")
        else:
            zcl=stats.norm.ppf(input_alpha/2); cul=floor(mu_u+zcl*sigma_u)
            zcu=stats.norm.ppf(1-input_alpha/2); cuu=ceil(mu_u+zcu*sigma_u) # Corrected upper critical
            if xs[xs<=cul].size > 0: ax.fill_between(xs[xs<=cul],ys[xs<=cul],color="red",alpha=0.3)
            if xs[xs>=cuu].size > 0: ax.fill_between(xs[xs>=cuu],ys[xs>=cuu],color="red",alpha=0.3,label="Reject H0")
            ax.axvline(cul,color="green",ls="--"); ax.axvline(cuu,color="green",ls="--")
            if not np.isnan(stats.norm.pdf(cul,mu_u,sigma_u)): place_label(ax,placed_u_p,cul,stats.norm.pdf(cul,mu_u,sigma_u)+0.005,f"Ucrit L≈{cul}",color="green")
            if not np.isnan(stats.norm.pdf(cuu,mu_u,sigma_u)): place_label(ax,placed_u_p,cuu,stats.norm.pdf(cuu,mu_u,sigma_u)+0.005,f"Ucrit U≈{cuu}",color="green")
        pdf_uc=stats.norm.pdf(u_calc,mu_u,sigma_u)
        if not np.isnan(pdf_uc):place_label(ax,placed_u_p,u_calc,pdf_uc+0.005,f"Ucalc={u_calc}",color="blue")
        ax.axvline(u_calc,color="blue",ls="--")
    except Exception as e: st.warning(f"U-Plot elements error: {e}")
    ax.set_xlabel("U (Normal Approx.)");ax.set_ylabel("Approx. density");ax.legend(loc='upper right');ax.set_title(f"Mann-Whitney U (Approx. $n_1$={n1},$n_2$={n2}, input $\\alpha$={input_alpha:.4f})");fig.tight_layout();return fig

def u_crit(n1:int, n2:int, input_alpha:float, tail:str)->any:
    mu_u=n1*n2/2.0; sigma_u=np.sqrt(n1*n2*(n1+n2+1)/12.0)
    if sigma_u<=1e-9: return np.nan
    from math import floor
    z_crit_table=stats.norm.ppf(input_alpha if tail.startswith("one") else input_alpha/2)
    return int(floor(mu_u+z_crit_table*sigma_u))

def build_u_table(n1:int, n2:int, input_alpha:float, tail:str)->str:
    rs_u=list(range(max(2,n1-2),n1+3+1)); cs_u=list(range(max(2,n2-2),n2+3+1)); ci_u=-1
    if n2 in cs_u: ci_u=cs_u.index(n2)+1
    hd_u="".join(f"<th>$n_2$={c}</th>" for c in cs_u);bd_u=""
    for r in rs_u:
        rc_u=f'<td id="u_{r}_0">$n_1$={r}</td>'
        for i,c_val in enumerate(cs_u,start=1):
            v_u=u_crit(r,c_val,input_alpha,tail); rc_u+=f'<td id="u_{r}_{i}">{v_u if not np.isnan(v_u) else "N/A"}</td>'
        bd_u+=f"<tr>{rc_u}</tr>"
    cde_u=f"<tr><th>$n_1 \\setminus n_2$</th>{hd_u}</tr>{bd_u}";ht_u=wrap_table(CSS_BASE,cde_u)
    if n1 in rs_u:
        for i in range(len(cs_u)+1):ht_u=style_cell(ht_u,f"u_{n1}_{i}")
    if ci_u!=-1 and 1 <= ci_u <= len(cs_u): # Added bounds check for ci_u
        for rr_u in rs_u: ht_u=style_cell(ht_u,f"u_{rr_u}_{ci_u}")
        if n1 in rs_u: ht_u=style_cell(ht_u,f"u_{n1}_{ci_u}",color="blue",px=3)
    return ht_u

def u_table(n1:int, n2:int, input_alpha:float, tail:str): code=build_u_table(n1,n2,input_alpha,tail);st.markdown(container(code,height=300),unsafe_allow_html=True)

def u_apa(u_val: int, n1: int, n2: int, input_alpha: float, tail: str):
    mu_u=n1*n2/2.0; sigma_u=np.sqrt(n1*n2*(n1+n2+1)/12.0)
    za,pc_u,cr_za_u,rj_u = np.nan,np.nan,np.nan,False
    if sigma_u>1e-9:
        uc=0.5 if u_val<mu_u else (-0.5 if u_val>mu_u else 0); za=(u_val-mu_u+uc)/sigma_u
        if tail.startswith("one"):
            pc_u=stats.norm.cdf(za) if za<=0 else stats.norm.sf(za)
            cr_za_u=stats.norm.ppf(input_alpha) if pc_u<0.5 else stats.norm.ppf(1-input_alpha)
            rj_u=(za<cr_za_u if pc_u<0.5 else za>cr_za_u)
        else: pc_u=2*stats.norm.sf(abs(za)); cr_za_u=stats.norm.ppf(1-input_alpha/2); rj_u=abs(za)>cr_za_u
    else: st.warning("Cannot calculate Z approx for U: σ=0.")
    dec_u="rejected" if rj_u else "failed to reject"; rs_u=f"$Z_{{approx}}$({za:.2f}) in rej. region (vs $Z_{{crit}} \\approx {cr_za_u:.2f}$)" if rj_u else f"$Z_{{approx}}$ not in rej. region"; rp_u=f"$p \\approx {pc_u:.3f} < \\alpha$" if rj_u else f"$p \\approx {pc_u:.3f} \\ge \\alpha$"
    
    # P-Value Calculation Explanation section
    st.subheader("P-value Calculation Explanation (Normal Approx.)")
    expl_u_list = [f"For $U_{{calc}} = {u_val}$ (with $n_1={n1}, n_2={n2}$, using your input $\\alpha = {input_alpha:.4f}$):"]
    if sigma_u > 1e-9:
        expl_u_list.append(f"Using Normal Approximation: $\\mu_U \\approx {mu_u:.2f}$, $\\sigma_U \\approx {sigma_u:.2f}$.")
        expl_u_list.append(f"The Z-statistic (with continuity correction) $Z_{{approx}} \\approx {za:.2f}$.")
        cdf_za = stats.norm.cdf(za) # CDF of the calculated Z_approx
        expl_u_list.append(f"The cumulative probability $P(Z \\le Z_{{approx}}) \\approx {cdf_za:.4f}$.")
        if tail.startswith("one"):
            # Explanation needs to match how p_calc_u was determined
            if za <=0 : # Implies U_calc was on the lower side, p-value is P(Z <= Z_approx)
                 expl_u_list.append(f"For a **one-tailed** test (e.g., testing if $U$ is significantly small), $p \\approx P(Z \\le Z_{{approx}}) \\approx {cdf_za:.4f}$.")
            else: # Implies U_calc was on the upper side, p-value is P(Z >= Z_approx)
                 expl_u_list.append(f"For a **one-tailed** test (e.g., testing if $U$ is significantly large), $p \\approx 1 - P(Z \\le Z_{{approx}}) \\approx {1-cdf_za:.4f}$.")
        else: # two-tailed
            expl_u_list.append(f"For a **two-tailed** test, $p \\approx 2 \\times P(Z \\ge |Z_{{approx}}|) \\approx {pc_u:.4f}$.")
        expl_u_list.append(f"The calculated p-value (from Z with cont. corr.) is $p \\approx {pc_u:.4f}$.")
    else: expl_u_list.append("Cannot provide detailed explanation as $\\sigma_U$ is zero.")
    st.write("\n\n".join(expl_u_list))

    # APA Interpretation section
    st.markdown(f"**APA interpretation (Normal Approximation for U)**\nMann-Whitney U = {u_val}, Z$\\approx${za:.2f}, approx. *$p$*={pc_u:.3f} ({tail}).\nCrit Z for your $\\alpha$={input_alpha:.4f} ({tail}): $Z_{{crit}} \\approx {cr_za_u:.2f}$.\nH₀ **{dec_u}** ({rs_u}) via Z.\nH₀ **{dec_u}** ({rp_u}) via p.\n**APA 7 (approx.):** Mann-Whitney U indicated null was **{dec_u}**, U={u_val}, Z$\\approx${za:.2f},*p*$\\approx${pc_u:.3f}({tail}), $\\alpha$={input_alpha:.2f}.")

def tab_u():
    st.subheader("Tab 5 • Mann-Whitney U Distribution");c1,c2=st.columns(2)
    with c1:u_v=st.number_input("U stat val",min_value=0,value=23,step=1,key="u_v_w");n1v=st.number_input("n1 (samp1)",min_value=1,value=8,step=1,key="u_n1_w")
    with c2:n2v=st.number_input("n2 (samp2)",min_value=1,value=10,step=1,key="u_n2_w");aiv=st.number_input("Your $\\alpha$ for U",value=0.05,step=0.001,min_value=0.0001,max_value=0.99,format="%.4f",key="u_ai_w");tiv=st.radio("Tail for U",["one-tailed","two-tailed"],key="u_ti_w",horizontal=True)
    if 'u_show_results' not in st.session_state: st.session_state.u_show_results = False
    if st.button("Generate Plot, Table & APA for Mann-Whitney U",key="u_gen_w"): st.session_state.u_show_results = True
    if st.session_state.u_show_results:
        if int(n1v)<1 or int(n2v)<1:st.error("n1 and n2 must be >= 1.")
        else:
            try: fig_u=plot_u(int(u_v),int(n1v),int(n2v),float(aiv),tiv);
                 if fig_u:st.pyplot(fig_u)
            except Exception as e:st.error(f"U-Plot error: {e}")
            st.write("**U-table** (Approx. Lower Crit U's for your input $\\alpha$)")
            ct_u,ce_u=st.columns([3,2]) # Keep table and explanation separate for layout consistency
            with ct_u:
                try: u_table(int(n1v),int(n2v),float(aiv),tiv); st.info("U-table shows approx. lower crit U's for *your input α*. Reject H₀ if obs. U ≤ table U (for appropriate tail).")
                except Exception as e:st.error(f"U-Table error: {e}");st.exception(e)
            with ce_u: # The u_apa function now contains the "P-value Calc Explanation" subheader
                try: u_apa(int(u_v),int(n1v),int(n2v),float(aiv),tiv) # This will print the subheader and then APA
                except Exception as e:st.error(f"U-APA error: {e}")

def tab_wilcoxon_t(): st.subheader("Tab 6 • Wilcoxon Signed-Rank T"); st.write("Wilcoxon T functionality to be implemented.")
def tab_binomial(): st.subheader("Tab 7 • Binomial Distribution"); st.write("Binomial functionality to be implemented.")

def main():
    st.set_page_config(layout="wide",page_title="Statistical Tables Explorer")
    st.title("Oli's - Statistical Table Explorer")
    tabs_list=["t-Dist","z-Dist","F-Dist","Chi-Square","Mann-Whitney U","Wilcoxon T","Binomial"]
    tabs_created=st.tabs(tabs_list)
    with tabs_created[0]: tab_t()
    with tabs_created[1]: tab_z()
    with tabs_created[2]: tab_f()
    with tabs_created[3]: tab_chi()
    with tabs_created[4]: tab_u()
    with tabs_created[5]: tab_wilcoxon_t()
    with tabs_created[6]: tab_binomial()

if __name__ == "__main__":
    main()
