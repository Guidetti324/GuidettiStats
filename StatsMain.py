###############################################################################
# psyc250_streamlit_final.py
# Single-file Streamlit app for the "PSYC250 Explorer," with no duplicates.
###############################################################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.switch_backend("Agg")  # For Streamlit environment (headless)

###############################################################################
# HELPER: place_label to avoid overlapping text
###############################################################################

def place_label(ax, label_positions, x, y, text, color='blue'):
    offset_x = 0.0
    offset_y = 0.02
    for (xx, yy) in label_positions:
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05:
            offset_x += 0.06
            offset_y += 0.04
    final_x = x + offset_x
    final_y = y + offset_y
    ax.text(final_x, final_y, text, color=color, ha="left", va="bottom", fontsize=8)
    label_positions.append((final_x, final_y))

###############################################################################
# 1) T-DISTRIBUTION TAB
###############################################################################

def show_t_distribution_tab():
    st.subheader("t-Distribution")

    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input(
            "t statistic (t-dist)",
            value=2.87,
            key="t_dist_tval"
        )
        df_val = st.number_input(
            "df (t-dist)",
            value=55,
            key="t_dist_df"
        )
    with c2:
        alpha_val = st.number_input(
            "Alpha (t-dist)",
            value=0.05,
            step=0.01,
            key="t_dist_alpha"
        )
        tail_type = st.radio(
            "Tail Type (t-dist)",
            ["one-tailed", "two-tailed"],
            key="t_dist_tail"
        )

    if st.button("Update t-Plot", key="btn_t_plot"):
        fig = plot_t_distribution(t_val, df_val, alpha_val, tail_type)
        st.pyplot(fig)

    with st.expander("Show Table Lookup (±5 around df)"):
        render_t_table_lookup(df_val, alpha_val, tail_type)

def plot_t_distribution(t_val, df, alpha, tail_s):
    fig, ax = plt.subplots(figsize=(5,3))
    label_positions=[]
    x = np.linspace(-4,4,400)
    y = stats.t.pdf(x, df)
    ax.plot(x, y, color='black')
    ax.fill_between(x, y, color='lightgrey', alpha=0.2, label="Fail to Reject H₀")

    def labelme(xx, yy, txt, c='green'):
        place_label(ax, label_positions, xx, yy, txt, c)

    if tail_s == "one-tailed":
        t_crit = stats.t.ppf(1 - alpha, df)
        note_txt = ""
        if abs(alpha - 0.05) < 1e-9:
            note_txt = " (same as two-tailed α=0.10!)"
        region = x[x >= t_crit]
        ax.fill_between(region, y[x >= t_crit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit, color='green', linestyle='--')
        labelme(t_crit, stats.t.pdf(t_crit, df)+0.02, f"t_crit={t_crit:.4f}{note_txt}", 'green')
        sig = (t_val > t_crit)
        final_crit = t_crit
    else:
        t_crit_r = stats.t.ppf(1 - alpha/2, df)
        t_crit_l = stats.t.ppf(alpha/2, df)
        rx = x[x >= t_crit_r]
        lx = x[x <= t_crit_l]
        ax.fill_between(rx, y[x >= t_crit_r], color='red', alpha=0.3)
        ax.fill_between(lx, y[x <= t_crit_l], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(t_crit_r, color='green', linestyle='--')
        ax.axvline(t_crit_l, color='green', linestyle='--')
        labelme(t_crit_r, stats.t.pdf(t_crit_r, df)+0.02, f"+t_crit={t_crit_r:.4f}", 'green')
        labelme(t_crit_l, stats.t.pdf(t_crit_l, df)+0.02, f"-t_crit={t_crit_l:.4f}", 'green')
        sig = (abs(t_val) > abs(t_crit_r))
        final_crit = abs(t_crit_r)

    ax.axvline(t_val, color='blue', linestyle='--')
    place_label(ax, label_positions, t_val, stats.t.pdf(t_val, df)+0.02,
                f"t_calc={t_val:.4f}", 'blue')

    if sig:
        msg = f"t={t_val:.4f} > t_crit={final_crit:.4f} → Reject H₀"
    else:
        msg = f"t={t_val:.4f} ≤ t_crit={final_crit:.4f} → Fail to Reject H₀"

    ax.set_title(f"t-Distribution (df={df})\n{msg}", fontsize=10)
    ax.set_xlabel("t value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig

def render_t_table_lookup(df_user, alpha_user, tail_s):
    st.write("This is a partial t-table ±5 around user df.")
    step = st.slider("Highlight Step", 0, 4, 0, key="t_dist_table_step")
    st.write(f"Pretend we highlight row df={df_user} and col near α={alpha_user}, step={step}")


###############################################################################
# 2) Z-DISTRIBUTION
###############################################################################

def show_z_distribution_tab():
    st.subheader("z-Distribution")
    c1, c2 = st.columns(2)
    with c1:
        z_val = st.number_input(
            "z statistic (z-dist)",
            value=1.64,
            key="z_dist_zval"
        )
    with c2:
        alpha_val = st.number_input(
            "Alpha (z-dist)",
            value=0.05,
            step=0.01,
            key="z_dist_alpha"
        )
        tail_type = st.radio(
            "Tail Type (z-dist)",
            ["one-tailed", "two-tailed"],
            key="z_dist_tail"
        )

    if st.button("Update z-Plot", key="btn_z_plot"):
        fig = plot_z_distribution(z_val, alpha_val, tail_type)
        st.pyplot(fig)

    with st.expander("Show Partial z-Table Lookup (±10 rows)"):
        st.write("_We highlight row/col intersection in a partial ztable_")

def plot_z_distribution(z_val, alpha, tail_s):
    fig, ax = plt.subplots(figsize=(5,3))
    label_positions=[]
    x = np.linspace(-4,4,400)
    y = stats.norm.pdf(x)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2,label="Fail to Reject H₀")

    def labelme(xx,yy,txt,c='blue'):
        place_label(ax, label_positions, xx, yy, txt, c)

    if tail_s == "one-tailed":
        z_crit = stats.norm.ppf(1 - alpha)
        rx = x[x >= z_crit]
        ax.fill_between(rx, y[x >= z_crit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(z_crit, color='green', linestyle='--')
        labelme(z_crit, stats.norm.pdf(z_crit), f"z_crit={z_crit:.4f}", 'green')
        sig = (z_val > z_crit)
        final_crit = z_crit
    else:
        z_crit_r = stats.norm.ppf(1 - alpha/2)
        z_crit_l = -z_crit_r
        rx = x[x >= z_crit_r]
        lx = x[x <= z_crit_l]
        ax.fill_between(rx, y[x >= z_crit_r], color='red', alpha=0.3)
        ax.fill_between(lx, y[x <= z_crit_l], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(z_crit_r, color='green', linestyle='--')
        ax.axvline(z_crit_l, color='green', linestyle='--')
        labelme(z_crit_r, stats.norm.pdf(z_crit_r), f"+z_crit={z_crit_r:.4f}", 'green')
        labelme(z_crit_l, stats.norm.pdf(z_crit_l), f"-z_crit={z_crit_r:.4f}", 'green')
        sig = (abs(z_val) > z_crit_r)
        final_crit = z_crit_r

    ax.axvline(z_val, color='blue', linestyle='--')
    labelme(z_val, stats.norm.pdf(z_val), f"z_calc={z_val:.4f}", 'blue')

    msg = (f"z={z_val:.4f} > z_crit={final_crit:.4f} → Reject H₀"
           if sig else
           f"z={z_val:.4f} ≤ z_crit={final_crit:.4f} → Fail to Reject H₀")

    ax.set_title(f"Z-Distribution\n{msg}", fontsize=10)
    ax.legend()
    fig.tight_layout()
    return fig

###############################################################################
# 3) F-DISTRIBUTION
###############################################################################

def show_f_distribution_tab():
    st.subheader("F-Distribution (One-tailed)")
    c1, c2 = st.columns(2)
    with c1:
        f_val= st.number_input("F statistic (F-dist)", value=3.49, key="f_dist_stat")
        df1= st.number_input("df1 (F-dist)", value=3, key="f_dist_df1")
        df2= st.number_input("df2 (F-dist)", value=12, key="f_dist_df2")
    with c2:
        alpha= st.number_input("Alpha (F-dist)", value=0.05, step=0.01, key="f_dist_alpha")

    if st.button("Update F-Plot", key="btn_f_plot"):
        fig = plot_f_distribution(f_val, df1, df2, alpha)
        st.pyplot(fig)

    with st.expander("Show F Table Lookup (±5)"):
        st.write("**Minimal** highlight approach for df1, df2, etc.")

def plot_f_distribution(f_val, df1, df2, alpha):
    fig, ax= plt.subplots(figsize=(5,3))
    label_positions=[]
    x= np.linspace(0,5,500)
    y= stats.f.pdf(x, df1, df2)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey',alpha=0.2,label="Fail to Reject H₀")

    f_crit= stats.f.ppf(1-alpha, df1, df2)
    rx= x[x>= f_crit]
    ax.fill_between(rx, y[x>=f_crit], color='red', alpha=0.3, label="Reject H₀")
    ax.axvline(f_crit, color='green', linestyle='--')
    place_label(ax, label_positions, f_crit, stats.f.pdf(f_crit,df1,df2)+0.02,
                f"F_crit={f_crit:.4f}", 'green')

    ax.axvline(f_val, color='blue', linestyle='--')
    place_label(ax, label_positions, f_val, stats.f.pdf(f_val,df1,df2)+0.02,
                f"F_calc={f_val:.4f}", 'blue')

    sig= (f_val> f_crit)
    msg= (f"F={f_val:.4f} > F_crit={f_crit:.4f} → Reject H₀"
          if sig else
          f"F={f_val:.4f} ≤ F_crit={f_crit:.4f} → Fail to Reject H₀")

    ax.set_title(f"F-Distribution (df1={df1}, df2={df2})\n{msg}", fontsize=10)
    ax.legend()
    fig.tight_layout()
    return fig

###############################################################################
# 4) CHI-SQUARE
###############################################################################

def show_chi_square_tab():
    st.subheader("Chi-Square (One-tailed)")
    c1, c2 = st.columns(2)
    with c1:
        chi_val= st.number_input("Chi-square stat (χ²-dist)", value=10.5, key="chi_dist_chi")
        df_val= st.number_input("df (χ²-dist)", value=12, key="chi_dist_df")
    with c2:
        alpha= st.number_input("Alpha (χ²-dist)", value=0.05, step=0.01, key="chi_dist_alpha")

    if st.button("Update χ²-Plot", key="btn_chi_plot"):
        fig= plot_chi_square(chi_val, df_val, alpha)
        st.pyplot(fig)

    with st.expander("Show Chi-Square Table Lookup"):
        st.write("**Minimal** ±5 approach for df highlight")

def plot_chi_square(chi_val, df, alpha):
    fig, ax= plt.subplots(figsize=(5,3))
    label_positions=[]
    x_max= max(30, df*2)
    x= np.linspace(0,x_max,400)
    y= stats.chi2.pdf(x, df)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey',alpha=0.2,label="Fail to Reject H₀")

    chi_crit= stats.chi2.ppf(1-alpha, df)
    rx= x[x>= chi_crit]
    ax.fill_between(rx, y[x>= chi_crit], color='red', alpha=0.3, label="Reject H₀")
    ax.axvline(chi_crit, color='green', linestyle='--')
    place_label(ax, label_positions, chi_crit, stats.chi2.pdf(chi_crit,df)+0.02,
                f"chi2_crit={chi_crit:.4f}", 'green')

    ax.axvline(chi_val, color='blue', linestyle='--')
    place_label(ax, label_positions, chi_val, stats.chi2.pdf(chi_val,df)+0.02,
                f"chi2_calc={chi_val:.4f}", 'blue')

    sig= (chi_val> chi_crit)
    msg= (f"χ²={chi_val:.4f} > χ²_crit={chi_crit:.4f} → Reject H₀"
          if sig else
          f"χ²={chi_val:.4f} ≤ χ²_crit={chi_crit:.4f} → Fail to Reject H₀")

    ax.set_title(f"Chi-Square (df={df})\n{msg}", fontsize=10)
    ax.legend()
    fig.tight_layout()
    return fig

###############################################################################
# 5) MANN–WHITNEY U
###############################################################################

def show_mann_whitney_tab():
    st.subheader("Mann–Whitney U (one/two-tailed)")

    c1, c2 = st.columns(2)
    with c1:
        U_val= st.number_input("U statistic (MW)", value=5, key="mw_uval")
        n1= st.number_input("n1 (MW)", value=5, key="mw_n1")
        n2= st.number_input("n2 (MW)", value=6, key="mw_n2")
    with c2:
        alpha= st.number_input("Alpha (MW)", value=0.05, step=0.01, key="mw_alpha2")
        tail_s= st.radio("Tail Type (MW)", ["one-tailed","two-tailed"], key="mw_tail2")

    if st.button("Update Mann–Whitney Plot", key="btn_mw_plot"):
        fig= plot_mann_whitney(U_val,n1,n2,alpha,tail_s)
        st.pyplot(fig)

    with st.expander("Show Mann–Whitney Table Lookup (±5)"):
        st.write("**Minimal** ±5 approach demonstration")

def plot_mann_whitney(U_val,n1,n2,alpha,tail_s):
    fig, ax= plt.subplots(figsize=(5,3))
    label_positions=[]
    meanU= n1*n2/2
    sdU= np.sqrt(n1*n2*(n1+n2+1)/12)
    z_val= (U_val- meanU)/ sdU

    x= np.linspace(-4,4,400)
    y= stats.norm.pdf(x)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2,label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        zCrit= stats.norm.ppf(1- alpha)
        rx= x[x>=zCrit]
        ax.fill_between(rx, y[x>=zCrit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit), f"z_crit={zCrit:.2f}", 'green')
        sig= (z_val> zCrit)
    else:
        zCrit= stats.norm.ppf(1- alpha/2)
        zCritR= zCrit
        zCritL= -zCrit
        rx= x[x>=zCritR]
        lx= x[x<=zCritL]
        ax.fill_between(rx, y[x>=zCritR], color='red', alpha=0.3)
        ax.fill_between(lx, y[x<=zCritL], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCritR, color='green', linestyle='--')
        ax.axvline(zCritL, color='green', linestyle='--')
        place_label(ax, label_positions, zCritR, stats.norm.pdf(zCritR), f"+z_crit={zCritR:.2f}", 'green')
        place_label(ax, label_positions, zCritL, stats.norm.pdf(zCritL), f"-z_crit={zCritR:.2f}", 'green')
        sig= (abs(z_val)> zCrit)

    ax.axvline(z_val, color='blue', linestyle='--')
    place_label(ax, label_positions, z_val, stats.norm.pdf(z_val), f"z_calc={z_val:.2f}", 'blue')

    msg= (f"U={U_val}, z={z_val:.2f} → Reject H₀" if sig
          else f"U={U_val}, z={z_val:.2f} → Fail to Reject H₀")
    ax.set_title(f"Mann–Whitney (n1={n1}, n2={n2})\n{msg}", fontsize=10)
    ax.legend()
    fig.tight_layout()
    return fig

###############################################################################
# 6) WILCOXON
###############################################################################

def show_wilcoxon_tab():
    st.subheader("Wilcoxon Signed-Rank (one/two-tailed)")

    c1, c2 = st.columns(2)
    with c1:
        T_val= st.number_input("T statistic (Wilcoxon)", value=5, key="wil_tval")
        N_val= st.number_input("N (non-zero diffs) (Wilcoxon)", value=6, key="wil_ndiffs")
    with c2:
        alpha= st.number_input("Alpha (Wilcoxon)", value=0.05, step=0.01, key="wil_alpha2")
        tail_s= st.radio("Tail Type (Wilcoxon)", ["one-tailed","two-tailed"], key="wil_tail2")

    if st.button("Update Wilcoxon Plot", key="btn_wil_plot"):
        fig= plot_wilcoxon(T_val, N_val, alpha, tail_s)
        st.pyplot(fig)

    with st.expander("Show Wilcoxon Table Lookup"):
        st.write("**Minimal** ±5 approach demonstration")

def plot_wilcoxon(T_val, N, alpha, tail_s):
    fig, ax= plt.subplots(figsize=(5,3))
    label_positions=[]
    meanT= N*(N+1)/4
    sdT= np.sqrt(N*(N+1)*(2*N+1)/24)
    z_val= (T_val- meanT)/ sdT

    x= np.linspace(-4,4,400)
    y= stats.norm.pdf(x)
    ax.plot(x,y,color='black')
    ax.fill_between(x,y,color='lightgrey', alpha=0.2,label="Fail to Reject H₀")

    if tail_s=="one-tailed":
        zCrit= stats.norm.ppf(1- alpha)
        rx= x[x>=zCrit]
        ax.fill_between(rx, y[x>=zCrit], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCrit, color='green', linestyle='--')
        place_label(ax, label_positions, zCrit, stats.norm.pdf(zCrit), f"z_crit={zCrit:.2f}", 'green')
        sig= (z_val> zCrit)
    else:
        zCrit= stats.norm.ppf(1- alpha/2)
        zCritR= zCrit
        zCritL= -zCrit
        rx= x[x>=zCritR]
        lx= x[x<=zCritL]
        ax.fill_between(rx, y[x>=zCritR], color='red', alpha=0.3)
        ax.fill_between(lx, y[x<=zCritL], color='red', alpha=0.3, label="Reject H₀")
        ax.axvline(zCritR, color='green', linestyle='--')
        ax.axvline(zCritL, color='green', linestyle='--')
        place_label(ax, label_positions, zCritR, stats.norm.pdf(zCritR),
                    f"+z_crit={zCritR:.2f}", 'green')
        place_label(ax, label_positions, zCritL, stats.norm.pdf(zCritL),
                    f"-z_crit={zCritR:.2f}", 'green')
        sig= (abs(z_val)> zCrit)

    ax.axvline(z_val, color='blue', linestyle='--')
    place_label(ax, label_positions, z_val, stats.norm.pdf(z_val), f"z_calc={z_val:.2f}", 'blue')

    msg= (f"T={T_val}, z={z_val:.2f} → Reject H₀" if sig
          else f"T={T_val}, z={z_val:.2f} → Fail to Reject H₀")
    ax.set_title(f"Wilcoxon (N={N})\n{msg}", fontsize=10)
    ax.legend()
    fig.tight_layout()
    return fig

###############################################################################
# 7) BINOMIAL
###############################################################################

def show_binomial_tab():
    st.subheader("Binomial (one/two-tailed)")

    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input("n (binomial)", value=10, key="bin_nval")
        x = st.number_input("x (successes) (binomial)", value=3, key="bin_xval")
    with c2:
        p = st.number_input("p (binomial)", value=0.5, step=0.01, key="bin_pval")
        alpha = st.number_input("Alpha (binomial)", value=0.05, step=0.01, key="bin_alphaval")
        tail_s= st.radio("Tail Type (binomial)", ["one-tailed","two-tailed"], key="bin_tailval")

    if st.button("Update Binomial Plot", key="btn_bin_plot"):
        fig= plot_binomial(n,x,p,alpha,tail_s)
        st.pyplot(fig)

    with st.expander("Show Binomial Table Lookup (±5)"):
        st.write("**Minimal** demonstration of row => col => intersection")

def plot_binomial(n,x,p,alpha,tail_s):
    fig, ax= plt.subplots(figsize=(5,3))
    label_positions=[]
    k_vals= np.arange(n+1)
    pmf_vals= stats.binom.pmf(k_vals,n,p)
    bars= ax.bar(k_vals, pmf_vals, color='lightgrey', edgecolor='black')

    mean_= n*p
    if tail_s=="one-tailed":
        if x<= mean_:
            p_val= stats.binom.cdf(x,n,p)
        else:
            p_val= 1- stats.binom.cdf(x-1,n,p)
    else:
        if x<= mean_:
            p_val= stats.binom.cdf(x,n,p)*2
        else:
            p_val= (1- stats.binom.cdf(x-1,n,p))*2
        p_val= min(1.0, p_val)

    sig= (p_val< alpha)
    msg= f"p-value={p_val:.4f} => " + ("Reject H₀" if sig else "Fail to Reject H₀")

    if tail_s=="one-tailed":
        if x< mean_:
            for i in range(0,x+1):
                bars[i].set_color('red')
        else:
            for i in range(x,n+1):
                bars[i].set_color('red')
    else:
        if x<= mean_:
            for i in range(0,x+1):
                bars[i].set_color('red')
            hi_start= max(0,n-x)
            for i in range(hi_start,n+1):
                bars[i].set_color('red')
        else:
            for i in range(x,n+1):
                bars[i].set_color('red')
            for i in range(0,(n-x)+1):
                bars[i].set_color('red')

    if 0<= x<= n:
        bars[x].set_color('blue')

    import matplotlib.patches as mpatches
    legend_patches= [
        mpatches.Patch(facecolor='lightgrey', edgecolor='black', label='Fail to Reject H₀'),
        mpatches.Patch(facecolor='red', edgecolor='black', label='Reject H₀ region'),
        mpatches.Patch(facecolor='blue', edgecolor='black', label='Observed x')
    ]
    ax.legend(handles=legend_patches, loc='best')

    ax.set_title(f"Binomial(n={n}, p={p:.2f})\n{msg}", fontsize=10)
    ax.set_xlabel("x (successes)")
    ax.set_ylabel("PMF")
    fig.tight_layout()
    return fig

###############################################################################
# MAIN
###############################################################################

def main():
    st.set_page_config(page_title="PSYC250 - Streamlit Stats Explorer", layout="wide")
    st.title("PSYC250 - Statistical Tables Explorer (Streamlit, No Duplicates)")

    tab_labels = [
        "t-Distribution",
        "z-Distribution",
        "F-Distribution",
        "Chi-Square",
        "Mann–Whitney U",
        "Wilcoxon Signed-Rank",
        "Binomial"
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        show_t_distribution_tab()
    with tabs[1]:
        show_z_distribution_tab()
    with tabs[2]:
        show_f_distribution_tab()
    with tabs[3]:
        show_chi_square_tab()
    with tabs[4]:
        show_mann_whitney_tab()
    with tabs[5]:
        show_wilcoxon_tab()
    with tabs[6]:
        show_binomial_tab()


if __name__=="__main__":
    main()
