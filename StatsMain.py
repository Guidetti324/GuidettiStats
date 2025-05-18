df,k,alpha_0.01,alpha_0.05,alpha_0.10
1,2,90.030,17.970,8.990
1,3,135.000,26.980,13.480
1,4,164.300,32.820,16.360
1,5,185.700,37.080,18.480
1,6,202.200,40.410,20.150
2,2,14.000,6.085,3.927
2,3,19.020,8.331,5.040
2,4,22.290,9.798,5.757
2,5,24.720,10.880,6.286
2,6,26.630,11.740,6.701
3,2,8.260,4.501,3.182
3,3,10.620,5.910,3.953
3,4,12.170,6.825,4.498
3,5,13.330,7.515,4.903
3,6,14.240,8.037,5.221
5,2,5.700,3.639,2.768
5,3,6.980,4.602,3.401
5,4,7.800,5.218,3.813
5,5,8.420,5.673,4.102
5,6,8.910,6.033,4.328
5,10,10.850,7.540,5.350
10,2,4.470,3.151,2.409
10,3,5.270,3.877,2.913
10,4,5.830,4.327,3.240
10,5,6.260,4.654,3.481
10,6,6.620,4.909,3.671
10,10,7.940,6.076,4.500
20,2,3.960,2.950,2.280
20,3,4.640,3.578,2.722
20,4,5.060,3.983,3.000
20,5,5.390,4.295,3.207
20,6,5.660,4.544,3.372
20,10,6.770,5.556,4.080
20,20,8.000,6.800,5.000
60,2,3.520,2.756,2.116
60,3,4.100,3.314,2.523
60,4,4.480,3.631,2.762
60,5,4.770,3.859,2.933
60,6,4.990,4.039,3.066
60,10,5.830,4.823,3.620
60,20,6.970,5.890,4.400
120,2,3.360,2.617,2.000
120,3,3.980,3.356,2.500
120,4,4.360,3.685,2.750
120,5,4.650,3.919,2.920
120,6,4.850,4.103,3.050
120,10,5.500,4.686,3.450
120,20,6.800,5.300,3.980
```
Now, here is the Python code for `app.py`.


```python
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from io import StringIO # For reading CSV string

# Helper function to create APA style p-value string
def apa_p_value(p_val):
    if p_val < 0.001:
        return "p < .001"
    else:
        return f"p = {p_val:.3f}"

# Embedded CSV data for Tukey HSD fallback
TUKEY_CSV_DATA = """df,k,alpha_0.01,alpha_0.05,alpha_0.10
1,2,90.030,17.970,8.990
1,3,135.000,26.980,13.480
1,4,164.300,32.820,16.360
1,5,185.700,37.080,18.480
1,6,202.200,40.410,20.150
2,2,14.000,6.085,3.927
2,3,19.020,8.331,5.040
2,4,22.290,9.798,5.757
2,5,24.720,10.880,6.286
2,6,26.630,11.740,6.701
3,2,8.260,4.501,3.182
3,3,10.620,5.910,3.953
3,4,12.170,6.825,4.498
3,5,13.330,7.515,4.903
3,6,14.240,8.037,5.221
5,2,5.700,3.639,2.768
5,3,6.980,4.602,3.401
5,4,7.800,5.218,3.813
5,5,8.420,5.673,4.102
5,6,8.910,6.033,4.328
5,10,10.850,7.540,5.350
10,2,4.470,3.151,2.409
10,3,5.270,3.877,2.913
10,4,5.830,4.327,3.240
10,5,6.260,4.654,3.481
10,6,6.620,4.909,3.671
10,10,7.940,6.076,4.500
20,2,3.960,2.950,2.280
20,3,4.640,3.578,2.722
20,4,5.060,3.983,3.000
20,5,5.390,4.295,3.207
20,6,5.660,4.544,3.372
20,10,6.770,5.556,4.080
20,20,8.000,6.800,5.000
60,2,3.520,2.756,2.116
60,3,4.100,3.314,2.523
60,4,4.480,3.631,2.762
60,5,4.770,3.859,2.933
60,6,4.990,4.039,3.066
60,10,5.830,4.823,3.620
60,20,6.970,5.890,4.400
120,2,3.360,2.617,2.000
120,3,3.980,3.356,2.500
120,4,4.360,3.685,2.750
120,5,4.650,3.919,2.920
120,6,4.850,4.103,3.050
120,10,5.500,4.686,3.450
120,20,6.800,5.300,3.980
"""

# Function to get Tukey q critical value from CSV
def get_tukey_q_from_csv(df_error, k, alpha):
    try:
        df_tukey = pd.read_csv(StringIO(TUKEY_CSV_DATA))
    except Exception as e:
        st.error(f"Error reading embedded Tukey CSV data: {e}")
        return None

    alpha_col_map = {0.01: 'alpha_0.01', 0.05: 'alpha_0.05', 0.10: 'alpha_0.10'}
    if alpha not in alpha_col_map:
        st.warning(f"Alpha value {alpha} not directly available in CSV (0.01, 0.05, 0.10). Using alpha=0.05 as default for lookup.")
        alpha_lookup = 0.05
    else:
        alpha_lookup = alpha
    
    target_col = alpha_col_map[alpha_lookup]

    # Filter for k
    df_filtered_k = df_tukey[df_tukey['k'] == k]
    if df_filtered_k.empty:
         # Find nearest lower k
        available_k = sorted(df_tukey['k'].unique())
        lower_k = [val for val in available_k if val < k]
        if not lower_k: 
            st.warning(f"k value {k} is too small for CSV lookup. Using smallest k.")
            k_to_use = min(available_k)
        else:
            k_to_use = max(lower_k)
        st.warning(f"Exact k={k} not found. Using nearest lower k={k_to_use} from CSV.")
        df_filtered_k = df_tukey[df_tukey['k'] == k_to_use]


    # Find nearest lower df
    df_filtered_k_sorted = df_filtered_k.sort_values('df')
    
    # Exact match for df
    exact_match = df_filtered_k_sorted[df_filtered_k_sorted['df'] == df_error]
    if not exact_match.empty:
        return exact_match.iloc[0][target_col]

    # Nearest lower df
    lower_dfs = df_filtered_k_sorted[df_filtered_k_sorted['df'] < df_error]
    if not lower_dfs.empty:
        chosen_row = lower_dfs.iloc[-1]
        st.warning(f"Exact df={df_error} not found for k={k_to_use}. Using nearest lower df={chosen_row['df']} from CSV.")
        return chosen_row[target_col]

    # Nearest higher df if no lower df
    higher_dfs = df_filtered_k_sorted[df_filtered_k_sorted['df'] > df_error]
    if not higher_dfs.empty:
        chosen_row = higher_dfs.iloc[0]
        st.warning(f"Exact df={df_error} not found for k={k_to_use}, no lower df available. Using nearest higher df={chosen_row['df']} from CSV.")
        return chosen_row[target_col]
        
    st.error(f"Could not find a suitable value in CSV for df={df_error}, k={k_to_use}, alpha={alpha_lookup}.")
    return None


# --- Tab 1: t-distribution ---
def tab_t_distribution():
    st.header("t-Distribution Explorer")
    col1, col2 = st.columns([2, 1.5]) # Adjusted column widths

    with col1:
        st.subheader("Inputs")
        alpha_t = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_t")
        df_t = st.number_input("Degrees of Freedom (df)", 1, 1000, 10, 1, key="df_t")
        tail_t = st.radio("Tail Selection", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_t")
        test_stat_t = st.number_input("Calculated t-statistic", value=0.0, format="%.3f", key="test_stat_t")

        st.subheader("Distribution Plot")
        fig_t, ax_t = plt.subplots(figsize=(8,5)) # Adjusted figure size
        x_t = np.linspace(stats.t.ppf(0.0001, df_t), stats.t.ppf(0.9999, df_t), 500)
        y_t = stats.t.pdf(x_t, df_t)
        ax_t.plot(x_t, y_t, 'b-', lw=2, label=f't-distribution (df={df_t})')

        crit_val_t_upper = None
        crit_val_t_lower = None

        if tail_t == "Two-tailed":
            crit_val_t_upper = stats.t.ppf(1 - alpha_t / 2, df_t)
            crit_val_t_lower = stats.t.ppf(alpha_t / 2, df_t)
            x_fill_upper = np.linspace(crit_val_t_upper, stats.t.ppf(0.9999, df_t), 100)
            x_fill_lower = np.linspace(stats.t.ppf(0.0001, df_t), crit_val_t_lower, 100)
            ax_t.fill_between(x_fill_upper, stats.t.pdf(x_fill_upper, df_t), color='red', alpha=0.5, label=f'α/2 = {alpha_t/2:.4f}')
            ax_t.fill_between(x_fill_lower, stats.t.pdf(x_fill_lower, df_t), color='red', alpha=0.5)
            ax_t.axvline(crit_val_t_upper, color='red', linestyle='--', lw=1)
            ax_t.axvline(crit_val_t_lower, color='red', linestyle='--', lw=1)
        elif tail_t == "One-tailed (right)":
            crit_val_t_upper = stats.t.ppf(1 - alpha_t, df_t)
            x_fill_upper = np.linspace(crit_val_t_upper, stats.t.ppf(0.9999, df_t), 100)
            ax_t.fill_between(x_fill_upper, stats.t.pdf(x_fill_upper, df_t), color='red', alpha=0.5, label=f'α = {alpha_t:.4f}')
            ax_t.axvline(crit_val_t_upper, color='red', linestyle='--', lw=1)
        else: # One-tailed (left)
            crit_val_t_lower = stats.t.ppf(alpha_t, df_t)
            x_fill_lower = np.linspace(stats.t.ppf(0.0001, df_t), crit_val_t_lower, 100)
            ax_t.fill_between(x_fill_lower, stats.t.pdf(x_fill_lower, df_t), color='red', alpha=0.5, label=f'α = {alpha_t:.4f}')
            ax_t.axvline(crit_val_t_lower, color='red', linestyle='--', lw=1)

        ax_t.axvline(test_stat_t, color='green', linestyle='-', lw=2, label=f'Test Stat = {test_stat_t:.3f}')
        ax_t.set_title(f't-Distribution (df={df_t}) with Critical Region(s)')
        ax_t.set_xlabel('t-value')
        ax_t.set_ylabel('Probability Density')
        ax_t.legend()
        ax_t.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_t)

        st.subheader("Critical Value Table Snippet")
        alphas_table = [0.10, 0.05, 0.01, alpha_t]
        alphas_table = sorted(list(set(alphas_table))) # Unique and sorted

        table_data_t = {"Alpha (One-Tail)": [], "Alpha (Two-Tail)": [], "Critical t (Upper)": []}
        if tail_t != "One-tailed (left)": # Show upper critical for right and two-tailed
             for a_val in alphas_table:
                table_data_t["Alpha (One-Tail)"].append(f"{a_val:.4f}")
                table_data_t["Alpha (Two-Tail)"].append(f"{a_val*2:.4f}" if a_val*2 <=1 else "-") # for two-tailed interpretation
                table_data_t["Critical t (Upper)"].append(f"{stats.t.ppf(1 - a_val, df_t):.3f}")
        if tail_t != "One-tailed (right)": # Show lower critical for left and two-tailed
            if not table_data_t["Critical t (Upper)"]: # if only left tail
                 for a_val in alphas_table:
                    table_data_t["Alpha (One-Tail)"].append(f"{a_val:.4f}")
                    table_data_t["Alpha (Two-Tail)"].append(f"{a_val*2:.4f}" if a_val*2 <=1 else "-")
                    table_data_t["Critical t (Lower)"] = [f"{stats.t.ppf(a_val, df_t):.3f}" for a_val in alphas_table]
            else: # if two-tailed, add lower column
                table_data_t["Critical t (Lower)"] = [f"{stats.t.ppf(a_val, df_t):.3f}" for a_val in alphas_table]


        df_table_t = pd.DataFrame(table_data_t)
        
        def highlight_alpha_row(row):
            is_selected_alpha_one_tail = abs(float(row["Alpha (One-Tail)"]) - alpha_t) < 1e-5
            is_selected_alpha_two_tail = abs(float(row["Alpha (Two-Tail)"]) - alpha_t) < 1e-5 if row["Alpha (Two-Tail)"] != "-" else False

            if (tail_t == "Two-tailed" and is_selected_alpha_two_tail) or \
               (tail_t != "Two-tailed" and is_selected_alpha_one_tail):
                return ['background-color: yellow'] * len(row)
            return [''] * len(row)

        st.markdown(df_table_t.style.apply(highlight_alpha_row, axis=1).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows critical t-values for df={df_t}. Highlighted row corresponds to your selected α={alpha_t:.4f}.")

        st.markdown("""
        **Cumulative Table Note:**
        * For **one-tailed tests**, the 'Alpha (One-Tail)' column gives the area in that single tail. Use the corresponding critical value.
        * For **two-tailed tests**, the 'Alpha (Two-Tail)' column represents the total area in *both* tails combined. Each tail would contain half of this alpha (i.e., use the 'Alpha (One-Tail)' value for ppf calculation but consider it as alpha/2). The critical values shown are for one tail; for a two-tailed test, you'd use both positive and negative critical values associated with alpha/2 in each tail.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value is the probability of observing a test statistic as extreme as, or more extreme than, the calculated statistic ({test_stat_t:.3f}), assuming the null hypothesis is true.
        * For a **two-tailed test**, it's the probability of observing a value ≤ -|{test_stat_t:.3f}| or ≥ |{test_stat_t:.3f}|. This is calculated as `2 * P(T ≥ |{test_stat_t:.3f}|)`.
        * For a **one-tailed (right) test**, it's `P(T ≥ {test_stat_t:.3f})`.
        * For a **one-tailed (left) test**, it's `P(T ≤ {test_stat_t:.3f})`.
        """)

        st.subheader("Summary")
        # Calculate p-value for the test statistic
        if test_stat_t >= 0: # For positive test stat
            p_val_t_one_right = stats.t.sf(test_stat_t, df_t)
            p_val_t_one_left = stats.t.cdf(test_stat_t, df_t)
        else: # For negative test stat
            p_val_t_one_right = stats.t.sf(test_stat_t, df_t) # P(T > test_stat_t)
            p_val_t_one_left = stats.t.cdf(test_stat_t, df_t) # P(T < test_stat_t)
        
        p_val_t_two = 2 * stats.t.sf(abs(test_stat_t), df_t)


        crit_val_display = "N/A"
        p_val_for_crit_val_display = alpha_t # This is alpha itself

        if tail_t == "Two-tailed":
            crit_val_display = f"±{crit_val_t_upper:.3f}"
            p_val_calc = p_val_t_two
            decision_crit = abs(test_stat_t) > crit_val_t_upper
            comparison_crit_str = f"|{test_stat_t:.3f}| ({abs(test_stat_t):.3f}) > {crit_val_t_upper:.3f}" if decision_crit else f"|{test_stat_t:.3f}| ({abs(test_stat_t):.3f}) ≤ {crit_val_t_upper:.3f}"
        elif tail_t == "One-tailed (right)":
            crit_val_display = f"{crit_val_t_upper:.3f}"
            p_val_calc = p_val_t_one_right
            decision_crit = test_stat_t > crit_val_t_upper
            comparison_crit_str = f"{test_stat_t:.3f} > {crit_val_t_upper:.3f}" if decision_crit else f"{test_stat_t:.3f} ≤ {crit_val_t_upper:.3f}"
        else: # One-tailed (left)
            crit_val_display = f"{crit_val_t_lower:.3f}"
            p_val_calc = p_val_t_one_left
            decision_crit = test_stat_t < crit_val_t_lower
            comparison_crit_str = f"{test_stat_t:.3f} < {crit_val_t_lower:.3f}" if decision_crit else f"{test_stat_t:.3f} ≥ {crit_val_t_lower:.3f}"

        decision_p_alpha = p_val_calc < alpha_t
        
        st.markdown(f"""
        1.  **Critical Value ({tail_t})**: {crit_val_display}
            * *Associated p-value (α)*: {p_val_for_crit_val_display:.4f}
        2.  **Calculated Test Statistic**: {test_stat_t:.3f}
            * *Calculated p-value*: {p_val_calc:.4f} ({apa_p_value(p_val_calc)})
        3.  **Decision (Critical Value Method)**: The null hypothesis is **{'rejected' if decision_crit else 'not rejected'}**.
            * *Reason*: Because t(calc) {comparison_crit_str} relative to t(crit).
        4.  **Decision (p-value Method)**: The null hypothesis is **{'rejected' if decision_p_alpha else 'not rejected'}**.
            * *Reason*: Because {apa_p_value(p_val_calc)} is {'less than' if decision_p_alpha else 'not less than'} α ({alpha_t:.4f}).
        5.  **APA 7 Style Report**:
            *t*({df_t}) = {test_stat_t:.2f}, {apa_p_value(p_val_calc)}. The null hypothesis was {'rejected' if decision_p_alpha else 'not rejected'} at the α = {alpha_t:.2f} level.
        """)

# --- Tab 2: z-distribution ---
def tab_z_distribution():
    st.header("z-Distribution (Normal) Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_z = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_z")
        # Z-distribution does not have df in the same way as t, F, chi2. It's for large samples or known population variance.
        tail_z = st.radio("Tail Selection", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_z")
        test_stat_z = st.number_input("Calculated z-statistic", value=0.0, format="%.3f", key="test_stat_z")

        st.subheader("Distribution Plot")
        fig_z, ax_z = plt.subplots(figsize=(8,5))
        x_z = np.linspace(stats.norm.ppf(0.0001), stats.norm.ppf(0.9999), 500)
        y_z = stats.norm.pdf(x_z)
        ax_z.plot(x_z, y_z, 'b-', lw=2, label='Standard Normal Distribution (z)')

        crit_val_z_upper = None
        crit_val_z_lower = None

        if tail_z == "Two-tailed":
            crit_val_z_upper = stats.norm.ppf(1 - alpha_z / 2)
            crit_val_z_lower = stats.norm.ppf(alpha_z / 2)
            x_fill_upper = np.linspace(crit_val_z_upper, stats.norm.ppf(0.9999), 100)
            x_fill_lower = np.linspace(stats.norm.ppf(0.0001), crit_val_z_lower, 100)
            ax_z.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, label=f'α/2 = {alpha_z/2:.4f}')
            ax_z.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.5)
            ax_z.axvline(crit_val_z_upper, color='red', linestyle='--', lw=1)
            ax_z.axvline(crit_val_z_lower, color='red', linestyle='--', lw=1)
        elif tail_z == "One-tailed (right)":
            crit_val_z_upper = stats.norm.ppf(1 - alpha_z)
            x_fill_upper = np.linspace(crit_val_z_upper, stats.norm.ppf(0.9999), 100)
            ax_z.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, label=f'α = {alpha_z:.4f}')
            ax_z.axvline(crit_val_z_upper, color='red', linestyle='--', lw=1)
        else: # One-tailed (left)
            crit_val_z_lower = stats.norm.ppf(alpha_z)
            x_fill_lower = np.linspace(stats.norm.ppf(0.0001), crit_val_z_lower, 100)
            ax_z.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.5, label=f'α = {alpha_z:.4f}')
            ax_z.axvline(crit_val_z_lower, color='red', linestyle='--', lw=1)

        ax_z.axvline(test_stat_z, color='green', linestyle='-', lw=2, label=f'Test Stat = {test_stat_z:.3f}')
        ax_z.set_title('Standard Normal Distribution with Critical Region(s)')
        ax_z.set_xlabel('z-value')
        ax_z.set_ylabel('Probability Density')
        ax_z.legend()
        ax_z.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_z)

        st.subheader("Critical Value Table Snippet")
        alphas_table_z = [0.10, 0.05, 0.01, alpha_z]
        alphas_table_z = sorted(list(set(alphas_table_z)))

        table_data_z = {"Alpha (One-Tail)": [], "Alpha (Two-Tail)": [], "Critical z (Upper)": []}
        if tail_z != "One-tailed (left)":
             for a_val in alphas_table_z:
                table_data_z["Alpha (One-Tail)"].append(f"{a_val:.4f}")
                table_data_z["Alpha (Two-Tail)"].append(f"{a_val*2:.4f}" if a_val*2 <=1 else "-")
                table_data_z["Critical z (Upper)"].append(f"{stats.norm.ppf(1 - a_val):.3f}")
        if tail_z != "One-tailed (right)":
            if not table_data_z["Critical z (Upper)"]:
                 for a_val in alphas_table_z:
                    table_data_z["Alpha (One-Tail)"].append(f"{a_val:.4f}")
                    table_data_z["Alpha (Two-Tail)"].append(f"{a_val*2:.4f}" if a_val*2 <=1 else "-")
                    table_data_z["Critical z (Lower)"] = [f"{stats.norm.ppf(a_val):.3f}" for a_val in alphas_table_z]
            else:
                table_data_z["Critical z (Lower)"] = [f"{stats.norm.ppf(a_val):.3f}" for a_val in alphas_table_z]
        
        df_table_z = pd.DataFrame(table_data_z)
        
        def highlight_alpha_row_z(row):
            is_selected_alpha_one_tail = abs(float(row["Alpha (One-Tail)"]) - alpha_z) < 1e-5
            is_selected_alpha_two_tail = abs(float(row["Alpha (Two-Tail)"]) - alpha_z) < 1e-5 if row["Alpha (Two-Tail)"] != "-" else False
            if (tail_z == "Two-tailed" and is_selected_alpha_two_tail) or \
               (tail_z != "Two-tailed" and is_selected_alpha_one_tail):
                return ['background-color: yellow'] * len(row)
            return [''] * len(row)

        st.markdown(df_table_z.style.apply(highlight_alpha_row_z, axis=1).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows critical z-values. Highlighted row corresponds to your selected α={alpha_z:.4f}.")
        st.markdown("""
        **Cumulative Table Note:** (Same as t-distribution)
        * For **one-tailed tests**, 'Alpha (One-Tail)' is the area in that tail.
        * For **two-tailed tests**, 'Alpha (Two-Tail)' is the total area. Each tail has alpha/2.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value is the probability of observing a z-statistic as extreme as, or more extreme than, {test_stat_z:.3f}, assuming H₀ is true.
        * **Two-tailed**: `2 * P(Z ≥ |{test_stat_z:.3f}|)`
        * **One-tailed (right)**: `P(Z ≥ {test_stat_z:.3f})`
        * **One-tailed (left)**: `P(Z ≤ {test_stat_z:.3f})`
        """)

        st.subheader("Summary")
        p_val_z_one_right = stats.norm.sf(test_stat_z)
        p_val_z_one_left = stats.norm.cdf(test_stat_z)
        p_val_z_two = 2 * stats.norm.sf(abs(test_stat_z))

        crit_val_z_display = "N/A"
        p_val_for_crit_val_z_display = alpha_z

        if tail_z == "Two-tailed":
            crit_val_z_display = f"±{crit_val_z_upper:.3f}"
            p_val_calc_z = p_val_z_two
            decision_crit_z = abs(test_stat_z) > crit_val_z_upper
            comparison_crit_str_z = f"|{test_stat_z:.3f}| ({abs(test_stat_z):.3f}) > {crit_val_z_upper:.3f}" if decision_crit_z else f"|{test_stat_z:.3f}| ({abs(test_stat_z):.3f}) ≤ {crit_val_z_upper:.3f}"
        elif tail_z == "One-tailed (right)":
            crit_val_z_display = f"{crit_val_z_upper:.3f}"
            p_val_calc_z = p_val_z_one_right
            decision_crit_z = test_stat_z > crit_val_z_upper
            comparison_crit_str_z = f"{test_stat_z:.3f} > {crit_val_z_upper:.3f}" if decision_crit_z else f"{test_stat_z:.3f} ≤ {crit_val_z_upper:.3f}"
        else: # One-tailed (left)
            crit_val_z_display = f"{crit_val_z_lower:.3f}"
            p_val_calc_z = p_val_z_one_left
            decision_crit_z = test_stat_z < crit_val_z_lower
            comparison_crit_str_z = f"{test_stat_z:.3f} < {crit_val_z_lower:.3f}" if decision_crit_z else f"{test_stat_z:.3f} ≥ {crit_val_z_lower:.3f}"

        decision_p_alpha_z = p_val_calc_z < alpha_z
        
        st.markdown(f"""
        1.  **Critical Value ({tail_z})**: {crit_val_z_display}
            * *Associated p-value (α)*: {p_val_for_crit_val_z_display:.4f}
        2.  **Calculated Test Statistic**: {test_stat_z:.3f}
            * *Calculated p-value*: {p_val_calc_z:.4f} ({apa_p_value(p_val_calc_z)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_z else 'not rejected'}**.
            * *Reason*: z(calc) {comparison_crit_str_z} relative to z(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_z else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_z)} is {'less than' if decision_p_alpha_z else 'not less than'} α ({alpha_z:.4f}).
        5.  **APA 7 Style Report**:
            *z* = {test_stat_z:.2f}, {apa_p_value(p_val_calc_z)}. The null hypothesis was {'rejected' if decision_p_alpha_z else 'not rejected'} at α = {alpha_z:.2f}.
        """)

# --- Tab 3: F-distribution ---
def tab_f_distribution():
    st.header("F-Distribution Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_f = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_f")
        dfn_f = st.number_input("Numerator Degrees of Freedom (df1)", 1, 1000, 3, 1, key="dfn_f")
        dfd_f = st.number_input("Denominator Degrees of Freedom (df2)", 1, 1000, 20, 1, key="dfd_f")
        # F-test is typically one-tailed (right-tailed) for ANOVA.
        # Can also be two-tailed for variance equality, but less common in basic explorers.
        # For simplicity, focusing on right-tailed, as it's the primary use for F-tables.
        tail_f = st.radio("Tail Selection", ("One-tailed (right)", "Two-tailed (for variance test)"), key="tail_f")
        test_stat_f = st.number_input("Calculated F-statistic", value=1.0, format="%.3f", min_value=0.0, key="test_stat_f")

        st.subheader("Distribution Plot")
        fig_f, ax_f = plt.subplots(figsize=(8,5))
        # Ensure range starts from a small positive value for F-distribution
        x_f_max = stats.f.ppf(0.999, dfn_f, dfd_f) if stats.f.ppf(0.999, dfn_f, dfd_f) > 10 else 10
        x_f = np.linspace(0.001, x_f_max, 500)
        y_f = stats.f.pdf(x_f, dfn_f, dfd_f)
        ax_f.plot(x_f, y_f, 'b-', lw=2, label=f'F-dist (df1={dfn_f}, df2={dfd_f})')

        crit_val_f_upper = None
        crit_val_f_lower = None

        if tail_f == "One-tailed (right)":
            crit_val_f_upper = stats.f.ppf(1 - alpha_f, dfn_f, dfd_f)
            x_fill_upper = np.linspace(crit_val_f_upper, x_f_max, 100)
            ax_f.fill_between(x_fill_upper, stats.f.pdf(x_fill_upper, dfn_f, dfd_f), color='red', alpha=0.5, label=f'α = {alpha_f:.4f}')
            ax_f.axvline(crit_val_f_upper, color='red', linestyle='--', lw=1)
        else: # Two-tailed (for variance test)
            crit_val_f_upper = stats.f.ppf(1 - alpha_f / 2, dfn_f, dfd_f)
            crit_val_f_lower = stats.f.ppf(alpha_f / 2, dfn_f, dfd_f)
            x_fill_upper = np.linspace(crit_val_f_upper, x_f_max, 100)
            x_fill_lower = np.linspace(0.001, crit_val_f_lower, 100)
            ax_f.fill_between(x_fill_upper, stats.f.pdf(x_fill_upper, dfn_f, dfd_f), color='red', alpha=0.5, label=f'α/2 = {alpha_f/2:.4f}')
            ax_f.fill_between(x_fill_lower, stats.f.pdf(x_fill_lower, dfn_f, dfd_f), color='red', alpha=0.5)
            ax_f.axvline(crit_val_f_upper, color='red', linestyle='--', lw=1)
            ax_f.axvline(crit_val_f_lower, color='red', linestyle='--', lw=1)


        ax_f.axvline(test_stat_f, color='green', linestyle='-', lw=2, label=f'Test Stat = {test_stat_f:.3f}')
        ax_f.set_title(f'F-Distribution (df1={dfn_f}, df2={dfd_f}) with Critical Region(s)')
        ax_f.set_xlabel('F-value')
        ax_f.set_ylabel('Probability Density')
        ax_f.legend()
        ax_f.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_f)

        st.subheader("Critical Value Table Snippet")
        alphas_table_f = [0.10, 0.05, 0.01, alpha_f]
        alphas_table_f = sorted(list(set(alphas_table_f)))

        table_data_f = {"Alpha": [], "Critical F (Upper Tail)": []}
        if tail_f == "Two-tailed (for variance test)":
            table_data_f = {"Alpha (Total)": [], "Alpha (Each Tail)": [], "Critical F (Lower)": [], "Critical F (Upper)": []}
            for a_val in alphas_table_f:
                table_data_f["Alpha (Total)"].append(f"{a_val:.4f}")
                table_data_f["Alpha (Each Tail)"].append(f"{a_val/2:.4f}")
                table_data_f["Critical F (Lower)"].append(f"{stats.f.ppf(a_val / 2, dfn_f, dfd_f):.3f}")
                table_data_f["Critical F (Upper)"].append(f"{stats.f.ppf(1 - a_val / 2, dfn_f, dfd_f):.3f}")
        else: # One-tailed (right)
            for a_val in alphas_table_f:
                table_data_f["Alpha"].append(f"{a_val:.4f}")
                table_data_f["Critical F (Upper Tail)"].append(f"{stats.f.ppf(1 - a_val, dfn_f, dfd_f):.3f}")
        
        df_table_f = pd.DataFrame(table_data_f)
        
        def highlight_alpha_row_f(row):
            alpha_to_check = alpha_f
            if tail_f == "Two-tailed (for variance test)":
                is_selected_alpha = abs(float(row["Alpha (Total)"]) - alpha_to_check) < 1e-5
            else:
                is_selected_alpha = abs(float(row["Alpha"]) - alpha_to_check) < 1e-5
            
            if is_selected_alpha:
                return ['background-color: yellow'] * len(row)
            return [''] * len(row)

        st.markdown(df_table_f.style.apply(highlight_alpha_row_f, axis=1).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows critical F-values for df1={dfn_f}, df2={dfd_f}. Highlighted row for α={alpha_f:.4f}.")
        st.markdown("""
        **Cumulative Table Note:**
        * F-distribution tables typically provide upper-tail critical values.
        * For **ANOVA (One-tailed right)**, use the alpha directly to find the critical F.
        * For **Two-tailed variance tests**, use alpha/2 for each tail. The lower critical value F<sub>L</sub>(α/2, df1, df2) can be found as 1 / F<sub>U</sub>(α/2, df2, df1) if not directly available. SciPy calculates it directly.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value is the probability of an F-statistic as extreme as, or more extreme than, {test_stat_f:.3f}.
        * **One-tailed (right)**: `P(F ≥ {test_stat_f:.3f})`
        * **Two-tailed (for variance test)**: If F<sub>calc</sub> > 1, then `2 * P(F ≥ F_calc)`. If F<sub>calc</sub> < 1, then `2 * P(F ≤ F_calc)`. This is complex; often, the test is structured so F > 1 by putting larger variance in numerator. Here, we calculate `2 * min(P(F ≤ F_calc), P(F ≥ F_calc))`.
        """)

        st.subheader("Summary")
        p_val_f_one_right = stats.f.sf(test_stat_f, dfn_f, dfd_f)
        
        # For two-tailed F-test, p-value is 2 * min(cdf, sf)
        cdf_f = stats.f.cdf(test_stat_f, dfn_f, dfd_f)
        sf_f = stats.f.sf(test_stat_f, dfn_f, dfd_f)
        p_val_f_two = 2 * min(cdf_f, sf_f)


        crit_val_f_display = "N/A"
        p_val_for_crit_val_f_display = alpha_f

        if tail_f == "One-tailed (right)":
            crit_val_f_display = f"{crit_val_f_upper:.3f}"
            p_val_calc_f = p_val_f_one_right
            decision_crit_f = test_stat_f > crit_val_f_upper
            comparison_crit_str_f = f"{test_stat_f:.3f} > {crit_val_f_upper:.3f}" if decision_crit_f else f"{test_stat_f:.3f} ≤ {crit_val_f_upper:.3f}"
        else: # Two-tailed
            crit_val_f_display = f"Lower: {crit_val_f_lower:.3f}, Upper: {crit_val_f_upper:.3f}"
            p_val_calc_f = p_val_f_two
            decision_crit_f = (test_stat_f > crit_val_f_upper) or (test_stat_f < crit_val_f_lower)
            comparison_crit_str_f = f"{test_stat_f:.3f} > {crit_val_f_upper:.3f} or {test_stat_f:.3f} < {crit_val_f_lower:.3f}" if decision_crit_f else f"{crit_val_f_lower:.3f} ≤ {test_stat_f:.3f} ≤ {crit_val_f_upper:.3f}"


        decision_p_alpha_f = p_val_calc_f < alpha_f
        
        st.markdown(f"""
        1.  **Critical Value(s) ({tail_f})**: {crit_val_f_display}
            * *Associated p-value (α or α/2 per tail)*: {p_val_for_crit_val_f_display:.4f}
        2.  **Calculated Test Statistic**: {test_stat_f:.3f}
            * *Calculated p-value*: {p_val_calc_f:.4f} ({apa_p_value(p_val_calc_f)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_f else 'not rejected'}**.
            * *Reason*: F(calc) {comparison_crit_str_f} relative to F(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_f else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_f)} is {'less than' if decision_p_alpha_f else 'not less than'} α ({alpha_f:.4f}).
        5.  **APA 7 Style Report**:
            *F*({dfn_f}, {dfd_f}) = {test_stat_f:.2f}, {apa_p_value(p_val_calc_f)}. The null hypothesis was {'rejected' if decision_p_alpha_f else 'not rejected'} at α = {alpha_f:.2f}.
        """)

# --- Tab 4: Chi-square distribution ---
def tab_chi_square_distribution():
    st.header("Chi-square (χ²) Distribution Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_chi2 = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_chi2")
        df_chi2 = st.number_input("Degrees of Freedom (df)", 1, 1000, 5, 1, key="df_chi2")
        # Chi-square tests (goodness-of-fit, independence) are typically right-tailed.
        # Can be two-tailed for variance tests (less common than F for that).
        tail_chi2 = st.radio("Tail Selection", ("One-tailed (right)", "Two-tailed (e.g. for variance)"), key="tail_chi2")
        test_stat_chi2 = st.number_input("Calculated χ²-statistic", value=1.0, format="%.3f", min_value=0.0, key="test_stat_chi2")

        st.subheader("Distribution Plot")
        fig_chi2, ax_chi2 = plt.subplots(figsize=(8,5))
        x_chi2_max = stats.chi2.ppf(0.999, df_chi2) if stats.chi2.ppf(0.999, df_chi2) > 10 else 10
        x_chi2 = np.linspace(0.001, x_chi2_max, 500) # Chi2 is non-negative
        y_chi2 = stats.chi2.pdf(x_chi2, df_chi2)
        ax_chi2.plot(x_chi2, y_chi2, 'b-', lw=2, label=f'χ²-distribution (df={df_chi2})')

        crit_val_chi2_upper = None
        crit_val_chi2_lower = None

        if tail_chi2 == "One-tailed (right)":
            crit_val_chi2_upper = stats.chi2.ppf(1 - alpha_chi2, df_chi2)
            x_fill_upper = np.linspace(crit_val_chi2_upper, x_chi2_max, 100)
            ax_chi2.fill_between(x_fill_upper, stats.chi2.pdf(x_fill_upper, df_chi2), color='red', alpha=0.5, label=f'α = {alpha_chi2:.4f}')
            ax_chi2.axvline(crit_val_chi2_upper, color='red', linestyle='--', lw=1)
        else: # Two-tailed
            crit_val_chi2_upper = stats.chi2.ppf(1 - alpha_chi2 / 2, df_chi2)
            crit_val_chi2_lower = stats.chi2.ppf(alpha_chi2 / 2, df_chi2)
            x_fill_upper_chi2 = np.linspace(crit_val_chi2_upper, x_chi2_max, 100)
            x_fill_lower_chi2 = np.linspace(0.001, crit_val_chi2_lower, 100)
            ax_chi2.fill_between(x_fill_upper_chi2, stats.chi2.pdf(x_fill_upper_chi2, df_chi2), color='red', alpha=0.5, label=f'α/2 = {alpha_chi2/2:.4f}')
            ax_chi2.fill_between(x_fill_lower_chi2, stats.chi2.pdf(x_fill_lower_chi2, df_chi2), color='red', alpha=0.5)
            ax_chi2.axvline(crit_val_chi2_upper, color='red', linestyle='--', lw=1)
            ax_chi2.axvline(crit_val_chi2_lower, color='red', linestyle='--', lw=1)


        ax_chi2.axvline(test_stat_chi2, color='green', linestyle='-', lw=2, label=f'Test Stat = {test_stat_chi2:.3f}')
        ax_chi2.set_title(f'χ²-Distribution (df={df_chi2}) with Critical Region(s)')
        ax_chi2.set_xlabel('χ²-value')
        ax_chi2.set_ylabel('Probability Density')
        ax_chi2.legend()
        ax_chi2.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_chi2)

        st.subheader("Critical Value Table Snippet")
        alphas_table_chi2 = [0.10, 0.05, 0.01, alpha_chi2]
        alphas_table_chi2 = sorted(list(set(alphas_table_chi2)))

        table_data_chi2 = {}
        if tail_chi2 == "Two-tailed (e.g. for variance)":
            table_data_chi2 = {"Alpha (Total)": [], "Alpha (Each Tail)": [], "Critical χ² (Lower)": [], "Critical χ² (Upper)": []}
            for a_val in alphas_table_chi2:
                table_data_chi2["Alpha (Total)"].append(f"{a_val:.4f}")
                table_data_chi2["Alpha (Each Tail)"].append(f"{a_val/2:.4f}")
                table_data_chi2["Critical χ² (Lower)"].append(f"{stats.chi2.ppf(a_val / 2, df_chi2):.3f}")
                table_data_chi2["Critical χ² (Upper)"].append(f"{stats.chi2.ppf(1- a_val / 2, df_chi2):.3f}")
        else: # One-tailed (right)
            table_data_chi2 = {"Alpha (Right Tail)": [], "Critical χ² (Upper)": []}
            for a_val in alphas_table_chi2:
                table_data_chi2["Alpha (Right Tail)"].append(f"{a_val:.4f}")
                table_data_chi2["Critical χ² (Upper)"].append(f"{stats.chi2.ppf(1 - a_val, df_chi2):.3f}")
        
        df_table_chi2 = pd.DataFrame(table_data_chi2)
        
        def highlight_alpha_row_chi2(row):
            alpha_to_check = alpha_chi2
            if tail_chi2 == "Two-tailed (e.g. for variance)":
                 is_selected_alpha = abs(float(row["Alpha (Total)"]) - alpha_to_check) < 1e-5
            else:
                 is_selected_alpha = abs(float(row["Alpha (Right Tail)"]) - alpha_to_check) < 1e-5
            
            if is_selected_alpha:
                return ['background-color: yellow'] * len(row)
            return [''] * len(row)

        st.markdown(df_table_chi2.style.apply(highlight_alpha_row_chi2, axis=1).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows critical χ²-values for df={df_chi2}. Highlighted row for α={alpha_chi2:.4f}.")
        st.markdown("""
        **Cumulative Table Note:**
        * Chi-square tables typically provide upper-tail critical values (area to the right).
        * For **goodness-of-fit or independence tests (One-tailed right)**, use alpha directly.
        * For **Two-tailed tests** (e.g., on variance), use alpha/2 for the lower critical value and 1-alpha/2 for the upper.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value is the probability of a χ²-statistic as extreme as, or more extreme than, {test_stat_chi2:.3f}.
        * **One-tailed (right)**: `P(χ² ≥ {test_stat_chi2:.3f})`
        * **Two-tailed**: `2 * min(P(χ² ≤ {test_stat_chi2:.3f}), P(χ² ≥ {test_stat_chi2:.3f}))`
        """)

        st.subheader("Summary")
        p_val_chi2_one_right = stats.chi2.sf(test_stat_chi2, df_chi2)
        
        cdf_chi2 = stats.chi2.cdf(test_stat_chi2, df_chi2)
        sf_chi2 = stats.chi2.sf(test_stat_chi2, df_chi2)
        p_val_chi2_two = 2 * min(cdf_chi2, sf_chi2)

        crit_val_chi2_display = "N/A"
        p_val_for_crit_val_chi2_display = alpha_chi2

        if tail_chi2 == "One-tailed (right)":
            crit_val_chi2_display = f"{crit_val_chi2_upper:.3f}"
            p_val_calc_chi2 = p_val_chi2_one_right
            decision_crit_chi2 = test_stat_chi2 > crit_val_chi2_upper
            comparison_crit_str_chi2 = f"{test_stat_chi2:.3f} > {crit_val_chi2_upper:.3f}" if decision_crit_chi2 else f"{test_stat_chi2:.3f} ≤ {crit_val_chi2_upper:.3f}"
        else: # Two-tailed
            crit_val_chi2_display = f"Lower: {crit_val_chi2_lower:.3f}, Upper: {crit_val_chi2_upper:.3f}"
            p_val_calc_chi2 = p_val_chi2_two
            decision_crit_chi2 = (test_stat_chi2 > crit_val_chi2_upper) or (test_stat_chi2 < crit_val_chi2_lower)
            comparison_crit_str_chi2 = f"{test_stat_chi2:.3f} > {crit_val_chi2_upper:.3f} or {test_stat_chi2:.3f} < {crit_val_chi2_lower:.3f}" if decision_crit_chi2 else f"{crit_val_chi2_lower:.3f} ≤ {test_stat_chi2:.3f} ≤ {crit_val_chi2_upper:.3f}"

        decision_p_alpha_chi2 = p_val_calc_chi2 < alpha_chi2
        
        st.markdown(f"""
        1.  **Critical Value(s) ({tail_chi2})**: {crit_val_chi2_display}
            * *Associated p-value (α or α/2 per tail)*: {p_val_for_crit_val_chi2_display:.4f}
        2.  **Calculated Test Statistic**: {test_stat_chi2:.3f}
            * *Calculated p-value*: {p_val_calc_chi2:.4f} ({apa_p_value(p_val_calc_chi2)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_chi2 else 'not rejected'}**.
            * *Reason*: χ²(calc) {comparison_crit_str_chi2} relative to χ²(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_chi2 else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_chi2)} is {'less than' if decision_p_alpha_chi2 else 'not less than'} α ({alpha_chi2:.4f}).
        5.  **APA 7 Style Report**:
            χ²({df_chi2}) = {test_stat_chi2:.2f}, {apa_p_value(p_val_calc_chi2)}. The null hypothesis was {'rejected' if decision_p_alpha_chi2 else 'not rejected'} at α = {alpha_chi2:.2f}.
        """)

# --- Tab 5: Mann-Whitney U Test ---
def tab_mann_whitney_u():
    st.header("Mann-Whitney U Test (Normal Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_mw = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_mw")
        n1_mw = st.number_input("Sample Size Group 1 (n1)", 5, 1000, 10, 1, key="n1_mw")
        n2_mw = st.number_input("Sample Size Group 2 (n2)", 5, 1000, 12, 1, key="n2_mw")
        tail_mw = st.radio("Tail Selection", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_mw")
        # User inputs their calculated U statistic
        u_stat_mw = st.number_input("Calculated U-statistic", value=float(n1_mw*n2_mw/2), format="%.1f", key="u_stat_mw")
        st.caption("Note: For normal approximation, typically n1, n2 > 8-10. The U statistic should be the smaller of U1 or U2 for one-tailed tests if direction is pre-specified, or use U based on alternative hypothesis.")


        # Normal approximation
        mu_u = (n1_mw * n2_mw) / 2
        sigma_u = np.sqrt((n1_mw * n2_mw * (n1_mw + n2_mw + 1)) / 12)
        
        # Continuity correction: +0.5 if U < mu_u, -0.5 if U > mu_u for two-tailed.
        # For one-tailed, depends on direction.
        # Simplified: for z_calc, if U is used (often smaller U for one-tailed test against smaller U critical)
        # If U < mu_U, z = (U + 0.5 - mu_U) / sigma_U
        # If U > mu_U, z = (U - 0.5 - mu_U) / sigma_U
        # For this explorer, we'll use the direct z from U without forcing U to be the smaller one.
        # The user provides U, we calculate z.
        if u_stat_mw < mu_u:
            z_calc_mw = (u_stat_mw + 0.5 - mu_u) / sigma_u if sigma_u > 0 else 0
        elif u_stat_mw > mu_u:
            z_calc_mw = (u_stat_mw - 0.5 - mu_u) / sigma_u if sigma_u > 0 else 0
        else: # U == mu_u
            z_calc_mw = 0.0


        st.markdown(f"**Normal Approximation Parameters:** μ<sub>U</sub> = {mu_u:.2f}, σ<sub>U</sub> = {sigma_u:.2f}")
        st.markdown(f"**Calculated z-statistic (from U):** {z_calc_mw:.3f}")

        st.subheader("Standard Normal Distribution Plot (for z_calc)")
        fig_mw, ax_mw = plt.subplots(figsize=(8,5))
        x_norm = np.linspace(stats.norm.ppf(0.0001), stats.norm.ppf(0.9999), 500)
        y_norm = stats.norm.pdf(x_norm)
        ax_mw.plot(x_norm, y_norm, 'b-', lw=2, label='Standard Normal Distribution')

        crit_z_upper_mw, crit_z_lower_mw = None, None
        if tail_mw == "Two-tailed":
            crit_z_upper_mw = stats.norm.ppf(1 - alpha_mw / 2)
            crit_z_lower_mw = stats.norm.ppf(alpha_mw / 2)
            x_fill_upper = np.linspace(crit_z_upper_mw, stats.norm.ppf(0.9999), 100)
            x_fill_lower = np.linspace(stats.norm.ppf(0.0001), crit_z_lower_mw, 100)
            ax_mw.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, label=f'α/2 = {alpha_mw/2:.4f}')
            ax_mw.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.5)
            ax_mw.axvline(crit_z_upper_mw, color='red', linestyle='--', lw=1)
            ax_mw.axvline(crit_z_lower_mw, color='red', linestyle='--', lw=1)
        elif tail_mw == "One-tailed (right)":
            crit_z_upper_mw = stats.norm.ppf(1 - alpha_mw)
            x_fill_upper = np.linspace(crit_z_upper_mw, stats.norm.ppf(0.9999), 100)
            ax_mw.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, label=f'α = {alpha_mw:.4f}')
            ax_mw.axvline(crit_z_upper_mw, color='red', linestyle='--', lw=1)
        else: # One-tailed (left)
            crit_z_lower_mw = stats.norm.ppf(alpha_mw)
            x_fill_lower = np.linspace(stats.norm.ppf(0.0001), crit_z_lower_mw, 100)
            ax_mw.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.5, label=f'α = {alpha_mw:.4f}')
            ax_mw.axvline(crit_z_lower_mw, color='red', linestyle='--', lw=1)

        ax_mw.axvline(z_calc_mw, color='green', linestyle='-', lw=2, label=f'z_calc = {z_calc_mw:.3f}')
        ax_mw.set_title('Normal Approx. for Mann-Whitney U: Critical z Region(s)')
        ax_mw.set_xlabel('z-value')
        ax_mw.set_ylabel('Probability Density')
        ax_mw.legend()
        ax_mw.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_mw)

        st.subheader("Critical z-Value Table Snippet (for U test)")
        # This is essentially the z-table snippet
        alphas_table_z_mw = [0.10, 0.05, 0.01, alpha_mw]
        alphas_table_z_mw = sorted(list(set(alphas_table_z_mw)))
        table_data_z_mw = {"Alpha (One-Tail)": [], "Alpha (Two-Tail)": [], "Critical z (Upper)": []}
        if tail_mw != "One-tailed (left)":
             for a_val in alphas_table_z_mw:
                table_data_z_mw["Alpha (One-Tail)"].append(f"{a_val:.4f}")
                table_data_z_mw["Alpha (Two-Tail)"].append(f"{a_val*2:.4f}" if a_val*2 <=1 else "-")
                table_data_z_mw["Critical z (Upper)"].append(f"{stats.norm.ppf(1 - a_val):.3f}")
        if tail_mw != "One-tailed (right)":
            if not table_data_z_mw["Critical z (Upper)"]:
                 for a_val in alphas_table_z_mw:
                    table_data_z_mw["Alpha (One-Tail)"].append(f"{a_val:.4f}")
                    table_data_z_mw["Alpha (Two-Tail)"].append(f"{a_val*2:.4f}" if a_val*2 <=1 else "-")
                    table_data_z_mw["Critical z (Lower)"] = [f"{stats.norm.ppf(a_val):.3f}" for a_val in alphas_table_z_mw]
            else:
                table_data_z_mw["Critical z (Lower)"] = [f"{stats.norm.ppf(a_val):.3f}" for a_val in alphas_table_z_mw]
        df_table_z_mw = pd.DataFrame(table_data_z_mw)
        
        def highlight_alpha_row_z_mw(row):
            is_selected_alpha_one_tail = abs(float(row["Alpha (One-Tail)"]) - alpha_mw) < 1e-5
            is_selected_alpha_two_tail = abs(float(row["Alpha (Two-Tail)"]) - alpha_mw) < 1e-5 if row["Alpha (Two-Tail)"] != "-" else False
            if (tail_mw == "Two-tailed" and is_selected_alpha_two_tail) or \
               (tail_mw != "Two-tailed" and is_selected_alpha_one_tail):
                return ['background-color: yellow'] * len(row)
            return [''] * len(row)
        st.markdown(df_table_z_mw.style.apply(highlight_alpha_row_z_mw, axis=1).to_html(), unsafe_allow_html=True)
        st.caption(f"Highlighted row for α={alpha_mw:.4f}. Compare calculated z from U to these critical z-values.")
        st.markdown("""
        **Cumulative Table Note:**
        * The Mann-Whitney U test (with normal approximation) converts the U statistic to a z-statistic.
        * This table shows critical z-values. Interpret as per standard z-distribution.
        * Small sample exact U tables are different and not shown here.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The U statistic ({u_stat_mw:.1f}) is converted to a z-statistic ({z_calc_mw:.3f}) using the normal approximation (μ<sub>U</sub>={mu_u:.2f}, σ<sub>U</sub>={sigma_u:.2f}). The p-value is then found from the standard normal distribution based on this z_calc_mw.
        * **Two-tailed**: `2 * P(Z ≥ |{z_calc_mw:.3f}|)`
        * **One-tailed (right)**: `P(Z ≥ {z_calc_mw:.3f})` (if H1 predicts U is large, meaning group 1 > group 2, or vice-versa depending on U definition)
        * **One-tailed (left)**: `P(Z ≤ {z_calc_mw:.3f})` (if H1 predicts U is small)
        The interpretation of "right" and "left" for U depends on which U (U1 or U2) is used and the alternative hypothesis. This explorer assumes the z_calc_mw directionality matches the tail selection.
        """)

        st.subheader("Summary")
        p_val_mw_one_right = stats.norm.sf(z_calc_mw)
        p_val_mw_one_left = stats.norm.cdf(z_calc_mw)
        p_val_mw_two = 2 * stats.norm.sf(abs(z_calc_mw))

        crit_val_z_display_mw = "N/A"
        p_val_for_crit_val_mw_display = alpha_mw

        if tail_mw == "Two-tailed":
            crit_val_z_display_mw = f"±{crit_z_upper_mw:.3f}"
            p_val_calc_mw = p_val_mw_two
            decision_crit_mw = abs(z_calc_mw) > crit_z_upper_mw
            comparison_crit_str_mw = f"|z_calc| ({abs(z_calc_mw):.3f}) > {crit_z_upper_mw:.3f}" if decision_crit_mw else f"|z_calc| ({abs(z_calc_mw):.3f}) ≤ {crit_z_upper_mw:.3f}"
        elif tail_mw == "One-tailed (right)":
            crit_val_z_display_mw = f"{crit_z_upper_mw:.3f}"
            p_val_calc_mw = p_val_mw_one_right
            decision_crit_mw = z_calc_mw > crit_z_upper_mw
            comparison_crit_str_mw = f"z_calc ({z_calc_mw:.3f}) > {crit_z_upper_mw:.3f}" if decision_crit_mw else f"z_calc ({z_calc_mw:.3f}) ≤ {crit_z_upper_mw:.3f}"
        else: # One-tailed (left)
            crit_val_z_display_mw = f"{crit_z_lower_mw:.3f}"
            p_val_calc_mw = p_val_mw_one_left
            decision_crit_mw = z_calc_mw < crit_z_lower_mw
            comparison_crit_str_mw = f"z_calc ({z_calc_mw:.3f}) < {crit_z_lower_mw:.3f}" if decision_crit_mw else f"z_calc ({z_calc_mw:.3f}) ≥ {crit_z_lower_mw:.3f}"

        decision_p_alpha_mw = p_val_calc_mw < alpha_mw
        
        st.markdown(f"""
        1.  **Critical z-value ({tail_mw}) for U test**: {crit_val_z_display_mw}
            * *Associated p-value (α)*: {p_val_for_crit_val_mw_display:.4f}
        2.  **Calculated U-statistic**: {u_stat_mw:.1f}
            * *Converted z-statistic (z_calc)*: {z_calc_mw:.3f}
            * *Calculated p-value (from z_calc)*: {p_val_calc_mw:.4f} ({apa_p_value(p_val_calc_mw)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_mw else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_mw} relative to z(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_mw else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_mw)} is {'less than' if decision_p_alpha_mw else 'not less than'} α ({alpha_mw:.4f}).
        5.  **APA 7 Style Report (based on z-approximation)**:
            *U* = {u_stat_mw:.1f}, *z* = {z_calc_mw:.2f}, {apa_p_value(p_val_calc_mw)}. The null hypothesis was {'rejected' if decision_p_alpha_mw else 'not rejected'} at α = {alpha_mw:.2f}. (Sample sizes: n1={n1_mw}, n2={n2_mw})
        """)

# --- Tab 6: Wilcoxon Signed-Rank T Test ---
def tab_wilcoxon_t():
    st.header("Wilcoxon Signed-Rank T Test (Normal Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_w = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_w")
        n_w = st.number_input("Sample Size (n, number of pairs with non-zero differences)", 6, 1000, 15, 1, key="n_w") # Normal approx good for n > ~10-15
        tail_w = st.radio("Tail Selection", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_w")
        # User inputs their calculated T statistic (sum of ranks for positive or negative differences, whichever is smaller for two-tailed)
        t_stat_w = st.number_input("Calculated T-statistic (sum of ranks)", value=float(n_w*(n_w+1)/4 / 2), format="%.1f", key="t_stat_w") # Default to smaller T approx
        st.caption("Note: For normal approximation, n should be > ~10-15. T is the sum of ranks of positive or negative differences.")

        # Normal approximation
        mu_t_w = n_w * (n_w + 1) / 4
        sigma_t_w = np.sqrt(n_w * (n_w + 1) * (2 * n_w + 1) / 24)
        
        # Continuity correction for T.
        # If T < mu_T, z = (T + 0.5 - mu_T) / sigma_T
        # If T > mu_T, z = (T - 0.5 - mu_T) / sigma_T
        if t_stat_w < mu_t_w:
            z_calc_w = (t_stat_w + 0.5 - mu_t_w) / sigma_t_w if sigma_t_w > 0 else 0
        elif t_stat_w > mu_t_w:
            z_calc_w = (t_stat_w - 0.5 - mu_t_w) / sigma_t_w if sigma_t_w > 0 else 0
        else: # T == mu_T
            z_calc_w = 0.0

        st.markdown(f"**Normal Approximation Parameters:** μ<sub>T</sub> = {mu_t_w:.2f}, σ<sub>T</sub> = {sigma_t_w:.2f}")
        st.markdown(f"**Calculated z-statistic (from T):** {z_calc_w:.3f}")

        st.subheader("Standard Normal Distribution Plot (for z_calc)")
        fig_w, ax_w = plt.subplots(figsize=(8,5))
        x_norm_w = np.linspace(stats.norm.ppf(0.0001), stats.norm.ppf(0.9999), 500)
        y_norm_w = stats.norm.pdf(x_norm_w)
        ax_w.plot(x_norm_w, y_norm_w, 'b-', lw=2, label='Standard Normal Distribution')

        crit_z_upper_w, crit_z_lower_w = None, None
        if tail_w == "Two-tailed":
            crit_z_upper_w = stats.norm.ppf(1 - alpha_w / 2)
            crit_z_lower_w = stats.norm.ppf(alpha_w / 2)
            x_fill_upper_w = np.linspace(crit_z_upper_w, stats.norm.ppf(0.9999), 100)
            x_fill_lower_w = np.linspace(stats.norm.ppf(0.0001), crit_z_lower_w, 100)
            ax_w.fill_between(x_fill_upper_w, stats.norm.pdf(x_fill_upper_w), color='red', alpha=0.5, label=f'α/2 = {alpha_w/2:.4f}')
            ax_w.fill_between(x_fill_lower_w, stats.norm.pdf(x_fill_lower_w), color='red', alpha=0.5)
            ax_w.axvline(crit_z_upper_w, color='red', linestyle='--', lw=1)
            ax_w.axvline(crit_z_lower_w, color='red', linestyle='--', lw=1)
        elif tail_w == "One-tailed (right)": # H1: T is large (e.g. sum of positive ranks is large)
            crit_z_upper_w = stats.norm.ppf(1 - alpha_w)
            x_fill_upper_w = np.linspace(crit_z_upper_w, stats.norm.ppf(0.9999), 100)
            ax_w.fill_between(x_fill_upper_w, stats.norm.pdf(x_fill_upper_w), color='red', alpha=0.5, label=f'α = {alpha_w:.4f}')
            ax_w.axvline(crit_z_upper_w, color='red', linestyle='--', lw=1)
        else: # One-tailed (left) H1: T is small (e.g. sum of positive ranks is small, or sum of negative ranks is large if T = T_neg)
            crit_z_lower_w = stats.norm.ppf(alpha_w)
            x_fill_lower_w = np.linspace(stats.norm.ppf(0.0001), crit_z_lower_w, 100)
            ax_w.fill_between(x_fill_lower_w, stats.norm.pdf(x_fill_lower_w), color='red', alpha=0.5, label=f'α = {alpha_w:.4f}')
            ax_w.axvline(crit_z_lower_w, color='red', linestyle='--', lw=1)

        ax_w.axvline(z_calc_w, color='green', linestyle='-', lw=2, label=f'z_calc = {z_calc_w:.3f}')
        ax_w.set_title('Normal Approx. for Wilcoxon T: Critical z Region(s)')
        ax_w.set_xlabel('z-value')
        ax_w.set_ylabel('Probability Density')
        ax_w.legend()
        ax_w.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_w)

        st.subheader("Critical z-Value Table Snippet (for T test)")
        alphas_table_z_w = [0.10, 0.05, 0.01, alpha_w]
        alphas_table_z_w = sorted(list(set(alphas_table_z_w)))
        table_data_z_w = {"Alpha (One-Tail)": [], "Alpha (Two-Tail)": [], "Critical z (Upper)": []}
        if tail_w != "One-tailed (left)":
             for a_val in alphas_table_z_w:
                table_data_z_w["Alpha (One-Tail)"].append(f"{a_val:.4f}")
                table_data_z_w["Alpha (Two-Tail)"].append(f"{a_val*2:.4f}" if a_val*2 <=1 else "-")
                table_data_z_w["Critical z (Upper)"].append(f"{stats.norm.ppf(1 - a_val):.3f}")
        if tail_w != "One-tailed (right)":
            if not table_data_z_w["Critical z (Upper)"]:
                 for a_val in alphas_table_z_w:
                    table_data_z_w["Alpha (One-Tail)"].append(f"{a_val:.4f}")
                    table_data_z_w["Alpha (Two-Tail)"].append(f"{a_val*2:.4f}" if a_val*2 <=1 else "-")
                    table_data_z_w["Critical z (Lower)"] = [f"{stats.norm.ppf(a_val):.3f}" for a_val in alphas_table_z_w]
            else:
                table_data_z_w["Critical z (Lower)"] = [f"{stats.norm.ppf(a_val):.3f}" for a_val in alphas_table_z_w]
        df_table_z_w = pd.DataFrame(table_data_z_w)
        
        def highlight_alpha_row_z_w(row):
            is_selected_alpha_one_tail = abs(float(row["Alpha (One-Tail)"]) - alpha_w) < 1e-5
            is_selected_alpha_two_tail = abs(float(row["Alpha (Two-Tail)"]) - alpha_w) < 1e-5 if row["Alpha (Two-Tail)"] != "-" else False
            if (tail_w == "Two-tailed" and is_selected_alpha_two_tail) or \
               (tail_w != "Two-tailed" and is_selected_alpha_one_tail):
                return ['background-color: yellow'] * len(row)
            return [''] * len(row)
        st.markdown(df_table_z_w.style.apply(highlight_alpha_row_z_w, axis=1).to_html(), unsafe_allow_html=True)
        st.caption(f"Highlighted row for α={alpha_w:.4f}. Compare calculated z from T to these critical z-values.")
        st.markdown("""
        **Cumulative Table Note:**
        * The Wilcoxon T statistic (with normal approximation) is converted to a z-statistic.
        * This table shows critical z-values. Small sample exact T tables are different.
        * For a two-tailed test, T is usually the smaller of T+ or T-. The p-value is doubled.
        * For one-tailed, T is T+ or T- based on H1. The z-score's sign reflects this.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The T statistic ({t_stat_w:.1f}) is converted to a z-statistic ({z_calc_w:.3f}) using μ<sub>T</sub>={mu_t_w:.2f}, σ<sub>T</sub>={sigma_t_w:.2f}. The p-value is from the standard normal distribution.
        * **Two-tailed**: `2 * P(Z ≥ |{z_calc_w:.3f}|)` (if T is the smaller rank sum) or `2 * P(Z ≤ -|{z_calc_w:.3f}|)` (if T is smaller rank sum and corresponds to negative z_calc). More generally, `2 * min(P(Z <= z_calc_w), P(Z >= z_calc_w))`.
        * **One-tailed (right)**: `P(Z ≥ {z_calc_w:.3f})` (if H1: median difference > 0, T = T_pos, z_calc_w should be positive)
        * **One-tailed (left)**: `P(Z ≤ {z_calc_w:.3f})` (if H1: median difference < 0, T = T_neg, or T_pos is small, z_calc_w should be negative)
        This explorer assumes the z_calc_w directionality matches the tail selection.
        """)

        st.subheader("Summary")
        # P-value calculation from z_calc_w
        # For Wilcoxon, a smaller T is more significant.
        # If T_calc is used as the smaller of T+ and T-, then for a two-tailed test, p = 2 * P(T <= T_calc)
        # This translates to z_calc_w being negative if T_calc is small.
        # So, for a two-tailed test, p = 2 * CDF(z_calc_w) if z_calc_w is based on the smaller T.
        # However, our z_calc_w can be positive or negative depending on T vs mu_T.
        # The standard approach is:
        p_val_w_one_right = stats.norm.sf(z_calc_w) # If H1: T is large (less common interpretation for T as sum of ranks)
        p_val_w_one_left = stats.norm.cdf(z_calc_w)  # If H1: T is small (more common, T = smaller rank sum)
        p_val_w_two = 2 * stats.norm.sf(abs(z_calc_w)) # This is standard for z-scores from any symmetric test.
                                                    # If T is defined as the smaller sum, then its z will typically be negative,
                                                    # and 2 * cdf(negative_z) = 2 * sf(positive_z)

        crit_val_z_display_w = "N/A"
        p_val_for_crit_val_w_display = alpha_w

        if tail_w == "Two-tailed":
            crit_val_z_display_w = f"±{crit_z_upper_w:.3f}"
            p_val_calc_w = p_val_w_two
            decision_crit_w = abs(z_calc_w) > crit_z_upper_w
            comparison_crit_str_w = f"|z_calc| ({abs(z_calc_w):.3f}) > {crit_z_upper_w:.3f}" if decision_crit_w else f"|z_calc| ({abs(z_calc_w):.3f}) ≤ {crit_z_upper_w:.3f}"
        elif tail_w == "One-tailed (right)": # H1: T is large, so z_calc should be positive and large
            crit_val_z_display_w = f"{crit_z_upper_w:.3f}"
            p_val_calc_w = p_val_w_one_right
            decision_crit_w = z_calc_w > crit_z_upper_w
            comparison_crit_str_w = f"z_calc ({z_calc_w:.3f}) > {crit_z_upper_w:.3f}" if decision_crit_w else f"z_calc ({z_calc_w:.3f}) ≤ {crit_z_upper_w:.3f}"
        else: # One-tailed (left) H1: T is small, so z_calc should be negative and large in magnitude
            crit_val_z_display_w = f"{crit_z_lower_w:.3f}"
            p_val_calc_w = p_val_w_one_left
            decision_crit_w = z_calc_w < crit_z_lower_w
            comparison_crit_str_w = f"z_calc ({z_calc_w:.3f}) < {crit_z_lower_w:.3f}" if decision_crit_w else f"z_calc ({z_calc_w:.3f}) ≥ {crit_z_lower_w:.3f}"

        decision_p_alpha_w = p_val_calc_w < alpha_w
        
        st.markdown(f"""
        1.  **Critical z-value ({tail_w}) for T test**: {crit_val_z_display_w}
            * *Associated p-value (α)*: {p_val_for_crit_val_w_display:.4f}
        2.  **Calculated T-statistic**: {t_stat_w:.1f}
            * *Converted z-statistic (z_calc)*: {z_calc_w:.3f}
            * *Calculated p-value (from z_calc)*: {p_val_calc_w:.4f} ({apa_p_value(p_val_calc_w)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_w else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_w} relative to z(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_w else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_w)} is {'less than' if decision_p_alpha_w else 'not less than'} α ({alpha_w:.4f}).
        5.  **APA 7 Style Report (based on z-approximation)**:
            *T* = {t_stat_w:.1f}, *z* = {z_calc_w:.2f}, {apa_p_value(p_val_calc_w)}. The null hypothesis was {'rejected' if decision_p_alpha_w else 'not rejected'} at α = {alpha_w:.2f}. (Sample size n={n_w})
        """)

# --- Tab 7: Binomial Test ---
def tab_binomial_test():
    st.header("Binomial Test Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_b = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_b")
        n_b = st.number_input("Number of Trials (n)", 1, 1000, 20, 1, key="n_b")
        p_null_b = st.number_input("Null Hypothesis Probability (p₀)", 0.01, 0.99, 0.5, 0.01, format="%.2f", key="p_null_b")
        k_success_b = st.number_input("Number of Successes (k)", 0, n_b, int(n_b * p_null_b), 1, key="k_success_b")
        tail_b = st.radio("Tail Selection (Alternative Hypothesis)", 
                          (f"Two-tailed (p ≠ {p_null_b})", 
                           f"One-tailed (right, p > {p_null_b})", 
                           f"One-tailed (left, p < {p_null_b})"), 
                          key="tail_b")

        st.subheader("Binomial Distribution Plot")
        fig_b, ax_b = plt.subplots(figsize=(8,5))
        x_b = np.arange(0, n_b + 1)
        y_b_pmf = stats.binom.pmf(x_b, n_b, p_null_b)
        ax_b.bar(x_b, y_b_pmf, label=f'Binomial PMF (n={n_b}, p₀={p_null_b})', alpha=0.7)
        
        # Highlighting critical region for binomial is complex as it's discrete.
        # We'll show where k_success_b falls.
        ax_b.scatter([k_success_b], [stats.binom.pmf(k_success_b, n_b, p_null_b)], color='green', s=100, zorder=5, label=f'k = {k_success_b}')
        
        # Approximate critical values by finding k such that P(X<=k_crit_low) or P(X>=k_crit_high) is alpha or alpha/2
        # This is illustrative rather than a sharp cutoff like continuous.
        temp_alpha_b = alpha_b
        if tail_b.startswith("Two-tailed"): temp_alpha_b = alpha_b / 2.0
        
        # Lower critical region (illustrative)
        k_crit_low_b = stats.binom.ppf(temp_alpha_b, n_b, p_null_b)
        # Upper critical region (illustrative)
        k_crit_high_b = stats.binom.isf(temp_alpha_b, n_b, p_null_b) # isf gives value k such that P(X>=k) = q
        
        if tail_b.startswith("Two-tailed"):
            ax_b.bar(x_b[x_b <= k_crit_low_b], y_b_pmf[x_b <= k_crit_low_b], color='red', alpha=0.5, label=f'Approx. Crit Region α/2')
            ax_b.bar(x_b[x_b >= k_crit_high_b], y_b_pmf[x_b >= k_crit_high_b], color='red', alpha=0.5)
        elif tail_b.startswith("One-tailed (right"):
            ax_b.bar(x_b[x_b >= k_crit_high_b], y_b_pmf[x_b >= k_crit_high_b], color='red', alpha=0.5, label=f'Approx. Crit Region α')
        elif tail_b.startswith("One-tailed (left"):
             ax_b.bar(x_b[x_b <= k_crit_low_b], y_b_pmf[x_b <= k_crit_low_b], color='red', alpha=0.5, label=f'Approx. Crit Region α')

        ax_b.set_title(f'Binomial Distribution (n={n_b}, p₀={p_null_b})')
        ax_b.set_xlabel('Number of Successes (k)')
        ax_b.set_ylabel('Probability Mass')
        ax_b.legend()
        ax_b.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_b)

        st.subheader("Probability Table Snippet")
        k_range_table = np.arange(max(0, k_success_b - 3), min(n_b, k_success_b + 3) + 1)
        table_data_b = {
            "k": k_range_table,
            "P(X=k)": [f"{stats.binom.pmf(k_val, n_b, p_null_b):.4f}" for k_val in k_range_table],
            "P(X≤k) (CDF)": [f"{stats.binom.cdf(k_val, n_b, p_null_b):.4f}" for k_val in k_range_table],
            "P(X≥k) (SF)": [f"{stats.binom.sf(k_val -1, n_b, p_null_b):.4f}" for k_val in k_range_table] # sf(k-1) = P(X >= k)
        }
        df_table_b = pd.DataFrame(table_data_b)
        
        def highlight_k_row_b(row):
            if int(row["k"]) == k_success_b:
                return ['background-color: yellow'] * len(row)
            return [''] * len(row)
        st.markdown(df_table_b.style.apply(highlight_k_row_b, axis=1).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows probabilities around k={k_success_b}. Highlighted row is your observed k.")
        st.markdown("""
        **Cumulative Table Note:**
        * `P(X=k)` is the probability of exactly k successes.
        * `P(X≤k)` (CDF) is the cumulative probability of k or fewer successes.
        * `P(X≥k)` (SF, survival function) is the cumulative probability of k or more successes.
        * These are used to calculate p-values for one-tailed or two-tailed tests.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value for a binomial test depends on the alternative hypothesis:
        * **Two-tailed (p ≠ {p_null_b})**: Sum of probabilities of outcomes as or more extreme than k={k_success_b} in *both* tails. This is typically `2 * min(P(X ≤ k), P(X ≥ k))` if symmetrical, or more generally, sum all P(X=i) where P(X=i) ≤ P(X=k_success_b). SciPy's `binomtest` handles this. For manual calculation:
            * If k < n*p₀: p-value = 2 * P(X ≤ k) (if symmetric, otherwise more complex)
            * If k > n*p₀: p-value = 2 * P(X ≥ k) (if symmetric, otherwise more complex)
            * A common method is to sum probabilities of all outcomes at least as "unlikely" as k.
        * **One-tailed (right, p > {p_null_b})**: `P(X ≥ {k_success_b}) = stats.binom.sf({k_success_b}-1, n_b, p_null_b)`
        * **One-tailed (left, p < {p_null_b})**: `P(X ≤ {k_success_b}) = stats.binom.cdf({k_success_b}, n_b, p_null_b)`
        """)

        st.subheader("Summary")
        # Calculate p-value using scipy.stats.binomtest or manual sum for two-tailed
        # For simplicity and consistency with other tabs, we'll calculate p-values based on cdf/sf
        
        p_val_b_one_left = stats.binom.cdf(k_success_b, n_b, p_null_b)
        p_val_b_one_right = stats.binom.sf(k_success_b - 1, n_b, p_null_b) # P(X >= k) = sf(k-1)

        # Two-tailed p-value: sum of P(X=i) for all i where P(X=i) <= P(X=k_success_b)
        # This is what binomtest does. A simpler (sometimes approximate) way:
        if k_success_b == n_b * p_null_b:
            p_val_b_two = 1.0
        elif k_success_b < n_b * p_null_b:
            p_val_b_two = 2 * p_val_b_one_left
        else: # k_success_b > n_b * p_null_b
            p_val_b_two = 2 * p_val_b_one_right
        p_val_b_two = min(p_val_b_two, 1.0) # Ensure p-value is not > 1

        # For binomial, critical value is a k, not a test statistic value directly.
        # The decision is based on whether p_calc < alpha.
        # We can state the k values that would be significant.
        
        crit_val_b_display = "N/A (discrete)" # No single critical value like continuous
        p_val_for_crit_val_b_display = alpha_b

        if tail_b.startswith("Two-tailed"):
            p_val_calc_b = p_val_b_two # Using simplified version. For exact, use stats.binomtest
            # For a more precise two-tailed p-value using binomtest logic:
            # res = stats.binomtest(k_success_b, n_b, p_null_b, alternative='two-sided')
            # p_val_calc_b = res.pvalue
            # For this explorer, we stick to the sf/cdf method for transparency
            crit_val_b_display = f"k ≤ {int(k_crit_low_b)} or k ≥ {int(k_crit_high_b)} (approx for α/2)"
        elif tail_b.startswith("One-tailed (right"):
            p_val_calc_b = p_val_b_one_right
            crit_val_b_display = f"k ≥ {int(k_crit_high_b)} (approx for α)"
        else: # One-tailed (left)
            p_val_calc_b = p_val_b_one_left
            crit_val_b_display = f"k ≤ {int(k_crit_low_b)} (approx for α)"

        decision_p_alpha_b = p_val_calc_b < alpha_b
        
        # Decision by critical value is trickier for discrete.
        # We compare k_success_b to the illustrative k_crit.
        decision_crit_b = False
        comparison_crit_str_b = "N/A for direct comparison of k to a single critical k."
        if tail_b.startswith("Two-tailed"):
            if k_success_b <= k_crit_low_b or k_success_b >= k_crit_high_b : decision_crit_b = True
            comparison_crit_str_b = f"k={k_success_b} falls in approx. critical region ({crit_val_b_display})" if decision_crit_b else f"k={k_success_b} does not fall in approx. critical region"
        elif tail_b.startswith("One-tailed (right"):
            if k_success_b >= k_crit_high_b : decision_crit_b = True
            comparison_crit_str_b = f"k={k_success_b} ≥ approx. k_crit ({int(k_crit_high_b)})" if decision_crit_b else f"k={k_success_b} < approx. k_crit ({int(k_crit_high_b)})"
        elif tail_b.startswith("One-tailed (left"):
            if k_success_b <= k_crit_low_b : decision_crit_b = True
            comparison_crit_str_b = f"k={k_success_b} ≤ approx. k_crit ({int(k_crit_low_b)})" if decision_crit_b else f"k={k_success_b} > approx. k_crit ({int(k_crit_low_b)})"


        st.markdown(f"""
        1.  **Approximate Critical Region ({tail_b})**: {crit_val_b_display}
            * *Associated significance level (α)*: {p_val_for_crit_val_b_display:.4f}
        2.  **Observed Number of Successes (k)**: {k_success_b}
            * *Calculated p-value*: {p_val_calc_b:.4f} ({apa_p_value(p_val_calc_b)})
        3.  **Decision (Approx. Critical Region Method)**: The null hypothesis is **{'rejected' if decision_crit_b else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_b}. (Note: This is illustrative for discrete distributions).
        4.  **Decision (p-value Method)**: The null hypothesis is **{'rejected' if decision_p_alpha_b else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_b)} is {'less than' if decision_p_alpha_b else 'not less than'} α ({alpha_b:.4f}).
        5.  **APA 7 Style Report**:
            A binomial test indicated that the observed number of successes (k={k_success_b}, n={n_b}) was {'' if decision_p_alpha_b else 'not '}significantly different from the proportion expected under the null hypothesis (p₀={p_null_b}), {apa_p_value(p_val_calc_b)}. The null hypothesis was {'rejected' if decision_p_alpha_b else 'not rejected'} at α = {alpha_b:.2f}.
        """)

# --- Tab 8: Tukey HSD ---
def tab_tukey_hsd():
    st.header("Tukey HSD (Honestly Significant Difference) Explorer")
    col1, col2 = st.columns([2, 1.5])
    
    tukey_message = ""

    with col1:
        st.subheader("Inputs")
        alpha_tukey = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_tukey_hsd")
        k_tukey = st.number_input("Number of Groups (k)", 2, 20, 3, 1, key="k_tukey_hsd")
        df_error_tukey = st.number_input("Degrees of Freedom for Error (within-group df)", 1, 1000, 20, 1, key="df_error_tukey_hsd")
        # Tukey HSD is inherently an upper-tailed test for the q statistic.
        # test_stat_tukey would be the calculated q from user's data: (mean_i - mean_j) / sqrt(MSE/n)
        test_stat_tukey_q = st.number_input("Calculated q-statistic (for a pair)", value=1.0, format="%.3f", min_value=0.0, key="test_stat_tukey_q")

        st.subheader("Studentized Range q Distribution (Conceptual)")
        st.markdown("""
        The Tukey HSD test uses the studentized range q distribution. Plotting this distribution directly is complex and often not standard in explorers like this.
        Instead, we focus on the critical q value. The q statistic is always positive.
        The test is inherently one-sided (upper tail) for the q value.
        """)
        
        # Attempt to use statsmodels, then fallback
        q_crit_tukey = None
        try:
            from statsmodels.stats.libqsturng import qsturng
            q_crit_tukey = qsturng(1 - alpha_tukey, k_tukey, df_error_tukey)
            tukey_message = f"Critical q value calculated using `statsmodels.stats.libqsturng`: q({alpha_tukey:.3f}, k={k_tukey}, df={df_error_tukey}) = {q_crit_tukey:.3f}"
        except ImportError:
            tukey_message = "Statsmodels `qsturng` not available. Attempting CSV fallback."
            st.warning(tukey_message)
            q_crit_tukey = get_tukey_q_from_csv(df_error_tukey, k_tukey, alpha_tukey)
            if q_crit_tukey is not None:
                tukey_message += f"\nCSV fallback used. This value ({q_crit_tukey:.3f}) may be an approximation."
            else:
                tukey_message += "\nCSV fallback failed to find a value."
                st.error("Could not determine critical q value.")
        except Exception as e: # Catch other errors from qsturng if it exists but fails
            tukey_message = f"Error using `statsmodels.stats.libqsturng`: {e}. Attempting CSV fallback."
            st.warning(tukey_message)
            q_crit_tukey = get_tukey_q_from_csv(df_error_tukey, k_tukey, alpha_tukey)
            if q_crit_tukey is not None:
                tukey_message += f"\nCSV fallback used. This value ({q_crit_tukey:.3f}) may be an approximation."
            else:
                tukey_message += "\nCSV fallback failed to find a value."
                st.error("Could not determine critical q value.")


        fig_tukey, ax_tukey = plt.subplots(figsize=(8,5))
        if q_crit_tukey is not None:
            # Illustrative plot: a generic chi-squared shape to represent a positive, skewed distribution
            # This is NOT the actual q-distribution pdf, just for visual aid.
            x_placeholder = np.linspace(0.01, max(q_crit_tukey * 2, test_stat_tukey_q * 2, 5), 100)
            y_placeholder = stats.chi2.pdf(x_placeholder, df=k_tukey) # Using chi2 as a stand-in shape
            ax_tukey.plot(x_placeholder, y_placeholder, 'b-', lw=2, label=f'Conceptual q-like distribution shape')
            ax_tukey.axvline(q_crit_tukey, color='red', linestyle='--', lw=1.5, label=f'Critical q = {q_crit_tukey:.3f}')
            
            # Shade critical region
            x_fill_crit = np.linspace(q_crit_tukey, max(q_crit_tukey * 2, test_stat_tukey_q * 2, 5), 50)
            y_fill_crit = stats.chi2.pdf(x_fill_crit, df=k_tukey)
            ax_tukey.fill_between(x_fill_crit, y_fill_crit, color='red', alpha=0.5)

            ax_tukey.axvline(test_stat_tukey_q, color='green', linestyle='-', lw=2, label=f'Test q = {test_stat_tukey_q:.3f}')
            ax_tukey.set_title(f'Conceptual q-Distribution with Critical Region (α={alpha_tukey:.3f})')
            ax_tukey.set_xlabel('q-value')
            ax_tukey.set_ylabel('Density (Illustrative)')
        else:
            ax_tukey.text(0.5, 0.5, "Critical q not available for plotting.", ha='center', va='center')
            ax_tukey.set_title('Plot Unavailable')

        ax_tukey.legend()
        ax_tukey.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_tukey)
        if "CSV fallback used" in tukey_message:
            st.warning("Plot uses a generic shape; it's not the true q-distribution PDF.")


        st.subheader("Critical Value Table Snippet")
        st.info(tukey_message)
        
        # Table of critical q values for common alphas
        alphas_table_tukey = [0.10, 0.05, 0.01]
        if alpha_tukey not in alphas_table_tukey:
            alphas_table_tukey.append(alpha_tukey)
        alphas_table_tukey = sorted(list(set(alphas_table_tukey)))

        table_data_tukey = {"Alpha": [], "Critical q": [], "Source": []}
        for a_val in alphas_table_tukey:
            q_c = None
            source = ""
            try:
                from statsmodels.stats.libqsturng import qsturng
                q_c = qsturng(1 - a_val, k_tukey, df_error_tukey)
                source = "statsmodels"
            except:
                q_c = get_tukey_q_from_csv(df_error_tukey, k_tukey, a_val)
                source = "CSV Fallback" if q_c is not None else "Not Found"
            
            table_data_tukey["Alpha"].append(f"{a_val:.4f}")
            table_data_tukey["Critical q"].append(f"{q_c:.3f}" if q_c is not None else "N/A")
            table_data_tukey["Source"].append(source)

        df_table_tukey = pd.DataFrame(table_data_tukey)
        def highlight_alpha_row_tukey(row):
            if abs(float(row["Alpha"]) - alpha_tukey) < 1e-5:
                return ['background-color: yellow'] * len(row)
            return [''] * len(row)
        st.markdown(df_table_tukey.style.apply(highlight_alpha_row_tukey, axis=1).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows critical q-values for k={k_tukey}, df_error={df_error_tukey}. Highlighted for your α.")

        st.markdown("""
        **Cumulative Table Note:**
        * Tukey's HSD uses the studentized range (q) statistic.
        * The critical q value is found for a given alpha, number of groups (k), and error df.
        * If your calculated q for a pair of means exceeds the critical q, that pair is significantly different.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value for a specific calculated q-statistic ({test_stat_tukey_q:.3f}) from Tukey's HSD is complex to compute directly as it involves the CDF of the studentized range distribution.
        * Software packages (like R's `ptukey`) can calculate this.
        * `statsmodels.stats.libqsturng.psturng(q_stat, k, df)` would give P(Q < q_stat), so p-value = `1 - psturng(q_stat, k, df)`.
        * This explorer does not directly calculate the p-value for the *test_stat_tukey_q* due to complexity without full statsmodels or similar. Instead, we compare q_calc to q_crit.
        * The "associated p-value" for the critical q IS alpha.
        """)

        st.subheader("Summary")
        p_val_for_crit_val_tukey_display = alpha_tukey
        
        # We don't calculate p-value for test_stat_tukey_q here, mark as N/A
        p_val_calc_tukey_display = "N/A (requires specialized function)"
        apa_p_val_calc_tukey_display = "p N/A"

        decision_crit_tukey = False
        comparison_crit_str_tukey = "Critical q not available"
        if q_crit_tukey is not None:
            decision_crit_tukey = test_stat_tukey_q > q_crit_tukey
            comparison_crit_str_tukey = f"q(calc) ({test_stat_tukey_q:.3f}) > q(crit) ({q_crit_tukey:.3f})" if decision_crit_tukey else f"q(calc) ({test_stat_tukey_q:.3f}) ≤ q(crit) ({q_crit_tukey:.3f})"
        
        # Decision by p-value vs alpha is not directly applicable here as we don't calculate p_calc
        decision_p_alpha_tukey_display = "N/A (p-value for q_calc not computed)"
        reason_p_alpha_tukey_display = "p-value for q_calc not computed by this tool."

        st.markdown(f"""
        1.  **Critical q-value**: {q_crit_tukey:.3f if q_crit_tukey is not None else "N/A"}
            * *Associated significance level (α)*: {p_val_for_crit_val_tukey_display:.4f}
        2.  **Calculated q-statistic (for one pair)**: {test_stat_tukey_q:.3f}
            * *Calculated p-value*: {p_val_calc_tukey_display} 
        3.  **Decision (Critical Value Method)**: For this pair, the null hypothesis (of no difference) is **{'rejected' if decision_crit_tukey else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_tukey}.
        4.  **Decision (p-value Method)**: {decision_p_alpha_tukey_display}
            * *Reason*: {reason_p_alpha_tukey_display}
        5.  **APA 7 Style Report (for this specific comparison, if q_crit available)**:
            A Tukey HSD test was performed. For the compared pair, the calculated q-statistic was {test_stat_tukey_q:.2f}. {'This exceeded the critical q value (' + f'{q_crit_tukey:.2f}' + ') at α = ' + f'{alpha_tukey:.2f}' + ', indicating a significant difference.' if decision_crit_tukey and q_crit_tukey is not None else 'This did not exceed the critical q value (' + (f'{q_crit_tukey:.2f}' if q_crit_tukey is not None else 'N/A') + ') at α = ' + f'{alpha_tukey:.2f}' + ', indicating no significant difference.'} (k={k_tukey}, df<sub>error</sub>={df_error_tukey}).
        """)
        st.caption("Note: A full Tukey HSD analysis involves pairwise comparisons for all groups. This tab focuses on a single comparison against the critical q.")

# --- Tab 9: Kruskal-Wallis Test ---
def tab_kruskal_wallis():
    st.header("Kruskal-Wallis H Test (Chi-square Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_kw = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_kw")
        k_groups_kw = st.number_input("Number of Groups (k)", 3, 20, 3, 1, key="k_groups_kw") # KW needs at least 3 groups
        # For KW, df = k - 1
        df_kw = k_groups_kw - 1
        st.markdown(f"Degrees of Freedom (df) = k - 1 = {df_kw}")
        # Kruskal-Wallis H statistic is compared against chi-square distribution.
        # It's an upper-tailed test.
        test_stat_h_kw = st.number_input("Calculated H-statistic", value=float(df_kw), format="%.3f", min_value=0.0, key="test_stat_h_kw")
        st.caption("Note: For the chi-square approximation to be good, each group size should ideally be ≥ 5.")

        st.subheader("Chi-square Distribution Plot (Approximation for H)")
        fig_kw, ax_kw = plt.subplots(figsize=(8,5))
        if df_kw > 0:
            x_chi2_max_kw = stats.chi2.ppf(0.999, df_kw) if stats.chi2.ppf(0.999, df_kw) > 10 else 10
            x_chi2_kw = np.linspace(0.001, x_chi2_max_kw, 500)
            y_chi2_kw = stats.chi2.pdf(x_chi2_kw, df_kw)
            ax_kw.plot(x_chi2_kw, y_chi2_kw, 'b-', lw=2, label=f'χ²-distribution (df={df_kw})')

            crit_val_chi2_kw = stats.chi2.ppf(1 - alpha_kw, df_kw)
            x_fill_upper_kw = np.linspace(crit_val_chi2_kw, x_chi2_max_kw, 100)
            ax_kw.fill_between(x_fill_upper_kw, stats.chi2.pdf(x_fill_upper_kw, df_kw), color='red', alpha=0.5, label=f'α = {alpha_kw:.4f}')
            ax_kw.axvline(crit_val_chi2_kw, color='red', linestyle='--', lw=1, label=f'χ²_crit = {crit_val_chi2_kw:.3f}')
            ax_kw.axvline(test_stat_h_kw, color='green', linestyle='-', lw=2, label=f'H_calc = {test_stat_h_kw:.3f}')
            ax_kw.set_title(f'χ²-Approximation for Kruskal-Wallis H (df={df_kw})')
            ax_kw.set_xlabel('χ²-value / H-statistic')
            ax_kw.set_ylabel('Probability Density')
        else:
            ax_kw.text(0.5, 0.5, "df must be > 0 (k > 1)", ha='center', va='center')
            ax_kw.set_title('Plot Unavailable')
            crit_val_chi2_kw = None


        ax_kw.legend()
        ax_kw.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_kw)

        st.subheader("Critical χ² Value Table Snippet (for H test)")
        if df_kw > 0:
            alphas_table_chi2_kw = [0.10, 0.05, 0.01, alpha_kw]
            alphas_table_chi2_kw = sorted(list(set(alphas_table_chi2_kw)))
            table_data_chi2_kw = {"Alpha (Right Tail)": [], "Critical χ² (Upper)": []}
            for a_val in alphas_table_chi2_kw:
                table_data_chi2_kw["Alpha (Right Tail)"].append(f"{a_val:.4f}")
                table_data_chi2_kw["Critical χ² (Upper)"].append(f"{stats.chi2.ppf(1 - a_val, df_kw):.3f}")
            df_table_chi2_kw = pd.DataFrame(table_data_chi2_kw)
            
            def highlight_alpha_row_chi2_kw(row):
                if abs(float(row["Alpha (Right Tail)"]) - alpha_kw) < 1e-5:
                    return ['background-color: yellow'] * len(row)
                return [''] * len(row)
            st.markdown(df_table_chi2_kw.style.apply(highlight_alpha_row_chi2_kw, axis=1).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows critical χ²-values for df={df_kw}. Highlighted for your α.")
        else:
            st.warning("df must be > 0 to generate table.")

        st.markdown("""
        **Cumulative Table Note:**
        * The Kruskal-Wallis H statistic is approximately chi-square distributed with df = k-1.
        * The test is typically right-tailed: a large H suggests differences between groups.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value for the Kruskal-Wallis test is the probability of observing an H statistic as large as, or larger than, the calculated H ({test_stat_h_kw:.3f}), assuming the null hypothesis (that all group medians are equal) is true. This is found using the chi-square survival function (sf):
        * `P(χ² ≥ H_calc) = stats.chi2.sf({test_stat_h_kw:.3f}, df={df_kw})`
        """)

        st.subheader("Summary")
        p_val_for_crit_val_kw_display = alpha_kw
        
        if df_kw > 0:
            p_val_calc_kw = stats.chi2.sf(test_stat_h_kw, df_kw)
            decision_crit_kw = test_stat_h_kw > crit_val_chi2_kw
            comparison_crit_str_kw = f"H({test_stat_h_kw:.3f}) > χ²_crit({crit_val_chi2_kw:.3f})" if decision_crit_kw else f"H({test_stat_h_kw:.3f}) ≤ χ²_crit({crit_val_chi2_kw:.3f})"
            decision_p_alpha_kw = p_val_calc_kw < alpha_kw
            apa_H_stat = f"*H*({df_kw}) = {test_stat_h_kw:.2f}" # Scipy returns H, sometimes χ² is used in report.
        else:
            p_val_calc_kw = float('nan')
            decision_crit_kw = False
            comparison_crit_str_kw = "df must be > 0"
            decision_p_alpha_kw = False
            apa_H_stat = "*H* = N/A (df=0)"


        st.markdown(f"""
        1.  **Critical χ²-value (df={df_kw})**: {crit_val_chi2_kw:.3f if df_kw > 0 else "N/A"}
            * *Associated p-value (α)*: {p_val_for_crit_val_kw_display:.4f}
        2.  **Calculated H-statistic**: {test_stat_h_kw:.3f}
            * *Calculated p-value (from χ² approx.)*: {p_val_calc_kw:.4f if df_kw > 0 else "N/A"} ({apa_p_value(p_val_calc_kw) if df_kw > 0 else "N/A"})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_kw else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_kw}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_kw else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_kw) if df_kw > 0 else "p N/A"} is {'less than' if decision_p_alpha_kw else 'not less than'} α ({alpha_kw:.4f}).
        5.  **APA 7 Style Report**:
            A Kruskal-Wallis H test showed that there was a {'' if decision_p_alpha_kw else 'not a '}statistically significant difference in medians between the k={k_groups_kw} groups, {apa_H_stat}, {apa_p_value(p_val_calc_kw) if df_kw > 0 else "p N/A"}. The null hypothesis was {'rejected' if decision_p_alpha_kw else 'not rejected'} at α = {alpha_kw:.2f}.
        """)

# --- Tab 10: Friedman Test ---
def tab_friedman_test():
    st.header("Friedman Test (Chi-square Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_fr = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_fr")
        k_conditions_fr = st.number_input("Number of Conditions/Treatments (k)", 3, 20, 3, 1, key="k_conditions_fr") # Friedman needs k >= 3 generally
        n_blocks_fr = st.number_input("Number of Blocks/Subjects (n)", 3, 100, 10, 1, key="n_blocks_fr")
        
        # For Friedman, df = k - 1
        df_fr = k_conditions_fr - 1
        st.markdown(f"Degrees of Freedom (df) = k - 1 = {df_fr}")
        
        # Friedman test statistic (Q or FR or χ²_r) is compared against chi-square. Upper-tailed.
        test_stat_q_fr = st.number_input("Calculated Friedman Q-statistic (or χ²_r)", value=float(df_fr), format="%.3f", min_value=0.0, key="test_stat_q_fr")

        if n_blocks_fr <= 10 :
            st.warning("Small sample (n ≤ 10). Friedman’s chi-square approximation may be less reliable. Exact methods or specialized tables are preferred where possible for small n or k.")

        st.subheader("Chi-square Distribution Plot (Approximation for Q)")
        fig_fr, ax_fr = plt.subplots(figsize=(8,5))

        if df_fr > 0:
            x_chi2_max_fr = stats.chi2.ppf(0.999, df_fr) if stats.chi2.ppf(0.999, df_fr) > 10 else 10
            x_chi2_fr = np.linspace(0.001, x_chi2_max_fr, 500)
            y_chi2_fr = stats.chi2.pdf(x_chi2_fr, df_fr)
            ax_fr.plot(x_chi2_fr, y_chi2_fr, 'b-', lw=2, label=f'χ²-distribution (df={df_fr})')

            crit_val_chi2_fr = stats.chi2.ppf(1 - alpha_fr, df_fr)
            x_fill_upper_fr = np.linspace(crit_val_chi2_fr, x_chi2_max_fr, 100)
            ax_fr.fill_between(x_fill_upper_fr, stats.chi2.pdf(x_fill_upper_fr, df_fr), color='red', alpha=0.5, label=f'α = {alpha_fr:.4f}')
            ax_fr.axvline(crit_val_chi2_fr, color='red', linestyle='--', lw=1, label=f'χ²_crit = {crit_val_chi2_fr:.3f}')
            ax_fr.axvline(test_stat_q_fr, color='green', linestyle='-', lw=2, label=f'Q_calc = {test_stat_q_fr:.3f}')
            ax_fr.set_title(f'χ²-Approximation for Friedman Q (df={df_fr})')
            ax_fr.set_xlabel('χ²-value / Q-statistic')
            ax_fr.set_ylabel('Probability Density')
        else:
            ax_fr.text(0.5, 0.5, "df must be > 0 (k > 1)", ha='center', va='center')
            ax_fr.set_title('Plot Unavailable')
            crit_val_chi2_fr = None

        ax_fr.legend()
        ax_fr.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_fr)

        st.subheader("Critical χ² Value Table Snippet (for Q test)")
        if df_fr > 0:
            alphas_table_chi2_fr = [0.10, 0.05, 0.01, alpha_fr]
            alphas_table_chi2_fr = sorted(list(set(alphas_table_chi2_fr)))
            table_data_chi2_fr = {"Alpha (Right Tail)": [], "Critical χ² (Upper)": []}
            for a_val in alphas_table_chi2_fr:
                table_data_chi2_fr["Alpha (Right Tail)"].append(f"{a_val:.4f}")
                table_data_chi2_fr["Critical χ² (Upper)"].append(f"{stats.chi2.ppf(1 - a_val, df_fr):.3f}")
            df_table_chi2_fr = pd.DataFrame(table_data_chi2_fr)
            
            def highlight_alpha_row_chi2_fr(row):
                if abs(float(row["Alpha (Right Tail)"]) - alpha_fr) < 1e-5:
                    return ['background-color: yellow'] * len(row)
                return [''] * len(row)
            st.markdown(df_table_chi2_fr.style.apply(highlight_alpha_row_chi2_fr, axis=1).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows critical χ²-values for df={df_fr}. Highlighted for your α.")
        else:
            st.warning("df must be > 0 to generate table.")

        st.markdown("""
        **Cumulative Table Note:**
        * The Friedman Q statistic is approximately chi-square distributed with df = k-1.
        * The test is right-tailed: a large Q suggests differences between condition medians.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value for the Friedman test is the probability of observing a Q statistic as large as, or larger than, the calculated Q ({test_stat_q_fr:.3f}), assuming the null hypothesis (that all condition medians are equal across blocks) is true. This is found using the chi-square survival function (sf):
        * `P(χ² ≥ Q_calc) = stats.chi2.sf({test_stat_q_fr:.3f}, df={df_fr})`
        """)

        st.subheader("Summary")
        p_val_for_crit_val_fr_display = alpha_fr
        
        if df_fr > 0:
            p_val_calc_fr = stats.chi2.sf(test_stat_q_fr, df_fr)
            decision_crit_fr = test_stat_q_fr > crit_val_chi2_fr
            comparison_crit_str_fr = f"Q({test_stat_q_fr:.3f}) > χ²_crit({crit_val_chi2_fr:.3f})" if decision_crit_fr else f"Q({test_stat_q_fr:.3f}) ≤ χ²_crit({crit_val_chi2_fr:.3f})"
            decision_p_alpha_fr = p_val_calc_fr < alpha_fr
            apa_Q_stat = f"χ²<sub>r</sub>({df_fr}) = {test_stat_q_fr:.2f}" # Common notation for Friedman stat
        else:
            p_val_calc_fr = float('nan')
            decision_crit_fr = False
            comparison_crit_str_fr = "df must be > 0"
            decision_p_alpha_fr = False
            apa_Q_stat = "χ²<sub>r</sub> = N/A (df=0)"

        st.markdown(f"""
        1.  **Critical χ²-value (df={df_fr})**: {crit_val_chi2_fr:.3f if df_fr > 0 else "N/A"}
            * *Associated p-value (α)*: {p_val_for_crit_val_fr_display:.4f}
        2.  **Calculated Q-statistic (χ²_r)**: {test_stat_q_fr:.3f}
            * *Calculated p-value (from χ² approx.)*: {p_val_calc_fr:.4f if df_fr > 0 else "N/A"} ({apa_p_value(p_val_calc_fr) if df_fr > 0 else "N/A"})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_fr else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_fr}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_fr else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_fr) if df_fr > 0 else "p N/A"} is {'less than' if decision_p_alpha_fr else 'not less than'} α ({alpha_fr:.4f}).
        5.  **APA 7 Style Report**:
            A Friedman test indicated that there was a {'' if decision_p_alpha_fr else 'not a '}statistically significant difference in medians across the k={k_conditions_fr} conditions for n={n_blocks_fr} blocks, {apa_Q_stat}, {apa_p_value(p_val_calc_fr) if df_fr > 0 else "p N/A"}. The null hypothesis was {'rejected' if decision_p_alpha_fr else 'not rejected'} at α = {alpha_fr:.2f}.
        """, unsafe_allow_html=True)


# --- Main app ---
def main():
    st.set_page_config(page_title="Statistical Table Explorer", layout="wide")
    st.title("🔢 Statistical Table Explorer")
    st.markdown("""
    This application provides an interactive way to explore various statistical distributions and tests. 
    Select a tab to begin. On each tab, you can adjust parameters like alpha, degrees of freedom, 
    and input a calculated test statistic to see how it compares to critical values and to understand p-value calculations.
    """)

    tab_names = [
        "t-Distribution", "z-Distribution", "F-Distribution", "Chi-square (χ²)",
        "Mann-Whitney U", "Wilcoxon Signed-Rank T", "Binomial Test",
        "Tukey HSD", "Kruskal-Wallis H", "Friedman Test"
    ]
    
    tabs = st.tabs(tab_names)

    with tabs[0]:
        tab_t_distribution()
    with tabs[1]:
        tab_z_distribution()
    with tabs[2]:
        tab_f_distribution()
    with tabs[3]:
        tab_chi_square_distribution()
    with tabs[4]:
        tab_mann_whitney_u()
    with tabs[5]:
        tab_wilcoxon_t()
    with tabs[6]:
        tab_binomial_test()
    with tabs[7]:
        tab_tukey_hsd()
    with tabs[8]:
        tab_kruskal_wallis()
    with tabs[9]:
        tab_friedman_test()

if __name__ == "__main__":
    main()
