#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Streamlit UI
# TO DEPLOY THE STREAMLIT APP, YOU MUST CONVERT THIS JUPYTER FILE TO A PYTHON SOURCE FILE
# RUN THIS: jupyter nbconvert --to script emissions_streamlit.ipynb

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import xgboost as xgb

# ---------------------------
# LOAD MODEL AND DATA
# ---------------------------
model_path = "xgboost_model.json"
model = xgb.XGBRegressor()
model.load_model(model_path)

df = pd.read_excel("emissions_forecasting_data.xlsx")
for col in df.columns:
    if col != "State":
        df[col] = pd.to_numeric(df[col], errors="coerce")

feature_cols = model.get_booster().feature_names

# ---------------------------
# CONFIGURATION
# ---------------------------
raw_cols = [
    "coal_use", "natural_gas_use", "petroleum_use", "nuclear_use",
    "biomass_use", "geothermal_use", "hydro_use", "solar_use", "wind_use"
]

pct_cols = [
    "%_coal", "%_natural_gas", "%_petroleum", "%_nuclear",
    "%_biomass", "%_geothermal", "%_hydro", "%_solar", "%_wind"
]

st.set_page_config(layout="wide")

# ---------------------------
# HEADER
# ---------------------------
st.markdown(
    """
    <h1 style="color:green">
        US CO₂ Emissions Simulator
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="width:90%; font-size:18px; line-height:1.5;">
        This simulation uses aggregated U.S. 2023 energy data as the baseline. Here we employ a counterfactual analysis, which asks: 
        How would 2023 emissions have been under different under altered conditions? Below you can adjust 2023 data 
        relating to energy usage, energy composition, and GDP to simulate hypothetical situations.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# GET U.S. 2023 BASELINE
# ---------------------------
df_2023 = df[df["Year"] == 2023].copy()

# Sum all numeric columns for USA aggregate (for slider defaults)
latestUSA = df_2023[raw_cols + ["total_energy", "real_gdp", "total_co2"]].sum(numeric_only=True)

# Recompute percent mix for USA
for r, p in zip(raw_cols, pct_cols):
    latestUSA[p] = latestUSA[r] / latestUSA["total_energy"] if latestUSA["total_energy"] != 0 else 0

latestUSA["Year"] = 2023

st.subheader("Adjust Inputs for USA 2023 Counterfactual")

# ---------------------------
# TOTAL ENERGY AND GDP INPUT
# ---------------------------
gdp_pct = st.number_input("U.S. GDP change (%)", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)
energy_pct = st.number_input("U.S. Total Energy change (%)", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)


# ---------------------------
# COLUMNS FOR MIX TABLE AND CHART
# ---------------------------
col1, col2 = st.columns([1, 2])  # 1:2 ratio, adjust width

# Column 1: Table
with col1:
    # ---------------------------
    # ENERGY MIX INPUT TABLE
    # ---------------------------
    st.write("**Energy Mix**")
    st.write(
    "Original: Percent breakdown for total primary energy consumption in 2023.\n"
    "Scenario: Click on the cells to modify the energy breakdown of 2023 and visualize how that change affects emissions."
    )

    
   # Prepare energy mix table
    mix_df = pd.DataFrame({
        "Energy Source": [c.replace("_use","").replace("_"," ").title() for c in raw_cols],
        "Original (%)": [round(float(latestUSA[p]*100), 2) for p in pct_cols],
        "Scenario (%)": [round(float(latestUSA[p]*100), 2) for p in pct_cols]
    })
    
    # Style function to highlight just the "Scenario (%)" column header
    def highlight_scenario_header(styler):
        # Apply styles to the header row
        styles = []
        for col in styler.data.columns:
            if col == "Scenario (%)":
                styles.append({'selector': f'th.col{i}', 'props': [('text-decoration', 'underline'), ('color', 'red')]})
        return styler
    
    # Apply styling and display editable table
    styled_df = mix_df.style.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]}  # optional: center all headers
    ]).apply(lambda x: ['' for _ in x], axis=1)  # keep cell styling empty
    
    # Now show in Streamlit
    edited_df = st.data_editor(
        styled_df,
        num_rows="fixed",
        use_container_width=True,
        key="energy_mix_table"
    )
    
    # Ensure numeric
    edited_df["Scenario (%)"] = pd.to_numeric(edited_df["Scenario (%)"], errors="coerce")
    
    # Validate sum = 100%
    total_pct = edited_df["Scenario (%)"].sum()
    if not np.isclose(total_pct, 100.0, atol=0.1):
        st.error(f"❌ Scenario percentages must sum to 100%. Current sum = {total_pct:.2f}%")
        st.stop()


# Column 2: Chart
with col2:
    # --------------------------- 
    # STACKED BAR — Energy Mix Percentages 
    # --------------------------- 
    st.write("### 🔥 Energy Mix: Actual 2023 vs Scenario 2023 (%)")
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Get actual and scenario percentages
    before_vals = [latestUSA[p]*100 for p in pct_cols]
    after_vals = [edited_df.loc[i, "Scenario (%)"] for i in range(len(raw_cols))]
    
    # FIXED:Proper stacking
    x_pos = [0, 1]
    bar_width = 0.6
    colors = plt.cm.tab10(np.linspace(0, 1, len(raw_cols)))
    
    bottom_before = 0
    bottom_after = 0
    
    for i, (before, after, color, source) in enumerate(zip(before_vals, after_vals, colors, raw_cols)):
        label = source.replace("_use","").replace("_"," ").title()
        ax3.bar(x_pos[0], before, bar_width, bottom=bottom_before, color=color, label=label, edgecolor='white', linewidth=0.5)
        ax3.bar(x_pos[1], after, bar_width, bottom=bottom_after, color=color, edgecolor='white', linewidth=0.5)
        bottom_before += before
        bottom_after += after
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(["Actual 2023", "Counterfactual 2023"], fontsize=12)
    ax3.set_ylabel("Energy Mix (%)", fontsize=12)
    ax3.set_ylim(0, 105)
    ax3.set_title("Energy Mix Comparison", fontsize=14, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    st.pyplot(fig3)    

# ---------------------------
# PER-STATE SCENARIO AND PREDICTIONS
# ---------------------------
state_predictions = []

for idx, row in df_2023.iterrows():
    scenario = row.copy()
    
    # Apply user adjustments (GDP and Energy Consumption)
    scenario["total_energy"] = row["total_energy"] * (1 + energy_pct / 100)
    scenario["real_gdp"] = row["real_gdp"] * (1 + gdp_pct / 100)

    
    # Update energy mix per state
    state_total_energy = scenario["total_energy"]
    for i, col in enumerate(raw_cols):
        pct_value = edited_df.loc[i, "Scenario (%)"]
        scenario[col] = state_total_energy * (pct_value / 100)
        scenario[pct_cols[i]] = scenario[col] / state_total_energy if state_total_energy != 0 else 0
    
    scenario["percent_sum"] = scenario[pct_cols].sum()
    
    # Use actual 2022 lag1 values
    state_2022 = df[(df["State"]==row["State"]) & (df["Year"]==2022)]
    if len(state_2022) > 0:
        lag_mapping = {
            'coal_use_lag1': 'coal_use',
            'natural_gas_use_lag1': 'natural_gas_use',
            'petroleum_use_lag1': 'petroleum_use',
            'nuclear_use_lag1': 'nuclear_use',
            'renewables_use_lag1': 'renewables_use',
            'biomass_use_lag1': 'biomass_use',
            'geothermal_use_lag1': 'geothermal_use',
            'hydro_use_lag1': 'hydro_use',
            'solar_use_lag1': 'solar_use',
            'wind_use_lag1': 'wind_use',
            'real_gdp_lag1': 'real_gdp',
            'carbon_intensity_lag1': 'carbon_intensity',
            'electricity_sales_lag1': 'electricity_sales',
            'total_consumption_lag1': 'total_consumption'
        }
        for lag_col, base_col in lag_mapping.items():
            scenario[lag_col] = state_2022.iloc[0][base_col]
    
    # rps_lag2 from 2021
    state_2021 = df[(df["State"]==row["State"]) & (df["Year"]==2021)]
    if len(state_2021) > 0:
        scenario["rps_lag2"] = state_2021.iloc[0]["rps"]
    
    # Fill missing features with 0
    for col in feature_cols:
        if col not in scenario.index:
            scenario[col] = 0
    
    # Align for model
    Xs = scenario[feature_cols].to_frame().T
    Xs = Xs.apply(pd.to_numeric, errors="coerce").fillna(0)
    
    # Predict
    pred = float(model.predict(Xs)[0])
    state_predictions.append(pred)

# Add predictions to df_2023
df_2023["predicted_emissions"] = state_predictions

# Aggregate total US emissions
total_us_emissions = df_2023["predicted_emissions"].sum()
st.write(f"### State CO₂ Emissions (Percent Change) ")

# ---------------------------
# STATE-LEVEL MAP
# ---------------------------
state_changes = []
for idx, row in df_2023.iterrows():
    actual = row["total_co2"]
    pred = row["predicted_emissions"]
    pct_change = 100 * (pred - actual) / actual if actual != 0 else 0
    state_changes.append({
        "State": row["State"],
        "Actual 2023": actual,
        "Counterfactual 2023": pred,
        "% Change": pct_change
    })
map_df = pd.DataFrame(state_changes)

fig_map = px.choropleth(
    map_df,
    locations="State",
    locationmode="USA-states",
    color="% Change",
    color_continuous_scale=["green","white","red"],
    color_continuous_midpoint=0,
    scope="usa",
    labels={"% Change":"% Change"},
    hover_data={
        "State": True,
        "% Change": ":.2f",
        "Actual 2023": ":,.0f",
        "Counterfactual 2023": ":,.0f"
    }
)
# Update layout: centered title and legend closer to plot
fig_map.update_layout(
    title={
        'text': "2023 Counterfactual vs Actual",
        'x':0.5,  # center
        'xanchor': 'center'
    },
    legend=dict(
        x=1.02,  # slightly to the right of plot
        y=1,
        xanchor='left',
        yanchor='top'
    ),
    margin=dict(l=20, r=20, t=60, b=20)  # optional, tighten margins
)
st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------
# KPI CARDS
# ---------------------------
st.write("## 📊 Emission Impact (USA)")

actual_co2 = df_2023["total_co2"].sum()
predicted_co2 = total_us_emissions
delta = predicted_co2 - actual_co2
delta_pct = 100 * delta / actual_co2 if actual_co2 != 0 else 0

c1, c2, c3 = st.columns(3)
c1.metric("Actual CO₂ (2023)", f"{actual_co2:,.0f} MMT")
c2.metric("Counterfactual CO₂ (2023)", f"{predicted_co2:,.0f} MMT", f"{delta_pct:+.1f}%", delta_color = 'inverse')
c3.metric("What-If Change", f"{delta:+,.0f} MMT")

# ---------------------------
# SCENARIO SUMMARY TABLE
# ---------------------------
st.write("### Scenario Summary")
with st.expander("📋 View Complete Scenario Inputs"):
    # Show per-state changes: total_energy, real_gdp, energy mix percentages
    summary_cols = ["State", "Year", "total_energy", "real_gdp"] + raw_cols + pct_cols
    # For clean display, round numeric values
    scenario_display = df_2023[summary_cols].copy()
    for col in summary_cols[2:]:  # skip State and Year
        scenario_display[col] = scenario_display[col].round(2)
    
    st.dataframe(scenario_display, use_container_width=True)


# In[ ]:




