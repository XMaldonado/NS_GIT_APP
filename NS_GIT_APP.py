import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from snowflake.snowpark.context import get_active_session


st.set_page_config(page_title="Multi-Page App", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Home", "Curve", "Fundamental", "Z-Score of Peers"])

# Home Page
if page == "Home":
    st.title("Welcome!")
    
    st.markdown("#### Your Gateway to Smarter Trading Decisions")
    st.markdown(
         """
        Use the navigation menu to get started.
        """
    )
    st.image("City Scape.jpg")
    st.balloons()

# Curve Page
elif page == "Curve":
    st.title("📈 Curve")
    tab1, tab2 = st.tabs(["IG", "HY"])
    with tab1:    
        st.subheader("IG - Nelson-Siegel")
        #st.set_option('snowflake.streamlitSleepTimeoutMinutes', 60)
        # --- Setup Streamlit Page ---
        #st.set_page_config(layout="wide")
        #st.title("IG - Nelson-Siegel")
        
        # --- Deviation Threshold ---
        deviation_threshold = 5
        # Get the current credentials
        session = get_active_session()
        # --- Fetch Bloomberg Data ---
        @st.cache_data(show_spinner="Fetching Bloomberg bond data...")
        def fetch_bloomberg_data(_session):
            conn = _session.connection  # Use the connection from the active session
            query = "SELECT * FROM CORPORATE.INVESTMENT_GRADE.DAILY_CURVE_IG_DATA"
            df = pd.read_sql(query, conn)
            return df
        df_bonds = fetch_bloomberg_data(session)
        # --- Process Bloomberg Data ---
        df_bonds['TICKER'] = df_bonds['TICKER'].astype(str)
        df_bonds['COUPON'] = df_bonds['COUPON'].astype(str)
        df_bonds['MATURDATE'] = pd.to_datetime(df_bonds['MATURDATE'], format='%Y%m%d').dt.strftime('%m/%d/%Y')
        df_bonds.insert(3, 'ID', df_bonds['TICKER'] + ' ' + df_bonds['COUPON'] + ' ' + df_bonds['MATURDATE'])
        df_bonds['DURADJMOD'] = pd.to_numeric(df_bonds['DURADJMOD'], errors='coerce')
        df_bonds['OAS_BP'] = pd.to_numeric(df_bonds['OAS_BP'], errors='coerce')
        df_bonds = df_bonds.dropna(subset=['DURADJMOD', 'OAS_BP'])
        
        # Remove low-frequency tickers
        df_bonds = df_bonds[df_bonds['TICKER'].map(df_bonds['TICKER'].value_counts()) >= 5]
        # --- Fetch Positions Data ---
        @st.cache_data(show_spinner="Fetching Position Data...")
        def fetch_positions(_session):
            conn = _session.connection  # Use the connection from the active session
            #query2 = """SELECT * FROM TRADERS.EAGLE.SELCTION_POSITIONS """
            query = """SELECT * FROM CORPORATE.GENERAL.CORP_CURRENT_POSITIONS"""
            df = pd.read_sql(query, conn)
            return df
        df_positions = fetch_positions(session)
        # --- Process Positions ---
        #for this data SHARE_PAR_VALUE used yo be PRICE_SOD
        df_positions.columns = [f"{col}_{i}" if df_positions.columns.duplicated()[i] else col for i, col in enumerate(df_positions.columns)]
        df_positions = df_positions[~df_positions['CRD_STRATEGY'].isin(['INS', 'MODEL', 'PLEDGE','UNSUP'])]
        
        
        box_map = (df_positions.groupby(['CUSIP', 'TICK'])['CRD_STRATEGY']
           .apply(lambda x: ', '.join(sorted(set(x))))
           .reset_index()
           .rename(columns={'CRD_STRATEGY': 'all_possible_strategies'})
        )
        # Step 3: Merge back and drop unnecessary columns
        df_positions = df_positions.drop(columns=['CRD_STRATEGY']).drop_duplicates()
        df_positions = df_positions.merge(box_map, on=['CUSIP', 'TICK'], how='left')
        
        
        
        df_positions = df_positions.groupby(['CUSIP', 'TICK', 'all_possible_strategies'])['SHARE_PAR_VALUE'].sum().reset_index()
        df_positions['SHARE_PAR_VALUE'] = pd.to_numeric(df_positions['SHARE_PAR_VALUE'], errors='coerce')
        df_positions = df_positions[df_positions['SHARE_PAR_VALUE'] > 2000000]
        # --- Merge Ownership Info ---
        df_bonds = df_bonds.merge(df_positions[['CUSIP', 'all_possible_strategies']], on='CUSIP', how='left', indicator=True)
        df_bonds['Own?'] = df_bonds['_merge'].map({'both': 'Y', 'left_only': 'N', 'right_only': 'N'})
        
        
        
        st.success(f"Fetched bloomberg and positions data.")
        
        # --- Nelson-Siegel function ---
        def ns_func(x, beta0, beta1, beta2, lambda1):
            term1 = beta0
            term2 = beta1 * (1 - np.exp(-x / lambda1)) / (x / lambda1)
            term3 = beta2 * ((1 - np.exp(-x / lambda1)) / (x / lambda1) - np.exp(-x / lambda1))
            return term1 + term2 + term3
        
        # --- Fit NS curve ---
        def fit_ns_curve(x, y):
            try:
                initial_params = [0.01, -0.01, 0.01, 1.0]
                params, _ = curve_fit(ns_func, x, y, p0=initial_params, maxfev=10000)
                y_fit = ns_func(x, *params)
                return params, y_fit
            except Exception as e:
                st.warning(f"Error fitting curve: {e}")
                return None, None
        
        
        trade_signals_tot = []
        for ticker in df_bonds['TICKER'].unique():
            df_filtered1 = df_bonds[df_bonds['TICKER'] == ticker]
            if len(df_filtered1) < 2:
                continue
            
            x = df_filtered1['DURADJMOD'].values
            y = df_filtered1['OAS_BP'].values
            params, y_fit = fit_ns_curve(x, y)
        
            if params is not None:
                df_filtered1['NS_FIT'] = ns_func(df_filtered1['DURADJMOD'], *params)
                df_filtered1['Deviation'] = df_filtered1['OAS_BP'] - df_filtered1['NS_FIT']
                df_filtered1['Above/Below'] = np.where(df_filtered1['Deviation'] > 0, 'Above', 'Below')
        
                df_below_owned1 = df_filtered1[
                    (df_filtered1['Own?'] == 'Y') &
                    (df_filtered1['Deviation'] < -deviation_threshold / 100 * df_filtered1['NS_FIT'])
                ].copy()
        
                df_above_unowned1 = df_filtered1[
                    (df_filtered1['Own?'] == 'N') &
                    (df_filtered1['Deviation'] > deviation_threshold / 100 * df_filtered1['NS_FIT'])
                ].copy()
        
                for _, row_below in df_below_owned1.iterrows():
                    for _, row_above in df_above_unowned1.iterrows():
                        if row_above['DURADJMOD'] > row_below['DURADJMOD']:
                            ratio = (row_above['OAS_BP'] - row_below['OAS_BP']) / (row_above['DURADJMOD'] - row_below['DURADJMOD'])
                            if ratio > 12:
                                trade_signals_tot.append({
                                    "CRD_STRATEGY1": row_below[ 'all_possible_strategies'],
                                    "Owned ID1": row_below['ID'],
                                    "Matched ID1": row_above['ID'],
                                    "Ratio OAS/Dur1": round(ratio, 2),
                                    "OAS Diff1": round(row_above['OAS_BP'] - row_below['OAS_BP'], 2),
                                    "Dur Diff1": round(row_above['DURADJMOD'] - row_below['DURADJMOD'], 2),
                                    "Deviation Owned1": round(row_below['Deviation'], 2),
                                    "Deviation Matched1": round(row_above['Deviation'], 2),
                                    "Dev Diff1": round(row_above['Deviation'] - row_below['Deviation'], 2)
                                })
        
                trade_signals_tot = sorted(trade_signals_tot, key=lambda x: (x['Owned ID1'], x['Ratio OAS/Dur1']), reverse=False)
                
        # --- Select Ticker ---
        selected_ticker = st.selectbox("Select a ticker", sorted(df_bonds['TICKER'].unique()))
        df_filtered = df_bonds[df_bonds['TICKER'] == selected_ticker]
        if len(df_filtered) < 2:
            st.warning("Not enough bonds to fit a curve.")
        else:
            x = df_filtered['DURADJMOD'].values
            y = df_filtered['OAS_BP'].values
            params, y_fit = fit_ns_curve(x, y)
        
            if params is not None:
                x_sorted_idx = np.argsort(x)
                x_sorted = x[x_sorted_idx]
                y_fit_sorted = y_fit[x_sorted_idx]
                y_fit_upper = y_fit_sorted * (1 + deviation_threshold / 100)
                y_fit_lower = y_fit_sorted * (1 - deviation_threshold / 100)
        
                # Assign colors
                colors = df_filtered['Own?'].map({'Y': 'red', 'N': 'blue'}).tolist()
        
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_filtered['DURADJMOD'], y=df_filtered['OAS_BP'],
                                         mode='markers', marker=dict(size=8, color=colors),
                                         name='Bonds', customdata=df_filtered[['ID']],
                                         hovertemplate="ID: %{customdata[0]}<br>Dur: %{x}<br>OAS: %{y}<extra></extra>"))
                fig.add_trace(go.Scatter(x=x_sorted, y=y_fit_sorted, mode='lines', name='NS Fit', line=dict(color='black')))
                fig.add_trace(go.Scatter(x=x_sorted, y=y_fit_upper, mode='lines', name='Upper Bound', line=dict(dash='dash')))
                fig.add_trace(go.Scatter(x=x_sorted, y=y_fit_lower, mode='lines', name='Lower Bound', line=dict(dash='dash')))
        
                fig.update_layout(title=f"{selected_ticker} Curve Fit",
                                  xaxis_title="Duration (DURADJMOD)",
                                  yaxis_title="OAS (bps)",
                                  height=600)
        
                st.plotly_chart(fig, use_container_width=True)
        
                # --- Outliers ---
                df_filtered['NS_FIT'] = ns_func(df_filtered['DURADJMOD'], *params)
                df_filtered['Deviation'] = df_filtered['OAS_BP'] - df_filtered['NS_FIT']
                df_filtered['Above/Below'] = np.where(df_filtered['Deviation'] > 0, 'Above', 'Below')
        
                df_below_owned = df_filtered[
                    (df_filtered['Own?'] == 'Y') &
                    (df_filtered['Deviation'] < -deviation_threshold / 100 * df_filtered['NS_FIT'])
                ].copy()
        
                df_above_unowned = df_filtered[
                    (df_filtered['Own?'] == 'N') &
                    (df_filtered['Deviation'] > deviation_threshold / 100 * df_filtered['NS_FIT'])
                ].copy()
        
                # --- Trade Generation ---
                trade_signals = []
                for _, row_below in df_below_owned.iterrows():
                    for _, row_above in df_above_unowned.iterrows():
                        if row_above['DURADJMOD'] > row_below['DURADJMOD']:
                            ratio = (row_above['OAS_BP'] - row_below['OAS_BP']) / (row_above['DURADJMOD'] - row_below['DURADJMOD'])
                            if ratio > 12:
                                trade_signals.append({
                                    "Owned ID": row_below['ID'],
                                    "Matched ID": row_above['ID'],
                                    "Ratio OAS/Dur": round(ratio, 2),
                                    "OAS Diff": round(row_above['OAS_BP'] - row_below['OAS_BP'], 2),
                                    "Dur Diff": round(row_above['DURADJMOD'] - row_below['DURADJMOD'], 2),
                                    "Deviation Owned": round(row_below['Deviation'], 2),
                                    "Deviation Matched": round(row_above['Deviation'], 2),
                                    "Dev Diff": round(row_above['Deviation'] - row_below['Deviation'], 2)
                                })
        
                # --- Tables ---
                st.subheader("Outliers (Owned & Below Line)")
                st.dataframe(df_below_owned[['ID', 'CUSIP', 'Deviation']].reset_index(drop=True), use_container_width=True)
                # Add a download button for the dataframe
        
                st.subheader("Potential Trade Targets")
                if trade_signals:
                    df_trades = pd.DataFrame(trade_signals)
                    st.dataframe(df_trades, use_container_width=True)
                else:
                    st.info("No qualifying trade signals found for this ticker.")
        
                st.subheader("Entire Universe Trade")
                if trade_signals_tot:
                    df_trades_tot = pd.DataFrame(trade_signals_tot)
                    st.dataframe(df_trades_tot, use_container_width=True)
                else:
                    st.info("No qualifying trade signals found for this ticker.")
        




    
    with tab2:
        st.subheader("Curve - High Yield (HY)")
        # --- Deviation Threshold ---
        deviation_threshold = 10
        
        # Get the current credentials
        session = get_active_session()
        
        # --- Fetch Bloomberg Data ---
        #@st.cache_data(show_spinner="Fetching Bloomberg bond data...",ttl=3600)
        @st.cache_data(show_spinner="Fetching Bloomberg bond data...")
        def fetch_bloomberg_data(_session):
            conn = _session.connection
            query = "SELECT * FROM CORPORATE.HIGH_YIELD.DAILY_CURVE_HY_DATA"
            df = pd.read_sql(query, conn)
            return df
        df_bonds = fetch_bloomberg_data(session)
        
        # --- Process Bloomberg Data ---
        df_bonds['TICKER'] = df_bonds['TICKER'].astype(str)
        df_bonds['COUPON'] = df_bonds['COUPON'].astype(str)
        df_bonds['MATURDATE'] = pd.to_datetime(df_bonds['MATURDATE'], format='%Y%m%d').dt.strftime('%m/%d/%Y')
        df_bonds.insert(3, 'ID', df_bonds['TICKER'] + ' ' + df_bonds['COUPON'] + ' ' + df_bonds['MATURDATE'])
        df_bonds['DURADJMOD'] = pd.to_numeric(df_bonds['DURADJMOD'], errors='coerce')
        df_bonds['OAS_BP'] = pd.to_numeric(df_bonds['OAS_BP'], errors='coerce')
        df_bonds = df_bonds.dropna(subset=['DURADJMOD', 'OAS_BP'])
        
        # Remove low-frequency tickers
        #df_bonds = df_bonds[df_bonds['TICKER'].map(df_bonds['TICKER'].value_counts()) >= 5]
        
        # --- Fetch Positions Data ---
        #@st.cache_data(show_spinner="Fetching Position Data...",ttl=3600)
        @st.cache_data(show_spinner="Fetching Position Data...")
        def fetch_positions(_session):
            conn = _session.connection  # Use the connection from the active session
            #query = """SELECT * FROM TRADERS.EAGLE.SELCTION_POSITIONS """
            query = """SELECT * FROM CORPORATE.GENERAL.CORP_CURRENT_POSITIONS"""
            df = pd.read_sql(query, conn)
            return df
        df_positions = fetch_positions(session)
        
        # --- Process Positions ---
        #for this data SHARE_PAR_VALUE used yo be PRICE_SOD
        df_positions.columns = [f"{col}_{i}" if df_positions.columns.duplicated()[i] else col for i, col in enumerate(df_positions.columns)]
        df_positions = df_positions.groupby(['CUSIP', 'TICK'])['SHARE_PAR_VALUE'].sum().reset_index()
        df_positions['SHARE_PAR_VALUE'] = pd.to_numeric(df_positions['SHARE_PAR_VALUE'], errors='coerce')
        #df_positions = df_positions[df_positions['SHARE_PAR_VALUE'] > 2000000]
        
        # --- Merge Ownership Info ---
        df_bonds = df_bonds.merge(df_positions[['CUSIP']], on='CUSIP', how='left', indicator=True)
        df_bonds['Own?'] = df_bonds['_merge'].map({'both': 'Y', 'left_only': 'N', 'right_only': 'N'})
        st.success(f"Fetched bloomberg and positions data.")
        
        # --- Trade Generation for All Tickers ---
        trade_signals_tot = []
        for ticker in df_bonds['TICKER'].unique():
            df_filtered1 = df_bonds[df_bonds['TICKER'] == ticker]
            if len(df_filtered1) < 2:
                continue
            df_below_owned1 = df_filtered1[(df_filtered1['Own?'] == 'Y')].copy()
            df_above_unowned1 = df_filtered1[(df_filtered1['Own?'] == 'N')].copy()
            for _, row_below in df_below_owned1.iterrows():
                for _, row_above in df_filtered1.iterrows():
                    if row_above['DURADJMOD'] > row_below['DURADJMOD']:
                        ratio = (row_above['OAS_BP'] - row_below['OAS_BP']) / (row_above['DURADJMOD'] - row_below['DURADJMOD'])
                        if ratio > 20:
                            trade_signals_tot.append({
                                "Cusip": row_below['CUSIP'],
                                "Owned": row_below['ID'],
                                "Matched": row_above['ID'],
                                "Cusip Matched": row_above['CUSIP'],
                                "Ratio OAS/Dur1": round(ratio, 2),
                                "OAS": round(row_above['OAS_BP'] - row_below['OAS_BP'], 2),
                                "Dur": round(row_above['DURADJMOD'] - row_below['DURADJMOD'], 2),
                            })
            trade_signals_tot = sorted(trade_signals_tot, key=lambda x: (x['Owned'], x['Ratio OAS/Dur1']), reverse=False)
        
        # --- Select Ticker ---
        selected_ticker = st.selectbox("Select a ticker", sorted(df_bonds['TICKER'].unique()))
        df_filtered = df_bonds[df_bonds['TICKER'] == selected_ticker]
        
        if len(df_filtered) < 2:
            st.warning("Not enough bonds to fit a curve.")
        else:
            x = df_filtered['DURADJMOD'].values
            y = df_filtered['OAS_BP'].values
            
            # Assign colors
            colors = df_filtered['Own?'].map({'Y': 'red', 'N': 'blue'}).tolist()
        
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_filtered['DURADJMOD'], y=df_filtered['OAS_BP'],
                                        mode='markers', marker=dict(size=8, color=colors),
                                        name='Bonds', customdata=df_filtered[['ID']],
                                        hovertemplate="ID: %{customdata[0]}<br>Dur: %{x}<br>OAS: %{y}<extra></extra>"))
        
            fig.update_layout(title=f"{selected_ticker} Plot",
                                  xaxis_title="Duration (DURADJMOD)",
                                  yaxis_title="OAS (bps)",
                                  height=600)
            st.plotly_chart(fig, use_container_width=True)
        
            # --- Outliers ---
            df_below_owned = df_filtered[(df_filtered['Own?'] == 'Y')].copy()
            df_above_unowned = df_filtered[(df_filtered['Own?'] == 'N')].copy()
            # --- Trade Generation ---
            trade_signals = []
            for _, row_below in df_below_owned.iterrows():
                for _, row_above in df_above_unowned.iterrows():
                    if row_above['DURADJMOD'] > row_below['DURADJMOD']:
                        ratio = (row_above['OAS_BP'] - row_below['OAS_BP']) / (row_above['DURADJMOD'] - row_below['DURADJMOD'])
                        if ratio > 20:
                            trade_signals.append({
                                "Owned ID": row_below['ID'],
                                "Matched ID": row_above['ID'],
                                "Ratio OAS/Dur": round(ratio, 2),
                                "OAS Diff": round(row_above['OAS_BP'] - row_below['OAS_BP'], 2),
                                "Dur Diff": round(row_above['DURADJMOD'] - row_below['DURADJMOD'], 2),
                            })
            # --- Tables ---
            st.subheader("Owned")
            st.dataframe(df_below_owned[['ID', 'CUSIP']].reset_index(drop=True), use_container_width=True)
        
            st.subheader("Potential Trade Targets")
            if trade_signals:
                df_trades = pd.DataFrame(trade_signals)
                st.dataframe(df_trades, use_container_width=True)
            else:
                st.info("No qualifying trade signals found for this ticker.")
                
        st.subheader("Overall HY Trade Universe")
        if trade_signals_tot:
            df_trades_tot = pd.DataFrame(trade_signals_tot)
            st.dataframe(df_trades_tot, use_container_width=True)
        else:
            st.info("No qualifying trade signals found for this ticker.")
    
# Fundamental Page
elif page == "Fundamental":
    st.title("📊 Fundamental")
    tab1, tab2 = st.tabs(["IG", "HY"])
    with tab1:
        st.subheader("Great things take time.")
        st.image("A realistic image of a bird wearing a construction hard hat and construction attire, standing on a c (1).jpeg")
    with tab2:
        st.subheader("We’re building this with turtle-speed precision")
        st.image("A realistic image of a turtle wearing a construction hard hat and construction attire, standing on a.jpeg")
        #st.subheader("We are known for our speed....")
# Z-Score
elif page == "Z-Score of Peers":
    st.title("📉 Z-Score of Peers")
    tab1, tab2 = st.tabs(["IG", "HY"])
    with tab1:
        st.subheader("The dog union demands more treats before continuing work")

        if st.button("Make it Rain!"):
            treats = ["🦴", "🍖"]
            for _ in range(3):
                col = st.columns(5)
                with col[random.randint(0, 2)]:
                    st.markdown(f"<h1 style='text-align: center;'>{random.choice(treats)}</h1>", unsafe_allow_html=True)
        
        st.image("A realistic image of a golden doodle wearing a construction hard hat and construction attire, standi.jpeg")
    with tab2:
        st.subheader("Ehhhh.... Sorry pal, we are still under construction")
        st.image("A realistic image of a bird wearing a construction hard hat and construction attire, standing on a c.jpeg")

