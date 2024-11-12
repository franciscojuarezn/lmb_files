import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
import plotly.express as px

# Set the page configuration
st.set_page_config(page_title="LMB Stats", layout="wide")

# --- Data Loading Functions ---
@st.cache_data
def load_players_data():
    players_data_files = glob.glob(os.path.join('stats_data', 'players_data_*.csv'))
    players_df_list = [pd.read_csv(file) for file in players_data_files]
    return pd.concat(players_df_list, ignore_index=True)

@st.cache_data
def load_standard_stats_batters():
    standard_stats_files = glob.glob(os.path.join('stats_data', 'df_standard_stats_*.csv'))
    standard_stats_df_list = [pd.read_csv(file) for file in standard_stats_files]
    return pd.concat(standard_stats_df_list, ignore_index=True)

@st.cache_data
def load_standard_stats_pitchers():
    standard_stats_files = glob.glob(os.path.join('stats_data_pitchers', 'df_standard_stats_*.csv'))
    standard_stats_df_list = [pd.read_csv(file) for file in standard_stats_files]
    return pd.concat(standard_stats_df_list, ignore_index=True)

@st.cache_data
def load_advanced_stats_batters():
    advanced_stats_files = glob.glob(os.path.join('stats_data', 'df_advanced_stats_*.csv'))
    advanced_stats_df_list = [pd.read_csv(file) for file in advanced_stats_files]
    return pd.concat(advanced_stats_df_list, ignore_index=True)

@st.cache_data
def load_advanced_stats_pitchers():
    advanced_stats_files = glob.glob(os.path.join('stats_data_pitchers', 'df_advanced_stats_*.csv'))
    advanced_stats_df_list = [pd.read_csv(file) for file in advanced_stats_files]
    return pd.concat(advanced_stats_df_list, ignore_index=True)

@st.cache_data
def load_hit_trajectory():
    hit_trajectory_files = glob.glob(os.path.join('stats_data', 'hit_trajectory_lmp_*.csv'))
    hit_trajectory_df_list = [pd.read_csv(file) for file in hit_trajectory_files]
    return pd.concat(hit_trajectory_df_list, ignore_index=True)

@st.cache_data
def load_stadium_data():
    return pd.read_csv(os.path.join('stats_data', 'stadium.csv'))

@st.cache_data
def load_headshots():
    conn = sqlite3.connect(os.path.join('stats_data', 'player_headshots.db'))
    headshots_df = pd.read_sql_query("SELECT playerId, headshot_url FROM player_headshots", conn)
    conn.close()
    return headshots_df

@st.cache_data
def load_batters_df():
    return pd.read_csv("batters_df.csv")

@st.cache_data
def load_pitchers_df():
    pitchers_df_files = glob.glob('pitchers_df*.csv')
    pitchers_df_list = [pd.read_csv(file) for file in pitchers_df_files]
    pitchers_df = pd.concat(pitchers_df_list, ignore_index=True)
    return pitchers_df

@st.cache_data
def load_league_averages_pitchers():
    league_avg_files = glob.glob('stats_data_pitchers/league_avg*.csv')
    league_avg_df_list = [pd.read_csv(file) for file in league_avg_files]
    league_avg_df = pd.concat(league_avg_df_list, ignore_index=True)
    return league_avg_df

@st.cache_data
def load_team_data_batters():
    team_data_std = pd.read_csv('team_data_std.csv')
    team_data_adv = pd.read_csv('team_data_adv.csv')
    return team_data_std, team_data_adv

@st.cache_data
def load_team_data_pitchers():
    team_data_std_files = glob.glob('stats_data_pitchers/team_data_std_p*.csv')
    team_data_std_df_list = [pd.read_csv(file) for file in team_data_std_files]
    team_data_std_df = pd.concat(team_data_std_df_list, ignore_index=True)

    team_data_adv_files = glob.glob('stats_data_pitchers/team_data_adv_p*.csv')
    team_data_adv_df_list = [pd.read_csv(file) for file in team_data_adv_files]
    team_data_adv_df = pd.concat(team_data_adv_df_list, ignore_index=True)

    return team_data_std_df, team_data_adv_df

# Load datasets
players_df = load_players_data()
# players_df_pitchers = load_pitchers_players_data()
standard_stats_df_batters = load_standard_stats_batters()
standard_stats_df_pitchers = load_standard_stats_pitchers()
advanced_stats_df_batters = load_advanced_stats_batters()
advanced_stats_df_pitchers = load_advanced_stats_pitchers()
hit_trajectory_df = load_hit_trajectory()
team_data = load_stadium_data()
headshots_df = load_headshots()
batters_df = load_batters_df()
pitchers_df = load_pitchers_df()
league_avg_df_pitchers = load_league_averages_pitchers()
team_data_std_batters, team_data_adv_batters = load_team_data_batters()
team_data_std_pitchers, team_data_adv_pitchers = load_team_data_pitchers()

# --- Main App ---
# Add a select box to choose between Batters and Pitchers
selected_category = st.selectbox("Select Category", ["Batters", "Pitchers"])

if selected_category == "Batters":
    # --- Batters Section ---
    # --- Calculate League Averages ---
    # Aggregate data for standard stats
    total_hits = team_data_std_batters['H'].sum()
    total_walks = team_data_std_batters['BB'].sum()
    total_hbp = team_data_std_batters['HBP'].sum()
    total_ab = team_data_std_batters['AB'].sum()
    total_sf = team_data_std_batters['SF'].sum()
    total_pa = team_data_std_batters['PA'].sum()
    total_hr = team_data_std_batters['HR'].sum()
    total_2b = team_data_std_batters['2B'].sum()
    total_3b = team_data_std_batters['3B'].sum()
    total_k = team_data_std_batters['K'].sum()

    # Standard Stats Calculations
    league_avg = round(total_hits / total_ab, 3)
    league_obp = round((total_hits + total_walks + total_hbp) / (total_ab + total_walks + total_hbp + total_sf), 3)
    league_slg = round((total_hits - total_2b - total_3b - total_hr + (total_2b * 2) + (total_3b * 3) + (total_hr * 4)) / total_ab, 3)
    league_ops = round(league_obp + league_slg, 3)
    league_babip = round((total_hits - total_hr) / (total_ab - total_k - total_hr + total_sf), 3)

    # Aggregate data for advanced stats
    total_swing_misses = team_data_adv_batters['swingAndMisses'].sum()
    total_pitches = team_data_adv_batters['numP'].sum()
    total_swings = team_data_adv_batters['totalSwings'].sum()
    total_fb = team_data_adv_batters['FO'].sum() + team_data_adv_batters['flyHits'].sum()
    total_bip = team_data_adv_batters['BIP'].sum()
    total_gb = team_data_adv_batters['GO'].sum() + team_data_adv_batters['groundHits'].sum()
    total_ld = team_data_adv_batters['lineOuts'].sum() + team_data_adv_batters['lineHits'].sum()
    total_pop_up = team_data_adv_batters['popOuts'].sum() + team_data_adv_batters['popHits'].sum()
    total_hr_fb = team_data_adv_batters['HR'].sum()

    # Advanced Stats Calculations
    league_k_percent = round((total_k / total_pa) * 100, 1)
    league_bb_percent = round((total_walks / total_pa) * 100, 1)
    league_bb_k = round(total_walks / total_k, 3)
    league_swstr_percent = round((total_swing_misses / total_pitches) * 100, 1)
    league_whiff_percent = round((total_swing_misses / total_swings) * 100, 1)
    league_fb_percent = round((total_fb / total_bip) * 100, 1)
    league_gb_percent = round((total_gb / total_bip) * 100, 1)
    league_ld_percent = round((total_ld / total_bip) * 100, 1)
    league_popup_percent = round((total_pop_up / total_bip) * 100, 1)
    league_hr_fb_percent = round((total_hr_fb / total_fb) * 100, 1)

    # Combined League Averages DataFrame
    league_averages = pd.DataFrame({
        'AVG': [league_avg],
        'OBP': [league_obp],
        'SLG': [league_slg],
        'OPS': [league_ops],
        'BABIP': [league_babip],
        'K%': [league_k_percent],
        'BB%': [league_bb_percent],
        'BB/K': [league_bb_k],
        'SwStr%': [league_swstr_percent],
        'Whiff%': [league_whiff_percent],
        'FB%': [league_fb_percent],
        'GB%': [league_gb_percent],
        'LD%': [league_ld_percent],
        'PopUp%': [league_popup_percent],
        'HR/FB%': [league_hr_fb_percent]
    })

    view_selection = st.radio("", ['Players', "Teams", "Leaderboard"], index=0, horizontal=True)

    if view_selection == "Players":
        players_df = players_df.copy()
        standard_stats_df = standard_stats_df_batters.copy()
        advanced_stats_df = advanced_stats_df_batters.copy()

        headshots_df['playerId'] = headshots_df['playerId'].astype(int)
        players_df = pd.merge(players_df, headshots_df, left_on='id', right_on='playerId', how='left')

        standard_stats_df['player_id'] = standard_stats_df['player_id'].astype(int)
        advanced_stats_df['player_id'] = advanced_stats_df['player_id'].astype(int)

        logo_and_title = """
            <div style="display: flex; align-items: center;">
                <img src="https://images.ctfassets.net/iiozhi00a8lc/5xDuNqUHZdXiLEzzTQnLj6/c9b2c619e6d5f8a527d4a4918bdbacaf/l125.svg" alt="LMP Logo" width="50" height="50">
                <h1 style="margin-left: 10px;">LMB Batting Stats</h1>
            </div>
        """

        st.markdown(logo_and_title, unsafe_allow_html=True)
        st.divider()

        non_pitchers_df = players_df[players_df['POS'] != 'P']
        non_pitchers_df_unique = non_pitchers_df.drop_duplicates(subset=['id'])
        non_pitchers_df_unique = non_pitchers_df_unique.sort_values('fullName')

        default_player = 'Franklin Barreto'
        default_index = next((i for i, name in enumerate(non_pitchers_df_unique['fullName']) if  name == default_player), 0)

        col1, col2 = st.columns([1,3])

        with col1:
            selected_batter = st.selectbox("Select a Batter", non_pitchers_df_unique['fullName'], index=default_index)

        player_data = non_pitchers_df[non_pitchers_df['fullName'] == selected_batter].iloc[0]

        def filter_2024_season_data(player_name, df):
            player_ops_data = df[df['FullName'] == player_name]
            
            # Filter for the 2024 season
            player_ops_data['Date'] = pd.to_datetime(player_ops_data['Date'])
            player_ops_data['season'] = player_ops_data['Date'].dt.year
            player_2024_ops_data = player_ops_data[player_ops_data['season'] == 2024]
            
            return player_2024_ops_data

        # Adjusted plotting function to handle missing 2024 season data
        def plot_player_ops_styled_2024(player_name, player_ops_data, league_avg_ops):
            if player_ops_data.empty:
                st.write("")
                return
            
            # Convert 'Date' column to datetime for proper sorting and plotting
            player_ops_data['Date'] = pd.to_datetime(player_ops_data['Date'])
            
            # Sort by date for accurate plotting
            player_ops_data = player_ops_data.sort_values('Date')
            
            # Plot the player's OPS over time with custom styles
            plt.figure(figsize=(8, 4))
            plt.gca().set_facecolor('beige')   # Set plot background color
            plt.gcf().set_facecolor('beige')   # Set figure background color

            plt.plot(
                player_ops_data['Date'], player_ops_data['OPS'], 
                color='blue',         # Line color
                linestyle='-',        # Solid line
                marker='o',           # Circle markers
                linewidth=2,          # Line width
                label='_nolegend_'    # Exclude from legend
            )
            
            # Add a horizontal line for the league average OPS with custom style
            plt.axhline(
                y=league_avg_ops, 
                color='red',          # Red color
                linestyle='--',       # Dashed line
                linewidth=2,          # Line width
                label=f'League Avg OPS: {league_avg_ops:.3f}'
            )
            
            # Add titles and labels
            plt.title(f'Rolling OPS {player_name}')
            # plt.xlabel('Date')
            # plt.ylabel('OPS')
            plt.legend()
            plt.grid(False)
            
            # Format the x-axis to show only the month and day
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
            
            # Format the y-axis to display 3 decimal places
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Show the plot
            plt.tight_layout()
            st.pyplot(plt)

        # Display player information in three columns
        st.subheader("Player Information")
        col1, col2, col3, col4 = st.columns([.5, .5, .5, .8])

        with col1:
            st.write(f"**Full Name:** {player_data['fullName']}")
            st.write(f"**Position:** {player_data['POS']}")
            st.write(f"**B/T:** {player_data['B/T']}")

        with col2:
            st.write(f"**Birthdate:** {player_data['birthDate']}")
            st.write(f"**Birthplace:** {player_data['Birthplace']}")

        # Filter for 2024 season data
        with col4:
            player_ops_data_2024 = filter_2024_season_data(selected_batter, batters_df)  # Filter data for the selected player in 2024
            league_avg_ops = league_averages['OPS'].iloc[0]  # Use the calculated league average OPS
            plot_player_ops_styled_2024(selected_batter, player_ops_data_2024, league_avg_ops)
        default_image_path = os.path.abspath(os.path.join('stats_data', 'current.jpg'))
        with col3:
            # Check if headshot_url exists and display the image
            if pd.notna(player_data['headshot_url']):
                st.image(player_data['headshot_url'], width=150)
            else:
                # Use the absolute path for the default image
                if os.path.exists(default_image_path):
                    st.image(default_image_path, width=150)
                else:
                    st.write("Default image not found at", default_image_path)
        # --- Standard Stats ---

        # --- Standard Stats ---
        # Filter stats for the selected player (can have multiple rows if player has stats for multiple seasons/teams)
        standard_stats = standard_stats_df[standard_stats_df['player_id'] == player_data['id']]

        # Convert 'season' to integer for proper sorting
        standard_stats.loc[:, 'season'] = standard_stats['season'].astype(int)

        # Select specific columns and order for standard stats
        standard_columns = ['season', 'Name', 'team', 'POS', 'G', 'PA', 'AB', 'H', 'RBI', 'SB', '2B', '3B', 'HR', 'R','TB', 'HBP', 'SF', 'K', 'BB', 'IBB', 'AVG', 'OBP', 'SLG', 'OPS']
        standard_stats_filtered = standard_stats[standard_columns].copy()

        # Sort by season in descending order and by team
        standard_stats_filtered = standard_stats_filtered.sort_values(by=['season', 'team'], ascending=[False, False])

        # Apply formatting to highlight rows where 'team' is '2 Teams'
        def highlight_multiple_teams(row):
            if row['team'] == '2 Teams':
                return ['background-color: #2E2E2E; color: white' for _ in row]
            elif row['team'] == '3 Teams':
                return ['background-color: #2E2E2E; color: white' for _ in row]
            else:
                return ['' for _ in row]

        # Format numeric columns in standard stats to three decimal places
        standard_stats_formatted = standard_stats_filtered.style.format({
            'AVG': '{:.3f}',
            'OBP': '{:.3f}',
            'SLG': '{:.3f}',
            'OPS': '{:.3f}'
        }).apply(highlight_multiple_teams, axis=1)

        # Display Standard Stats table
        st.subheader("Standard Stats", divider='gray')
        st.dataframe(standard_stats_formatted, hide_index=True, use_container_width=True)

        # --- Advanced Stats ---
        # Filter stats for the selected player (can have multiple rows if player has stats for multiple seasons/teams)
        advanced_stats = advanced_stats_df[advanced_stats_df['player_id'] == player_data['id']]

        # Convert 'season' to integer for proper sorting
        advanced_stats.loc[:, 'season'] = advanced_stats['season'].astype(int)

        # Select specific columns and order for advanced stats
        advanced_columns = ['season', 'Name', 'team', 'BABIP', 'K%', 'BB%', 'HR/PA', 'BB/K', 'HR/FB%', 'SwStr%', 'Whiff%', 'FB%', 'GB%', 'LD%', 'PopUp%']
        advanced_stats_filtered = advanced_stats[advanced_columns].copy()

        # Sort by season in descending order and by team
        advanced_stats_filtered = advanced_stats_filtered.sort_values(by=['season', 'team'], ascending=[False, False])

        # Apply formatting to highlight rows where 'team' is '2 Teams'
        def highlight_multiple_teams(row):
            if row['team'] == '2 Teams':
                return ['background-color: #2E2E2E; color: white' for _ in row]
            elif row['team'] == '3 Teams':
                return ['background-color: #2E2E2E; color: white' for _ in row]
            else:
                return ['' for _ in row]
            
        # Format numeric columns in advanced stats to the appropriate decimal places
        advanced_stats_formatted = advanced_stats_filtered.style.format({
            'BABIP': '{:.3f}',
            'K%': '{:.1f}',
            'BB%': '{:.1f}',
            'HR/PA': '{:.3f}',
            'BB/K': '{:.3f}',
            'HR/FB%': '{:.1f}',
            'SwStr%': '{:.1f}',
            'Whiff%': '{:.1f}',
            'FB%': '{:.1f}',
            'GB%': '{:.1f}',
            'LD%': '{:.1f}',
            'PopUp%': '{:.1f}'
        }).apply(highlight_multiple_teams, axis=1)

        # Display Advanced Stats table
        st.subheader("Advanced Stats & Batted Ball", divider='gray')
        st.dataframe(advanced_stats_formatted, hide_index=True, use_container_width=True)

        # Batted Ball Distribution Section
        st.subheader(f"Batted Ball Distribution for {selected_batter}")

        # Create season column from date in hit_trajectory_df
        hit_trajectory_df['date'] = pd.to_datetime(hit_trajectory_df['date'])
        hit_trajectory_df['season'] = hit_trajectory_df['date'].dt.year

        # Get available seasons
        available_seasons = sorted(hit_trajectory_df['season'].unique(), reverse=True)

        col1, col2 =st.columns([1,3])
        with col1:
            selected_season = st.selectbox("Select Season", available_seasons)

        # Filter the hit trajectory data based on the selected season and batter
        filtered_hit_trajectory = hit_trajectory_df[
            (hit_trajectory_df['season'] == selected_season) &
            (hit_trajectory_df['batter_name'] == selected_batter)
        ]

        # Event types
        event_types = ['single', 'double', 'triple', 'home_run', 'out']
        col1, col2 =st.columns([1,2])
        with col1:
            selected_events = st.multiselect("Select Event Types", event_types, default=event_types)

        # All 'outs'
        out_events = ['field_out', 'double_play', 'force_out', 'sac_bunt', 'grounded_into_double_play', 'sac_fly', 'fielders_choice_out', 'field_error', 'sac_fly_double_play']
        filtered_hit_trajectory.loc[:, 'event'] = filtered_hit_trajectory['event'].apply(lambda x: 'out' if x in out_events else x)


        # Define splits for LHP and RHP
        vs_LHP = filtered_hit_trajectory[filtered_hit_trajectory['split_batter'] == 'vs_LHP']
        vs_RHP = filtered_hit_trajectory[filtered_hit_trajectory['split_batter'] == 'vs_RHP']

        # Filter the data for the selected events
        vs_LHP = vs_LHP[vs_LHP['event'].isin(selected_events)]
        vs_RHP = vs_RHP[vs_RHP['event'].isin(selected_events)]

        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)

        # Function to plot the field and hit outcomes
        def plot_field_and_hits(team_data, hit_data, selected_column, palette, plot_title):
            plt.figure(figsize=(8,8))
            y_offset = 275
            excluded_segments = ['outfield_inner']
            
            # Plot the field layout
            for segment_name in team_data['segment'].unique():
                if segment_name not in excluded_segments:
                    segment_data = team_data[team_data['segment'] == segment_name]
                    plt.plot(segment_data['x'], -segment_data['y'] + y_offset, linewidth=4, zorder=1, color='forestgreen', alpha=0.5)

            # Adjust hit coordinates and plot the hits
            hit_data['adj_coordY'] = -hit_data['coordY'] + y_offset
            sns.scatterplot(data=hit_data, x='coordX', y='adj_coordY', hue=selected_column, palette=palette, edgecolor='black', s=100, alpha=0.7)

            plt.text(295, 23, 'Created by: @iamfrankjuarez', fontsize=8, color='grey', alpha=0.3, ha='right')

            plt.title(plot_title, fontsize=15)
            plt.xlabel("")
            plt.ylabel("")
            plt.legend(title=selected_column, title_fontsize='11', fontsize='11', borderpad=1)
            plt.xticks([])
            plt.yticks([])
            plt.xlim(-50, 300)
            plt.ylim(20, 300)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(False)
            st.pyplot(plt)

        # Plot for vs LHP
        with col1:
            if not vs_LHP.empty:
                plot_title = f"Batted Ball Outcomes vs LHP for {selected_batter}"
                plot_field_and_hits(team_data, vs_LHP, 'event', {
                    'single': 'darkorange', 'double': 'purple', 'triple': 'yellow', 'home_run': 'red', 'out': 'grey'
                }, plot_title)
            else:
                st.write("No data available for vs LHP.")

        # Plot for vs RHP
        with col2:
            if not vs_RHP.empty:
                plot_title = f"Batted Ball Outcomes vs RHP for {selected_batter}"
                plot_field_and_hits(team_data, vs_RHP, 'event', {
                    'single': 'darkorange', 'double': 'purple', 'triple': 'yellow', 'home_run': 'red', 'out': 'grey'
                }, plot_title)
            else:
                st.write("No data available for vs RHP.")

    elif view_selection == "Teams":
        # Teams view for League Averages and Team Stats Dashboard
        st.markdown("<h1 style='text-align: center;'>League & Teams</h1>", unsafe_allow_html=True)

        # Display Combined League Averages
        st.subheader("League Averages", divider='gray')
        st.dataframe(league_averages, use_container_width=True, hide_index=True)
        team_abbreviations = {
            'PUE': 'Pericos de Puebla',
            'CAM': 'Piratas de Campeche',
            'AGS': 'Rieleros de Aguascalientes',
            'TIJ': 'Toros de Tijuana',
            'MEX': 'Diablos Rojos del Mexico',
            'LAR': 'Tecos de los Dos Laredos',
            'QRO': 'Conspiradores de Queretaro',
            'MVA': 'Acereros del Norte',
            'LEO': 'Bravos de Leon',
            'TIG': 'Tigres de Quintana Roo',
            'TAB': 'Olmecas de Tabasco',
            'LAG': 'Algodoneros Union Laguna',
            'CHI': 'Dorados de Chihuahua',
            'VER': 'El Aguila de Veracruz',
            'OAX': 'Guerreros de Oaxaca',
            'DUR': 'Caliente de Durango',
            'YUC': 'Leones de Yucatan',
            'SLT': 'Saraperos de Saltillo',
            'JAL': 'Charros de Jalisco',
            'MTY': 'Sultanes de Monterrey'
        }

        # Replace team abbreviations with full team names in both DataFrames
        team_data_std_batters['team'] = team_data_std_batters['team'].replace(team_abbreviations)
        team_data_adv_batters['team'] = team_data_adv_batters['team'].replace(team_abbreviations)

        # Team-Level Stats Dashboard for Standard and Advanced Stats
        st.subheader("Team Standard Stats", divider='gray')
        standard_columns = ['team', 'PA', 'AB', 'H', 'RBI', 'SB', '2B', '3B', 'HR', 'R', 'TB', 'HBP', 'SF', 'K', 'BB', 'IBB', 'AVG', 'OBP', 'SLG', 'OPS','1B', 'G', 'season', 'BABIP']
        team_standard_formatted = team_data_std_batters[standard_columns].drop(columns=['season', 'G', '1B', 'BABIP']).style.format({
            'AVG': '{:.3f}',
            'OBP': '{:.3f}',
            'SLG': '{:.3f}',
            'OPS': '{:.3f}'
        })
        st.dataframe(team_standard_formatted, use_container_width=True, hide_index=True)

        st.subheader("Team Advanced Stats", divider='gray')
        advanced_columns = ['team', 'BABIP', 'K%', 'BB%', 'HR/PA', 'BB/K', 'SwStr%', 'Whiff%', 'FB%', 'GB%', 'LD%', 'PopUp%', 'HR/FB%']
        team_advanced_formatted = team_data_adv_batters[advanced_columns].style.format({
            'BABIP': '{:.3f}', 'K%': '{:.1f}', 'BB%': '{:.1f}', 'HR/PA': '{:.3f}', 'BB/K': '{:.3f}', 'HR/FB%': '{:.1f}', 'SwStr%': '{:.1f}', 'Whiff%': '{:.1f}', 'FB%': '{:.1f}', 'GB%': '{:.1f}', 'LD%': '{:.1f}', 'PopUp%': '{:.1f}'
        })
        st.dataframe(team_advanced_formatted, use_container_width=True, hide_index=True)

    elif view_selection == "Leaderboard":
        # Load the data
        standard_stats_df = standard_stats_df_batters.copy()
        advanced_stats_df = advanced_stats_df_batters.copy()

        # Merge data on player ID and season, with suffixes to avoid conflicts
        merged_df = pd.merge(
            standard_stats_df,
            advanced_stats_df,
            on=['player_id', 'season'],
            how='outer',
            suffixes=('', '_adv')
        )

        # Separate the "2 Teams" entries
        multiple_teams_df = merged_df[merged_df['team'].isin(['2 Teams', '3 Teams'])]
        individual_teams_df = merged_df[~merged_df['team'].isin(['2 Teams', '3 Teams'])]

        # Keep only unique players by dropping individual team entries if "2 Teams" or "3 Teams" exists
        merged_df = pd.concat([multiple_teams_df, individual_teams_df]).drop_duplicates(subset=['player_id', 'season'], keep='first')

        # Convert season to integer for consistent filtering
        merged_df['season'] = merged_df['season'].astype(int)

        # List of columns to display (excluding 'season' for the final display)
        display_columns = [
            'Name', 'team', 'G', 'PA', 'AB', 'H', 'RBI', 'SB', '2B', '3B', 'HR', 'R',
            'TB', 'HBP', 'SF', 'K', 'BB', 'IBB', 'AVG', 'OBP', 'SLG', 'OPS', 'K%', 'BB%', 'BABIP'
        ]

        # Set up Streamlit layout
        st.title("LMB Batting Leaderboard")
        st.divider()

        # Layout: Select Year, Minimum AB, and Qualified Players toggle
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1.5, 1, 1])
        with col1:
            available_years = merged_df['season'].unique()
            selected_year = st.selectbox("Year", sorted(available_years, reverse=True))
        with col3:
            # Filter data based on the selected year
            filtered_df = merged_df[merged_df['season'] == selected_year]
            max_ab = filtered_df['AB'].max()
            min_ab = st.slider("Minimum AB", min_value=0, max_value=int(max_ab), value=0)

        # Calculate the qualified player threshold for the selected year
        max_games = filtered_df['G'].max()
        pa_threshold = int(max_games * 2.72)

        with col5:
            # Toggle for "All Players" and "Qualified Players" using radio buttons
            player_filter = st.radio("Player Filter", ["All Players", "Qualified Players"], horizontal=True)

        # Apply the PA filter if "Qualified Players" is selected
        if player_filter == "Qualified Players":
            filtered_df = filtered_df[filtered_df['PA'] >= pa_threshold]

        # Apply the AB filter and drop the season column
        filtered_df = filtered_df[filtered_df['AB'] >= min_ab]
        filtered_df = filtered_df[display_columns]  # Drop season for display

        # Sort by AVG in descending order
        filtered_df = filtered_df.sort_values(by='AVG', ascending=False)

        # Format columns for specific decimal places
        format_dict = {
            'AVG': '{:.3f}',
            'OBP': '{:.3f}',
            'SLG': '{:.3f}',
            'OPS': '{:.3f}',
            'BABIP': '{:.3f}',
            'K%': '{:.1f}',
            'BB%': '{:.1f}'
        }
        filtered_df = filtered_df.style.format(format_dict)

        # Display filtered leaderboard with 20 players per page
        st.dataframe(filtered_df, height=600, use_container_width=True, hide_index=True)

elif selected_category == "Pitchers":
    # --- Pitchers Section ---
    # Load datasets specific to pitchers
    players_df = load_players_data()
    standard_stats_df = standard_stats_df_pitchers.copy()
    advanced_stats_df = advanced_stats_df_pitchers.copy()
    league_avg_df = league_avg_df_pitchers.copy()
    team_data_std_df = team_data_std_pitchers.copy()
    team_data_adv_df = team_data_adv_pitchers.copy()





    view_selection = st.radio("", ['Players', "Teams", "Leaderboard"], index=0, horizontal=True)

    if view_selection == "Players":
        logo_and_title = """
            <div style="display: flex; align-items: center;">
                <img src="https://images.ctfassets.net/iiozhi00a8lc/5xDuNqUHZdXiLEzzTQnLj6/c9b2c619e6d5f8a527d4a4918bdbacaf/l125.svg" alt="LMP Logo" width="50" height="50">
                <h1 style="margin-left: 10px;">Liga Mexicana de Beisbol Pitching Stats</h1>
            </div>
        """

        # Display the logo and title using st.markdown
        st.markdown(logo_and_title, unsafe_allow_html=True)
        st.divider()
        # Ensure 'playerId' and 'id' are of the same type
        headshots_df['playerId'] = headshots_df['playerId'].astype(int)
        players_df['id'] = players_df['id'].astype(int)
        players_df = pd.merge(players_df, headshots_df, left_on='id', right_on='playerId', how='left')

            # Ensure 'player_id' in stats DataFrames is of type integer
        standard_stats_df['player_id'] = standard_stats_df['player_id'].astype(int)
        advanced_stats_df['player_id'] = advanced_stats_df['player_id'].astype(int)
        # Get team and player info from `standard_stats_df`
        pitchers_with_teams = standard_stats_df[['player_id', 'team', 'Name']].drop_duplicates()

        # Merge the team information with the player dataset
        # Select only 'player_id' and 'team' to avoid column conflicts
        # pitchers_with_teams = pitchers_with_teams[['player_id', 'team', 'Name']]

        # Merge without overlapping columns
        players_with_teams = pd.merge(players_df, pitchers_with_teams, left_on='id', right_on='player_id', how='inner')

        # Filter to exclude non-pitcher positions
        pos_to_ignore = ['OF', 'IF', 'C', 'SS', '2B', '1B', '3B']
        non_pitchers_df = players_with_teams[~players_with_teams['POS'].isin(pos_to_ignore)]
        pitchers_unique = non_pitchers_df.drop_duplicates(subset=['player_id'])

        pitchers_unique['fullName'] = pitchers_unique['fullName'].astype(str)
        pitchers_unique = pitchers_unique.sort_values('fullName')

        # Set Manny Barreda as the default pitcher
        default_pitcher = 'Manny Barreda'
        default_pitcher_index = next((i for i, name in enumerate(pitchers_unique['fullName']) if name == default_pitcher), 0)

        # Add "ALL" to the list of teams
        teams_unique = ["ALL"] + pitchers_unique['team'].unique().tolist()

        # Layout adjustments for pitcher and team selectboxes
        col1, col2, empty_col1, empty_col2 = st.columns([1, 1, 1, 1])  # Adjust the layout

        # Select a team with the "ALL" option
        with col2:
            selected_team = st.selectbox("Filter by Team", teams_unique, index=0)

        # Filter the pitchers based on the selected team or show all pitchers if "ALL" is selected
        if selected_team == "ALL":
            team_pitchers = pitchers_unique
        else:
            team_pitchers = pitchers_unique[pitchers_unique['team'] == selected_team]

        # Update the pitcher selectbox to show only pitchers from the selected team (or all if "ALL" is selected)
        with col1:
            selected_pitcher = st.selectbox("Select a Pitcher", team_pitchers['fullName'].tolist(), index=default_pitcher_index if selected_team == "ALL" else 0)

        # --- K-BB% Plotting Function ---
        def plot_pitcher_kbb_styled(pitcher_name, pitcher_data, league_avg_kbb):
            if pitcher_data.empty:
                st.write(f"No data available for {pitcher_name}.")
                return
            
            # Check if 'Date' column exists and is formatted correctly
            if 'Date' in pitcher_data.columns:
                try:
                    pitcher_data['Date'] = pd.to_datetime(pitcher_data['Date'], errors='coerce')
                    pitcher_data = pitcher_data.dropna(subset=['Date'])  # Remove rows with invalid dates
                    pitcher_data = pitcher_data.sort_values('Date')  # Sort by date for accurate plotting
                except Exception as e:
                    st.write(f"Error in date formatting: {e}")
                    return
            
            plt.figure(figsize=(8.5, 4))
            
            # Set the facecolor of the plot to beige
            plt.gca().set_facecolor('beige')   # Set plot background color
            plt.gcf().set_facecolor('beige')   # Set figure background color
            
            # Plot the pitcher's K-BB% over time (if 'Date' exists) or index otherwise
            if 'Date' in pitcher_data.columns:
                plt.plot(
                    pitcher_data['Date'], pitcher_data['K-BB%'], 
                    color='blue',         # Line color
                    linestyle='-',        # Solid line
                    marker='o',           # Circle markers
                    linewidth=2,          # Line width
                    label='_nolegend_'    # Exclude from legend
                )
            else:
                plt.plot(
                    pitcher_data.index, pitcher_data['K-BB%'], 
                    color='blue',         # Line color
                    linestyle='-',        # Solid line
                    marker='o',           # Circle markers
                    linewidth=2,          # Line width
                    label='_nolegend_'    # Exclude from legend
                )
            
            # Add a horizontal line for the league average K-BB% with custom style
            plt.axhline(
                y=league_avg_kbb, 
                color='red',          # Red color
                linestyle='--',       # Dashed line
                linewidth=2,          # Line width
                label=f'League Avg K-BB%: {league_avg_kbb}%'
            )
            
            # Add titles and labels
            plt.title(f'Rolling K-BB% for {pitcher_name}', fontsize=14)
            # plt.xlabel('Date' if 'Date' in pitcher_data.columns else 'Index', fontsize=12)
            # plt.ylabel('K-BB%', fontsize=12)
            
            # Remove the grid
            plt.grid(False)
            
            # Format the x-axis to show only the month and day if 'Date' exists
            if 'Date' in pitcher_data.columns:
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
            
            # Format the y-axis to display 1 decimal place
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Add the legend to display both the pitcher's K-BB% and league average
            plt.legend(loc='best')

            # Show the plot
            plt.tight_layout()
            st.pyplot(plt)

        # Get player data for the selected pitcher
        if not team_pitchers.empty:
            player_data = team_pitchers[team_pitchers['fullName'] == selected_pitcher].iloc[0]
            
            # Display player information in three columns and the plot in the fourth
            st.subheader("Player Information")
            col1, col2, col3, col4 = st.columns([.5, .5, .5, .8])  # Adjust column widths

            # Player information in col1 and col2
            with col1:
                st.write(f"**Full Name:** {player_data['fullName']}")
                st.write(f"**Position:** {player_data['POS']}")
                st.write(f"**B/T:** {player_data['B/T']}")

            with col2:
                st.write(f"**Birthdate:** {player_data['birthDate']}")
                st.write(f"**Birthplace:** {player_data['Birthplace']}")

            # Display headshot in col3
            with col3:
                if pd.notna(player_data['headshot_url']):
                    st.image(player_data['headshot_url'], width=150)
                else:
                    st.image(os.path.join('stats_data', 'current.jpg'), width=150)

            # Plot K-BB% in col4
            with col4:
                league_avg_kbb = league_avg_df['K-BB%'].iloc[0]  # League average K-BB%
                pitcher_data_filtered = pitchers_df[pitchers_df['FullName'] == selected_pitcher]  # Filter data for the selected pitcher
                plot_pitcher_kbb_styled(selected_pitcher, pitcher_data_filtered, league_avg_kbb)

        # --- Standard Stats ---
        # Filter stats for the selected player
        standard_stats = standard_stats_df[standard_stats_df['player_id'] == player_data['id']]
        standard_stats.loc[:, 'season'] = standard_stats['season'].astype(int)

        standard_columns = ['season', 'Name', 'team', 'POS', 'G', 'GS', 'IP', 'QS','ERA','WHIP', 'W', 'L', 'SV', 'SVOpp','HLD', 'K', 'BB', 'IBB', 'BF', 'H', 'HR', 'ER', 'HBP', 'GIDP', 'WP']
        standard_stats_filtered = standard_stats[standard_columns].copy()

        # Sort by season in descending order and by team
        standard_stats_filtered = standard_stats_filtered.sort_values(by=['season', 'team'], ascending=[False, False])

        # Apply formatting to highlight rows where 'team' is '2 Teams'
        def highlight_multiple_teams(row):
            if row['team'] == '2 Teams':
                return ['background-color: #2E2E2E; color: white' for _ in row]
            elif row['team'] == '3 Teams':
                return ['background-color: #2E2E2E; color: white' for _ in row]
            else:
                return ['' for _ in row]
        # Format numeric columns in standard stats to three decimal places
        standard_stats_formatted = standard_stats_filtered.style.format({
            'IP': '{:.1f}',
            'ERA': '{:.2f}',
            'WHIP': '{:.2f}',
            'OPS': '{:.3f}'
        }).apply(highlight_multiple_teams, axis=1)

        # Display Standard Stats table
        st.subheader("Standard Stats", divider='gray')
        st.dataframe(standard_stats_formatted, hide_index=True, use_container_width=True)

        # --- Advanced Stats ---
        # Load FIP data from the CSV file
        @st.cache_data
        def load_fip_data():
            fip_df = pd.read_csv(os.path.join('stats_data_pitchers', 'FIP_files.csv'))
            fip_df = fip_df.rename(columns={'x_FIPFB': 'xFIP'})
            # Select only relevant columns to reduce memory usage
            return fip_df[['player_id', 'season', 'FIP', 'xFIP', 'team']]
        advanced_stats = advanced_stats_df[advanced_stats_df['player_id'] == player_data['id']]
        advanced_stats.loc[:, 'season'] = advanced_stats['season'].astype(int)

        # Load the FIP data
        fip_df = load_fip_data()
        # Filter stats for the selected player
        advanced_stats = advanced_stats_df[advanced_stats_df['player_id'] == player_data['id']]
        advanced_stats.loc[:, 'season'] = advanced_stats['season'].astype(int)

        advanced_stats_df = advanced_stats_df.merge(fip_df, on=['player_id', 'season','team'], how='left')


        # Separate aggregated data (multiple teams) and individual team data
        aggregated_stats_df = advanced_stats_df[advanced_stats_df['team'].isin(['2 Teams', '3 Teams'])]
        individual_team_stats_df = advanced_stats_df[~advanced_stats_df['team'].isin(['2 Teams', '3 Teams'])]

        # Combine aggregated and individual team stats with duplicates removed
        advanced_stats_df = pd.concat([aggregated_stats_df, individual_team_stats_df]).drop_duplicates(subset=['player_id', 'season', 'team'], keep='first')

        # Filter for the selected player only after deduplication
        advanced_stats = advanced_stats_df[advanced_stats_df['player_id'] == player_data['id']]
        advanced_stats.loc[:, 'season'] = advanced_stats['season'].astype(int)

        # Define the columns to display and format
        advanced_columns = [
            'season', 'Name', 'team', 'POS', 'FIP', 'xFIP', 'BABIP', 'AVG', 'OBP', 'SLG', 'OPS',
            'K%', 'BB%', 'K-BB%', 'K/9', 'BB/9', 'K/BB', 'HR/9', 'HR/FB%'
        ]
        available_columns = [col for col in advanced_columns if col in advanced_stats.columns]
        missing_columns = [col for col in advanced_columns if col not in advanced_stats.columns]

        if missing_columns:
            st.warning(f"The following columns are missing and won't be displayed: {', '.join(missing_columns)}")

        # Filter and sort advanced stats for display
        advanced_stats_filtered = advanced_stats[available_columns].sort_values(by=['season', 'team'], ascending=[False, False])

        # Formatting for display
        format_dict = {
            'BABIP': '{:.3f}', 'K%': '{:.1f}', 'BB%': '{:.1f}', 'K-BB%': '{:.1f}',
            'AVG': '{:.3f}', 'OBP': '{:.3f}', 'SLG': '{:.3f}', 'OPS': '{:.3f}',
            'K/9': '{:.2f}', 'BB/9': '{:.2f}', 'K/BB': '{:.2f}', 'HR/9': '{:.2f}',
            'HR/FB%': '{:.1f}', 'xFIP': '{:.2f}', 'FIP': '{:.2f}'
        }
        advanced_stats_formatted = advanced_stats_filtered.style.format(format_dict).apply(highlight_multiple_teams, axis=1)

        # Display in Streamlit
        st.subheader("Advanced Stats", divider='gray')
        st.dataframe(advanced_stats_formatted, hide_index=True, use_container_width=True)

        # --- Batted Ball Data ---
        batted_ball_data = advanced_stats_df[advanced_stats_df['player_id'] == player_data['id']]

        batted_ball_data.loc[:, 'season'] = batted_ball_data['season'].astype(int)

        batted_columns = ['season', 'Name', 'team', 'POS', 'LD%', 'GB%', 'FB%', 'PopUp%', 'P/IP', 'Str%', 'SwStr%', 'Whiff%', 'CSW%', 'CStr%', 'F-Strike%']
        batted_ball_data_filtered = batted_ball_data[batted_columns].copy()

        # Sort and format batted ball data
        batted_ball_data_filtered = batted_ball_data_filtered.sort_values(by=['season', 'team'], ascending=[False, False])

        batted_ball_formatted = batted_ball_data_filtered.style.format({
            'LD%': '{:.1f}', 'GB%': '{:.1f}', 'FB%': '{:.1f}', 'PopUp%': '{:.1f}',
            'P/IP': '{:.1f}', 'SwStr%': '{:.1f}', 'Whiff%': '{:.1f}', 'Str%': '{:.1f}',
            'CSW%': '{:.1f}', 'CStr%': '{:.1f}', 'F-Strike%': '{:.1f}',
        }).apply(highlight_multiple_teams, axis=1)

        st.subheader("Batted Ball & Plate Discipline", divider='gray')
        st.dataframe(batted_ball_formatted, hide_index=True, use_container_width=True)
        # Batted Ball Distribution Section
        st.subheader(f"Batted Ball Distribution for {selected_pitcher}")

        # Create season column from date in hit_trajectory_df
        hit_trajectory_df['date'] = pd.to_datetime(hit_trajectory_df['date'])
        hit_trajectory_df['season'] = hit_trajectory_df['date'].dt.year

        # Get available seasons
        available_seasons = sorted(hit_trajectory_df['season'].unique(), reverse=True)

        col1, col2 =st.columns([1,3])
        with col1:
            selected_season = st.selectbox("Select Season", available_seasons)

        # Filter the hit trajectory data based on the selected season and batter
        filtered_hit_trajectory = hit_trajectory_df[
            (hit_trajectory_df['season'] == selected_season) &
            (hit_trajectory_df['pitchername'] == selected_pitcher)
        ]

        # Event types
        event_types = ['single', 'double', 'triple', 'home_run', 'out']
        col1, col2 =st.columns([1,2])
        with col1:
            selected_events = st.multiselect("Select Event Types", event_types, default=event_types)

        # All 'outs'
        out_events = ['field_out', 'double_play', 'force_out', 'sac_bunt', 'grounded_into_double_play', 'sac_fly', 'fielders_choice_out', 'field_error', 'sac_fly_double_play']
        filtered_hit_trajectory.loc[:, 'event'] = filtered_hit_trajectory['event'].apply(lambda x: 'out' if x in out_events else x)


        # Define splits for LHP and RHP
        vs_LHP = filtered_hit_trajectory[filtered_hit_trajectory['split_pitcher'] == 'vs_LHB']
        vs_RHP = filtered_hit_trajectory[filtered_hit_trajectory['split_pitcher'] == 'vs_RHB']

        # Filter the data for the selected events
        vs_LHP = vs_LHP[vs_LHP['event'].isin(selected_events)]
        vs_RHP = vs_RHP[vs_RHP['event'].isin(selected_events)]

        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)

        # Function to plot the field and hit outcomes
        def plot_field_and_hits(team_data, hit_data, selected_column, palette, plot_title):
            plt.figure(figsize=(8,8))
            y_offset = 275
            excluded_segments = ['outfield_inner']
            
            # Plot the field layout
            for segment_name in team_data['segment'].unique():
                if segment_name not in excluded_segments:
                    segment_data = team_data[team_data['segment'] == segment_name]
                    plt.plot(segment_data['x'], -segment_data['y'] + y_offset, linewidth=4, zorder=1, color='forestgreen', alpha=0.5)

            # Adjust hit coordinates and plot the hits
            hit_data['adj_coordY'] = -hit_data['coordY'] + y_offset
            sns.scatterplot(data=hit_data, x='coordX', y='adj_coordY', hue=selected_column, palette=palette, edgecolor='black', s=100, alpha=0.7)

            plt.text(295, 23, 'Created by: @iamfrankjuarez', fontsize=8, color='grey', alpha=0.3, ha='right')

            plt.title(plot_title, fontsize=15)
            plt.xlabel("")
            plt.ylabel("")
            plt.legend(title=selected_column, title_fontsize='11', fontsize='11', borderpad=1)
            plt.xticks([])
            plt.yticks([])
            plt.xlim(-50, 300)
            plt.ylim(20, 300)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(False)
            st.pyplot(plt)

        # Plot for vs LHP
        with col1:
            if not vs_LHP.empty:
                plot_title = f"Batted Ball Outcomes vs LHP for {selected_pitcher}"
                plot_field_and_hits(team_data, vs_LHP, 'event', {
                    'single': 'darkorange', 'double': 'purple', 'triple': 'yellow', 'home_run': 'red', 'out': 'grey'
                }, plot_title)
            else:
                st.write("No data available for vs LHP.")

        # Plot for vs RHP
        with col2:
            if not vs_RHP.empty:
                plot_title = f"Batted Ball Outcomes vs RHP for {selected_pitcher}"
                plot_field_and_hits(team_data, vs_RHP, 'event', {
                    'single': 'darkorange', 'double': 'purple', 'triple': 'yellow', 'home_run': 'red', 'out': 'grey'
                }, plot_title)
            else:
                st.write("No data available for vs RHP.")

    elif view_selection == "Teams":
        st.markdown("<h1 style='text-align: center;'>League & Teams</h1>", unsafe_allow_html=True)
        # Display Combined League Averages
        st.subheader("League Averages", divider='gray')

        league_avg_df.insert(2, 'FIP', 5.16)
        league_avg_df.insert(3, 'xFIP', 4.45)
        league_columns = ['ERA', 'WHIP', 'FIP', 'xFIP','K%', 'BB%', 'K-BB%', 'SwStr%', 'Whiff%', 'Str%', 'CSW%', 'CStr%', 'F-Strike%', 'LD%', 'GB%', 'FB%', 'PopUp%', 'HR/FB%', 'BABIP', 'AVG', 'OBP', 'SLG', 'OPS',
                          'K/9', 'BB/9', 'H/9', 'R/9', 'HR/9', 'K/BB']
        league_avg_formatted = league_avg_df[league_columns].style.format({
            'xFIP': '{:.2f}', 'FIP': '{:.2f}',
            'ERA': '{:.2f}', 'WHIP': '{:.2f}',
            'BABIP': '{:.3f}', 'K%': '{:.1f}', 'BB%': '{:.1f}', 'K-BB%': '{:.1f}',
            'AVG': '{:.3f}', 'OBP': '{:.3f}', 'SLG': '{:.3f}', 'OPS': '{:.3f}',
            'K/9': '{:.2f}', 'BB/9': '{:.2f}', 'K/BB': '{:.2f}', 'HR/9': '{:.2f}',
            'H/9': '{:.2f}', 'R/9': '{:.2f}','HR/FB%': '{:.1f}',
            'LD%': '{:.1f}',
            'GB%': '{:.1f}',
            'FB%': '{:.1f}',
            'PopUp%': '{:.1f}',
            'P/IP': '{:.1f}',
            'SwStr%': '{:.1f}',
            'Whiff%': '{:.1f}',
            'Str%': '{:.1f}',
            'CSW%': '{:.1f}',
            'CStr%': '{:.1f}',
            'F-Strike%': '{:.1f}',
        })
        
        st.dataframe(league_avg_formatted, use_container_width=True, hide_index=True)

        team_abbreviations = {
            'PUE': 'Pericos de Puebla',
            'CAM': 'Piratas de Campeche',
            'AGS': 'Rieleros de Aguascalientes',
            'TIJ': 'Toros de Tijuana',
            'MEX': 'Diablos Rojos del Mexico',
            'LAR': 'Tecos de los Dos Laredos',
            'QRO': 'Conspiradores de Queretaro',
            'MVA': 'Acereros del Norte',
            'LEO': 'Bravos de Leon',
            'TIG': 'Tigres de Quintana Roo',
            'TAB': 'Olmecas de Tabasco',
            'LAG': 'Algodoneros Union Laguna',
            'CHI': 'Dorados de Chihuahua',
            'VER': 'El Aguila de Veracruz',
            'OAX': 'Guerreros de Oaxaca',
            'DUR': 'Caliente de Durango',
            'YUC': 'Leones de Yucatan',
            'SLT': 'Saraperos de Saltillo',
            'JAL': 'Charros de Jalisco',
            'MTY': 'Sultanes de Monterrey'
        }
        team_data_std_df['team'] = team_data_std_df['team'].replace(team_abbreviations)
        team_data_adv_df['team'] = team_data_adv_df['team'].replace(team_abbreviations)

        st.subheader("Team Standard Stats", divider='gray')
        standard_columns = ['team', 'W', 'L', 'ERA', 'WHIP', 'GS', 'QS', 'SV', 'SVOpp','BS', 'HLD', 'IP', 'BF', 'AB', 'R', 'ER', 'K', '2B', '3B', 'HR', 'TB', 'BB', 'IBB', 'HBP', 'BK', 'WP', 'SF']
        team_standard_formatted = team_data_std_df[standard_columns].style.format({
            'ERA': '{:.2f}',
            'WHIP': '{:.2f}',
            'IP': '{:.1f}'
        })
        st.dataframe(team_standard_formatted, use_container_width=True, hide_index=True)

        st.subheader("Team Advanced Stats", divider='gray')
        advanced_columns = ['team','K%', 'BB%', 'K-BB%', 'BABIP', 'AVG', 'OBP', 'SLG', 'OPS', 'K/9', 'BB/9', 'K/BB', 'H/9', 'R/9', 'HR/9']
        team_advanced_formatted = team_data_adv_df[advanced_columns].style.format({
            'BABIP': '{:.3f}', 'K%': '{:.1f}', 'BB%': '{:.1f}', 'K-BB%': '{:.1f}',
            'AVG': '{:.3f}', 'OBP': '{:.3f}', 'SLG': '{:.3f}', 'OPS': '{:.3f}',
            'K/9': '{:.2f}', 'BB/9': '{:.2f}', 'K/BB': '{:.2f}', 'HR/9': '{:.2f}',
            'H/9': '{:.2f}', 'R/9': '{:.2f}'        
        })
        st.dataframe(team_advanced_formatted, use_container_width=True, hide_index=True)

        st.subheader("Team Batted Ball & Plate Discipline", divider='gray')
        batted_columns = ['team', 'LD%', 'GB%', 'FB%', 'PopUp%', 'HR/FB%', 'Str%', 'SwStr%', 'Whiff%', 'CSW%', 'CStr%', 'F-Strike%']
        team_batted_formatted = team_data_adv_df[batted_columns].style.format({
            'LD%': '{:.1f}',
            'GB%': '{:.1f}',
            'FB%': '{:.1f}',
            'PopUp%': '{:.1f}',
            'P/IP': '{:.1f}',
            'SwStr%': '{:.1f}',
            'Whiff%': '{:.1f}',
            'Str%': '{:.1f}',
            'CSW%': '{:.1f}',
            'CStr%': '{:.1f}',
            'F-Strike%': '{:.1f}',
            'HR/FB%': '{:.1f}'
        })
        st.dataframe(team_batted_formatted, use_container_width=True, hide_index=True)

    elif view_selection == "Leaderboard":
        # Load the data
        @st.cache_data
        def load_standard_stats():
            standard_stats_files = glob.glob(os.path.join('stats_data_pitchers', 'df_standard_stats_*.csv'))
            standard_stats_df_list = [pd.read_csv(file) for file in standard_stats_files]
            return pd.concat(standard_stats_df_list, ignore_index=True)
        
        @st.cache_data
        def load_advanced_stats():
            advanced_stats_files = glob.glob(os.path.join('stats_data_pitchers', 'df_advanced_stats_*.csv'))
            advanced_stats_df_list = [pd.read_csv(file) for file in advanced_stats_files]
            return pd.concat(advanced_stats_df_list, ignore_index=True)
        
        @st.cache_data
        def load_fip_data():
            fip_df = pd.read_csv(os.path.join('stats_data_pitchers', 'FIP_files.csv'))
            fip_df = fip_df.rename(columns={'x_FIPFB': 'xFIP'})
            return fip_df[['player_id', 'season', 'FIP', 'xFIP']]
        
        @st.cache_data
        def load_data():
            team_data_std_files = glob.glob('stats_data_pitchers/team_data_std_p*.csv')
            team_data_std_df_list = [pd.read_csv(file) for file in team_data_std_files]
            team_data_std_df = pd.concat(team_data_std_df_list, ignore_index=True)

            return team_data_std_df
        
        standard_stats_df = load_standard_stats()
        advanced_stats_df = load_advanced_stats()
        fip_df = load_fip_data()
        team_data_df = load_data()

        advanced_stats_df = advanced_stats_df.merge(fip_df, on=['player_id', 'season'], how='left')
        merged_df = pd.merge(standard_stats_df, advanced_stats_df, on=['player_id', 'season'], how='outer', suffixes=('', '_adv'))
        
        multiple_teams_df = merged_df[merged_df['team'].isin(['2 Teams', '3 Teams'])]
        individual_teams_df = merged_df[~merged_df['team'].isin(['2 Teams', '3 Teams'])]

        # Keep only unique players by dropping individual team entries if "2 Teams" or "3 Teams" exists
        merged_df = pd.concat([multiple_teams_df, individual_teams_df]).drop_duplicates(subset=['player_id', 'season'], keep='first')

        merged_df['season'] = merged_df['season'].astype(int)

        display_columns = ['Name', 'team', 'W', 'L',  'ERA', 'WHIP', 'FIP', 'xFIP', 'G', 'GS', 'IP', 'QS', 'SV', 'SVOpp', 'BS', 'HLD', 'BF', 'R', 'ER', 'K', 'HR', 'BB', 'HBP', 'BK', 'WP']
        st.title("LMB Pitching Leaderboard")
        st.divider()

        col1, col2, col3, col4, col5 = st.columns([1,1,1.5,1,1])

        with col1:
            available_years = merged_df['season'].unique()
            selected_year = st.selectbox("Year", sorted(available_years, reverse=True))
        
        with col3:
            filtered_df = merged_df[merged_df['season'] == selected_year]
            max_ip = filtered_df['IP'].max()
            min_ip = st.slider("Min IP", min_value=0, max_value=int(max_ip), value=0)

        max_games = team_data_df['GS'].max()
        ip_threshold = int(max_games * 0.8)

        with col5:
            player_filter = st.radio("Player Filter", ['All Pitchers', "Qualified Pitchers"], horizontal=True)

        # Apply the filter for Qualified Pitchers if selected
        if player_filter == "Qualified Pitchers":
            filtered_df = filtered_df[filtered_df['IP'] >= ip_threshold]

        # Apply the minimum IP filter
        filtered_df = filtered_df[filtered_df['IP'] >= min_ip]
        # Select columns for the main dashboard
        main_dashboard_df = filtered_df[display_columns]
        main_dashboard_df = main_dashboard_df.sort_values(by='ERA', ascending=True)

        # Format columns in the main dashboard
        format_dict_main = {
            'ERA': '{:.2f}',
            'WHIP': '{:.2f}',
            'IP': '{:.1f}',
            'K-BB%': '{:.2f}',
            'FIP': '{:.2f}',
            'xFIP': '{:.2f}',
        }
        main_dashboard_df = main_dashboard_df.style.format(format_dict_main)
        st.dataframe(main_dashboard_df, height=600, use_container_width=True, hide_index=True)

        # --- Second Dashboard for K%, BB%, and K-BB% ---
        st.subheader("Advanced Stats")
        # Select columns for the second dashboard
        second_dashboard_columns = ['Name', 'team', 'K%', 'BB%', 'K-BB%', 'K/9', 'BB/9', 'GB%', 'FB%', 'LD%', 'SwStr%', 'Whiff%', 'CSW%', 'AVG', 'OBP', 'SLG', 'OPS']
        second_dashboard_df = filtered_df[second_dashboard_columns]
        second_dashboard_df = second_dashboard_df.sort_values(by='K-BB%', ascending=False)

        # Format columns in the second dashboard
        format_dict_second = {
            'K%': '{:.2f}',
            'BB%': '{:.2f}',
            'K-BB%': '{:.2f}',
            'AVG': '{:.3f}', 'OBP': '{:.3f}', 'SLG': '{:.3f}', 'OPS': '{:.3f}',
            'K/9': '{:.2f}', 'BB/9': '{:.2f}', 'K/BB': '{:.2f}', 'HR/9': '{:.2f}',
            'H/9': '{:.2f}', 'R/9': '{:.2f}',
            'LD%': '{:.1f}',
            'GB%': '{:.1f}',
            'FB%': '{:.1f}',
            'PopUp%': '{:.1f}',
            'P/IP': '{:.1f}',
            'SwStr%': '{:.1f}',
            'Whiff%': '{:.1f}',
            'Str%': '{:.1f}',
            'CSW%': '{:.1f}',

        }
        second_dashboard_df = second_dashboard_df.style.format(format_dict_second)
        st.dataframe(second_dashboard_df, height=400, use_container_width=True, hide_index=True)
