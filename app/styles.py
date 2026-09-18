import streamlit as st


def apply_health_theme():
  """Injects Apple Health / Samsung Health design system into the Streamlit runtime."""
  st.markdown(
      """
    <style>
        /* Import clean system typography */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        /* Root Canvas & Typography */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            letter-spacing: -0.015em;
        }

        /* Clean canvas & hide default Streamlit chrome */
        #MainMenu, header, footer {visibility: hidden;}
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 4rem !important;
            max-width: 1100px !important;
        }

        /* Apple / One UI Bento Cards */
        .health-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 24px;
            margin-bottom: 20px;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .health-card:hover {
            border-color: rgba(255, 255, 255, 0.16);
        }

        /* Metric Tile Card */
        .metric-tile {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 20px;
            padding: 18px 20px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .metric-label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #8E8E93;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.1;
        }
        .metric-unit {
            font-size: 0.9rem;
            font-weight: 500;
            color: #8E8E93;
            margin-left: 4px;
        }

        /* Health Pill Badges */
        .health-pill {
            display: inline-flex;
            align-items: center;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            width: fit-content;
        }
        .pill-low {
            background: rgba(46, 213, 115, 0.15);
            color: #2ed573;
            border: 1px solid rgba(46, 213, 115, 0.3);
        }
        .pill-moderate {
            background: rgba(255, 171, 0, 0.15);
            color: #ffab00;
            border: 1px solid rgba(255, 171, 0, 0.3);
        }
        .pill-high {
            background: rgba(255, 71, 87, 0.15);
            color: #ff4757;
            border: 1px solid rgba(255, 71, 87, 0.3);
        }

        /* Hero Clinical Header */
        .app-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        .app-title {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #FFFFFF 30%, #8E8E93 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }
        .app-subtitle {
            font-size: 0.88rem;
            color: #8E8E93;
            margin-top: 4px;
        }

        /* Streamlit Native Widget Overrides */
        .stButton > button {
            background: #0071E3 !important;
            color: white !important;
            border: none !important;
            border-radius: 9999px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            background: #0077ED !important;
            transform: scale(1.01);
            box-shadow: 0 4px 14px rgba(0, 113, 227, 0.35) !important;
        }
    </style>
    """,
      unsafe_allow_html=True,
  )