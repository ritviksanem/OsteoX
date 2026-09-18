import streamlit as st


def render_header(patient_name: str | None = None, patient_id: str | None = None):
  """Renders the top navigation and active patient status pill."""
  patient_tag = (
      f'<div class="health-pill pill-low">👤 {patient_name} ({patient_id})</div>'
      if patient_name
      else '<div class="health-pill" style="background: rgba(255,255,255,0.08); color: #8E8E93;">No Active Patient</div>'
  )

  st.markdown(
      f"""
    <div class="app-header">
        <div>
            <h1 class="app-title">OsteoX Health</h1>
            <div class="app-subtitle">Intelligent Musculoskeletal & Kinematic Triage</div>
        </div>
        <div>
            {patient_tag}
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )


def render_metric_tile(title: str, value: str, unit: str = "", delta: str = ""):
  """Renders a Samsung Health / Apple Health style metric squircle card."""
  delta_html = (
      f'<div style="font-size: 0.78rem; color: #8E8E93; margin-top: 4px;">{delta}</div>'
      if delta
      else ""
  )
  st.markdown(
      f"""
    <div class="metric-tile">
        <div class="metric-label">{title}</div>
        <div>
            <span class="metric-value">{value}</span>
            <span class="metric-unit">{unit}</span>
        </div>
        {delta_html}
    </div>
    """,
      unsafe_allow_html=True,
  )


def render_risk_card(risk_level: str, confidence: float, recommendation: str):
  """Renders the primary clinical diagnosis summary card."""
  pill_class = {
      "LOW": "pill-low",
      "MODERATE": "pill-moderate",
      "HIGH": "pill-high",
  }.get(risk_level, "pill-low")

  st.markdown(
      f"""
    <div class="health-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <span style="font-size: 0.85rem; font-weight: 700; color: #8E8E93; text-transform: uppercase;">Screening Assessment</span>
            <span class="health-pill {pill_class}">{risk_level} RISK</span>
        </div>
        <div style="font-size: 1.7rem; font-weight: 800; margin-bottom: 8px;">
            {confidence:.1f}% Match Confidence
        </div>
        <div style="font-size: 0.95rem; color: #D1D1D6; line-height: 1.5;">
            {recommendation}
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )