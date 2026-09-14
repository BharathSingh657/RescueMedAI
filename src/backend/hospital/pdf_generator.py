"""
Hospital Pre-Arrival PDF Handoff Report Generator.
Produces formatted, professional emergency pre-arrival PDF reports
adhering to emergency medical command center standards.
"""
import io
import time
from typing import Optional, List, Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)

from src.backend.state import (
    IncidentState,
    HospitalNotification,
    MEDICAL_SAFETY_DISCLAIMER,
    SIMULATED_DATA_NOTICE
)


class HospitalPDFReportGenerator:
    """
    Generates a high-contrast, professional emergency handoff PDF report.
    """

    @staticmethod
    def generate(state: IncidentState, notification: Optional[HospitalNotification] = None) -> bytes:
        if notification is None:
            from src.backend.hospital.notification import HospitalNotificationGenerator
            notification = HospitalNotificationGenerator.generate(state)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        # Custom palette
        COLOR_NAVY = colors.HexColor("#0f172a")
        COLOR_PRIMARY = colors.HexColor("#0284c7")
        COLOR_SLATE = colors.HexColor("#334155")
        COLOR_TEXT = colors.HexColor("#1e293b")
        COLOR_MUTED = colors.HexColor("#64748b")
        COLOR_BG_LIGHT = colors.HexColor("#f8fafc")
        COLOR_BORDER = colors.HexColor("#cbd5e1")

        triage_prio = notification.triage_color.upper() if notification.triage_color else "RED"
        if triage_prio == "RED":
            PRIO_COLOR = colors.HexColor("#dc2626")
            PRIO_BG = colors.HexColor("#fef2f2")
        elif triage_prio == "YELLOW":
            PRIO_COLOR = colors.HexColor("#d97706")
            PRIO_BG = colors.HexColor("#fffbeb")
        else:
            PRIO_COLOR = colors.HexColor("#059669")
            PRIO_BG = colors.HexColor("#ecfdf5")

        # Custom Paragraph Styles
        header_title_style = ParagraphStyle(
            'HeaderTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            textColor=colors.white
        )

        header_sub_style = ParagraphStyle(
            'HeaderSub',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#93c5fd")
        )

        section_heading = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=14,
            textColor=COLOR_NAVY,
            spaceBefore=8,
            spaceAfter=4
        )

        label_style = ParagraphStyle(
            'LabelStyle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=11,
            textColor=COLOR_SLATE
        )

        val_style = ParagraphStyle(
            'ValStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=11,
            textColor=COLOR_TEXT
        )

        badge_style = ParagraphStyle(
            'BadgeStyle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=12,
            textColor=PRIO_COLOR
        )

        small_disclaimer = ParagraphStyle(
            'SmallDisclaimer',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=7.5,
            leading=9.5,
            textColor=COLOR_MUTED
        )

        story = []

        # 1. HEADER BANNER TABLE
        header_text = Paragraph("RESCUEMED AI — EMERGENCY MEDICAL COMMAND", header_title_style)
        header_sub = Paragraph("CRITICAL DISASTER PRE-ARRIVAL HOSPITAL HANDOFF REPORT", header_sub_style)
        case_badge = Paragraph(f"<b>CASE REF:</b> {notification.case_id}<br/><b>TIMESTAMP:</b> {notification.timestamp}", ParagraphStyle('Hb', fontName='Helvetica', fontSize=8, leading=10, textColor=colors.white, alignment=2))

        header_data = [[
            [header_text, Spacer(1, 2), header_sub],
            case_badge
        ]]

        header_table = Table(header_data, colWidths=[380, 160])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLOR_NAVY),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 10))

        # 2. INCIDENT & SITUATION SUMMARY
        story.append(Paragraph("1. INCIDENT & SITUATION OVERVIEW", section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceBefore=1, spaceAfter=6))

        sit_data = [
            [
                Paragraph("<b>Disaster Type:</b>", label_style),
                Paragraph(state.disaster_type, val_style),
                Paragraph("<b>Triage Acuity:</b>", label_style),
                Paragraph(f"<b>{triage_prio} LEVEL</b>", badge_style)
            ],
            [
                Paragraph("<b>Incident Location:</b>", label_style),
                Paragraph(f"{state.location_name} (Node: {state.incident_node})", val_style),
                Paragraph("<b>Destination Facility:</b>", label_style),
                Paragraph(f"{notification.hospital_name} ({notification.hospital_id})", val_style)
            ],
            [
                Paragraph("<b>Disaster Severity:</b>", label_style),
                Paragraph(f"{state.severity.label} (Score: {state.severity.score:.2f}/1.0)" if state.severity else "N/A", val_style),
                Paragraph("<b>Inbound Unit & ETA:</b>", label_style),
                Paragraph(f"<b>{notification.assigned_transport.get('unit', 'AMB-01')}</b> | ETA ~<b>{notification.assigned_transport.get('eta_min', 10)} min</b>", val_style)
            ]
        ]
        sit_table = Table(sit_data, colWidths=[100, 170, 110, 160])
        sit_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_LIGHT),
            ('BOX', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(sit_table)
        story.append(Spacer(1, 10))

        # 3. PATIENT & PHYSIOLOGICAL VITALS
        story.append(Paragraph("2. PATIENT PROFILE & FIELD VITALS", section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceBefore=1, spaceAfter=6))

        pt_summary = notification.patient_vitals_summary or {}
        vitals_data = [
            [
                Paragraph("<b>Age / Gender:</b>", label_style),
                Paragraph(f"{pt_summary.get('age', '30')} yrs / Adult", val_style),
                Paragraph("<b>Blood Pressure:</b>", label_style),
                Paragraph(str(pt_summary.get('bp', '120/80 mmHg')), val_style)
            ],
            [
                Paragraph("<b>SpO2 Saturation:</b>", label_style),
                Paragraph(str(pt_summary.get('spo2', '98%')), val_style),
                Paragraph("<b>Heart Rate:</b>", label_style),
                Paragraph(str(pt_summary.get('heart_rate', '80 bpm')), val_style)
            ],
            [
                Paragraph("<b>Respiratory Rate:</b>", label_style),
                Paragraph(str(pt_summary.get('respiratory_rate', '18/min')), val_style),
                Paragraph("<b>Consciousness (AVPU):</b>", label_style),
                Paragraph(str(pt_summary.get('consciousness', 'Alert')), val_style)
            ],
            [
                Paragraph("<b>Observed Trauma:</b>", label_style),
                Paragraph(str(pt_summary.get('visible_trauma', 'None')), val_style),
                Paragraph("<b>Reported Symptoms:</b>", label_style),
                Paragraph(", ".join(pt_summary.get('symptoms', [])) if pt_summary.get('symptoms') else "None", val_style)
            ]
        ]
        vitals_table = Table(vitals_data, colWidths=[100, 170, 110, 160])
        vitals_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('BOX', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(vitals_table)
        story.append(Spacer(1, 10))

        # 4. AI-ASSISTED ASSESSMENT & DECISION SUPPORT
        story.append(Paragraph("3. AI-ASSISTED ASSESSMENT & CLINICAL TRIAGE DECISION SUPPORT", section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceBefore=1, spaceAfter=6))

        triage_obj = state.triage
        conf_str = f"{triage_obj.confidence_score:.0%}" if triage_obj else "N/A"
        expl_str = triage_obj.explanation if triage_obj else "Rule & Bayesian synthesis completed."

        ai_summary_data = [
            [
                Paragraph("<b>Hybrid Triage Classification:</b>", label_style),
                Paragraph(f"<b>{triage_prio}</b> (Level: {triage_obj.priority_level if triage_obj else 'Immediate'})", val_style),
                Paragraph("<b>Decision Confidence:</b>", label_style),
                Paragraph(conf_str, val_style)
            ],
            [
                Paragraph("<b>Clinical Rationale:</b>", label_style),
                Paragraph(expl_str, val_style),
                Paragraph("<b>Safety Escalation:</b>", label_style),
                Paragraph("Applied (Conservative Override)" if (triage_obj and triage_obj.conservative_escalation_applied) else "Standard Thresholds Met", val_style)
            ]
        ]
        ai_table = Table(ai_summary_data, colWidths=[130, 140, 110, 160])
        ai_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_LIGHT),
            ('BOX', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(ai_table)
        story.append(Spacer(1, 6))

        # Bayesian Table
        if notification.suspected_pathologies:
            story.append(Paragraph("<b>Bayesian Pathological Posterior Probabilities:</b>", label_style))
            bayes_rows = [[Paragraph("<b>Candidate Injury / Condition</b>", label_style), Paragraph("<b>Posterior Probability</b>", label_style)]]
            for item in notification.suspected_pathologies:
                bayes_rows.append([
                    Paragraph(item['condition'], val_style),
                    Paragraph(f"{item['posterior_probability']:.1%}", val_style)
                ])
            bayes_table = Table(bayes_rows, colWidths=[340, 200])
            bayes_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), COLOR_SLATE),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('BOX', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(bayes_table)
            story.append(Spacer(1, 10))

        # 5. RESOURCE ALLOCATION & TRANSPORT TELEMETRY
        story.append(Paragraph("4. RESOURCE ALLOCATION & ROUTING TELEMETRY", section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceBefore=1, spaceAfter=6))

        t_info = notification.assigned_transport
        alloc_data = [
            [
                Paragraph("<b>Dispatched Ambulance:</b>", label_style),
                Paragraph(f"{t_info.get('unit', 'AMB-ALS-01')} ({t_info.get('type', 'ALS')})", val_style),
                Paragraph("<b>Specialist Rescue Team:</b>", label_style),
                Paragraph(str(t_info.get('rescue_team', 'RESCUE-01')), val_style)
            ],
            [
                Paragraph("<b>A* Route Effective Cost:</b>", label_style),
                Paragraph(f"{t_info.get('route_cost', 12.0)} min", val_style),
                Paragraph("<b>Road Navigation Distance:</b>", label_style),
                Paragraph(f"{t_info.get('distance_km', 5.0)} km", val_style)
            ]
        ]
        alloc_table = Table(alloc_data, colWidths=[120, 150, 120, 150])
        alloc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('BOX', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(alloc_table)
        story.append(Spacer(1, 10))

        # 6. FACILITY PREPAREDNESS DIRECTIVES
        story.append(Paragraph("5. RECEIVING FACILITY PREPAREDNESS DIRECTIVES", section_heading))
        story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceBefore=1, spaceAfter=6))

        prep_rows = []
        for p_item in notification.recommended_preparedness:
            prep_rows.append([
                Paragraph("☑", ParagraphStyle('Chk', fontName='Helvetica-Bold', fontSize=10, textColor=COLOR_PRIMARY)),
                Paragraph(p_item, val_style)
            ])
        if prep_rows:
            prep_table = Table(prep_rows, colWidths=[20, 520])
            prep_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(prep_table)
        story.append(Spacer(1, 14))

        # 7. DISCLAIMERS & SIMULATION NOTICES
        story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_BORDER, spaceBefore=6, spaceAfter=6))
        story.append(Paragraph(f"<b>CLINICAL SAFETY DISCLAIMER:</b> {MEDICAL_SAFETY_DISCLAIMER}", small_disclaimer))
        story.append(Spacer(1, 3))
        story.append(Paragraph(f"<b>PROTOTYPE DATA NOTICE:</b> {SIMULATED_DATA_NOTICE}", small_disclaimer))

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
