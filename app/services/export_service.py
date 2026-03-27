"""
Export service for generating reports and exports.
"""
import io
import json
import zipfile
from datetime import datetime
from html import escape
from typing import Iterable, List

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

class ExportService:
    """Service for exporting prediction records in several formats."""

    def _format_datetime(self, value):
        if value is None:
            return "N/A"
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return str(value)

    def _format_processing_time(self, value):
        if value in (None, ""):
            return "N/A"
        try:
            return f"{float(value):.0f} ms"
        except (TypeError, ValueError):
            return str(value)

    def _sanitize_filename(self, value, fallback="prediction"):
        cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(value or fallback))
        cleaned = cleaned.strip("_")
        return cleaned or fallback

    def _build_prediction_rows(self, predictions: Iterable):
        rows = []
        for pred in predictions:
            row = {
                "ID": pred.id,
                "Image Name": pred.image_name,
                "Prediction": pred.prediction,
                "Confidence": pred.confidence,
                "Processing Time (ms)": pred.processing_time,
                "Date": self._format_datetime(pred.created_at),
            }

            for prob in pred.get_probabilities():
                row[f'Probability_{prob.get("class", "Unknown")}'] = prob.get("probability", 0)

            rows.append(row)
        return rows

    def generate_pdf(self, prediction):
        """Generate a PDF report for a single prediction."""
        if not HAS_REPORTLAB:
            raise RuntimeError("PDF export is unavailable because reportlab is not installed")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#2ecc71"),
            alignment=TA_CENTER,
            spaceAfter=30,
        )

        story.append(Paragraph("Plant Disease Classification Report", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        story.append(Spacer(1, 20))

        details = [
            ["Field", "Value"],
            ["Image", escape(str(prediction.image_name))],
            ["Prediction", escape(str(prediction.prediction))],
            ["Confidence", f"{prediction.confidence:.2%}"],
            ["Processing Time", self._format_processing_time(prediction.processing_time)],
            ["Date", self._format_datetime(prediction.created_at)],
        ]

        details_table = Table(details, colWidths=[2 * inch, 3 * inch])
        details_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(details_table)
        story.append(Spacer(1, 20))

        probability_rows = [["Class", "Probability"]]
        for prob in prediction.get_probabilities():
            probability_rows.append(
                [escape(str(prob.get("class", "Unknown"))), f"{float(prob.get('probability', 0)):.2%}"]
            )

        probability_table = Table(probability_rows, colWidths=[3 * inch, 2 * inch])
        probability_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(probability_table)
        story.append(Spacer(1, 20))

        story.append(Paragraph("Recommendations:", styles["Heading2"]))
        for recommendation in self.get_recommendations(prediction.prediction):
            story.append(Paragraph(f"â€¢ {escape(recommendation)}", styles["Normal"]))

        doc.build(story)
        buffer.seek(0)
        return buffer

    def generate_pdf_bundle(self, predictions: List):
        """Generate a ZIP bundle containing one PDF per prediction."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for prediction in predictions:
                filename = self._sanitize_filename(f"prediction_{prediction.id}_{prediction.prediction}")
                archive.writestr(f"{filename}.pdf", self.generate_pdf(prediction).getvalue())

            manifest = {
                "generated_at": datetime.now().isoformat(),
                "total_predictions": len(predictions),
                "prediction_ids": [prediction.id for prediction in predictions],
            }
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))

        buffer.seek(0)
        return buffer

    def generate_csv(self, predictions):
        """Generate CSV export for one or more predictions."""
        df = pd.DataFrame(self._build_prediction_rows(predictions))
        text_buffer = io.StringIO()
        df.to_csv(text_buffer, index=False)
        buffer = io.BytesIO(text_buffer.getvalue().encode("utf-8"))
        buffer.seek(0)
        return buffer

    def generate_excel(self, predictions):
        """Generate Excel export with summary and prediction sheets."""
        prediction_list = list(predictions)
        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            summary_data = {
                "Metric": ["Total Predictions", "Average Confidence", "Date Range"],
                "Value": [
                    len(prediction_list),
                    sum(p.confidence for p in prediction_list) / len(prediction_list) if prediction_list else 0,
                    (
                        f"{self._format_datetime(min(p.created_at for p in prediction_list))[:10]} - "
                        f"{self._format_datetime(max(p.created_at for p in prediction_list))[:10]}"
                    )
                    if prediction_list
                    else "N/A",
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame(self._build_prediction_rows(prediction_list)).to_excel(
                writer, sheet_name="Predictions", index=False
            )

        buffer.seek(0)
        return buffer

    def get_recommendations(self, disease):
        """Get recommendations based on disease."""
        normalized = str(disease or "").lower()

        if "healthy" in normalized:
            return [
                "No dominant disease signal detected; continue routine scouting",
                "Keep periodic reference images to track field changes",
                "Maintain balanced irrigation and nutrition practices",
            ]

        if "rust" in normalized:
            return [
                "Inspect both leaf surfaces to verify rust pressure",
                "Track neighboring plants to evaluate spread dynamics",
                "Apply crop-specific integrated rust management guidance",
            ]

        if "blight" in normalized:
            return [
                "Review lesion progression across nearby plants",
                "Adjust canopy and moisture conditions where feasible",
                "Follow local extension guidance before intervention",
            ]

        if any(token in normalized for token in ("spot", "mold", "lesion")):
            return [
                "Increase monitoring frequency on affected blocks",
                "Reduce prolonged leaf wetness risk factors",
                "Validate with additional images before treatment decisions",
            ]

        return [
            "Consult local agricultural extension officer",
            "Monitor field regularly",
            "Maintain good agricultural practices",
        ]
