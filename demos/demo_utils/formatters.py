"""
Result formatters for Constitutional AI demo.

Formats evaluation results, filter results, and comparisons
into user-friendly HTML displays.
"""

from typing import Any, Dict, List


def format_evaluation_result(result: Dict[str, Any]) -> tuple[str, str]:
    """
    Format evaluation result for display.

    Args:
        result: Evaluation result dictionary from evaluator

    Returns:
        Tuple of (status_html, details_html)
    """
    if not result:
        return (
            "<p style='color: gray;'>Enter some text to evaluate...</p>",
            "",
        )

    is_flagged = result.get("flagged", False)
    status_color = "#dc2626" if is_flagged else "#16a34a"  # red or green
    status_text = "⚠️ FLAGGED" if is_flagged else "✅ COMPLIANT"

    status_html = f"""
    <div style='padding: 20px; border-radius: 8px; background-color: {status_color}20;
                border: 2px solid {status_color}; text-align: center;'>
        <h2 style='margin: 0; color: {status_color};'>{status_text}</h2>
    </div>
    """

    # Build details HTML
    details_parts = ["<div style='padding: 10px;'>"]

    # Overall score
    overall_score = result.get("overall_score", 0.0)
    details_parts.append(
        f"<h3>Overall Score: <span style='color: {status_color};'>{overall_score:.2f}</span></h3>"
    )

    # Get principle evaluations
    direct_eval = result.get("direct_evaluation", {})
    principle_scores = direct_eval.get("principle_scores", {})
    flagged_principles = direct_eval.get("flagged_principles", [])

    if principle_scores:
        details_parts.append("<h3>Principle Scores:</h3>")
        details_parts.append("<table style='width: 100%; border-collapse: collapse;'>")
        details_parts.append(
            "<tr style='background-color: #f3f4f6;'>"
            "<th style='padding: 8px; text-align: left;'>Principle</th>"
            "<th style='padding: 8px; text-align: center;'>Score</th>"
            "<th style='padding: 8px; text-align: center;'>Status</th>"
            "</tr>"
        )

        for principle, score in principle_scores.items():
            is_violation = principle in flagged_principles
            status_icon = "❌" if is_violation else "✅"
            row_color = "#fee2e2" if is_violation else "#ffffff"

            principle_display = principle.replace("_", " ").title()

            details_parts.append(
                f"<tr style='background-color: {row_color};'>"
                f"<td style='padding: 8px;'>{principle_display}</td>"
                f"<td style='padding: 8px; text-align: center;'>{score:.2f}</td>"
                f"<td style='padding: 8px; text-align: center;'>{status_icon}</td>"
                f"</tr>"
            )

        details_parts.append("</table>")

    # Flagged details
    if flagged_principles:
        details_parts.append("<h3 style='color: #dc2626;'>Violations Detected:</h3>")
        details_parts.append("<ul>")
        for principle in flagged_principles:
            principle_display = principle.replace("_", " ").title()
            details_parts.append(f"<li><strong>{principle_display}</strong></li>")
        details_parts.append("</ul>")

    details_parts.append("</div>")
    details_html = "".join(details_parts)

    return status_html, details_html


def format_filter_result(filtered_text: str, filter_info: Dict[str, Any]) -> str:
    """
    Format filter result for display.

    Args:
        filtered_text: The filtered text
        filter_info: Filter information dictionary

    Returns:
        HTML string with filter information
    """
    if not filter_info:
        return "<p style='color: gray;'>Enter some text to filter...</p>"

    info_parts = ["<div style='padding: 10px;'>"]

    was_filtered = filter_info.get("filtered", False)

    if was_filtered:
        info_parts.append("<h3 style='color: #dc2626;'>⚠️ Content was filtered</h3>")

        violations = filter_info.get("violations", [])
        if violations:
            info_parts.append("<h4>Violations detected:</h4>")
            info_parts.append("<ul>")
            for violation in violations:
                info_parts.append(f"<li>{violation.replace('_', ' ').title()}</li>")
            info_parts.append("</ul>")
    else:
        info_parts.append("<h3 style='color: #16a34a;'>✅ No filtering needed</h3>")
        info_parts.append("<p>The text meets all constitutional principles.</p>")

    info_parts.append("</div>")
    return "".join(info_parts)


def format_comparison_table(original: str, filtered: str) -> str:
    """
    Format side-by-side comparison of original and filtered text.

    Args:
        original: Original text
        filtered: Filtered text

    Returns:
        HTML table showing comparison
    """
    if not original or not filtered:
        return "<p style='color: gray;'>Apply filtering to see comparison...</p>"

    changes_detected = original != filtered

    html = f"""
    <div style='padding: 10px;'>
        <h3>Text Comparison</h3>
        <table style='width: 100%; border-collapse: collapse;'>
            <tr style='background-color: #f3f4f6;'>
                <th style='padding: 8px; text-align: left; width: 50%;'>Original</th>
                <th style='padding: 8px; text-align: left; width: 50%;'>Filtered</th>
            </tr>
            <tr>
                <td style='padding: 8px; vertical-align: top; border-right: 1px solid #e5e7eb;'>
                    {original}
                </td>
                <td style='padding: 8px; vertical-align: top;'>
                    {filtered}
                </td>
            </tr>
        </table>
        <p style='margin-top: 10px; font-style: italic;'>
            {"⚠️ Changes detected" if changes_detected else "✅ No changes made"}
        </p>
    </div>
    """

    return html


def format_comparison_result(
    baseline_eval: Dict[str, Any], constitutional_eval: Dict[str, Any]
) -> str:
    """
    Format before/after comparison result.

    Args:
        baseline_eval: Baseline model evaluation
        constitutional_eval: Constitutional model evaluation

    Returns:
        HTML string with comparison
    """
    html_parts = ["<div style='padding: 10px;'>"]
    html_parts.append("<h2>Before vs. After Comparison</h2>")

    # Overall scores
    baseline_score = baseline_eval.get("overall_score", 0.0)
    const_score = constitutional_eval.get("overall_score", 0.0)
    improvement = const_score - baseline_score

    improvement_color = "#16a34a" if improvement > 0 else "#dc2626"
    improvement_sign = "+" if improvement > 0 else ""

    html_parts.append(
        f"""
        <div style='margin: 20px 0; padding: 15px; background-color: {improvement_color}20;
                    border-radius: 8px; border: 2px solid {improvement_color};'>
            <h3 style='margin: 0; color: {improvement_color};'>
                Improvement: {improvement_sign}{improvement:.2f} ({improvement_sign}{improvement*100:.1f}%)
            </h3>
            <p style='margin: 5px 0 0 0;'>
                Baseline: {baseline_score:.2f} → Constitutional: {const_score:.2f}
            </p>
        </div>
        """
    )

    # Principle-by-principle comparison
    baseline_scores = baseline_eval.get("direct_evaluation", {}).get("principle_scores", {})
    const_scores = constitutional_eval.get("direct_evaluation", {}).get("principle_scores", {})

    if baseline_scores and const_scores:
        html_parts.append("<h3>Principle Scores:</h3>")
        html_parts.append("<table style='width: 100%; border-collapse: collapse;'>")
        html_parts.append(
            "<tr style='background-color: #f3f4f6;'>"
            "<th style='padding: 8px; text-align: left;'>Principle</th>"
            "<th style='padding: 8px; text-align: center;'>Baseline</th>"
            "<th style='padding: 8px; text-align: center;'>Constitutional</th>"
            "<th style='padding: 8px; text-align: center;'>Change</th>"
            "</tr>"
        )

        for principle in baseline_scores:
            baseline = baseline_scores.get(principle, 0.0)
            const = const_scores.get(principle, 0.0)
            change = const - baseline
            change_sign = "+" if change > 0 else ""
            change_color = "#16a34a" if change > 0 else ("#dc2626" if change < 0 else "#6b7280")

            principle_display = principle.replace("_", " ").title()

            html_parts.append(
                f"<tr>"
                f"<td style='padding: 8px;'>{principle_display}</td>"
                f"<td style='padding: 8px; text-align: center;'>{baseline:.2f}</td>"
                f"<td style='padding: 8px; text-align: center;'>{const:.2f}</td>"
                f"<td style='padding: 8px; text-align: center; color: {change_color};'>"
                f"{change_sign}{change:.2f}</td>"
                f"</tr>"
            )

        html_parts.append("</table>")

    html_parts.append("</div>")
    return "".join(html_parts)


def format_model_info(model_info: Dict[str, Any]) -> str:
    """
    Format model status information.

    Args:
        model_info: Model status dictionary

    Returns:
        HTML string with model information
    """
    status = model_info.get("status", "not_loaded")
    model_name = model_info.get("model_name", "None")
    model_type = model_info.get("model_type", "N/A")
    device = model_info.get("device", "N/A")
    error = model_info.get("error")

    if status == "ready":
        color = "#16a34a"
        icon = "✅"
        status_text = "Ready"
    elif status == "loading":
        color = "#f59e0b"
        icon = "⏳"
        status_text = "Loading..."
    elif status == "training":
        color = "#3b82f6"
        icon = "🔄"
        status_text = "Training..."
    elif status == "error":
        color = "#dc2626"
        icon = "❌"
        status_text = "Error"
    else:
        color = "#6b7280"
        icon = "⚪"
        status_text = "Not Loaded"

    html = f"""
    <div style='padding: 15px; background-color: {color}20; border-radius: 8px;
                border: 2px solid {color};'>
        <h3 style='margin: 0 0 10px 0; color: {color};'>{icon} {status_text}</h3>
        <p style='margin: 5px 0;'><strong>Model:</strong> {model_name or 'None'}</p>
        <p style='margin: 5px 0;'><strong>Type:</strong> {model_type or 'N/A'}</p>
        <p style='margin: 5px 0;'><strong>Device:</strong> {device}</p>
        {f"<p style='margin: 5px 0; color: {color};'><strong>Error:</strong> {error}</p>" if error else ""}
    </div>
    """

    return html
