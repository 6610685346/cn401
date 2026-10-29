"""Small helpers for the page templates:  {% load ui %}

    {% icon "cctv" %}            an SVG icon from static/img/icons.svg (Lucide)
    {% icon "plus" "i-lg" %}     with an extra CSS class
    {{ name|initial }}           first letter for an avatar circle
"""

from django import template
from django.templatetags.static import static
from django.utils.html import format_html

register = template.Library()


@register.simple_tag
def icon(name: str, extra: str = ""):
    return format_html('<svg class="i {}" aria-hidden="true"><use href="{}#i-{}"></use></svg>',
                       extra, static("img/icons.svg"), name)


@register.filter
def initial(text) -> str:
    text = str(text or "").strip()
    return text[:1].upper() if text else "?"


@register.filter
def pct(value, digits: int = 1) -> str:
    """0.873 -> "87.3%"; None -> "–" (nothing to divide by yet)."""
    if value is None or value == "":
        return "–"
    try:
        return f"{float(value) * 100:.{int(digits)}f}%"
    except (TypeError, ValueError):
        return "–"


@register.filter
def num(value, digits: int = 2) -> str:
    if value is None or value == "":
        return "–"
    try:
        return f"{float(value):.{int(digits)}f}"
    except (TypeError, ValueError):
        return str(value)


@register.filter
def clock(seconds) -> str:
    """Video time 75.4 -> "1:15.4"."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "–"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:04.1f}"
