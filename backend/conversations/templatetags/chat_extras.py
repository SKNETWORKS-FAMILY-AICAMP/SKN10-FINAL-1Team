import base64
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter(name='encode_chart_data')
def encode_chart_data(chart_html):
    """Encodes the chart HTML content into Base64."""
    if not chart_html:
        return ""
    # Ensure the string is properly encoded to bytes before base64 encoding
    encoded_bytes = base64.b64encode(chart_html.encode('utf-8'))
    return encoded_bytes.decode('utf-8')
