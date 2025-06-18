import json
import uuid

def generate_chart_html(chart_type: str, data: dict, options: dict = None, chart_id: str = None) -> str:
    """
    Generates a self-contained HTML/JavaScript snippet for a Chart.js chart.

    Args:
        chart_type (str): Type of chart (e.g., 'bar', 'line', 'pie').
        data (dict): Data for the chart (labels, datasets).
                     Example: {'labels': ['A', 'B'], 'datasets': [{'label': 'Series 1', 'data': [10, 20]}]}
        options (dict, optional): Chart.js options. Defaults to {}.
        chart_id (str, optional): HTML ID for the canvas element. If None, a random UUID is generated.

    Returns:
        str: HTML string for the chart.
    """
    if chart_id is None:
        chart_id = f"chart-{uuid.uuid4().hex[:8]}"

    if options is None:
        options = {}

    # Ensure data and options are proper JSON strings for embedding in JavaScript
    data_json = json.dumps(data)
    options_json = json.dumps(options)

    html_template = f"""
    <div>
      <canvas id=\"{chart_id}\"></canvas>
    </div>
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
    <script>
      const ctx_{chart_id} = document.getElementById('{chart_id}');
      if (ctx_{chart_id}) {{
        new Chart(ctx_{chart_id}, {{
          type: '{chart_type}',
          data: {data_json},
          options: {options_json}
        }});
      }} else {{
        console.error('Failed to find canvas element with ID: {chart_id}');
      }}
    </script>
    """
    return html_template

if __name__ == '__main__':
    # Example Usage
    sample_bar_data = {
        'labels': ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
        'datasets': [{
            'label': '# of Votes',
            'data': [12, 19, 3, 5, 2, 3],
            'backgroundColor': [
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(153, 102, 255, 0.2)',
                'rgba(255, 159, 64, 0.2)'
            ],
            'borderColor': [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)'
            ],
            'borderWidth': 1
        }]
    }
    sample_bar_options = {
        'scales': {
            'y': {
                'beginAtZero': True
            }
        },
        'responsive': True,
        'maintainAspectRatio': False
    }

    bar_chart_html = generate_chart_html('bar', sample_bar_data, sample_bar_options)
    print("--- Bar Chart HTML ---")
    print(bar_chart_html)

    # Example for a Pie chart
    sample_pie_data = {
        'labels': ['Work', 'Eat', 'Commute', 'Watch TV', 'Sleep'],
        'datasets': [{
            'label': 'My Daily Activities',
            'data': [8, 2, 2, 4, 8],
            'backgroundColor': [
                'rgba(255, 99, 132, 0.8)',
                'rgba(54, 162, 235, 0.8)',
                'rgba(255, 206, 86, 0.8)',
                'rgba(75, 192, 192, 0.8)',
                'rgba(153, 102, 255, 0.8)',
            ]
        }]
    }
    sample_pie_options = {
        'responsive': True,
        'maintainAspectRatio': False,
        'plugins': {
            'legend': {
                'position': 'top',
            },
            'title': {
                'display': True,
                'text': 'My Daily Activities'
            }
        }
    }
    pie_chart_html = generate_chart_html('pie', sample_pie_data, sample_pie_options)
    print("\n--- Pie Chart HTML ---")
    print(pie_chart_html)

    # Save to a test HTML file
    with open("test_charts.html", "w", encoding="utf-8") as f:
        f.write("<h1>Test Charts</h1>")
        f.write(bar_chart_html)
        f.write(pie_chart_html)
    print("\nCharts saved to test_charts.html")
