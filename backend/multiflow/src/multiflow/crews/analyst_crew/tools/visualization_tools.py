from crewai.tools import BaseTool
from typing import Type, Any, Dict
import json
import uuid

# Assuming chart_utils.py is in the same directory or accessible in PYTHONPATH
# If it's in the same directory, a relative import might work depending on execution context.
# For robustness, especially if running from different directories, ensure analyst_crew.tools is a package.
try:
    from .chart_utils import generate_chart_html
except ImportError:
    # Fallback for cases where the script might be run directly or module structure isn't fully recognized
    from chart_utils import generate_chart_html

class ChartGenerationTool(BaseTool):
    name: str = "Chart Generator"
    description: str = (
        "Generates a self-contained HTML/JavaScript snippet for a Chart.js chart. "
        "Input must be a JSON string with 'chart_type', 'data', and optional 'options'."
    )

    def _run(self, input_json: str) -> str:
        """ 
        Parses the input JSON string, calls generate_chart_html, and returns the HTML string.
        Input JSON structure:
        {
            "chart_type": "bar", 
            "data": {"labels": [...], "datasets": [{...}]}, 
            "options": {...} 
        }
        """
        try:
            params = json.loads(input_json)
            chart_type = params.get('chart_type')
            data = params.get('data')
            options = params.get('options', {})

            if not chart_type or not data:
                return "Error: 'chart_type' and 'data' are required in the input JSON."

            return generate_chart_html(chart_type=chart_type, data=data, options=options)
        except json.JSONDecodeError:
            return "Error: Invalid JSON input."
        except Exception as e:
            return f"Error during chart generation: {str(e)}"

if __name__ == '__main__':
    tool = ChartGenerationTool()
    
    # Example usage for a bar chart
    bar_chart_input = {
        "chart_type": "bar",
        "data": {
            'labels': ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
            'datasets': [{
                'label': '# of Votes',
                'data': [12, 19, 3, 5, 2, 3],
                'borderWidth': 1
            }]
        },
        "options": {
            'scales': {
                'y': {
                    'beginAtZero': True
                }
            }
        }
    }
    print("--- Bar Chart Tool Output ---")
    print(tool.run(json.dumps(bar_chart_input)))

    # Example usage for a pie chart
    pie_chart_input = {
        "chart_type": "pie",
        "data": {
            'labels': ['Work', 'Eat', 'Commute', 'Watch TV', 'Sleep'],
            'datasets': [{
                'label': 'My Daily Activities',
                'data': [8, 2, 2, 4, 8],
            }]
        },
        "options": {
            'responsive': True,
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
    }
    print("\n--- Pie Chart Tool Output ---")
    print(tool.run(json.dumps(pie_chart_input)))
