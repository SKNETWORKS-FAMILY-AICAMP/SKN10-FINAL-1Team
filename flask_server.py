from flask import Flask, send_from_directory
import os

app = Flask(__name__)

# HTML 파일들이 들어있는 폴더 (product/)
HTML_FOLDER = os.path.join(os.getcwd(), "product")

@app.route("/product/<path:filename>")
def serve_html(filename):
    return send_from_directory(HTML_FOLDER, filename)

if __name__ == "__main__":
    app.run(port=8800, debug=True)
