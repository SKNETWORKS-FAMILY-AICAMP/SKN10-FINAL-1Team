from flask import Flask, request, jsonify
import subprocess
import os
import hmac
import hashlib

app = Flask(__name__)

# GitHub Webhook Secret을 환경 변수에서 가져옵니다.
GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "your_default_secret_token")

# 재시작 스크립트의 절대 경로를 지정합니다.
APP_RESTART_SCRIPT = os.path.join(os.path.dirname(__file__), 'run_fastapi.sh')

@app.route('/webhook', methods=['POST'])
def github_webhook():
    if request.method == 'POST':
        # 서명 확인
        if GITHUB_WEBHOOK_SECRET:
            signature = request.headers.get('X-Hub-Signature-256')
            if not signature:
                return jsonify({"message": "Missing X-Hub-Signature-256 header"}), 400

            payload = request.get_data()
            expected_signature = "sha256=" + hmac.new(
                GITHUB_WEBHOOK_SECRET.encode('utf-8'),
                payload,
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(expected_signature, signature):
                print(f"Invalid signature. Expected: {expected_signature}, Got: {signature}")
                return jsonify({"message": "Invalid signature"}), 403

        # 이벤트 타입 확인 (push만 처리)
        event = request.headers.get('X-GitHub-Event')
        if event != 'push':
            print(f"Received non-push event: {event}")
            return jsonify({"message": f"Event {event} ignored"}), 200

        # dev 브랜치에 대한 푸시인지 확인
        payload_data = request.json
        ref = payload_data.get('ref', '')
        if ref != 'refs/heads/dev':
            print(f"Push to non-dev branch ({ref}) ignored.")
            return jsonify({"message": f"Push to branch {ref} ignored"}), 200

        try:
            print(f"Received webhook for dev branch. Executing {APP_RESTART_SCRIPT}...")
            # 스크립트를 백그라운드에서 실행
            subprocess.Popen([APP_RESTART_SCRIPT],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)

            print("Restart script initiated.")
            return jsonify({"message": "Webhook received, application restart initiated."}), 200
        except Exception as e:
            print(f"Error executing script: {e}")
            return jsonify({"message": f"Error executing script: {e}"}), 500
            
    return jsonify({"message": "Method Not Allowed"}), 405

if __name__ == '__main__':
    print("Starting Webhook Listener on port 5000...")
    # Gunicorn을 사용하는 것이 좋지만, 테스트용으로 직접 실행할 수 있습니다.
    app.run(host='0.0.0.0', port=5000)
