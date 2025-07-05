import os
import time
import tempfile
import traceback
import psycopg2
import whisperx
import boto3
from dotenv import load_dotenv

# ----------------------------
# 초기화
# ----------------------------
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", "5432")),
}

AWS_SESSION = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

s3 = AWS_SESSION.client("s3")

WHISPER_MODEL_SIZE = "tiny"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

S3_BUCKET = "saeroun-meet"

model = None

# ----------------------------
# 모델 초기화
# ----------------------------
def init_model():
    global model
    if model is None:
        print("[INFO] Loading WhisperX model...")
        model = whisperx.load_model(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )
        print("[INFO] Model loaded.")

# ----------------------------
# 작업 조회 및 상태 선점
# ----------------------------
def get_next_target():
    with psycopg2.connect(**DB_CONFIG) as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id
                FROM ims.meet_recording
                WHERE status = 2
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                rec_id = row[0]
                cur.execute("UPDATE ims.meet_recording SET status = 10 WHERE id = %s", (rec_id,))
                conn.commit()
                return rec_id
            conn.commit()
    return None

# ----------------------------
# 작업 결과 갱신
# ----------------------------
def update_result(rec_id, status, webvtt=None, error=None):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE ims.meet_recording
                SET status = %s,
                    web_vtt = %s,
                    stacktrace = %s
                WHERE id = %s
            """, (status, webvtt, error, rec_id))

# ----------------------------
# 오디오 다운로드
# ----------------------------
def download_audio(rec_id, dest_path):
    s3_key = f"{rec_id}/audio.m4a"
    s3.download_file(S3_BUCKET, s3_key, dest_path)

# ----------------------------
# WhisperX 결과를 VTT로 변환
# ----------------------------
def convert_to_vtt(segments):
    lines = ["WEBVTT\n"]
    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip().replace("\n", " ")
        lines.append(f"{i+1}")
        lines.append(f"{format_time(start)} --> {format_time(end)}")
        lines.append(text + "\n")
    return "\n".join(lines)

def format_time(seconds):
    ms = int((seconds % 1) * 1000)
    s = int(seconds)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

# ----------------------------
# STT 처리
# ----------------------------
def process_recording(rec_id):
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.m4a")
        download_audio(rec_id, audio_path)

        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=1)
        segments = result["segments"]

        vtt = convert_to_vtt(segments)
        return vtt

# ----------------------------
# 데몬 루프
# ----------------------------
def daemon_loop():
    init_model()
    while True:
        rec_id = get_next_target()
        if not rec_id:
            time.sleep(5)
            continue

        try:
            print(f"[INFO] Processing: {rec_id}")
            vtt = process_recording(rec_id)
            update_result(rec_id, 3, webvtt=vtt)
            print(f"[INFO] Success: {rec_id}")
        except Exception as e:
            traceback_str = traceback.format_exc(limit=5)
            update_result(rec_id, -1, error=traceback_str)
            print(f"[ERROR] Failed {rec_id}: {e}")

if __name__ == "__main__":
    daemon_loop()
