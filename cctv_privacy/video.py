"""FFmpeg normalization, bounded-memory decoding, and atomic MP4 output."""

from __future__ import annotations
import json
import hashlib
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
import cv2


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command):
    result = subprocess.run(list(map(str, command)), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("명령 실행 실패:\n" + result.stderr[-5000:])
    return result.stdout


def probe(path, count_frames=False):
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe가 없습니다. 설치 셀을 먼저 실행하세요.")
    command = ["ffprobe", "-v", "error"]
    if count_frames:
        command += ["-count_frames"]
    command += ["-show_streams", "-show_format", "-of", "json", str(path)]
    data = json.loads(run(command))
    video = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("영상 스트림이 없는 파일입니다.")
    rate = video.get("avg_frame_rate", "0/0")
    try:
        fps = Fraction(rate)
        if not 1 <= float(fps) <= 120:
            raise ValueError()
    except (ValueError, ZeroDivisionError):
        rate, fps = "30/1", Fraction(30)
    count = video.get("nb_read_frames") or video.get("nb_frames")
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": float(fps),
        "fps_fraction": str(fps),
        "frame_count": int(count) if count and count != "N/A" else None,
        "duration": float(video.get("duration") or data.get("format", {}).get("duration") or 0),
        "has_audio": any(s.get("codec_type") == "audio" for s in data["streams"]),
    }


def prepare_video(source, project, start_seconds=0.0, end_seconds=60.0, fps=None):
    """Normalize VFR/orientation once; retain normalized audio for optional later use.

    All downstream indices and coordinates refer to prepared.mp4, at native resolution
    with at most a one-pixel pad for the H.264 even-dimension requirement.
    """
    source, project = Path(source).expanduser().resolve(), Path(project).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg가 없습니다. 설치 셀을 먼저 실행하세요.")
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("시작 시각은 0 이상의 유한한 숫자여야 합니다.")
    if end_seconds is not None and (not math.isfinite(end_seconds) or end_seconds <= start_seconds):
        raise ValueError("종료 시각은 시작 시각보다 커야 합니다.")
    info = probe(source)
    if info["duration"] and start_seconds >= info["duration"]:
        raise ValueError("시작 시각이 영상 길이보다 큽니다.")
    rate = Fraction(str(fps)) if fps is not None else Fraction(info["fps_fraction"])
    if not 1 <= float(rate) <= 120:
        raise ValueError("FPS는 1~120이어야 합니다.")
    project.mkdir(parents=True, exist_ok=True)
    output = project / "prepared.mp4"
    manifest = project / "manifest.json"
    if output.exists() or manifest.exists() or (project / "analysis.sqlite").exists():
        raise FileExistsError(
            "기존 프로젝트를 덮어쓰지 않습니다. 새 PROJECT 경로를 사용하거나 기존 프로젝트를 불러오세요."
        )
    partial = project / "prepared.partial.mp4"
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        str(source),
    ]
    if end_seconds is not None:
        command += ["-t", str(end_seconds - start_seconds)]
    command += [
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        f"fps={rate},pad=ceil(iw/2)*2:ceil(ih/2)*2,setsar=1",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-af",
        "aresample=async=1:first_pts=0",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    try:
        run(command)
        prepared = probe(partial, count_frames=True)
        if not prepared["frame_count"]:
            raise ValueError("선택한 구간에서 영상 프레임을 읽지 못했습니다.")
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    data = {
        "schema_version": 1,
        "source_name": source.name,
        "source_start_seconds": start_seconds,
        "prepared_name": output.name,
        **prepared,
    }
    manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def read_frame(video, index):
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"{index}번 프레임을 읽을 수 없습니다.")
        return frame
    finally:
        cap.release()


class Encoder:
    def __init__(self, output, info, audio_source, keep_audio=False, crf=20):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists():
            raise FileExistsError(f"기존 출력 파일입니다: {self.output}. 새 이름을 지정하세요.")
        self.partial = self.output.with_name(self.output.stem + ".partial.mp4")
        self.log_path = self.partial.with_suffix(".ffmpeg.log")
        self.log = self.log_path.open("wb")
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{info['width']}x{info['height']}",
            "-r",
            info["fps_fraction"],
            "-i",
            "pipe:0",
        ]
        if keep_audio and info["has_audio"]:
            command += [
                "-i",
                str(audio_source),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-af",
                "apad",
                "-shortest",
            ]
        else:
            command += ["-map", "0:v:0", "-an"]
        command += [
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(self.partial),
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log
        )
        self.count = 0

    def write(self, frame):
        try:
            self.process.stdin.write(frame.tobytes())
            self.count += 1
        except BrokenPipeError as exc:
            raise RuntimeError("FFmpeg 인코딩 실패. " + str(self.log_path)) from exc

    def finish(self, expected_count):
        self.process.stdin.close()
        code = self.process.wait()
        self.log.close()
        if code or self.count != expected_count:
            error = self.log_path.read_text(errors="replace")[-3000:]
            self.partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"인코딩 실패 또는 프레임 수 불일치 ({self.count}/{expected_count}): {error}"
            )
        result = probe(self.partial, count_frames=True)
        if result["frame_count"] != expected_count:
            self.partial.unlink(missing_ok=True)
            raise RuntimeError("저장된 영상 프레임 수가 다릅니다. 결과 파일을 확정하지 않았습니다.")
        self.partial.replace(self.output)
        self.log_path.unlink(missing_ok=True)
        return result

    def abort(self):
        try:
            self.process.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait()
        if not self.log.closed:
            self.log.close()
        self.partial.unlink(missing_ok=True)
