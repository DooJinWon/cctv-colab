"""Build a deterministic, self-contained Colab notebook from current source."""

from pathlib import Path
import base64
import hashlib
import io
import json
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def source_files():
    for directory in ("cctv_privacy", "tests", "docs", "scripts", ".github"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                yield path
    for name in (
        "README.md",
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        ".gitignore",
    ):
        yield ROOT / name


def source_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_files()):
            entry = zipfile.ZipInfo(
                path.relative_to(ROOT).as_posix(), date_time=(2000, 1, 1, 0, 0, 0)
            )
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, path.read_bytes())
    return buffer.getvalue()


def build():
    cells = []

    def add(kind, source):
        cell = {
            "cell_type": kind,
            "id": f"cell-{len(cells):02d}",
            "metadata": {},
            "source": textwrap.dedent(source).strip() + "\n",
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add(
        "markdown",
        """
    # CCTV Privacy — Colab

    [GitHub source and English documentation](https://github.com/DooJinWon/cctv-colab)

    **녹화 영상 → 모든 얼굴 가림 → 한 명 선택 → 수동 보정 → MP4 저장**

    이 노트북 안에 Python 소스와 테스트가 모두 들어 있습니다. GitHub·API 키가 필요하지 않습니다.
    GPU가 없으면 작은 영상을 CPU로 테스트할 수 있습니다. 모델 추가 학습은 하지 않습니다.

    **실행 순서:** 설치 → 진단/테스트 → 영상 설정·업로드 → 준비 → 분석 → 검수 → 출력.
    **전체 실행으로 건너뛰지 말고, 검수 화면에서 멈춰 수정한 뒤 출력 셀을 실행하세요.**

    자동 탐지는 얼굴을 놓치거나 추적 대상을 혼동할 수 있습니다. 결과 전체를 직접 확인해야 합니다.
    기본 설정은 첫 60초, 모든 얼굴 모자이크, 출력 음성 제외입니다.
    영상은 이 클라우드 런타임에 업로드됩니다. 공개 가능한 짧은 샘플부터 사용하세요.
    """,
    )
    add(
        "markdown",
        """
    ## 1. Install and extract the complete source package

    첫 실행은 패키지 설치 때문에 시간이 걸립니다. GPU 연결은 나중에 해도 됩니다.
    다운로드되는 모델 가중치는 코드 묶음에 포함되지 않습니다.
    """,
    )
    payload = source_zip()
    encoded = base64.b64encode(payload).decode()
    bootstrap = """
    from pathlib import Path
    import base64, hashlib, io, os, subprocess, sys, zipfile

    IN_COLAB = Path('/content').is_dir()
    CODE_ROOT = Path('/content/cctv-colab') if IN_COLAB else Path.cwd() / 'cctv-colab-runtime'
    CODE_ROOT.mkdir(parents=True, exist_ok=True)
    OVERWRITE_MODIFIED_CODE = False
    INSTALL_DEPENDENCIES = True
    SOURCE_ZIP_B64 = '@@PAYLOAD@@'
    payload = base64.b64decode(SOURCE_ZIP_B64)
    assert hashlib.sha256(payload).hexdigest() == '@@HASH@@', 'Source package checksum mismatch'
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            destination = (CODE_ROOT / name).resolve()
            if not destination.is_relative_to(CODE_ROOT.resolve()):
                raise ValueError('Unsafe archive path')
            content = archive.read(name)
            if destination.exists() and destination.read_bytes() != content and not OVERWRITE_MODIFIED_CODE:
                raise FileExistsError(f'Modified source exists: {destination}. Choose another CODE_ROOT.')
        archive.extractall(CODE_ROOT)
    (CODE_ROOT / 'cache').mkdir(exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(CODE_ROOT / 'cache')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['MPLCONFIGDIR'] = str(CODE_ROOT / 'cache')
    if INSTALL_DEPENDENCIES:
        # OpenCV distributions share cv2; retain only the pinned main wheel.
        subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y',
            'opencv-python-headless', 'opencv-contrib-python', 'opencv-contrib-python-headless'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
            '-r', str(CODE_ROOT / 'requirements-dev.txt')])
    import shutil
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        if IN_COLAB:
            subprocess.check_call(['apt-get', '-qq', 'update'])
            subprocess.check_call(['apt-get', '-qq', 'install', '-y', 'ffmpeg'])
        else:
            raise RuntimeError('Install FFmpeg/ffprobe for this Jupyter environment.')
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    if IN_COLAB:
        from google.colab import output
        output.enable_custom_widget_manager()
    print('Source ready:', CODE_ROOT)
    print('Re-run diagnostics after restarting the session if package versions were already loaded.')
    """
    add(
        "code",
        bootstrap.replace("@@PAYLOAD@@", encoded).replace(
            "@@HASH@@", hashlib.sha256(payload).hexdigest()
        ),
    )
    add(
        "markdown",
        "## 2. Runtime diagnostics and offline tests\nGPU가 아직 없어도 CPU로 진행할 수 있습니다.",
    )
    add(
        "code",
        """
    import cv2, torch, platform, importlib.metadata
    print('Python:', platform.python_version())
    print('OpenCV:', cv2.__version__)
    print('Ultralytics:', importlib.metadata.version('ultralytics'))
    print('PyTorch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    else:
        print('CPU mode: use a short test clip; analysis can be slow.')
    print(subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0])
    assert cv2.__version__.startswith('4.11.'), 'Restart the Colab session after installation, then run setup again.'
    RUN_OFFLINE_TESTS = True
    if RUN_OFFLINE_TESTS:
        subprocess.check_call([sys.executable, '-m', 'pytest', '-q', str(CODE_ROOT / 'tests')], cwd=CODE_ROOT)
    """,
    )
    add(
        "markdown",
        """
    ## 3. Input and project settings

    왼쪽 **Files** 패널로 영상을 올리고 `INPUT_VIDEO`를 수정하세요.
    큰 영상은 아래 선택형 업로더 대신 Files 패널이나 본인의 비공개 Drive 경로를 사용하세요.
    Colab에서 개인 Drive를 연결하려면 별도 셀에서 `from google.colab import drive; drive.mount('/content/drive')`를 실행할 수 있습니다.
    Drive 연결은 이 노트북이 자동으로 실행하지 않습니다.

    `END_SECONDS=None`이면 끝까지 처리합니다. 처음에는 30~60초로 테스트하세요.
    `PROJECT`는 영상 구간 하나의 작업 폴더입니다. 다른 영상을 처리할 때 새 이름을 쓰세요.
    """,
    )
    add(
        "code",
        """
    from datetime import datetime
    WORKSPACE = Path('/content/cctv-workspace') if IN_COLAB else Path.cwd() / 'cctv-workspace'
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    INPUT_VIDEO = Path('/content/input.mp4') if IN_COLAB else WORKSPACE / 'input.mp4'
    PROJECT = WORKSPACE / 'projects' / 'demo-001'
    START_SECONDS = 0.0
    END_SECONDS = 60.0  # None = full remaining video
    NORMALIZE_FPS = None  # None = inferred input average FPS; no resolution downscale
    DEVICE = 'auto'  # CUDA when available, otherwise CPU; optionally 'cpu' or 'cuda:0'
    print('Input:', INPUT_VIDEO)
    print('Project:', PROJECT)
    """,
    )
    add(
        "code",
        """
    # Optional small-file uploader. It temporarily holds the upload in memory.
    USE_SMALL_FILE_UPLOADER = False
    if USE_SMALL_FILE_UPLOADER:
        if not IN_COLAB:
            raise RuntimeError('Use the Jupyter file browser outside Colab.')
        from google.colab import files
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError('Upload exactly one video.')
        name, content = next(iter(uploaded.items()))
        INPUT_VIDEO = WORKSPACE / Path(name).name
        INPUT_VIDEO.write_bytes(content)
        del content, uploaded
        print('Input:', INPUT_VIDEO)
    else:
        print('Upload using the Files panel, then verify INPUT_VIDEO above.')
    """,
    )
    add(
        "markdown",
        """
    ## 4. Prepare the working video

    영상 회전·가변 FPS를 정규화하고 선택 구간을 저장합니다. 이후 좌표·프레임 번호는 이 작업 영상 기준입니다.
    기존 프로젝트를 다시 열 때는 준비 단계를 재실행하지 않고 manifest를 읽습니다.
    다른 원본이나 구간을 처리하려면 PROJECT 이름을 바꾸세요.
    """,
    )
    add(
        "code",
        """
    import json
    from cctv_privacy.video import prepare_video
    if (PROJECT / 'manifest.json').exists():
        info = json.loads((PROJECT / 'manifest.json').read_text())
        print('Reusing existing prepared project. Input/trim changes are not applied to this project.')
    else:
        if not INPUT_VIDEO.is_file():
            raise FileNotFoundError(f'Upload a video and set INPUT_VIDEO first: {INPUT_VIDEO}')
        info = prepare_video(INPUT_VIDEO, PROJECT, START_SECONDS, END_SECONDS, NORMALIZE_FPS)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    """,
    )
    add(
        "markdown",
        """
    ## 5. Analyze every prepared frame

    처음 실행하면 모델을 내려받습니다. `FAST_SMOKE_MODE=True`는 설치 확인용으로만 사용하세요.
    회전·분할 탐지를 끄면 작은 얼굴·누운 얼굴을 더 놓칠 수 있습니다.
    분석이 중단됐다면 `RESTART_INCOMPLETE_ANALYSIS=True`로 바꾸고 재실행할 수 있습니다.
    완료된 분석은 보존되며, 추적 상태 때문에 중간 프레임부터 이어서 분석하지 않습니다.
    """,
    )
    add(
        "code",
        """
    from cctv_privacy.pipeline import AnalysisConfig, analyze, reset_incomplete_analysis
    from cctv_privacy.store import Store
    from tqdm.auto import tqdm
    FAST_SMOKE_MODE = False
    RESTART_INCOMPLETE_ANALYSIS = False
    config = AnalysisConfig(device=DEVICE)
    if FAST_SMOKE_MODE:
        config.person_image_size = 640
        config.face_rotations = (0,)
        config.face_tile_size = 0
    database = PROJECT / 'analysis.sqlite'
    already_complete = False
    if database.exists():
        existing = Store(database)
        already_complete = existing.get_meta('complete', False)
        existing.close()
        if not already_complete:
            if not RESTART_INCOMPLETE_ANALYSIS:
                raise RuntimeError('Incomplete analysis exists. Set RESTART_INCOMPLETE_ANALYSIS=True to restart metadata analysis.')
            reset_incomplete_analysis(PROJECT)
    if already_complete:
        print('Reusing completed analysis. Config changes require a new analysis project.')
        summary = json.loads((PROJECT / 'analysis_summary.json').read_text())
    else:
        with tqdm(total=info['frame_count'], desc='Analyze', unit='frame') as bar:
            def analysis_progress(current, total):
                bar.update(current - bar.n)
            summary = analyze(PROJECT, config, progress=analysis_progress)
    print(json.dumps(summary, indent=2))
    """,
    )
    add(
        "markdown",
        """
    ## 6. Review, select one target, and add manual corrections

    **여기서 멈춰 검수하세요.** 프레임 슬라이더와 다음/이전 버튼으로 확인합니다.

    - **한 명 선택:** Target의 사람 구간을 선택 → 시작/끝 프레임 조정 → 같은 대상 확인 체크 → 구간 추가 → Mode 변경.
    - **추적이 끊겼다가 나타난 경우:** 새 구간을 다시 확인해서 추가합니다. 단순 ID만으로 자동 연결하지 않습니다.
    - **수동 가림:** 좌표 격자 체크 → Left/Top/Right/Bottom 조절 → 시작 위치 저장 → 필요하면 끝 위치 저장 → 수동 가림 추가.
    - 수동 가림은 선택 대상 위에도 적용됩니다. 모든 변경은 `edits.json`에 자동 저장됩니다.
    - **원본 표시**는 검수 미리보기에만 적용되고, 출력 영상은 가림 정책으로 생성됩니다.

    자동 검토 이벤트는 누락 전체를 찾아내지 못합니다. 최종 영상 전체도 확인해야 합니다.
    """,
    )
    add(
        "code",
        """
    from cctv_privacy.review import ReviewUI
    if 'review' in globals():
        review.close()
    review = ReviewUI(PROJECT)
    review.show()
    """,
    )
    add(
        "markdown",
        """
    ## 7. Render the reviewed settings

    검수 설정을 적용해 원본 작업 영상에서 다시 생성합니다. 분석 모델은 다시 실행하지 않습니다.
    파일이 생성되었다는 사실은 모든 얼굴이 안전하게 가려졌다는 보장이 아닙니다.
    검수 화면의 음성 옵션이 꺼져 있으면 무음으로 출력합니다.
    """,
    )
    add(
        "code",
        """
    from cctv_privacy.policy import EditPlan
    from cctv_privacy.render import render
    ARTIFACTS = WORKSPACE / 'artifacts'
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    plan = EditPlan.load(PROJECT / 'edits.json') if (PROJECT / 'edits.json').exists() else review.plan
    OUTPUT_VIDEO = ARTIFACTS / f"cctv_{plan.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
    with tqdm(total=info['frame_count'], desc='Render', unit='frame') as bar:
        def render_progress(current, total):
            bar.update(current - bar.n)
        report = render(PROJECT, OUTPUT_VIDEO, plan, progress=render_progress)
    print('Video:', OUTPUT_VIDEO)
    print('Report:', OUTPUT_VIDEO.with_suffix('.report.json'))
    print('Status:', report['review_status'])
    print('Saved frames:', report['frames'])
    """,
    )
    add(
        "markdown",
        """
    ## 8. Download and optional preview

    결과 ZIP에는 처리된 MP4·보고서·편집 설정·검토 이벤트가 들어 있습니다. 원본 작업 영상은 포함하지 않습니다.
    노트북을 Git에 저장하기 전에는 출력 이미지·영상을 모두 지우세요.
    """,
    )
    add(
        "code",
        """
    import zipfile
    result_zip = OUTPUT_VIDEO.with_suffix('.results.zip')
    with zipfile.ZipFile(result_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in [OUTPUT_VIDEO, OUTPUT_VIDEO.with_suffix('.report.json'),
                     PROJECT / 'edits.json', PROJECT / 'analysis_summary.json', PROJECT / 'review_events.csv']:
            if file.exists():
                archive.write(file, arcname=file.name)
    print('Result bundle:', result_zip)
    DOWNLOAD_RESULTS = True
    SHOW_INLINE_VIDEO = False  # Embeds processed video in notebook output; clear it before Git commits.
    if SHOW_INLINE_VIDEO:
        from IPython.display import Video, display
        display(Video(str(OUTPUT_VIDEO), embed=True, width=900))
    if IN_COLAB and DOWNLOAD_RESULTS:
        from google.colab import files
        files.download(str(result_zip))
    """,
    )
    add(
        "markdown",
        """
    ## 9. Optional private project backup

    세션 종료 후 이어서 편집하려면 작업 영상을 포함한 프로젝트 백업이 필요합니다.
    **이 백업에는 가리지 않은 얼굴과 음성이 들어 있는 prepared.mp4가 포함됩니다. 공개 저장소에 올리지 마세요.**
    새 세션에서 비공개 작업 폴더에 압축을 풀고 PROJECT를 해당 폴더로 지정하면 완료된 분석을 다시 사용할 수 있습니다.
    """,
    )
    add(
        "code",
        """
    CREATE_PRIVATE_PROJECT_BACKUP = False
    if CREATE_PRIVATE_PROJECT_BACKUP:
        import sqlite3
        backup = WORKSPACE / f"PRIVATE_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        snapshot = WORKSPACE / 'analysis-backup.sqlite'
        # SQLite backup includes committed WAL data without copying a live database file.
        source_db = sqlite3.connect(str(PROJECT / 'analysis.sqlite'))
        target_db = sqlite3.connect(str(snapshot))
        source_db.backup(target_db)
        target_db.close(); source_db.close()
        with zipfile.ZipFile(backup, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in ['prepared.mp4','manifest.json','edits.json','analysis_summary.json','review_events.csv']:
                file = PROJECT / name
                if file.exists():
                    archive.write(file, arcname=f'project/{name}')
            archive.write(snapshot, arcname='project/analysis.sqlite')
        snapshot.unlink()
        print('Private backup:', backup)
        if IN_COLAB:
            from google.colab import files
            files.download(str(backup))
    else:
        print('Private project backup is disabled.')
    """,
    )
    add(
        "markdown",
        """
    ## Source and Git workflow

    Extracted code: `CODE_ROOT/cctv_privacy/`. English flow, architecture, and validation notes: `CODE_ROOT/README.md` and `CODE_ROOT/docs/`.
    Commit source and a cleared notebook, not videos, model weights, SQLite files, or image outputs.
    After editing source in the repository, run `python scripts/build_notebook.py` to refresh the embedded source package.

    This notebook does not create a public app, start a paid resource, fine-tune a model, or upload files to GitHub.
    """,
    )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": "CCTV_Privacy_Colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "cells": cells,
    }
    output = ROOT / "notebooks/CCTV_Privacy_Colab.ipynb"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Built {output.name}: {len(cells)} cells, {len(payload)} embedded source bytes")


if __name__ == "__main__":
    build()
