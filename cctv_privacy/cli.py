from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Offline CCTV face masking pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("source")
    prepare.add_argument("project")
    prepare.add_argument("--start", type=float, default=0)
    prepare.add_argument("--end", type=float, default=60)
    prepare.add_argument("--full", action="store_true")
    analysis = commands.add_parser("analyze")
    analysis.add_argument("project")
    analysis.add_argument("--device", default="auto")
    analysis.add_argument(
        "--fast",
        action="store_true",
        help="One face orientation, no face tiling; for smoke tests only",
    )
    rendering = commands.add_parser("render")
    rendering.add_argument("project")
    rendering.add_argument("output")
    rendering.add_argument("--edits")
    args = parser.parse_args()
    if args.command == "prepare":
        from .video import prepare_video

        result = prepare_video(
            args.source, args.project, args.start, None if args.full else args.end
        )
    elif args.command == "analyze":
        from .pipeline import AnalysisConfig, analyze

        config = AnalysisConfig(device=args.device)
        if args.fast:
            config.face_rotations = (0,)
            config.face_tile_size = 0
            config.person_image_size = 640
        result = analyze(args.project, config)
    else:
        from .policy import EditPlan
        from .render import render

        result = render(
            args.project, args.output, EditPlan.load(args.edits) if args.edits else EditPlan()
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
