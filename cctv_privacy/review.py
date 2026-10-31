"""Private, notebook-native review controls: no public web-server tunnel."""

from __future__ import annotations
import html
import json
from pathlib import Path
import cv2
from .policy import EditPlan, mask_boxes
from .render import redact, validate_project_plan
from .store import Store
from .video import read_frame


class ReviewUI:
    def __init__(self, project):
        import ipywidgets as w

        self.w = w
        self.project = Path(project).resolve()
        self.store = Store(self.project / "analysis.sqlite")
        self.info = self.store.get_meta("manifest")
        if not self.store.get_meta("complete"):
            self.store.close()
            raise RuntimeError("Complete the analysis before opening the review screen.")
        self.plan_path = self.project / "edits.json"
        self.plan = (
            EditPlan.load(self.plan_path)
            if self.plan_path.exists()
            else EditPlan(analysis_id=self.store.get_meta("analysis_id"))
        )
        validate_project_plan(self.store, self.plan)
        self._segments = {s["id"]: s for s in self.store.segments()}
        self.start_box = None
        self.end_box = None
        self._frame_cache = None
        layout = w.Layout(width="auto")
        self.frame = w.IntSlider(
            value=0,
            min=0,
            max=self.info["frame_count"] - 1,
            description="Frame",
            continuous_update=False,
            layout=w.Layout(width="95%"),
        )
        self.prev = w.Button(description="◀ Previous")
        self.next = w.Button(description="Next ▶")
        self.prev.on_click(lambda _: setattr(self.frame, "value", max(0, self.frame.value - 1)))
        self.next.on_click(
            lambda _: setattr(self.frame, "value", min(self.frame.max, self.frame.value + 1))
        )
        self.mode = w.Dropdown(
            options=[("모든 얼굴 가리기", "all"), ("선택한 대상만 가림 제외", "selected")],
            value=self.plan.mode,
            description="Mode",
            layout=w.Layout(width="420px"),
        )
        self.style = w.Dropdown(
            options=[("모자이크", "mosaic"), ("강한 블러", "blur"), ("검은색 가림", "solid")],
            value=self.plan.style,
            description="Style",
        )
        self.audio = w.Checkbox(value=self.plan.keep_audio, description="출력에 음성 포함")
        self.original = w.Checkbox(value=False, description="검수용 원본 표시")
        self.grid = w.Checkbox(value=False, description="좌표 격자·보정 박스 표시")
        self.status = w.HTML()
        self.message = w.HTML()
        self.image = w.Image(format="jpeg", layout=w.Layout(max_width="100%", width="960px"))
        options = [("대상 구간을 선택하세요", None)] + [
            (
                f"{s['id']} | {s['first_frame']/self.info['fps']:.2f}–{s['last_frame']/self.info['fps']:.2f}s",
                s["id"],
            )
            for s in self._segments.values()
        ]
        self.target = w.Dropdown(
            options=options, description="Target", layout=w.Layout(width="95%")
        )
        self.start = w.BoundedIntText(value=0, min=0, max=self.frame.max, description="Start frame")
        self.end = w.BoundedIntText(value=0, min=0, max=self.frame.max, description="End frame")
        self.set_start = w.Button(description="현재 프레임 → 시작")
        self.set_end = w.Button(description="현재 프레임 → 끝")
        self.set_start.on_click(lambda _: setattr(self.start, "value", self.frame.value))
        self.set_end.on_click(lambda _: setattr(self.end, "value", self.frame.value))
        self.confirm = w.Checkbox(value=False, description="같은 대상과 선택 구간을 직접 확인했음")
        self.add_reveal = w.Button(description="선택 구간 가림 제외 추가", button_style="primary")
        self.add_reveal.on_click(self._add_reveal)
        self.target.observe(self._target_changed, names="value")
        self.coords = [
            w.FloatSlider(
                value=value,
                min=0,
                max=1,
                step=0.005,
                description=label,
                continuous_update=False,
                readout_format=".3f",
                layout=w.Layout(width="46%"),
            )
            for label, value in [("Left", 0.1), ("Top", 0.1), ("Right", 0.3), ("Bottom", 0.4)]
        ]
        self.capture_start = w.Button(description="현재 박스 → 시작 위치")
        self.capture_end = w.Button(description="현재 박스 → 끝 위치")
        self.capture_start.on_click(lambda _: self._capture_box("start"))
        self.capture_end.on_click(lambda _: self._capture_box("end"))
        self.add_mask = w.Button(description="수동 가림 추가", button_style="primary")
        self.add_mask.on_click(self._add_mask)
        self.reveal_list = w.Select(options=[], rows=4, layout=w.Layout(width="95%"))
        self.mask_list = w.Select(options=[], rows=4, layout=w.Layout(width="95%"))
        remove_reveal = w.Button(description="선택한 가림 제외 삭제")
        remove_mask = w.Button(description="선택한 수동 가림 삭제")
        remove_reveal.on_click(lambda _: self._remove("reveal_intervals", self.reveal_list.value))
        remove_mask.on_click(lambda _: self._remove("manual_masks", self.mask_list.value))
        events = self.store.events(limit=300)
        self.events = w.Dropdown(
            options=[("검토 이벤트로 이동", None)]
            + [(f"{e['frame']/self.info['fps']:.2f}s | {e['kind']}", e["frame"]) for e in events],
            description="Review",
            layout=w.Layout(width="95%"),
        )
        self.events.observe(lambda change: self._event_jump(change["new"]), names="value")
        self.mode.observe(self._settings_changed, names="value")
        self.style.observe(self._settings_changed, names="value")
        self.audio.observe(self._settings_changed, names="value")
        for control in [self.frame, self.original, self.grid, *self.coords]:
            control.observe(lambda _: self.refresh(), names="value")
        reveal_panel = w.VBox(
            [
                self.target,
                w.HBox([self.start, self.end]),
                w.HBox([self.set_start, self.set_end]),
                self.confirm,
                self.add_reveal,
                self.reveal_list,
                remove_reveal,
            ]
        )
        mask_panel = w.VBox(
            [
                w.HTML(
                    "좌표는 화면 폭·높이에 대한 0~1 비율입니다. 시작/끝 프레임은 위 입력을 함께 사용합니다."
                ),
                w.HBox(self.coords[:2], layout=w.Layout(flex_flow="row wrap")),
                w.HBox(self.coords[2:], layout=w.Layout(flex_flow="row wrap")),
                w.HBox([self.capture_start, self.capture_end]),
                w.HTML(
                    "시작 위치만 저장하면 고정 박스, 끝 위치도 저장하면 두 위치 사이를 선형 이동합니다."
                ),
                self.add_mask,
                self.mask_list,
                remove_mask,
            ]
        )
        tabs = w.Tab(children=[reveal_panel, mask_panel])
        tabs.set_title(0, "한 명 선택·구간 확인")
        tabs.set_title(1, "수동 가림 보정")
        self.widget = w.VBox(
            [
                w.HTML("<b>CCTV 얼굴 가림 검수</b>"),
                w.HBox([self.mode, self.style], layout=w.Layout(flex_flow="row wrap")),
                w.HBox(
                    [self.audio, self.original, self.grid], layout=w.Layout(flex_flow="row wrap")
                ),
                self.frame,
                w.HBox([self.prev, self.next]),
                self.status,
                self.image,
                self.events,
                tabs,
                self.message,
            ]
        )
        self._refresh_lists()
        self.refresh()

    def _display_error(self, exc):
        self.message.value = f"<b>{html.escape(str(exc))}</b>"

    def _save(self):
        self.plan.validate(self.info["frame_count"])
        validate_project_plan(self.store, self.plan)
        self.plan.save(self.plan_path)
        self._refresh_lists()

    def _refresh_lists(self):
        self.reveal_list.options = [
            (f"{i}: {r['segment_id']} · frame {r['start_frame']}–{r['end_frame']}", i)
            for i, r in enumerate(self.plan.reveal_intervals)
        ]
        self.mask_list.options = [
            (f"{i}: frame {r['keyframes'][0]['frame']}–{r['keyframes'][-1]['frame']}", i)
            for i, r in enumerate(self.plan.manual_masks)
        ]

    def _settings_changed(self, _):
        self.plan.mode = self.mode.value
        self.plan.style = self.style.value
        self.plan.keep_audio = self.audio.value
        self._save()
        self.refresh()

    def _target_changed(self, change):
        if change["new"]:
            segment = self._segments[change["new"]]
            self.start.value = segment["first_frame"]
            self.end.value = segment["last_frame"]
            self.frame.value = segment["first_frame"]
            self.confirm.value = False

    def _event_jump(self, index):
        if index is not None:
            self.frame.value = index

    def _add_reveal(self, _):
        try:
            if not self.target.value or not self.confirm.value:
                raise ValueError("대상과 구간을 확인하고 확인란을 체크하세요.")
            candidate = {
                "segment_id": self.target.value,
                "start_frame": self.start.value,
                "end_frame": self.end.value,
                "confirmed_same_person": True,
            }
            self.plan.reveal_intervals.append(candidate)
            try:
                self._save()
            except Exception:
                self.plan.reveal_intervals.pop()
                raise
            self.confirm.value = False
            self.message.value = (
                "저장했습니다. Mode를 선택 대상 모드로 바꾸면 미리보기에 적용됩니다."
            )
            self.refresh()
        except Exception as exc:
            self._display_error(exc)

    def _capture_box(self, which):
        try:
            box = [c.value for c in self.coords]
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("Left < Right, Top < Bottom이어야 합니다.")
            setattr(self, which + "_box", box)
            self.message.value = f"{which} box saved: {box}"
        except Exception as exc:
            self._display_error(exc)

    def _add_mask(self, _):
        try:
            if self.start_box is None:
                raise ValueError("시작 위치 저장 버튼을 먼저 누르세요.")
            a, b = self.start.value, self.end.value
            if b < a:
                raise ValueError("끝 프레임은 시작 프레임 이상이어야 합니다.")
            keys = [{"frame": a, "box": self.start_box.copy()}]
            if b > a:
                keys.append({"frame": b, "box": (self.end_box or self.start_box).copy()})
            self.plan.manual_masks.append({"keyframes": keys})
            try:
                self._save()
            except Exception:
                self.plan.manual_masks.pop()
                raise
            self.start_box = self.end_box = None
            self.message.value = "수동 가림을 저장했습니다."
            self.refresh()
        except Exception as exc:
            self._display_error(exc)

    def _remove(self, field, index):
        if index is not None:
            getattr(self.plan, field).pop(index)
            self._save()
            self.refresh()

    def refresh(self):
        index = self.frame.value
        if self._frame_cache is None or self._frame_cache[0] != index:
            self._frame_cache = (
                index,
                read_frame(self.project / self.info["prepared_name"], index),
            )
        original = self._frame_cache[1]
        record = self.store.get(index)
        height, width = original.shape[:2]
        frame = (
            original.copy()
            if self.original.value
            else redact(
                original,
                mask_boxes(record, self.plan, width, height),
                self.plan.style,
                self.plan.mosaic_block_pixels,
            )
        )
        # English labels avoid missing CJK fonts in OpenCV; widget instructions are Korean.
        for person in record["people"]:
            x1, y1, x2, y2 = person["box"]
            cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (40, 210, 50), 2)
            label = person["segment_id"] or "UNTRACKED"
            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 210, 50),
                2,
            )
        if self.grid.value:
            for i in range(1, 10):
                x, y = round(width * i / 10), round(height * i / 10)
                cv2.line(frame, (x, 0), (x, height - 1), (130, 130, 130), 1)
                cv2.line(frame, (0, y), (width - 1, y), (130, 130, 130), 1)
                cv2.putText(
                    frame, f"{i/10:.1f}", (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1
                )
                cv2.putText(
                    frame, f"{i/10:.1f}", (2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1
                )
            a, b, c, d = [control.value for control in self.coords]
            cv2.rectangle(
                frame,
                (round(a * width), round(b * height)),
                (round(c * width), round(d * height)),
                (0, 200, 255),
                3,
            )
        scale = min(1, 960 / width)
        if scale < 1:
            frame = cv2.resize(frame, (960, max(1, round(height * scale))))
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Preview encoding failed.")
        self.image.value = encoded.tobytes()
        time = index / self.info["fps"]
        flags = ", ".join(sorted(set(record["flags"]))) or "none"
        self.status.value = f"Frame {index} · clip {time:.2f}s · source {time+self.info['source_start_seconds']:.2f}s · flags: {html.escape(flags)}"

    def show(self):
        from IPython.display import display

        display(self.widget)
        return self

    def close(self):
        self.store.close()
        self.widget.close()
