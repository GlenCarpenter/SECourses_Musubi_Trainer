import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, Optional

import gradio as gr
import psutil

from .class_gui_config import GUIConfig
from .common_gui import (
    generate_script_content,
    get_model_file_path,
    get_saveasfilename_path,
    save_executed_script,
    scriptdir,
    utf8_subprocess_options,
)
from .custom_logging import setup_logging


PYTHON = sys.executable
WORKER_SCRIPT = os.path.join(scriptdir, "musubi_tuner_gui", "krea2_checkpoint_merge_worker.py")
log = setup_logging()


def _checkpoint_b_weight(weight_a_percent: float):
    return gr.update(value=100.0 - float(weight_a_percent))


class Krea2CheckpointMerger:
    def __init__(self, headless: bool, config: Optional[GUIConfig]) -> None:
        self.headless = headless
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.cancel_requested = False

    @staticmethod
    def _is_running(process: Optional[subprocess.Popen]) -> bool:
        return process is not None and process.poll() is None

    @staticmethod
    def _terminate_process(process: Optional[subprocess.Popen]) -> bool:
        if not process or process.poll() is not None:
            return False
        try:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.Error:
                    continue
            parent.kill()
            return True
        except psutil.Error:
            return False

    def merge(
        self,
        checkpoint_a: str,
        checkpoint_b: str,
        weight_a_percent: float,
        output_path: str,
        device_choice: str,
        overwrite: bool,
    ) -> str:
        if self._is_running(self.process):
            return "A checkpoint merge is already running. Cancel it before starting another."
        if not os.path.isfile(WORKER_SCRIPT):
            return f"Checkpoint merge worker was not found: {WORKER_SCRIPT}"

        payload: Dict[str, object] = {
            "checkpoint_a": checkpoint_a,
            "checkpoint_b": checkpoint_b,
            "weight_a": float(weight_a_percent) / 100.0,
            "output_path": output_path,
            "device_choice": device_choice,
            "overwrite": bool(overwrite),
        }
        payload_fd, payload_path = tempfile.mkstemp(suffix=".json", prefix="krea2_checkpoint_merge_")
        result_fd, result_path = tempfile.mkstemp(suffix=".json", prefix="krea2_checkpoint_result_")
        os.close(result_fd)
        try:
            with os.fdopen(payload_fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            command = [PYTHON, WORKER_SCRIPT, "--input", payload_path, "--output", result_path]
            save_executed_script(
                script_content=generate_script_content(command, "Krea 2 checkpoint merge"),
                config_name=None,
                script_type="krea2_checkpoint_merge",
            )
            self.cancel_requested = False
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                **utf8_subprocess_options(),
            )
            output_lines = []
            if self.process.stdout:
                for raw_line in self.process.stdout:
                    line = raw_line.rstrip("\r\n")
                    output_lines.append(line)
                    if line:
                        log.info("[Krea 2 Checkpoint Merge] %s", line)
            self.process.wait()
            if self.cancel_requested:
                return "Checkpoint merge cancelled. No partial output was installed."
            try:
                with open(result_path, "r", encoding="utf-8") as handle:
                    result = json.load(handle)
            except (OSError, json.JSONDecodeError):
                tail = "\n".join(output_lines[-8:])
                return f"Checkpoint merge worker did not return a result.\n{tail}"
            status = result.get("status", "error")
            message = result.get("message", "Worker returned no message.")
            if status == "skip":
                return f"Skipped: {message}"
            if status == "error":
                return f"Error: {message}"
            return str(message)
        finally:
            self.process = None
            for path in (payload_path, result_path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def cancel(self) -> str:
        self.cancel_requested = True
        if self._terminate_process(self.process):
            return "Cancellation requested. The checkpoint merge is stopping."
        return "No checkpoint merge is currently running."


def krea2_checkpoint_merge_tab(headless: bool, config: Optional[GUIConfig]) -> None:
    merger = Krea2CheckpointMerger(headless, config)

    gr.Markdown("# Krea 2 Checkpoint Merger")
    gr.Markdown(
        "Blend two compatible Krea 2 ConvRot INT8 DiT checkpoints. Quantized layers are "
        "dequantized one at a time, blended, and requantized into a portable ConvRot INT8 checkpoint."
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                checkpoint_a = gr.Textbox(
                    label="Checkpoint A",
                    placeholder="First Krea 2 ConvRot INT8 .safetensors checkpoint",
                )
                checkpoint_a_button = gr.Button(
                    "Browse File", size="lg", elem_classes=["mbtn", "mbtn-blue"], visible=not headless
                )
            with gr.Row():
                checkpoint_b = gr.Textbox(
                    label="Checkpoint B",
                    placeholder="Second compatible Krea 2 ConvRot INT8 .safetensors checkpoint",
                )
                checkpoint_b_button = gr.Button(
                    "Browse File", size="lg", elem_classes=["mbtn", "mbtn-violet"], visible=not headless
                )
            with gr.Row():
                output_path = gr.Textbox(
                    label="Merged Output Path",
                    placeholder="Path for the merged ConvRot INT8 .safetensors checkpoint",
                )
                output_button = gr.Button(
                    "Save As", size="lg", elem_classes=["mbtn", "mbtn-navy"], visible=not headless
                )

            merge_status = gr.Textbox(
                label="Merge Log",
                lines=14,
                max_lines=40,
                interactive=False,
            )
            with gr.Row():
                merge_button = gr.Button(
                    "Merge Checkpoints", variant="primary", elem_classes=["mbtn", "mbtn-emerald"]
                )
                cancel_button = gr.Button(
                    "Cancel Merge", variant="secondary", elem_classes=["mbtn", "mbtn-stone"]
                )

        with gr.Column(scale=1):
            with gr.Accordion("Merge Settings", open=True):
                weight_a = gr.Slider(
                    label="Checkpoint A Contribution (%)",
                    minimum=0,
                    maximum=100,
                    value=70,
                    step=0.1,
                )
                weight_b = gr.Number(
                    label="Checkpoint B Contribution (%)",
                    value=30,
                    precision=1,
                    interactive=False,
                )
                device = gr.Radio(
                    label="Merge Device",
                    choices=["Auto", "CUDA", "CPU"],
                    value="Auto",
                    info="CUDA is faster; CPU needs more system RAM but no VRAM.",
                )
                overwrite = gr.Checkbox(label="Overwrite Existing Output", value=False)

    checkpoint_a_button.click(
        fn=lambda current: get_model_file_path(current),
        inputs=[checkpoint_a],
        outputs=[checkpoint_a],
        show_progress=False,
    )
    checkpoint_b_button.click(
        fn=lambda current: get_model_file_path(current),
        inputs=[checkpoint_b],
        outputs=[checkpoint_b],
        show_progress=False,
    )
    output_button.click(
        fn=lambda current: get_saveasfilename_path(
            current, extensions="*.safetensors", extension_name="Checkpoint files"
        ),
        inputs=[output_path],
        outputs=[output_path],
        show_progress=False,
    )
    weight_a.change(
        fn=_checkpoint_b_weight,
        inputs=[weight_a],
        outputs=[weight_b],
        show_progress=False,
    )
    merge_button.click(
        fn=merger.merge,
        inputs=[checkpoint_a, checkpoint_b, weight_a, output_path, device, overwrite],
        outputs=[merge_status],
        show_progress=True,
    )
    cancel_button.click(fn=merger.cancel, inputs=[], outputs=[merge_status], show_progress=False)