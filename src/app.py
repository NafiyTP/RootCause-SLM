"""
Small web app around the pipeline, with the real fine-tuned model (run it on a GPU, e.g. Colab).

Two tabs:
  - Pipeline: paste raw HDFS lines, every block is scored by the detector, and the top
    flagged blocks get their missing/rare events and a live explanation from the model
  - One line: give one log line and a label, get the model's explanation

Usage:
    pip install gradio
    python src/app.py            # local, http://127.0.0.1:7860
    python src/app.py --share    # public link (needed on Colab)
"""

import argparse
import time

import gradio as gr

from inference import explain, load_model
from pipeline import (describe, load_detector, missing_events, pick_line, predict_proba,
                      rare_events, read_lines)

VEC, COEFS, INTERCEPT, NORMAL_FREQ = load_detector()
MODEL, TOKENIZER = None, None


def get_model():
    global MODEL, TOKENIZER
    if MODEL is None:
        MODEL, TOKENIZER = load_model()
    return MODEL, TOKENIZER


def short(t):
    return t.replace("INFO dfs.", "").replace("WARN dfs.", "WARN ")


def run_pipeline(text, top, use_llm):
    blocks = read_lines(text.splitlines())
    if not blocks:
        return "No block id found. HDFS lines about a block contain an id like `blk_-1608999687919862906`."
    ids = list(blocks)
    X = VEC.transform([[t for t, _ in blocks[b]] for b in ids])
    scores = predict_proba(X, COEFS, INTERCEPT)
    flagged = sorted([(s, b) for s, b in zip(scores, ids) if s > 0.5], reverse=True)

    out = [f"**{len(ids)} blocks, {len(flagged)} flagged** (score > 0.5)\n"]
    for rank, (score, b) in enumerate(flagged[:int(top)]):
        lines = blocks[b]
        missing = missing_events(lines, NORMAL_FREQ)
        rare = rare_events(lines, NORMAL_FREQ)
        line = pick_line(lines, VEC, COEFS)
        out.append(f"### {b}  ·  score {score:.3f}  ·  {len(lines)} lines")
        out.append(f"**{describe(missing, rare)}**\n")
        for t in missing:
            out.append(f"- missing: `{short(t)}`")
        for t in rare:
            out.append(f"- rare: `{short(t)}`")
        out.append(f"\nLine sent to the model:\n```\n{line}\n```")
        if use_llm:
            model, tok = get_model()
            t0 = time.perf_counter()
            r = explain(line, "Anomaly", model, tok)
            out.append("Fine-tuned model (sees only this line, not reliable):\n\n"
                       f"**Cause:** {r.get('cause')}\n\n**Raisonnement:** {r.get('raisonnement')}\n\n"
                       f"_{time.perf_counter() - t0:.1f} s_\n")
    normal = len(ids) - len(flagged)
    if normal:
        out.append(f"\n{normal} block(s) scored normal.")
    return "\n".join(out)


def run_line(line, label):
    model, tok = get_model()
    t0 = time.perf_counter()
    r = explain(line.strip(), label, model, tok)
    return (f"**Cause:** {r.get('cause')}\n\n**Raisonnement:** {r.get('raisonnement')}\n\n"
            f"_{time.perf_counter() - t0:.1f} s_\n\nRaw output:\n```\n{r.get('raw')}\n```")


EXAMPLE = """081109 204403 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811092030_0001_m_000337_0/part-00337. blk_5679332564404011413
081109 204404 471 INFO dfs.DataNode$DataXceiver: Receiving block blk_5679332564404011413 src: /10.251.106.10:34539 dest: /10.251.106.10:50010"""


def build():
    with gr.Blocks(title="RootCause-SLM") as demo:
        gr.Markdown("# RootCause-SLM\nDetect anomalous HDFS blocks, then explain them with the "
                    "fine-tuned Qwen2.5-1.5B (LoRA).")
        with gr.Tab("Pipeline"):
            text = gr.Textbox(label="Raw HDFS log lines", lines=14, value=EXAMPLE)
            with gr.Row():
                top = gr.Slider(1, 10, value=3, step=1, label="Explain the top k flagged blocks")
                use_llm = gr.Checkbox(value=True, label="Run the language model")
            go = gr.Button("Run", variant="primary")
            out = gr.Markdown()
            go.click(run_pipeline, [text, top, use_llm], out)
        with gr.Tab("One line"):
            line = gr.Textbox(label="Log line", value=EXAMPLE.splitlines()[0])
            label = gr.Radio(["Anomaly", "Normal"], value="Anomaly", label="Label")
            go2 = gr.Button("Explain", variant="primary")
            out2 = gr.Markdown()
            go2.click(run_line, [line, label], out2)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build().launch(share=args.share)
