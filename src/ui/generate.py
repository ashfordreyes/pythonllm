"""Build streaming `plan_fn` / `code_fn` callables from a loaded HF model.

`src/ui/console.py` renders whatever two callables it is handed; this is the
adapter that turns a `PeftModel` + tokenizer into those callables so a Colab
cell is three lines instead of thirty.

The generation itself mirrors `04_eval_pipeline.ipynb` section 4 -- same chat
template, same greedy defaults, same `skip_special_tokens=False` so `<PLAN>`,
`<STEP>` and `</PLAN>` stay visible instead of being silently swallowed. The
difference is streaming: eval wants a final string, a chat UI wants the text as
it arrives, because a 7B model on an L4 takes tens of seconds per answer and a
frozen cell looks indistinguishable from a hung one.

torch and transformers are imported inside the function, not at module scope,
so `ui.console` stays importable without them.
"""

from __future__ import annotations

from typing import Callable, Iterator

PLANNER_SYSTEM_PROMPT = (
    "You are a planner that turns a task description into a pseudocode plan "
    "using the pythonllm DSL. Respond with only the plan, wrapped in "
    "<PLAN>...</PLAN> and made of <STEP> lines."
)

CODER_SYSTEM_PROMPT = (
    "You are a coder that turns a pythonllm DSL pseudocode plan into Python. "
    "Respond with only the code."
)


def _clean(text: str, tokenizer) -> str:
    """Drop the end/pad markers `skip_special_tokens=False` leaves behind.

    The DSL tokens have to survive decoding, which rules out
    `skip_special_tokens=True`, so the control tokens come back too and are
    stripped by name here.
    """
    for token in (tokenizer.eos_token, tokenizer.pad_token):
        if token:
            text = text.replace(token, "")
    return text.strip()


def make_generator(
    model,
    tokenizer,
    system_prompt: str,
    *,
    max_new_tokens: int = 512,
    stream: bool = True,
    strip_fences: bool = False,
    **generate_kwargs,
) -> Callable[[str], "str | Iterator[str]"]:
    """Return `f(user_text)` producing the model's reply.

    With `stream=True` the return value is an iterator of *cumulative*
    snapshots, which is the shape `Console.ask` re-renders on. With
    `stream=False` it is a single string.

    `strip_fences=True` removes a leading ```python fence and its closing
    counterpart -- a code-pretrained base often emits one, and it would land
    inside the console's Python panel as literal backticks.
    """
    import torch

    def _postprocess(text: str) -> str:
        text = _clean(text, tokenizer)
        if strip_fences and text.startswith("```"):
            text = text.split("\n", 1)[-1] if "\n" in text else text
            # Only trim the closing fence once the model has actually emitted
            # it; mid-stream the text is legitimately unterminated.
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        return text.strip()

    def _encode(user_text: str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,  # some transformers versions return a BatchEncoding
        ).to(model.device)

    def _kwargs(encoded) -> dict:
        kwargs = dict(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        kwargs.update(generate_kwargs)
        return kwargs

    def generate_blocking(user_text: str) -> str:
        encoded = _encode(user_text)
        with torch.no_grad():
            out = model.generate(**_kwargs(encoded))
        new_tokens = out[0][encoded["input_ids"].shape[1]:]
        return _postprocess(tokenizer.decode(new_tokens, skip_special_tokens=False))

    def generate_streaming(user_text: str) -> Iterator[str]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        encoded = _encode(user_text)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=False
        )
        kwargs = _kwargs(encoded)
        kwargs["streamer"] = streamer

        # `generate` has to run off the main thread for the streamer's queue to
        # be drained while it is still producing.
        error: list[BaseException] = []

        def worker() -> None:
            try:
                with torch.no_grad():
                    model.generate(**kwargs)
            except BaseException as exc:  # surfaced after the loop below drains
                error.append(exc)

        thread = Thread(target=worker, daemon=True)
        thread.start()

        accumulated = ""
        for chunk in streamer:
            accumulated += chunk
            yield _postprocess(accumulated)
        thread.join()
        if error:
            raise error[0]

    return generate_streaming if stream else generate_blocking


def make_planner(model, tokenizer, **kwargs):
    """`make_generator` with the Stage 1 system prompt."""
    kwargs.setdefault("system_prompt", PLANNER_SYSTEM_PROMPT)
    return make_generator(model, tokenizer, kwargs.pop("system_prompt"), **kwargs)


def make_coder(model, tokenizer, **kwargs):
    """`make_generator` with the Stage 2 system prompt and fence stripping."""
    kwargs.setdefault("system_prompt", CODER_SYSTEM_PROMPT)
    kwargs.setdefault("strip_fences", True)
    return make_generator(model, tokenizer, kwargs.pop("system_prompt"), **kwargs)
