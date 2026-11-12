# CLAUDE.md

## Project
In-ear AI race engineer for Le Mans Ultimate (and rFactor 2). Two paths, both spoken to the driver:
- **Ask**: the driver holds a wheel/controller button and asks a question → speech-to-text → LLM agent answers using live telemetry and strategy → text-to-speech in the ear.
- **Events**: telemetry is watched continuously; race situations (pit window, fuel low, gaps, flags, …) are detected and read out loud without being asked.

- Windows only, Python 3.12 (`requires-python = ">=3.12,<3.13"`), environment in `.venv/`.
- Game data comes from the shared-memory readers in `thirdparty/pyLMUSharedMemory` and `thirdparty/pyRfactor2SharedMemory` (MIT git submodules from the TinyPedal org; update with `git submodule update --remote`). Only the `telemetry/` layer imports them.
- Third-party, never edit: `thirdparty/`, `models/vosk-*`, anything vendored.
- `legacy/`: old code kept for reference while restructuring (e.g. `legacy/tinypedal_local/`, my former changes inside a TinyPedal fork). Nothing imports from it; move pieces into `src/` when asked.

## Role
I own the system architecture. You are the implementer: you design the smallest set of building blocks that fulfils a request and write them in my style. When a request is ambiguous or would require changing the architecture, ask before writing code.

## Layout
```
src/ai_race_engineer/
  telemetry/    game data sources wrapping the thirdparty readers (LMU, rF2); base: TelemetrySource
  strategy/     pure calculations: fuel, lap/sector deltas, pit windows
  events/       detect race situations from telemetry; base: Event
  voice/        audio devices, speech-to-text, text-to-speech; bases: SpeechToText, TextToSpeech
  agent/        LLM agent (langgraph), its tools and prompts
  recorders/    write telemetry/session data to disk (not "logging", that shadows stdlib)
scripts/        entry points; wiring only (e.g. run_engineer.py)
configs/        YAML: game, car class, track, voice, LLM model and keys
data/           raw/ recorded sessions, processed/ derived tables (.fuel, .csv)
models/         downloaded models (vosk)
tests/          pytest; mirrors src/ai_race_engineer/
notebooks/      exploration only; nothing imports from here
thirdparty/     git submodules: pyLMUSharedMemory, pyRfactor2SharedMemory (never edit)
legacy/         old code for reference during the restructure (nothing imports from here)
```

## Core principles
1. **One file, one purpose.** Every module serves a single responsibility (e.g. `strategy/fuel.py` only computes fuel). Its public classes/functions are imported and composed by higher-level code.
2. **Blocks compose upward.** Low-level blocks (telemetry readers, calculations, STT/TTS engines) → mid-level components (event monitor, agent, voice loop) → thin entry-point scripts. Entry scripts only wire blocks together; they contain no logic.
3. **Same-level components are interchangeable.** Alternatives (rF2 vs LMU source, Vosk vs Google STT, ElevenLabs vs local TTS, Mistral vs another chat model) share one abstract interface, so swapping them is a one-line change in the entry script or config. Add a thin wrapper when a library object doesn't fit the interface.
4. **Additive changes only.** Do not modify existing code. Add new files/classes. If an existing file truly must change (e.g. registering a new component, one import line), state the exact line and why, and keep the change minimal. Never refactor, rename, or reformat code I didn't ask about.
5. **Simplest design wins.** No speculative features, no premature abstraction, no configuration nobody asked for. If a library already does it, call the library.

## Code style
- **Strict typing everywhere**: full type hints on all arguments, returns and class attributes. Use dataclasses for telemetry snapshots and results, `pd.DataFrame` for lap tables, `numpy.typing.NDArray` for arrays, `ABC` for interfaces.
- **Input → output**: functions take only what they need and return only what is needed. Strategy and event logic is pure: it takes telemetry values and returns results, without reading shared memory or audio itself.
- **One docstring per function/class**, nothing else. State purpose, then each input/output with type and **unit**:
```python
  def laps_of_fuel(fuel_l: float, consumption_l_per_lap: float) -> float:
      """Laps remaining with the current fuel.
      In:  fuel_l                 float  litres
           consumption_l_per_lap  float  litres/lap
      Out:                        float  laps
      """
```
  No other comments.
- **Top-down inside a file**: public class/function first, the helpers it calls below it.
- **Trust inputs inside the core**: no asserts, no input validation, no defensive code in `strategy/`, `events/` or other pure logic.
- **Handle failure only at I/O boundaries**: game not running or shared memory unavailable, audio device missing, network/LLM/TTS API errors, file read/write. Handle it once at that boundary (retry, wait, or tell the driver) and nowhere else.
- **Real-time loop**: never block the telemetry/event loop with LLM, TTS or network calls; run those in a separate thread or async task.
- **Short functions**: as long as they need to be and no longer.
- **Classic OOP, library style**: constructor holds parameters, a small set of public methods (`read`, `update`, `check`, `transcribe`, `speak`, `__call__`…), inheritance from a shared base class for interchangeable components. Composition over deep inheritance.
- **Use established libraries**: pyLMUSharedMemory/pyRfactor2SharedMemory for game data, pandas/numpy for calculations, vosk/SpeechRecognition, sounddevice/pyaudio, elevenlabs, langgraph/langchain, pydantic, typer for CLIs. Never re-implement what they provide.
- **Settings in `configs/`**: thresholds, track/class parameters, model names and voices come from config or constructor args, never hardcoded in logic. Secrets come from environment variables.

### Interchangeable component pattern
```python
# src/ai_race_engineer/voice/base.py
class SpeechToText(ABC):
    """Common interface for all speech-to-text engines."""
    @abstractmethod
    def transcribe(self, audio: NDArray[np.int16]) -> str: ...

# scripts/run_engineer.py — swapping is one line
stt: SpeechToText = VoskSpeechToText(model_dir="models/vosk-small-en")   # or: GoogleSpeechToText()
```

## Tooling
- Lint and format: `ruff check --fix .` and `ruff format .` (line length 120, config in `pyproject.toml`).
- Tests: `pytest tests/`. Pure logic (strategy, events) gets tests; I/O wrappers don't need them.
- Run ruff and pytest on the changed files before reporting a task done.

## Workflow for every request
Before writing code, answer briefly in chat:
1. **Goal**: what is requested, and what is the concrete result?
2. **Inventory**: which existing blocks already cover part of it (file + class)? What is missing?
3. **Design, top-down**:
   a. Structure: which folders/files are needed and where they sit in the layout.
   b. Entry script: the few lines that import and wire the blocks.
   c. Blocks: each class/function the entry script uses (file path, typed signature, interface it implements), then what those blocks need, one level down at a time.
4. **Check**: does it meet the style above? Is every block single-purpose, composable and swappable with its siblings? Is anything existing modified? Is there a simpler design?

For non-trivial requests (more than one new file, a new interface, or anything touching existing code), wait for my OK on the design before implementing. For small, clear requests, state the design and implement right away.

After implementing, report only: files added, the one-line usage in the entry script, and any existing line changed.

## Subagents
Use these for non-trivial requests:
- **architect**: maps existing blocks and produces the design (steps 1–4). Read-only.
- **implementer**: writes the approved design, nothing more.
- **reviewer**: checks the result against this file, the design and the git diff; reports violations, doesn't fix them.

Default flow: architect → (my approval) → implementer → reviewer → fix reported issues.
Subagents start without the conversation: pass the approved design to the implementer word for word, and pass the design plus the implementer's list of changed files to the reviewer.
