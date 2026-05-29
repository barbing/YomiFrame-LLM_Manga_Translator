# YomiFrame Technical Notes

This document describes the technical contracts behind the current specialized modular pipeline. The README is the broad project overview; this file owns implementation-level responsibilities, authorization fields, cache contracts, validation mechanics, and stage-boundary rules.

## Architecture Principles

YomiFrame separates semantic authority from pixel evidence and from downstream execution:

- BubbleDetection/TextAreaPlan own semantic authority.
- BubbleDetection normalizes model evidence from the bubble/text-area ensemble.
- TextAreaPlan adjudicates that evidence into typed semantic text units and standardized downstream eligibility fields.
- ComicTextDetector/TextForegroundSegmentation supplies text pixels inside scoped regions.
- CleanupMask consumes upstream authorization and foreground projection; it does not infer speech, background, SFX, art, or review semantics from local component geometry.
- OCR, translation, cleanup, and rendering require explicit TextAreaPlan fields instead of accepting route intent alone. OCR may be allowed for review/conservation while translation, cleanup, and rendering remain blocked.

The target default chain is:

```text
BubbleDetection typed evidence
  -> TextAreaPlan semantic units
  -> scoped CTD/TextForegroundSegmentation projection
  -> component authorization map
  -> OCR and translation routing
  -> SourceGlyphMask
  -> CleanupJob
  -> CleanupMask
  -> CleanupPlan
  -> CleanupBackend
  -> CleanupResult
  -> CleanupProof
  -> renderer final composition
```

## Stage Ownership

### UI and Controller

The desktop UI gathers user selections and calls the pipeline controller. The controller orchestrates module execution and preserves the TextAreaPlan contract when regions are serialized, rehydrated, filtered, or routed.

The controller should not turn route intent into translation authority. A region is translatable only when TextAreaPlan has explicitly marked OCR, translation, cleanup, and render eligibility as appropriate for a cleanup-translatable semantic state. Review-only OCR conservation does not create translation or cleanup authority.

### BubbleDetection

`app/pipeline/bubble_detection.py` is the upstream model-evidence provider. It combines speech-bubble and text-area evidence from the available bubble detection models, including Kitsumed-style speech-bubble output and Ogkalu labels such as `bubble`, `text_bubble`, and `text_free`.

BubbleDetection is responsible for:

- preserving raw model labels
- normalizing candidate kind
- stamping evidence strength
- recording source evidence IDs
- auditing edge and clipping context
- computing neighboring speech context
- exposing first-class reason codes for speech and free-text evidence
- including semantic contract identity in runtime and cache metadata

BubbleDetection does not by itself authorize cleanup. It supplies typed evidence to TextAreaPlan.

### TextAreaPlan

`app/pipeline/text_area_plan.py` is the semantic authority layer. It converts evidence into typed text units before CTD/component projection.

TextAreaPlan is responsible for:

- speech-bubble text authorization
- background/title/narration text authorization
- caption authorization
- SFX/decorative protection
- art/non-text protection
- review/unknown quarantine
- deterministic eligibility fields for OCR, translation, cleanup, and rendering
- explicit authorization state, basis, and origin

TextAreaPlan must distinguish a candidate or review state from executable semantic authority. A high-confidence single-model result may be promoted only through documented constraints, such as Ogkalu speech evidence with page-edge/clipping support, neighboring speech context, and no protected conflict.

### Text Pixel Projection

ComicTextDetector/TextForegroundSegmentation provides text-pixel evidence within TextAreaPlan scopes. It may refine component boundaries and foreground pixels, but it must not become the semantic owner of speech, background, SFX, or review classifications.

The projection output feeds component authorization and cleanup masks. Projection quality may affect executability, but projection readiness should not recolor semantic state.

### OCR

OCR consumes TextAreaPlan-eligible regions and projected text areas. Some compatibility or review-conservation regions may be OCR-eligible while still blocked from translation, cleanup, and rendering. Known SFX/decorative/art regions should not enter the normal translation path unless a future feature explicitly defines SFX translation support.

OCR errors should be diagnosed separately from semantic authorization errors. If visible text was never authorized upstream, OCR cannot fix the missed region.

### Translation and NLP

Translation consumes OCR text from eligible semantic units. The NLP layer handles glossary, name memory, style guidance, and consistency. It should not promote review/unknown regions into translation merely because text was detected.

### Cleanup

Cleanup begins only after upstream semantic authorization and projection have produced approved source-glyph pixels.

The cleanup chain is:

```text
SourceGlyphMask
  -> CleanupJob
  -> CleanupMask
  -> CleanupPlan
  -> CleanupBackend
  -> CleanupResult
  -> CleanupProof
```

`CleanupMask` is a strict consumer. It should only erase components that upstream authorization made executable. Unknown, protected SFX/decorative, art, and non-text components must remain non-executable.

### Rendering

Rendering composes translated text after cleanup. It must preserve the full translated text or produce explicit evidence when text cannot fit. It should not silently drop characters, overflow unreadably, or reinterpret semantic scope.

## Semantic Authorization Contract

Downstream modules receive standardized fields rather than reading historical confidence-tier names or marker strings. The contract includes:

- CTD scope
- OCR eligibility
- translation eligibility
- render eligibility
- cleanup executability
- explicit authorization state
- authorization basis
- origin/provider metadata

Important semantic states include:

- `cleanup_translate_speech`
- `cleanup_translate_background`
- `cleanup_translate_caption`
- `protect_sfx_decorative`
- `protect_art_or_non_text`
- `review_unknown_not_cleanup`
- `outside_cleanup_scope`
- `ambiguous_component_owner`

The required invariant is that executable downstream work must come from explicit TextAreaPlan authority. Candidate-only signals, stale artifacts, SourceGlyph/projection artifacts, page geometry, bbox overlap, and cleanup jobs do not create semantic authority.

## BubbleDetection Cache Contract

BubbleDetection cache entries must include the semantic evidence contract identity. Cache validation should reject payloads produced by an older semantic contract, even when model file paths and generic detector settings match.

This prevents stale evidence from a previous contract from being reused after changes to:

- semantic evidence schema
- neighboring speech context computation
- normalized Ogkalu/Kitsumed evidence interpretation
- authority-related provider behavior

The current cache contract is versioned in `app/pipeline/bubble_detection.py`. Cache invalidation is part of the semantic contract, not only detector performance.

## Validation Policy

Validation level depends on the changed module.

For BubbleDetection/TextAreaPlan/CTD/CleanupMask authorization changes:

- use a fresh mask-only run when explicitly approved
- inspect raw visual artifacts page by page
- compare original page, refined segmentation, semantic-unit/component authorization overlays, projection-quality overlay, protected/unknown overlay, clean foreground union, and clean erase union
- do not use contact-sheet prose, summary JSON, counters, or reports as proof

For cleanup runtime or inpainting changes:

- validate CleanupPlan, CleanupBackend, CleanupResult, and CleanupProof
- inspect inpainted output against original and masks
- confirm protected material is preserved and approved source text is erased

For OCR, translation, or rendering changes:

- run a real translation validation cycle when feasible
- inspect project JSON, OCR text, translated text, rendered output, overlays, and full pages
- check that rendered text contains the full `translated_text`

Mask-only validation cannot by itself prove end-to-end cleanup runtime, inpainting quality, rendering quality, or full translation readiness.

## Cleanup/Inpainting Readiness Boundary

Cleanup and inpainting validation should start only after upstream semantic authorization and text-pixel projection are accepted. Cleanup validation then needs to prove runtime execution, proof artifacts, inpainting quality, and final render composition.

It should not reopen BubbleDetection/TextAreaPlan ownership unless raw visual evidence shows that an upstream semantic unit or component authorization is still wrong.

## Auto-Glossary and Name Memory

The glossary system supports chapter-level translation consistency for names, titles, organizations, places, nicknames, and forms of address. It consumes OCR and translation context after region authorization has already happened.

Glossary enforcement must not mask upstream OCR or detection failures. If a name is missing because the text region was never authorized or OCR failed, the issue belongs upstream of glossary enforcement.

## Models and Environment

The project is Windows-first. For public setup, use Python 3.10 and install `requirements.txt` in an isolated virtual environment. Conda is also acceptable when users need GPU-specific Torch, PaddlePaddle, or llama-cpp-python builds.

Local assets and caches are preferred over environment changes. Heavy new dependencies or mandatory extra models should not be added unless the roadmap or user explicitly authorizes them.

Main model families used by the pipeline include:

- bubble/text-area detection models for BubbleDetection evidence
- ComicTextDetector/TextForegroundSegmentation for text-pixel projection
- OCR models for approved regions
- local LLMs through Ollama-compatible translation paths
- the fixed cleanup inpainting backend model
- optional NLP resources for glossary and name memory

## Performance Expectations

The default workflow must remain practical for local use. Translation-quality improvements should not make average processing time exceed roughly 30 seconds per page unless the user explicitly approves a slower path.

Performance-sensitive work should report:

- total runtime
- page count
- average time per page
- bottleneck stage when identifiable

## Development Guidelines

- Identify the owning stage before editing code.
- Do not fix semantic authorization defects inside CleanupMask.
- Do not fix OCR failures by changing translation prompts.
- Do not fix rendering overflow by changing OCR or semantic routing.
- Do not route SFX/decorative/art areas through normal OCR, translation, cleanup, or render paths unless explicitly required.
- Keep compatibility paths visible and temporary; do not let them become hidden alternate authority.
- Prefer deterministic contracts and explicit fallback states over ad hoc heuristics.

## Recommended Lightweight Checks

Syntax checks for edited Python modules:

```powershell
python -m py_compile app/pipeline/bubble_detection.py app/pipeline/text_area_plan.py app/pipeline/controller.py
```

Contract-focused unit tests, when relevant:

```powershell
python -m unittest app.tests.test_semantic_authority_contract
```

Full validation requires the task-specific visual or translation workflow described above.
