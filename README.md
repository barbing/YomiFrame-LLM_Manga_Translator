# YomiFrame

YomiFrame is a Windows desktop app for local manga translation. The current pipeline is modular: upstream text-area planning decides what visible text means, pixel detectors supply text foreground, and cleanup, OCR, translation, and rendering consume explicit contracts instead of inferring semantics locally.

![Screenshot](assets/screenshot.png)

## What It Does

- Provides a Windows GUI for selecting manga folders, source language, target language, and local models.
- Uses a BubbleDetection/TextAreaPlan stage to identify speech, narration, background/title text, SFX/decorative text, and non-text artifacts before cleanup.
- Combines Kitsumed-style speech-bubble evidence with Ogkalu text-area labels to improve coverage where a single bubble model is insufficient.
- Uses ComicTextDetector/TextForegroundSegmentation as a scoped text-pixel provider, not as the owner of speech/background/SFX semantics.
- Runs OCR on TextAreaPlan-eligible regions, including review-only OCR conservation paths where translation, cleanup, and render remain blocked.
- Translates with local Ollama-compatible LLMs and optional glossary/name-memory context.
- Removes source glyphs through the cleanup module: SourceGlyphMask, CleanupJob, CleanupMask, CleanupPlan, cleanup backend, CleanupResult, and CleanupProof.
- Renders translated text back into the intended speech, narration, caption, or background areas.

## Current Architecture

The default architecture is a specialized module chain rather than a monolithic detect/OCR/erase/render pass:

```text
BubbleDetection typed model evidence
  -> TextAreaPlan semantic authorization
  -> scoped CTD/TextForegroundSegmentation projection
  -> component authorization map
  -> OCR and translation eligibility
  -> SourceGlyphMask and CleanupJob
  -> CleanupMask and CleanupPlan
  -> cleanup backend / inpainting
  -> CleanupResult and CleanupProof
  -> renderer final composition
```

The key rule is ownership separation. BubbleDetection and TextAreaPlan own semantic authority. CTD supplies text pixels. CleanupMask is a strict consumer of upstream authorization and must not reclassify speech, background text, SFX, art, or unknown components from local geometry alone.

## Module Responsibilities

- `app/pipeline/bubble_detection.py`: runs the bubble/text-area model ensemble, normalizes raw model labels, stamps evidence strength, edge/clipping context, neighbor context, and semantic contract identity.
- `app/pipeline/text_area_plan.py`: adjudicates upstream evidence into typed text units and standardized downstream fields such as OCR eligibility, translation eligibility, render eligibility, cleanup executability, authorization state, basis, and origin.
- `app/pipeline/controller.py`: orchestrates the stage chain and preserves the TextAreaPlan authorization contract when regions are rehydrated or routed.
- `app/detect/` and CTD integrations: provide scoped text foreground and component pixels for authorized text areas.
- `app/ocr/`: reads approved text regions without deciding whether a component is semantic text.
- `app/translate/` and `app/nlp/`: translate eligible OCR text and maintain glossary/name-memory context.
- `app/pipeline/source_glyph_mask.py` and cleanup modules: build provenance-aware cleanup jobs and masks from upstream authorization and foreground projection.
- `app/render/`: composes translated text with cleanup results and must not silently drop translated content.

## Semantic Authorization

Text areas should reach downstream modules with explicit state instead of ambiguous route intent. The important states are:

- `cleanup_translate_speech`: normal dialogue or bubble text eligible for OCR, translation, cleanup, and render.
- `cleanup_translate_background`: title, sign, narration, or background text eligible for cleanup/translation when general constraints pass.
- `cleanup_translate_caption`: caption-style text eligible for normal downstream handling.
- `protect_sfx_decorative`: SFX or decorative lettering that should be protected from normal translation cleanup unless a future feature explicitly handles it.
- `protect_art_or_non_text`: artwork, panel structure, or non-text marks that must not become executable cleanup.
- `review_unknown_not_cleanup`: uncertain material that remains visible for review but is not executable cleanup authority.
- `outside_cleanup_scope`: material outside the current cleanup/translation scope.

Downstream stages should require explicit TextAreaPlan fields for OCR, translation, cleanup, and rendering. OCR eligibility can exist for review/conservation, but translation, cleanup, and render authority require a cleanup-translatable semantic state. Candidate-only evidence, stale cache artifacts, geometry, overlap, and route intent are not semantic authority by themselves.

## Cleanup and Inpainting

The cleanup module starts from accepted upstream semantic units and scoped foreground projection. Its job is to erase approved source glyph pixels while preserving SFX, decorative text, art, and unknown material.

The cleanup chain is intentionally explicit:

```text
SourceGlyphMask
  -> CleanupJob
  -> CleanupMask
  -> CleanupPlan
  -> CleanupBackend
  -> CleanupResult
  -> CleanupProof
```

Mask-only validation can prove semantic authorization and cleanup-mask readiness. It does not by itself prove cleanup runtime, inpainting quality, rendering quality, Gate 1/2/3, Phase 6, or full translation readiness.

## Recommended Local Setup

- Platform: Windows.
- Python: use the existing conda environment for this repo, normally `manga-llm`.
- Local LLM: Ollama-compatible model for translation, for example `qwen2.5:14b`.
- Dataset for validation: `Test Manga` unless a task names another dataset.

Start the app from the repository root:

```powershell
python -m app.main
```

Quick syntax check:

```powershell
python -m py_compile app/pipeline/bubble_detection.py app/pipeline/text_area_plan.py app/pipeline/controller.py
```

## Models and Assets

The app uses repository-local assets, caches, and downloaded models where possible:

- Bubble/text-area models for Kitsumed and Ogkalu evidence.
- ComicTextDetector/TextForegroundSegmentation for scoped text-pixel projection.
- OCR models such as MangaOCR/PaddleOCR depending on the active path.
- Optional Japanese NER/glossary resources.
- Optional cleanup/inpainting backend assets such as Big-LAMA.

Avoid downloading new tools or changing the environment unless the task explicitly requires it.

## Output

Typical runs write translated images and project artifacts under `output/`, including:

- translated page images
- project JSON with regions and translations
- style guide and glossary artifacts when enabled
- debug overlays, masks, and validation artifacts when the selected workflow produces them

Validation artifacts are evidence, not acceptance by themselves. Full-page visual inspection remains required for cleanup, rendering, and translation-quality decisions.

## Documentation

- `TECHNICAL.md`: detailed architecture and contract notes.
- `docs/architecture.md`: project architecture authority.
- `docs/modules.md`: module ownership and interfaces.
- `docs/current-issues-and-roadmap.md`: active roadmap and unresolved work.
- `docs/testing-and-validation.md`: validation rules and visual-review requirements.

## Development Rules

- Keep changes small and targeted.
- Do not route known SFX/decorative/art areas through normal OCR, translation, or cleanup paths unless the roadmap or user explicitly asks for that behavior.
- Do not treat reviewer counters, JSON summaries, or contact-sheet prose as proof of visual correctness.
- For mask decisions, inspect raw original pages, CTD/refined segmentation, authorization overlays, projection-quality overlays, protected/unknown overlays, foreground union, and erase union.
- For translation-affecting code changes, validate real output images and not only logs or metadata.

## License

Apache-2.0. See `LICENSE`.
