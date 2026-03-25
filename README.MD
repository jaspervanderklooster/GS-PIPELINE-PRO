# GS_PIPELINE-PRO

Branch note (setup/colmap-fallback-and-config): worker paths/presets are now config-driven and COLMAP has automatic GPU→CPU fallback.

GS_PIPELINE-PRO is an internal, human-friendly pipeline for turning photo sets and videos into high-quality Gaussian Splats.

The goal is simple:

**Input goes in on one side.  
A reliable, high-quality Gaussian Splat comes out on the other side.**

This project is built around one core belief:

**The technology must adapt to the human, not the other way around.**

That means the pipeline should understand common human behavior, reduce friction, and make sensible decisions where possible, instead of expecting users to think like developers.

---

## Core Promise

A colleague should be able to:

- drop a video, photo set, or project folder into the pipeline
- choose a quality preset
- let the system do the heavy lifting
- receive a high-quality Gaussian Splat with clear status and predictable behavior

The system should feel calm, stable, and trustworthy.

---

## What We Are Building

GS_PIPELINE is not just a collection of scripts.

It is a **robust production pipeline** with one clear purpose:

- ingest human-supplied input
- normalize that input into a clean job
- preprocess it safely
- reconstruct a scene
- train/export a Gaussian Splat
- deliver the result in a predictable way

The long-term goal is not “more clever code”.

The long-term goal is:

**reliable output, understandable behavior, and stable operation**

---

## Stability First

Stability is a non-negotiable requirement.

This project must prefer:

- stability over speed
- quality over raw throughput
- predictable behavior over clever hacks
- safe fallback behavior over hard crashes
- clear logs and understandable states over silent failure

If a run takes longer but gives a better and more dependable result, that is acceptable.

If a fast shortcut increases the chance of broken output, unstable runs, or confusing behavior, it is not acceptable.

**A Gaussian Splat that is made slower but correctly is better than a fast result that cannot be trusted.**

---

## Design Principles

### 1. Human-first intake
Users should not have to think in terms of internal pipeline architecture.

Examples:

- If a user drops a single loose video into an `inbox/<colleague>` folder, the system should understand that this is probably one project and automatically place it into a proper job structure.
- If a user places multiple related videos inside one project folder, the system should understand that these belong together.
- If a user provides a photo set, the system should treat it as one coherent input set unless there is a clear reason not to.

The pipeline should reduce friction, not create it.

---

### 2. One clear job per project
Internally, everything should become a clean, explicit job.

Even if users provide messy input, the system should normalize it into a stable internal structure so the rest of the pipeline can work consistently.

That means:

- explicit job metadata
- explicit job states
- explicit artifacts
- explicit logs
- explicit reasons when something is rejected or needs attention

---

### 3. Safe automation
Automation is good, but blind automation is not.

The system should automatically help where it can, but it must remain conservative when input is ambiguous, risky, or too large.

Good automation:

- bundling a loose video into a proper job
- treating multiple videos inside one project folder as one project
- extracting representative frames from video
- removing obvious duplicate frames
- applying safe preset fallbacks when a requested run is too heavy

Bad automation:

- guessing wildly
- silently changing the nature of the job
- masking errors
- pretending success when output quality is not acceptable

---

### 4. Quality-aware preprocessing
Preprocessing exists to improve the final Gaussian Splat, not just to reduce file count.

The system should aim to keep:

- useful frames
- sharp frames
- frames with enough variation and overlap
- data that supports reliable reconstruction

And it should aim to remove or reduce:

- duplicate frames
- near-duplicate frames
- weak or low-value frames
- unnecessary memory pressure
- avoidable noise in the training input

---

### 5. Presets must stay human-readable
Presets are there to help humans make sensible quality choices without needing to understand the full technical stack.

Users should not have to think in terms of hundreds of internal parameters.

Presets should communicate intent, for example:

- a faster preview-oriented run
- a balanced standard run
- a high-quality run

Internally, the pipeline may translate those choices into technical settings, safe variants, or fallback logic.

But externally, the preset system should remain simple and understandable.

---

### 6. Fallbacks should protect the result
If a requested run is too heavy for safe execution, the pipeline should prefer a controlled downgrade over a hard failure, as long as the output remains meaningful and the behavior is clearly recorded.

This means the system may use:

- safer COLMAP variants
- memory-conscious settings
- conservative preprocessing behavior
- explicit notes in metadata or logs when fallback behavior was used

Fallback behavior is not a weakness.

It is part of building a resilient pipeline.

---

## User Experience We Want

The ideal user experience should feel like this:

1. The user drops input into the pipeline.
2. The pipeline understands what the user most likely meant.
3. The pipeline creates a clean internal job.
4. The user can follow simple status updates.
5. The system produces a Gaussian Splat of the requested quality, or clearly explains why it could not.

The user should not need to understand:

- folder normalization rules
- job contracts
- reconstruction stages
- memory tuning
- GPU behavior
- internal fallback logic

Those are implementation details.

The pipeline should carry that burden.

---

## Input Philosophy

GS_PIPELINE should support real-world human input patterns, not only ideal lab conditions.

Examples of intended behavior:

### Single loose video in a colleague inbox folder
If a user drops one video file directly into their inbox folder, the pipeline should recognize that this is likely a project and automatically turn it into one.

### Multiple videos inside one project folder
If multiple videos are placed inside a folder, the pipeline should assume they belong together unless there is a strong reason not to.

### Photo sets
A photo set should be treated as one coherent capture session, normalized where needed, and passed through preprocessing and reconstruction in a predictable way.

### Archives
Archives may be unpacked and normalized if that helps the user flow, but only in a safe and explicit manner.

### Ambiguous or risky input
If input is too mixed, too large, or too unclear, the system should not bluff. It should stop safely and explain what needs human attention.

---

## What “Done” Means

A job is only truly successful when it results in a usable, high-quality output.

Success is not:

- “the script finished”
- “the logs contain no Python errors”
- “a file exists somewhere”

Success means:

- the input was understood correctly
- preprocessing behaved sensibly
- reconstruction succeeded well enough
- the final Gaussian Splat is actually usable
- the process was logged clearly
- the result can be delivered with confidence

---

## What We Value More Than Speed

We explicitly value these things more than raw speed:

- reconstruction quality
- stability of the pipeline
- reproducibility
- understandable behavior
- predictable outputs
- safe defaults
- clean recovery from failure

A slower but stable pipeline is acceptable.

A faster but fragile pipeline is not.

---

## Technical Direction

Internally, the pipeline is expected to evolve around clear stages such as:

- intake / watcher
- job normalization
- preprocessing
- reconstruction
- Gaussian Splat training
- delivery / archiving / reporting

This architecture should remain modular, explicit, and readable.

We want a system where:

- one stage can be improved without collapsing the whole pipeline
- logs and metadata make debugging easier
- AI-generated code stays grounded in a stable structure
- future improvements can be added without turning the project into a house of cards

---

## Rules for Human and AI Contributors

All contributors, human or AI, should follow these rules.

### Do
- protect stability
- prefer simple solutions over clever ones
- keep behavior explicit
- write readable code
- preserve the human-first philosophy
- keep preset logic understandable
- log important decisions
- build for maintainability
- improve reliability before adding complexity

### Do not
- optimize for speed at the expense of quality
- add hidden magic that makes behavior harder to understand
- introduce brittle shortcuts
- silently change user intent
- overcomplicate the architecture
- treat AI-generated code as correct by default
- build features that make the system harder to trust

---

## Definition of Good Progress

A change is good progress if it makes the pipeline:

- more stable
- more predictable
- easier to reason about
- more tolerant of normal human input
- better at producing high-quality Gaussian Splats

A change is not good progress if it only makes the code look more advanced while increasing fragility or confusion.

---

## Scope

This repository is for building a stable internal pipeline.

It is not primarily about:

- flashy demos
- experimental one-off scripts
- chasing every new release immediately
- optimizing benchmark speed at all costs

Those things may matter later.

First, we build a pipeline we can trust.

---

## Final Principle

If there is ever doubt between two approaches, choose the one that better supports this sentence:

**A normal user should be able to put input into the pipeline with confidence, and receive a high-quality Gaussian Splat from a stable system.**
