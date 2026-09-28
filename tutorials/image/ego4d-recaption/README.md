# Ego4D image recaption

Generate structured English captions for frames sampled from Ego4D FHO and
narration annotations. This pipeline is independent of image filtering and
deduplication. It reuses the existing VLM clients, concurrent requests, service
launcher and Parquet batch recovery.

## Run

Use the existing Curator environment. Set `data_root` to the directory containing
the original `<video_uid>.mp4` files. By default the annotation files are
`v2/ego4d.json`, `v2/annotations/fho_main.json` and
`v2/annotations/narration.json` below that directory; their paths can be
overridden with `metadata`, `fho_annotations` and `narration_annotations`.

```bash
python tutorials/image/ego4d-recaption/run.py --config /path/to/config.json
python tutorials/image/ego4d-recaption/report.py --output /path/to/output
```

`config.example.json` calls an existing OpenAI-compatible service. The served
model is discovered unless `model` is configured. `config.managed.example.json`
starts replicas using a separately configured vLLM executable; Curator does not
install or modify that environment. Replica placement and engine arguments are
the existing VLM service launcher's configuration.

For a small trial, set `video_uids` to a list of original video IDs. Set
`extract_only` to `true` to test sampling and extraction without a model service.
`sources` selects `fho`, `narration`, or both. A video in the full FHO annotation
collection is excluded from narration sampling even when only narration is run.

## 1,000-frame review experiment

Copy `config.experiment.example.json` and set the data, output and service paths.
It selects 500 FHO frames and 500 narration frames by default. `sample_counts`
controls the allocation; `sampling_seed` fixes the selection. Each source is
sampled uniformly without replacement from the eligible frames produced by its
approved sampling rules across all available videos, rather than taking the
first videos. Only selected frames are decoded and sent to the model.

```bash
python tutorials/image/ego4d-recaption/experiment.py --config /path/to/experiment.json
```

Selected per-video plans and `source=<name>/selection.json` are saved before
extraction. Resuming uses those same plans. The requested count includes failed
samples; failures remain visible and are not replaced with different frames.
If a source has too few eligible frames, preparation raises an error before
decoding. Full-dataset runs without `sample_counts` retain their normal behavior.

The experiment entry runs the existing pipeline and then generates
`report/recaption-<caption_version>/index.html` with **every result**. The report
groups FHO and narration separately, separates failure types including `length`,
and shows planned/written/pending counts, the model and source system prompts.
Each card contains the image, complete caption, structured elements, model input
annotations, original annotation associations and raw response. Source/status
links provide navigation; images load lazily. It is static HTML.

To regenerate the complete report after interruption or a retry:

```bash
python tutorials/image/ego4d-recaption/report.py --output /path/to/output --all
```

Serve the report directory on the remote host:

```bash
python -m http.server 5501 --bind 127.0.0.1 --directory /path/to/output/report/recaption-v1
```

Forward port 5501 in VS Code and open `http://127.0.0.1:5501/` in the local browser.

## Sampling and input

FHO uses the seven available critical-frame roles and four uniformly spaced
frames in each valid action's original frame interval. Repeated frames are
merged per video; all action associations are preserved. Object hints use only
annotations at the exact selected frame. The selected primary action supplies
the model context.

Narration combines both passes, removes empty, unsure, invalid and redacted
points, and keeps at most one annotated frame per fixed five-second interval:
the narration closest to the interval center. Its own pass supplies the covering
summary, or `null` when absent. It does not impose a minimum gap or a video quota.

Each video is opened once by an extraction worker, with frame targets sorted
before seeking. OpenCV extracts the frames into JPEG TAR shards; their Parquet
manifests record member offsets for direct access. The image reader passes RGB
`ImageBatch` objects to the recaption stage. Requests contain the fixed
source-specific system prompt, one image, and serialized annotation context.
The model returns only `comprehensive_description` and `prominent_elements`.

## Outputs and recovery

```text
output/
  samples/schema-v1/run.json
  samples/schema-v1/plans/source=<name>/<video_uid>.parquet
  samples/schema-v1/manifests/source=<name>/<video_uid>-<batch>.parquet
  media/source=<name>/<video_uid>-<batch>.tar
  annotations/recaption/schema-v1/prompt-<caption_version>/run.json
  annotations/recaption/schema-v1/prompt-<caption_version>/source=<name>/part-*.parquet
  checkpoints/
  report/recaption-<caption_version>/index.html
```

The stable key is `source|video_uid|original_frame_number`. `caption_raw` remains
null; generated captions carry `caption_version`. Source-partitioned files omit
the duplicate `source` column. Original annotations and all associations remain
in the sample manifests, separately from generated captions. Raw model responses
are saved in caption results.

Each result row has `status=ok` or `failed`; failures retain `error_kind` and
`error`, including decoding, request, output-length and invalid-JSON failures.
Parquet files are written atomically per batch. Normal reruns resume using
Curator checkpoints and skip completed files. Set `retry_failed=true` to bypass
checkpoints, regenerate extraction batches containing decoding failures, and
retry failed rows in existing caption batches while preserving successful rows.
Use a normal rerun for caption batches not yet written.

Keep sampling and batch size unchanged when resuming. Changed prompt or model
settings require a new caption version; incompatible saved settings raise an
error instead of mixing results. The static report samples each source and
success/failure category and shows the image, complete model context, original
annotations, structured caption and raw response. Serve its report directory
with a static HTTP server and forward that port in VS Code.
