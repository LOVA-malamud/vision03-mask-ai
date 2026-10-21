> Historical verification from 18 October 2026. For the final 21 October audit, security dependency changes, and unresolved phone checks, see [FINAL_REVIEW_HE.md](FINAL_REVIEW_HE.md).

# Verification — Colab model 20261017T175219Z

Story: select a face image → preview → optionally save the original → classify through FastAPI → invoke the saved/reloaded CNN → display its label and probabilities.

## Training and downloaded artifacts

- The user's Colab training had already completed: 20 epochs on T4 GPU. Recovery did not retrain or replace the saved weights.
- Best checkpoint: epoch 16 selected by validation loss. Validation accuracy 95.5994%, loss 0.1239853. Held-out test accuracy 94.1711% (1,244/1,321); loss 0.1541201. Confusion matrix in `[with_mask, without_mask]` order: `[[670,43],[34,574]]`.
- SHA-256 of the downloaded `.keras`: `14e1221d9bb28a8ece1278c3200e007f4ccde6650d642c65570a9c16bb785d79`. Verified against its metadata before installation.
- After reconnecting Drive, preprocessing/helper definitions were restored and the existing model reloaded. The user selected the public held-out `with_mask.jpg` fixture in Colab. Output: `with_mask`, P(with_mask)=0.9979398373980075. See `reports/single_image_prediction.json`.
- The downloaded notebook contains 22 cells, genuine preserved training outputs, the completed image prediction, three embedded PNG outputs, and no error outputs. Execution counters reflect multiple runtime sessions; they were not fabricated or renumbered.
- Google Drive reported a full storage quota and could not save the latest notebook edits. The complete updated notebook was downloaded locally and included in the submission. Cloud notebook synchronization remains separate from the completed local submission.

## Local inference and HTTP

- Backup created and checksum-verified before replacement: `local pre-migration backup (outside repository)`. It contains the prior project artifacts and prior submission ZIP, with `backup-manifest.json`.
- Downloaded model trained with TensorFlow 2.21.0 / Keras 3.12.0 successfully reloads on the local TensorFlow 2.16.2 / Keras 3.12.0 environment.
- The Colab and local preprocessed-image SHA-256 values match exactly. Probability difference on the same image: `9.313225746154785e-10`. See `reports/local_inference_verification.json`.
- **13 tests passed**, three dependency deprecation warnings, in 10.58 seconds. Actual saved weights used throughout. Tests cover real HTTP/direct inference parity for both class fixtures, one startup model instance, RGB/L/RGBA, normalization and EXIF, invalid and oversized uploads, mapping validation, training/inference pixel parity, primary-frame MPO JPEG decoding, and rejection of animated PNG.
- Live Uvicorn HTTP checks passed: `/api/health` identifies the new Colab model; with-mask fixture returns P(with_mask)=0.9979398383293301; without-mask fixture returns P(without_mask)=0.8163670897483826. Invalid image bytes return 422. See `reports/http_verification.json`.
- The frontend build was reused unchanged. The shared decoder was extended for still MPO JPEGs after a real phone image exposed a format rejection. It decodes only the primary image; RGB/EXIF/resize/normalization and model weights are unchanged. The notebook includes the same decoder correction and explicitly notes it was applied after training; original training outputs remain genuine. The untouched downloaded Colab notebook remains in Downloads.

## Browser verification

The current Colab model passed the real user-selected image → preview → Classify → FastAPI → reloaded CNN → rendered result flow in the Codex in-app browser. The uploaded JPEG was identified as MPO with two images and initially returned 422; after the decoder fix, the same selection returned HTTP 200 and the UI displayed **without mask 76.0% / with mask 24.0%**, 0.21 seconds. No console errors were recorded. This is model output, not a claim that the photograph was independently labeled. The user's private photo is not included in the submission.

Save image remains a separate, unchanged button. Previous verification created a byte-identical downloaded fixture (SHA-256 `83cdeb38911b00fd6959d84f60a56197a7037d496c96fa77812abcd8d8631e30`); the file still exists and its hash was rechecked. For the current private photo, clicking Save in the Codex embedded browser did not produce a confirmed download: the event timed out and no matching file was found in Downloads. Current embedded-browser saving is therefore **unverified**, not reported as passing. Use Chrome for a fresh manual save check. Gallery/camera attributes remain correct. See `reports/browser_verification.json`.

## Limits

Physical iOS/Android camera capture was not tested. Prior browser checks verified separate camera/gallery inputs (`accept=image/* capture=user`) and a 390×844 responsive layout; those controls were unchanged. GitHub publication, instructor submission, ngrok, public deployment and unknown-under-60% are not completed or claimed. The dataset split does not guarantee separation by person identity; this is a whole-image classifier, not a face detector.
