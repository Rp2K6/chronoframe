# ChronoFrame — Master Bug Hunt, Fix & Validation Report

**Date:** October 2026  
**Scope:** Complete Backend, API, Math, Video, Security, and Frontend Quality Assurance Pass  
**Status:** All Discovered Bugs Resolved & Verified  

---

## 1. Bug Inventory & Detailed Classifications

### BUG #1: Unsafe `float('inf')` / `Infinity` JSON Serialization Crash
- **BUG ID:** `BUG-01`
- **SEVERITY:** **CRITICAL**
- **COMPONENT:** `utils/metrics.py`, `models/pipeline.py`, `web/app.py`
- **SYMPTOM:** When PSNR calculation produces infinite output (MSE = 0), standard Python `json.dumps()` or Flask `jsonify()` outputs the non-standard JSON token `Infinity` (`"psnr": Infinity`). The browser's native JavaScript `JSON.parse()` crashes with `SyntaxError: Unexpected token 'I', "... "psnr":Infinity ..." is not valid JSON`.
- **REPRODUCTION:** Generate a video where spatial keyframe reconstruction achieves identical values, then poll `/api/status/<job_id>`.
- **ROOT CAUSE:** Division by zero in PSNR ($10 \cdot \log_{10}(1 / 0)$) returns Python `float('inf')`. Python's JSON serializer outputs non-RFC 8259 compliant `Infinity` keywords.
- **FIX:**
  1. Updated `utils/metrics.py` to return `None` (JSON `null`) whenever MSE = 0 or value is infinite/NaN.
  2. Implemented `sanitize_json_obj(data)` in `web/app.py` to recursively scrub any floating-point `Inf` or `NaN` values before returning API responses.
  3. Updated `web/static/app.js` to render `null` PSNR as `"∞ dB (Identical)"` or `"Perfect (∞ dB)"`.
- **VERIFICATION STATUS:** **FIXED & VERIFIED** (API returns standard `null`; browser JSON parsing succeeds cleanly).

---

### BUG #2: Identical Reference Tensors in PSNR / SSIM Calculation
- **BUG ID:** `BUG-02`
- **SEVERITY:** **HIGH**
- **COMPONENT:** `models/pipeline.py`
- **SYMPTOM:** PSNR was perpetually reported as `inf` and SSIM as `1.0000` for both Baseline and Proposed modes.
- **REPRODUCTION:** Run `run_demo.py` and observe that PSNR/SSIM metrics showed no distinction between modes.
- **ROOT CAUSE:** Line 115 in `models/pipeline.py` computed `ref_rgb = decoder_kf(full_latents)`, using the exact same latents and decoder as `video_rgb = self.temporal_decoder(full_latents)`. Comparing a tensor to itself produced MSE = 0 every time.
- **FIX:** Updated `models/pipeline.py` to compare generated video frames at keyframe timestamps ($t = 0, 4, 8, 12, ...$) against the original spatial keyframe decoded RGB images `keyframe_rgb`.
- **VERIFICATION STATUS:** **FIXED & VERIFIED** (PSNR and SSIM evaluate real keyframe preservation quality).

---

### BUG #3: Windows Console Charmap Codec Encoding Crash
- **BUG ID:** `BUG-03`
- **SEVERITY:** **HIGH**
- **COMPONENT:** `models/pipeline.py`
- **SYMPTOM:** CLI execution on Windows crashed with `UnicodeEncodeError: 'charmap' codec can't encode character '\u221e' in position 42`.
- **REPRODUCTION:** Run `python run_demo.py` on standard Windows Command Prompt or PowerShell with default `cp1252` encoding.
- **ROOT CAUSE:** Terminal `print()` statement contained literal unicode infinity character `∞` (`\u221e`), which is unmappable in Windows `cp1252` code page.
- **FIX:** Replaced literal `∞` symbol in terminal logging strings with ASCII-compatible `"inf dB (Identical)"`.
- **VERIFICATION STATUS:** **FIXED & VERIFIED** (CLI executes without unicode charmap errors on Windows).

---

### BUG #4: Directory Traversal Security Vulnerability in File Serving Endpoint
- **BUG ID:** `BUG-04`
- **SEVERITY:** **HIGH**
- **COMPONENT:** `web/app.py` (`serve_output`)
- **SYMPTOM:** Crafted requests such as `GET /outputs/../README.md` could traverse outside the intended `outputs/` directory.
- **REPRODUCTION:** Request `/outputs/../README.md` via HTTP client.
- **ROOT CAUSE:** `web/app.py` joined `filename` directly into `os.path.join(res_dir, filename)` without sanitization.
- **FIX:** Added `werkzeug.utils.secure_filename` validation and strict root path restriction check `os.path.commonpath([res_dir, file_path]) == res_dir`.
- **VERIFICATION STATUS:** **FIXED & VERIFIED** (Path traversal requests return `400 Invalid filename` or `404 File not found`).

---

### BUG #5: Unhandled Input Parameter Type Error in API Endpoints
- **BUG ID:** `BUG-05`
- **SEVERITY:** **MEDIUM**
- **COMPONENT:** `web/app.py` (`api_generate`, `api_compare`)
- **SYMPTOM:** Passing non-numeric string values or `null` for numerical parameters (e.g., `{"keyframes": "invalid"}`) triggered an unhandled 500 Internal Server Error.
- **REPRODUCTION:** Send `POST /api/generate` with `{"keyframes": "invalid_number"}`.
- **ROOT CAUSE:** Unprotected `int(data.get("keyframes"))` call without try-except error handling.
- **FIX:** Created `parse_int_param(val, default_val, min_val, max_val)` helper in `web/app.py` providing robust type parsing and value clamping ($1 \le T \le 8$, $64 \le \text{res} \le 512$, $1 \le \text{fps} \le 60$).
- **VERIFICATION STATUS:** **FIXED & VERIFIED** (Invalid parameters fall back safely to defaults; returns 200 OK).

---

### BUG #6: Concurrency Race Condition in Job Dictionary
- **BUG ID:** `BUG-06`
- **SEVERITY:** **MEDIUM**
- **COMPONENT:** `web/app.py`
- **SYMPTOM:** Simultaneous background generation requests mutating global `jobs` dictionary caused potential dictionary mutation race conditions.
- **REPRODUCTION:** Trigger multiple simultaneous `POST /api/generate` requests.
- **ROOT CAUSE:** Unsynchronized shared state in multi-threaded Flask environment.
- **FIX:** Wrapped `jobs` dictionary reads and writes with `jobs_lock = threading.Lock()`.
- **VERIFICATION STATUS:** **FIXED & VERIFIED** (Thread-safe concurrent job processing verified).

---

### BUG #7: Tensor Slice Crash on Single Keyframe ($T=1$) Edge Case
- **BUG ID:** `BUG-07`
- **SEVERITY:** **MEDIUM**
- **COMPONENT:** `models/interpolation.py`
- **SYMPTOM:** Setting `num_keyframes = 1` crashed with `IndexError` during keyframe pair tensor slicing.
- **REPRODUCTION:** Execute `pipeline.generate(..., num_keyframes=1)`.
- **ROOT CAUSE:** `keyframes[:, :-1]` evaluates to an empty tensor when $T=1$.
- **FIX:** Added guard condition in `JointInterpolator.forward`: `if T <= 1: return keyframes`.
- **VERIFICATION STATUS:** **FIXED & VERIFIED** ($T=1$ returns 1 keyframe without crashing).

---

### BUG #8: Unreset Progress Bar Background Styling on Error Recovery
- **BUG ID:** `BUG-08`
- **SEVERITY:** **LOW**
- **COMPONENT:** `web/static/app.js`
- **SYMPTOM:** If a generation failed, the progress bar background turned red (`#ef4444`). Subsequent generation attempts left the red background styling active.
- **REPRODUCTION:** Trigger an error, then click Generate again.
- **ROOT CAUSE:** `startGeneration()` did not reset `progressBar.style.background`.
- **FIX:** Added explicit `progressBar.style.background = 'var(--accent-gradient)'` reset in `startGeneration()`.
- **VERIFICATION STATUS:** **FIXED & VERIFIED** (Progress bar style resets cleanly).

---

## 2. Quantitative Summary Statistics

- **TOTAL BUGS FOUND:** 8
- **TOTAL BUGS FIXED:** 8
- **CRITICAL REMAINING:** 0
- **HIGH REMAINING:** 0
- **MEDIUM REMAINING:** 0
- **LOW REMAINING:** 0

| Category / Component | Status | Metrics / Details |
|---|---|---|
| **PYTEST TEST SUITE** | **PASSED** | 17 passed in 2.35s |
| **API & BACKEND** | **PASSED** | 100% RFC 8259 JSON compliant (No Inf/NaN tokens) |
| **SECURITY** | **PASSED** | Path traversal protected |
| **PIPELINE & MATH** | **PASSED** | $T \to 4T-3$ verified ($1\to1$, $2\to5$, $4\to13$, $6\to21$) |
| **VIDEO GENERATION** | **PASSED** | Verified FMP4 / mp4v container (100% Playable) |
| **METRICS EVALUATION**| **PASSED** | PSNR, SSIM, Temporal Flicker, CLIP |
| **FRONTEND UI** | **PASSED** | Dark-first responsive UI on `http://localhost:5000` |
| **DOWNLOAD & BENCHMARK**| **PASSED** | Direct MP4 downloads and side-by-side mode benchmark |
| **CPU LATENCY** | **PASSED** | < 0.8s total generation time |

---

## 3. Verification Checklist

[x] Current Infinity JSON bug fixed  
[x] No NaN / Infinity in API JSON  
[x] No critical bugs  
[x] No high bugs  
[x] All pytest unit & integration tests pass (17/17)  
[x] API endpoints work cleanly  
[x] Frontend Web UI runs on `http://localhost:5000`  
[x] Baseline pipeline works  
[x] ChronoFrame pipeline works  
[x] Temporal mathematics ($T \to 4T-3$) verified  
[x] Generated MP4 videos are valid and playable (Verified codec: FMP4 / mp4v)  
[x] Evaluation metrics validated  
[x] Direct MP4 download works  
[x] Benchmark comparison works  
[x] Reset and error recovery work  
[x] Concurrency does not corrupt jobs  
[x] CPU execution verified  
[x] No personal machine secrets or credentials tracked  
[x] Documentation matches implementation  
