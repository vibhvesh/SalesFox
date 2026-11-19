import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL = "onnx-community/whisper-tiny";
let transcriber = null;
let loading = false;
let audioContext = null;
let micStream = null;
let source = null;
let processor = null;
let silentGain = null;
let isRecording = false;
let chunks = [];
let sampleRate = 48000;
let timer = null;
let startedAt = 0;
const queue = [];
let processing = false;

function emit(state, detail = {}) {
  window.dispatchEvent(new CustomEvent("salesfox-audio", { detail: { state, ...detail } }));
}
function feedback(freq, duration) {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.08, ctx.currentTime + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration / 1000);
    osc.connect(gain); gain.connect(ctx.destination);
    osc.start(); osc.stop(ctx.currentTime + duration / 1000 + 0.02);
    osc.onended = () => ctx.close();
  } catch (_) {}
}
async function loadEngine() {
  if (transcriber || loading) return;
  loading = true;
  emit("model-loading", { message: "Loading on-device Whisper speech engine…" });
  try {
    env.allowLocalModels = false;
    const opts = { progress_callback: p => {
      if (p.status === "progress" && p.total) emit("model-progress", { message: "Downloading Whisper " + Math.round(p.loaded / p.total * 100) + "%" });
      else if (p.status === "ready") emit("model-progress", { message: "Preparing speech engine…" });
    }};
    if (navigator.gpu) {
      try { transcriber = await pipeline("automatic-speech-recognition", MODEL, { ...opts, device: "webgpu", dtype: "q4" }); }
      catch (e) { console.warn("WebGPU unavailable; using WASM", e); }
    }
    if (!transcriber) transcriber = await pipeline("automatic-speech-recognition", MODEL, { ...opts, dtype: "q8" });
    emit("model-ready", { message: "Whisper ready · audio is transcribed on this device" });
    processQueue();
  } catch (e) {
    emit("model-error", { message: "Whisper could not load: " + e.message });
  } finally { loading = false; }
}
function mergeChunks() {
  const length = chunks.reduce((n, a) => n + a.length, 0);
  const out = new Float32Array(length);
  let offset = 0;
  for (const c of chunks) { out.set(c, offset); offset += c.length; }
  return out;
}
function resample(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const length = Math.floor(input.length * toRate / fromRate);
  const output = new Float32Array(length);
  const ratio = fromRate / toRate;
  for (let i = 0; i < length; i++) {
    const pos = i * ratio, left = Math.floor(pos), frac = pos - left;
    const a = input[left] || 0, b = input[Math.min(left + 1, input.length - 1)] || 0;
    output[i] = a + (b - a) * frac;
  }
  return output;
}
async function startCapture() {
  if (isRecording) return;
  try {
    emit("mic-opening");
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx || !navigator.mediaDevices?.getUserMedia) throw new Error("Audio capture is not supported. Open this HTTPS page in Chrome.");
    audioContext = new Ctx();
    await audioContext.resume();
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    sampleRate = audioContext.sampleRate;
    source = audioContext.createMediaStreamSource(micStream);
    processor = audioContext.createScriptProcessor(2048, 1, 1);
    silentGain = audioContext.createGain();
    silentGain.gain.value = 0;
    chunks = [];
    processor.onaudioprocess = event => {
      if (!isRecording) return;
      chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    source.connect(processor);
    processor.connect(silentGain);
    silentGain.connect(audioContext.destination);
    isRecording = true;
    startedAt = Date.now();
    feedback(880, 90);
    emit("recording", { startedAt, sampleRate });
    timer = setTimeout(stopCapture, 5000);
  } catch (e) {
    cleanup();
    emit("mic-error", { message: e.message || String(e) });
  }
}
function cleanup() {
  if (timer) { clearTimeout(timer); timer = null; }
  if (processor) { processor.onaudioprocess = null; try { processor.disconnect(); } catch (_) {} processor = null; }
  if (source) { try { source.disconnect(); } catch (_) {} source = null; }
  if (silentGain) { try { silentGain.disconnect(); } catch (_) {} silentGain = null; }
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  if (audioContext) { audioContext.close().catch(() => {}); audioContext = null; }
}
function stopCapture() {
  if (!isRecording) return;
  isRecording = false;
  const durationMs = Date.now() - startedAt;
  const audio = resample(mergeChunks(), sampleRate, 16000);
  chunks = [];
  cleanup();
  feedback(520, 120);
  if (!audio.length) { emit("capture-error", { message: "No microphone audio frames were captured. Check microphone permission." }); return; }
  const job = { id: String(startedAt), audio, durationMs, sampleRate: 16000, queuedAt: Date.now() };
  queue.push(job);
  emit("queued", { id: job.id, durationMs, samples: audio.length, queueLength: queue.length });
  processQueue();
}
async function processQueue() {
  if (processing || !transcriber || !queue.length) return;
  processing = true;
  const job = queue[0];
  emit("transcribing", { id: job.id, message: "Whisper is transcribing the recorded audio…" });
  try {
    const result = await transcriber(job.audio, { task: "transcribe" });
    const text = (result.text || "").trim();
    queue.shift();
    emit("transcript", { id: job.id, text, durationMs: job.durationMs, queueLength: queue.length });
  } catch (e) {
    queue.shift();
    emit("transcription-error", { id: job.id, message: e.message || String(e), queueLength: queue.length });
  } finally {
    processing = false;
    if (queue.length) processQueue();
  }
}
window.SalesFoxAudio = { start: startCapture, stop: stopCapture, isRecording: () => isRecording };
const ptt = document.getElementById("speak");
if (ptt) ptt.addEventListener("click", () => { if (isRecording) stopCapture(); else startCapture(); });
loadEngine();
