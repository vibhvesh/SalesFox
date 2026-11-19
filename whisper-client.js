import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL = "onnx-community/whisper-tiny";
let transcriber = null;
let loading = false;
let recorder = null;
let chunks = [];
let currentMode = "";
let currentLanguage = "auto";

const state = document.getElementById("whisperState");
const loadButton = document.getElementById("loadWhisper");

function setState(message) {
  if (state) state.textContent = message;
  ["trainingStatus", "micStatus"].forEach(id => {
    const el = document.getElementById(id);
    if (el && message) el.dataset.whisper = message;
  });
}

async function loadWhisper() {
  if (transcriber || loading) return;
  loading = true;
  loadButton.disabled = true;
  loadButton.textContent = "Loading Whisper…";
  setState("Downloading multilingual Whisper Tiny model. Keep this tab open…");
  try {
    env.allowLocalModels = false;
    const progress_callback = (p) => {
      if (p.status === "progress" && p.total) {
        const pct = Math.round((p.loaded / p.total) * 100);
        setState("Downloading model: " + pct + "%");
      } else if (p.status === "progress") {
        setState("Downloading Whisper model files…");
      } else if (p.status === "ready") {
        setState("Preparing local speech engine…");
      }
    };
    if (navigator.gpu) {
      try {
        transcriber = await pipeline("automatic-speech-recognition", MODEL, {
          device: "webgpu",
          dtype: "q4",
          progress_callback
        });
      } catch (gpuError) {
        console.warn("Whisper WebGPU unavailable; falling back to WASM", gpuError);
        transcriber = null;
      }
    }
    if (!transcriber) {
      transcriber = await pipeline("automatic-speech-recognition", MODEL, {
        dtype: "q8",
        progress_callback
      });
    }
    window.salesFoxWhisperReady = true;
    loadButton.textContent = "✓ Whisper ready";
    setState("Whisper is ready on this device. Model files are cached by the browser.");
    document.dispatchEvent(new CustomEvent("salesfox-whisper-ready"));
  } catch (error) {
    console.error("Whisper model load failed", error);
    loadButton.disabled = false;
    loadButton.textContent = "Retry loading Whisper";
    setState("Could not load Whisper: " + error.message);
  } finally {
    loading = false;
  }
}

async function audioTo16k(blob) {
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await blob.arrayBuffer());
    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    return rendered.getChannelData(0);
  } finally {
    await context.close();
  }
}

async function transcribe(audio, language) {
  if (!transcriber) throw new Error("Load Whisper on this device first.");
  const options = { task: "transcribe" };
  if (language && language !== "auto") options.language = language;
  const result = await transcriber(audio, options);
  return (result.text || "").trim();
}

function playFeedbackTone(frequency, duration) {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const context = new AudioCtx();
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.frequency.value = frequency;
    oscillator.type = "sine";
    gain.gain.setValueAtTime(0.0001, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.12, context.currentTime + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + duration / 1000);
    oscillator.connect(gain);
    gain.connect(context.destination);
    oscillator.start();
    oscillator.stop(context.currentTime + duration / 1000 + 0.02);
    oscillator.onended = () => context.close();
  } catch (error) { console.debug("PTT audio feedback unavailable", error); }
}

function startRecording(mode) {
  if (!transcriber) {
    setState("Tap Load Whisper on this device first.");
    return;
  }
  if (!window.salesFoxMicStream) {
    setState("Tap Enable microphone and allow access first.");
    return;
  }
  if (!window.MediaRecorder) {
    setState("Audio recording is not supported in this browser.");
    return;
  }
  if (recorder && recorder.state === "recording") return;
  currentMode = mode;
  const languageSelect = document.getElementById(mode === "training" ? "trainingLanguage" : "billingLanguage");
  currentLanguage = languageSelect ? languageSelect.value : "auto";
  chunks = [];
  const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "";
  recorder = new MediaRecorder(window.salesFoxMicStream, mime ? { mimeType: mime } : {});
  recorder.ondataavailable = event => {
    if (event.data && event.data.size) chunks.push(event.data);
  };
  recorder.onstop = async () => {
    const captured = chunks;
    chunks = [];
    const activeMode = currentMode;
    recorder = null;
    setState("Transcribing on this device with Whisper…");
    try {
      const blob = new Blob(captured, { type: captured[0]?.type || "audio/webm" });
      const audio = await audioTo16k(blob);
      const text = await transcribe(audio, currentLanguage);
      if (!text) throw new Error("No speech detected. Try speaking closer to the microphone.");
      document.dispatchEvent(new CustomEvent("salesfox-transcript", {
        detail: { mode: activeMode, text }
      }));
    } catch (error) {
      console.error(error);
      const id = activeMode === "training" ? "trainingStatus" : "micStatus";
      const el = document.getElementById(id);
      if (el) el.textContent = "Whisper error: " + error.message;
      playFeedbackTone(220, 260);
    }
  };
  recorder.start();
  playFeedbackTone(880, 90);
  const status = document.getElementById(mode === "training" ? "trainingStatus" : "micStatus");
  if (status) status.textContent = "● RECORDING · Speak one product, then press PTT again. Audio stays on this device.";
  const button = document.getElementById(mode === "training" ? "trainingMic" : "mic");
  if (button) button.classList.add("active");
}

function stopRecording(mode) {
  if (recorder && recorder.state === "recording" && currentMode === mode) {
    recorder.stop();
    playFeedbackTone(520, 120);
    const button = document.getElementById(mode === "training" ? "trainingMic" : "mic");
    if (button) button.classList.remove("active");
    const status = document.getElementById(mode === "training" ? "trainingStatus" : "micStatus");
    if (status) status.textContent = "PTT OFF · Processing locally…";
  }
}

loadButton?.addEventListener("click", loadWhisper);
// Download and initialize Whisper automatically during page startup. The button remains available for retry.
loadWhisper();
document.getElementById("trainingMic")?.addEventListener("click", () => {
  if (recorder?.state === "recording") stopRecording("training");
  else startRecording("training");
});
document.getElementById("mic")?.addEventListener("click", () => {
  if (recorder?.state === "recording") stopRecording("billing");
  else startRecording("billing");
});

document.addEventListener("salesfox-transcript", event => {
  const { mode, text } = event.detail;
  if (mode === "training") {
    if (typeof window.submitTrainingAnswer === "function") window.submitTrainingAnswer(text);
    const status = document.getElementById("trainingStatus");
    if (status) status.textContent = "✓ Whisper transcribed: " + text;
  } else {
    const status = document.getElementById("micStatus");
    if (status) status.textContent = "✓ Whisper transcribed: " + text;
    if (typeof window.salesFoxCommand === "function") window.salesFoxCommand(text);
  }
});
