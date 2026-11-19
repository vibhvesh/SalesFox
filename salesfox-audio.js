import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const RATE = 12000;
const MODEL = "onnx-community/whisper-tiny";
let engine = null, loading = false, audioContext = null, stream = null, source = null, processor = null, mute = null;
let active = false, samples = [], startedAt = 0, maxTimer = null, queue = [], busy = false;
const button = document.getElementById("speak");
const emit = (state, data={}) => window.dispatchEvent(new CustomEvent("salesfox-audio",{detail:{state,...data}}));
function tone(freq, ms) {
  try {
    const C = window.AudioContext || window.webkitAudioContext, c = new C(), o = c.createOscillator(), g = c.createGain();
    o.frequency.value=freq; g.gain.setValueAtTime(.0001,c.currentTime); g.gain.exponentialRampToValueAtTime(.07,c.currentTime+.01);
    g.gain.exponentialRampToValueAtTime(.0001,c.currentTime+ms/1000); o.connect(g);g.connect(c.destination);o.start();o.stop(c.currentTime+ms/1000+.02);o.onended=()=>c.close();
  } catch (_) {}
}
function downsampleInt16(input, fromRate) {
  const ratio=fromRate/RATE, n=Math.floor(input.length/ratio), out=new Int16Array(n);
  for(let i=0;i<n;i++){const v=Math.max(-1,Math.min(1,input[Math.floor(i*ratio)]||0));out[i]=v<0?v*32768:v*32767;}
  return out;
}
function wavFromChunks(chunks) {
  const n=chunks.reduce((a,c)=>a+c.length,0), buffer=new ArrayBuffer(44+n*2), v=new DataView(buffer);let p=0;
  const str=x=>{for(let i=0;i<x.length;i++)v.setUint8(p++,x.charCodeAt(i));};
  const u16=x=>{v.setUint16(p,x,true);p+=2;},u32=x=>{v.setUint32(p,x,true);p+=4;};
  str("RIFF");u32(36+n*2);str("WAVE");str("fmt ");u32(16);u16(1);u16(1);u32(RATE);u32(RATE*2);u16(2);u16(16);str("data");u32(n*2);
  chunks.forEach(c=>c.forEach(x=>{v.setInt16(p,x,true);p+=2;}));
  return new Blob([buffer],{type:"audio/wav"});
}
async function loadWhisper(){
  if(engine||loading)return;loading=true;emit("model-loading",{message:"Loading on-device Whisper…"});
  try{
    env.allowLocalModels=false;
    const opts={progress_callback:p=>{if(p.status==="progress"&&p.total)emit("model-progress",{message:"Downloading Whisper "+Math.round(p.loaded/p.total*100)+"%"});}};
    if(navigator.gpu){try{engine=await pipeline("automatic-speech-recognition",MODEL,{...opts,device:"webgpu",dtype:"q4"});}catch(e){console.warn("WebGPU fallback",e);}}
    if(!engine)engine=await pipeline("automatic-speech-recognition",MODEL,{...opts,dtype:"q8"});
    emit("model-ready",{message:"Whisper ready · local transcription"});
    processQueue();
  }catch(e){emit("model-error",{message:"Whisper load failed: "+e.message});}
  finally{loading=false;}
}
async function openMic(){
  if(stream&&audioContext&&audioContext.state!=="closed")return;
  const C=window.AudioContext||window.webkitAudioContext;if(!C||!navigator.mediaDevices?.getUserMedia)throw new Error("Microphone capture unavailable. Open this HTTPS page in Chrome.");
  audioContext=new C();await audioContext.resume();
  stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
  source=audioContext.createMediaStreamSource(stream);processor=audioContext.createScriptProcessor(4096,1,1);mute=audioContext.createGain();mute.gain.value=0;
  processor.onaudioprocess=e=>{if(!active)return;const input=e.inputBuffer.getChannelData(0);samples.push(downsampleInt16(input,audioContext.sampleRate));};
  source.connect(processor);processor.connect(mute);mute.connect(audioContext.destination);
}
function releaseMic(){
  if(processor){processor.onaudioprocess=null;try{processor.disconnect();}catch(_){}processor=null;}
  if(source){try{source.disconnect();}catch(_){}source=null;}
  if(mute){try{mute.disconnect();}catch(_){}mute=null;}
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
  if(audioContext){audioContext.close().catch(()=>{});audioContext=null;}
}
async function begin(){
  if(active)return;
  emit("mic-opening");
  try{
    await openMic();
    samples=[];startedAt=Date.now();active=true;
    tone(880,90);emit("recording",{startedAt});
    maxTimer=setTimeout(()=>end("five-second limit"),5000);
  }catch(e){releaseMic();emit("mic-error",{message:e.message||String(e)});}
}
function end(reason="PTT released"){
  if(!active)return;
  active=false;if(maxTimer){clearTimeout(maxTimer);maxTimer=null;}
  const chunks=samples;samples=[];const wav=wavFromChunks(chunks);
  const id=String(startedAt),job={id,startedAt,endedAt:Date.now(),durationMs:Date.now()-startedAt,wav};
  releaseMic();tone(520,120);
  if(wav.size<=44){emit("capture-error",{id,message:"No microphone audio frames captured. Check microphone permission."});return;}
  queue.push(job);emit("queued",{id,durationMs:job.durationMs,bytes:wav.size,queueLength:queue.length});
  processQueue();
}
async function processQueue(){
  if(busy||!engine||!queue.length)return;busy=true;const job=queue[0];
  emit("transcribing",{id:job.id,message:"Whisper is transcribing the captured PTT audio…"});
  try{
    const decoded=await audioContextForDecode(job.wav);
    const result=await engine(decoded,{sampling_rate:RATE,task:"transcribe"});
    const text=(result.text||"").trim();queue.shift();emit("transcript",{id:job.id,text,durationMs:job.durationMs,queueLength:queue.length});
  }catch(e){queue.shift();emit("transcription-error",{id:job.id,message:e.message||String(e),queueLength:queue.length});}
  finally{busy=false;if(queue.length)processQueue();}
}
async function audioContextForDecode(blob){
  const C=window.AudioContext||window.webkitAudioContext,c=new C();
  try{const decoded=await c.decodeAudioData(await blob.arrayBuffer());return decoded.getChannelData(0).slice();}
  finally{await c.close();}
}
function pttDown(e){if(e){e.preventDefault();if(button.setPointerCapture&&e.pointerId!==undefined){try{button.setPointerCapture(e.pointerId);}catch(_){}}}if(active)return;begin();}
function pttUp(e){if(e)e.preventDefault();if(active)end("PTT released");}
if(button){
  button.addEventListener("pointerdown",pttDown);
  ["pointerup","pointercancel","lostpointercapture"].forEach(type=>button.addEventListener(type,pttUp));
  button.addEventListener("contextmenu",e=>e.preventDefault());
  button.addEventListener("keydown",e=>{if((e.code==="Space"||e.code==="Enter")&&!e.repeat){e.preventDefault();begin();}});
  button.addEventListener("keyup",e=>{if(e.code==="Space"||e.code==="Enter"){e.preventDefault();end("key released");}});
}
window.SalesFoxAudio={start:begin,stop:end,isRecording:()=>active};
loadWhisper();
