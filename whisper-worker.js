// ブラウザ内でAIを動かすワーカー。音声は外部に送られない。
// - 高精度文字起こし(Whisper)
// - 話者分け用の声の特徴(WavLM)
import { pipeline, AutoProcessor, WavLMForXVector } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1';

const SV_MODEL = 'Xenova/wavlm-base-plus-sv';
let asr = null, loadedKey = '';
let sv = null, svProc = null, svKey = '';

function progress() {
  const files = {};
  return p => {
    if (p.status !== 'progress' || !p.total) return;
    files[p.file] = { l: p.loaded, t: p.total };
    let l = 0, t = 0;
    for (const f of Object.values(files)) { l += f.l; t += f.t; }
    self.postMessage({ type: 'progress', loaded: l, total: t });
  };
}

self.onmessage = async ({ data: d }) => {
  try {
    if (d.type === 'load') {
      const key = `${d.model}|${d.device}|${JSON.stringify(d.dtype)}`;
      if (key !== loadedKey) {
        try { await asr?.dispose?.(); } catch { }
        asr = null; loadedKey = '';
        asr = await pipeline('automatic-speech-recognition', d.model, {
          device: d.device, dtype: d.dtype, progress_callback: progress(),
        });
        loadedKey = key;
      }
      self.postMessage({ type: 'ready' });
    } else if (d.type === 'run') {
      const out = await asr(d.audio, { language: d.language, task: 'transcribe', return_timestamps: true });
      self.postMessage({ type: 'result', text: out.text || '', chunks: out.chunks || [] });
    } else if (d.type === 'sv-load') {
      const key = `${d.device}|${d.dtype}`;
      if (key !== svKey) {
        try { await sv?.dispose?.(); } catch { }
        sv = null; svKey = '';
        svProc ||= await AutoProcessor.from_pretrained(SV_MODEL);
        sv = await WavLMForXVector.from_pretrained(SV_MODEL, {
          device: d.device, dtype: d.dtype, progress_callback: progress(),
        });
        svKey = key;
      }
      self.postMessage({ type: 'ready' });
    } else if (d.type === 'embed') {
      const inputs = await svProc(d.audio);
      const { embeddings } = await sv(inputs);
      const v = Float32Array.from(embeddings.data);
      let n = 0;
      for (const x of v) n += x * x;
      n = Math.sqrt(n) || 1;
      for (let i = 0; i < v.length; i++) v[i] /= n;
      self.postMessage({ type: 'embedding', vec: v }, [v.buffer]);
    }
  } catch (e) {
    self.postMessage({ type: 'error', message: String(e?.message || e) });
  }
};
