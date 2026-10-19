// 高精度文字起こし(Whisper)をブラウザ内で動かすワーカー。音声は外部に送られない。
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1';

let asr = null, loadedKey = '';

self.onmessage = async ({ data: d }) => {
  try {
    if (d.type === 'load') {
      const key = `${d.model}|${d.device}|${JSON.stringify(d.dtype)}`;
      if (key !== loadedKey) {
        try { await asr?.dispose?.(); } catch { }
        asr = null; loadedKey = '';
        const files = {};
        asr = await pipeline('automatic-speech-recognition', d.model, {
          device: d.device,
          dtype: d.dtype,
          progress_callback: p => {
            if (p.status !== 'progress' || !p.total) return;
            files[p.file] = { l: p.loaded, t: p.total };
            let l = 0, t = 0;
            for (const f of Object.values(files)) { l += f.l; t += f.t; }
            self.postMessage({ type: 'progress', loaded: l, total: t });
          },
        });
        loadedKey = key;
      }
      self.postMessage({ type: 'ready' });
    } else if (d.type === 'run') {
      const out = await asr(d.audio, { language: d.language, task: 'transcribe', return_timestamps: true });
      self.postMessage({ type: 'result', text: out.text || '', chunks: out.chunks || [] });
    }
  } catch (e) {
    self.postMessage({ type: 'error', message: String(e?.message || e) });
  }
};
