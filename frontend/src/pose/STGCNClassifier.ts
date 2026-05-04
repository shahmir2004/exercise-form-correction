export const WINDOW = 30;
export const N_JOINTS = 17;
export const COORD_DIM = 3;
const HIDDEN = 64;
const N_CLASSES = 4;

export const KEY_JOINT_INDICES = [11,12,13,14,15,16,23,24,25,26,27,28,0,7,8,9,10] as const;

type ProbMap = Record<string, number> | null;

interface WeightsJSON {
  gc1_W: number[][];   // (COORD_DIM, HIDDEN)
  gc1_b: number[][];   // (N_JOINTS, HIDDEN)
  gc2_W: number[][];   // (HIDDEN, HIDDEN)
  gc2_b: number[][];   // (N_JOINTS, HIDDEN)
  fc_W:  number[][];   // (N_JOINTS*HIDDEN, N_CLASSES)
  fc_b:  number[];     // (N_CLASSES,)
  adjacency: number[][];
  labels: string[];
}

interface ScalerJSON {
  mean: number[];
  std:  number[];
}

interface LoadedWeights {
  gc1_W: Float32Array;
  gc1_b: Float32Array;
  gc2_W: Float32Array;
  gc2_b: Float32Array;
  fc_W:  Float32Array;
  fc_b:  Float32Array;
  A:     Float32Array;
  labels: string[];
  mean:  Float32Array;
  std:   Float32Array;
}

export class STGCNClassifier {
  private weights: LoadedWeights | null = null;

  get isReady(): boolean {
    return this.weights !== null;
  }

  async loadWeights(weightsUrl: string, scalerUrl?: string): Promise<void> {
    const [wJson, sJson] = await Promise.all([
      fetch(weightsUrl).then(r => r.json() as Promise<WeightsJSON & { mean?: number[]; std?: number[] }>),
      scalerUrl ? fetch(scalerUrl).then(r => r.json() as Promise<ScalerJSON>) : Promise.resolve(null),
    ]);
    const combined = {
      ...wJson,
      mean: sJson?.mean ?? wJson.mean ?? new Array(N_JOINTS * COORD_DIM).fill(0),
      std:  sJson?.std  ?? wJson.std  ?? new Array(N_JOINTS * COORD_DIM).fill(1),
    };
    this._parseWeights(combined as WeightsJSON & ScalerJSON);
  }

  loadWeightsFromObject(obj: WeightsJSON & { mean: number[]; std: number[] }): void {
    this._parseWeights(obj);
  }

  private _parseWeights(obj: WeightsJSON & { mean: number[]; std: number[] }): void {
    this.weights = {
      gc1_W: new Float32Array(obj.gc1_W.flat()),
      gc1_b: new Float32Array(obj.gc1_b.flat()),
      gc2_W: new Float32Array(obj.gc2_W.flat()),
      gc2_b: new Float32Array(obj.gc2_b.flat()),
      fc_W:  new Float32Array(obj.fc_W.flat()),
      fc_b:  new Float32Array(obj.fc_b),
      A:     new Float32Array(obj.adjacency.flat()),
      labels: obj.labels,
      mean:  new Float32Array(obj.mean),
      std:   new Float32Array(obj.std),
    };
  }

  infer(frames: Float32Array[]): ProbMap {
    if (!this.weights) return null;
    if (frames.length !== WINDOW) return null;
    const W = this.weights;

    // Normalise: input (WINDOW, N_JOINTS*COORD_DIM) -> normalised
    const x = new Float32Array(WINDOW * N_JOINTS * COORD_DIM);
    for (let t = 0; t < WINDOW; t++) {
      const frame = frames[t];
      for (let f = 0; f < N_JOINTS * COORD_DIM; f++) {
        const s = W.std[f];
        x[t * N_JOINTS * COORD_DIM + f] = s !== 0 ? (frame[f] - W.mean[f]) / s : 0;
      }
    }

    // GC Layer 1: (WINDOW, N_JOINTS, COORD_DIM) -> (WINDOW, N_JOINTS, HIDDEN)
    const H1 = this._gcLayer(x, W.A, W.gc1_W, W.gc1_b, COORD_DIM, HIDDEN);

    // GC Layer 2: (WINDOW, N_JOINTS, HIDDEN) -> (WINDOW, N_JOINTS, HIDDEN)
    const H2 = this._gcLayer(H1, W.A, W.gc2_W, W.gc2_b, HIDDEN, HIDDEN);

    // Global avg pool over time -> (N_JOINTS * HIDDEN,)
    const pooled = new Float32Array(N_JOINTS * HIDDEN);
    for (let t = 0; t < WINDOW; t++) {
      for (let i = 0; i < N_JOINTS * HIDDEN; i++) {
        pooled[i] += H2[t * N_JOINTS * HIDDEN + i] / WINDOW;
      }
    }

    // FC -> (N_CLASSES,)
    const logits = new Float32Array(N_CLASSES);
    for (let c = 0; c < N_CLASSES; c++) {
      let sum = W.fc_b[c];
      for (let i = 0; i < N_JOINTS * HIDDEN; i++) {
        sum += pooled[i] * W.fc_W[i * N_CLASSES + c];
      }
      logits[c] = sum;
    }

    const probs = this._softmax(logits);

    const out: Record<string, number> = {};
    for (let c = 0; c < N_CLASSES; c++) {
      out[W.labels[c]] = probs[c];
    }
    return out;
  }

  private _gcLayer(
    x: Float32Array,
    A: Float32Array,
    W_mat: Float32Array,
    b_mat: Float32Array,
    C_in: number,
    C_out: number,
  ): Float32Array {
    const out = new Float32Array(WINDOW * N_JOINTS * C_out);
    const AX = new Float32Array(N_JOINTS * C_in);
    for (let t = 0; t < WINDOW; t++) {
      const xOffset = t * N_JOINTS * C_in;

      // AX = A @ x[t]: (N_JOINTS, C_in)
      AX.fill(0);
      for (let i = 0; i < N_JOINTS; i++) {
        for (let k = 0; k < N_JOINTS; k++) {
          const a_ik = A[i * N_JOINTS + k];
          if (a_ik === 0) continue;
          for (let c = 0; c < C_in; c++) {
            AX[i * C_in + c] += a_ik * x[xOffset + k * C_in + c];
          }
        }
      }

      // H = AX @ W + b: (N_JOINTS, C_out) with ReLU
      const outOffset = t * N_JOINTS * C_out;
      for (let j = 0; j < N_JOINTS; j++) {
        for (let co = 0; co < C_out; co++) {
          // b_mat is (N_JOINTS, C_out) flattened
          let val = b_mat[j * C_out + co];
          for (let ci = 0; ci < C_in; ci++) {
            val += AX[j * C_in + ci] * W_mat[ci * C_out + co];
          }
          out[outOffset + j * C_out + co] = Math.max(0, val);
        }
      }
    }
    return out;
  }

  private _softmax(logits: Float32Array): Float32Array {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > max) max = logits[i];
    }
    const e = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      e[i] = Math.exp(logits[i] - max);
      sum += e[i];
    }
    for (let i = 0; i < e.length; i++) e[i] /= sum;
    return e;
  }
}
