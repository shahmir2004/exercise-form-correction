import { describe, it, expect, beforeAll } from 'vitest';
import { STGCNClassifier, WINDOW, N_JOINTS, COORD_DIM } from '../STGCNClassifier';

const LABELS = ['curl-stand', 'curl-seat', 'alt-stand', 'alt-seat'];
const N_CLASSES = 4;
const HIDDEN = 64;

function makeTestWeights() {
  const gc1_W = Array.from({ length: COORD_DIM }, () =>
    Array.from({ length: HIDDEN }, () => 0.01)
  );
  const gc2_W = Array.from({ length: HIDDEN }, () =>
    Array.from({ length: HIDDEN }, () => 0.01)
  );
  const gc1_b = Array.from({ length: N_JOINTS }, () =>
    Array.from({ length: HIDDEN }, () => 0.0)
  );
  const gc2_b = Array.from({ length: N_JOINTS }, () =>
    Array.from({ length: HIDDEN }, () => 0.0)
  );
  const fc_W = Array.from({ length: N_JOINTS * HIDDEN }, () =>
    Array.from({ length: N_CLASSES }, () => 0.01)
  );
  const fc_b = Array.from({ length: N_CLASSES }, () => 0.0);

  const adjacency = Array.from({ length: N_JOINTS }, (_, i) =>
    Array.from({ length: N_JOINTS }, (_, j) => (i === j ? 1.0 / N_JOINTS : 0.0))
  );
  const mean = Array.from({ length: N_JOINTS * COORD_DIM }, () => 0.0);
  const std = Array.from({ length: N_JOINTS * COORD_DIM }, () => 1.0);

  return { gc1_W, gc1_b, gc2_W, gc2_b, fc_W, fc_b, adjacency, labels: LABELS, mean, std };
}

function makeRandomWindow(): Float32Array[] {
  return Array.from({ length: WINDOW }, () => {
    const f = new Float32Array(N_JOINTS * COORD_DIM);
    for (let i = 0; i < f.length; i++) f[i] = Math.random() * 0.1;
    return f;
  });
}

describe('STGCNClassifier', () => {
  let clf: STGCNClassifier;

  beforeAll(() => {
    clf = new STGCNClassifier();
    clf.loadWeightsFromObject(makeTestWeights());
  });

  it('exports WINDOW=30, N_JOINTS=17, COORD_DIM=3', () => {
    expect(WINDOW).toBe(30);
    expect(N_JOINTS).toBe(17);
    expect(COORD_DIM).toBe(3);
  });

  it('isReady is true after loadWeightsFromObject', () => {
    expect(clf.isReady).toBe(true);
  });

  it('infer returns probabilities that sum to 1', () => {
    const w = makeRandomWindow();
    const probs = clf.infer(w);
    const sum = LABELS.reduce((s, l) => s + (probs as Record<string,number>)[l], 0);
    expect(sum).toBeCloseTo(1.0, 5);
  });

  it('all probabilities are non-negative', () => {
    const w = makeRandomWindow();
    const probs = clf.infer(w);
    for (const l of LABELS) {
      expect((probs as Record<string,number>)[l]).toBeGreaterThanOrEqual(0);
    }
  });

  it('infer returns all 4 label keys', () => {
    const w = makeRandomWindow();
    const probs = clf.infer(w);
    for (const l of LABELS) {
      expect(probs).toHaveProperty(l);
    }
  });

  it('infer returns null-like object before weights loaded', () => {
    const empty = new STGCNClassifier();
    expect(empty.isReady).toBe(false);
    const w = makeRandomWindow();
    const probs = empty.infer(w);
    expect(probs).toBeNull();
  });

  it('infer returns null when window has wrong length', () => {
    const bad = makeRandomWindow().slice(0, 5);
    const probs = clf.infer(bad);
    expect(probs).toBeNull();
  });
});
