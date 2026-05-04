import { describe, it, expect } from 'vitest';
import { buildPoseMessage } from '../usePoseStream';

describe('buildPoseMessage', () => {
  it('includes landmarks and timestamp', () => {
    const lms = [{ x: 0.1, y: 0.2, z: 0.0, visibility: 1.0 }];
    const msg = JSON.parse(buildPoseMessage(lms, 100, null));
    expect(msg.landmarks).toHaveLength(1);
    expect(msg.timestamp).toBe(100);
    expect(msg.client_probs).toBeUndefined();
  });

  it('includes client_probs when provided', () => {
    const lms = [{ x: 0.1, y: 0.2, z: 0.0, visibility: 1.0 }];
    const probs = { 'curl-stand': 0.8, 'curl-seat': 0.1, 'alt-stand': 0.05, 'alt-seat': 0.05 };
    const msg = JSON.parse(buildPoseMessage(lms, 200, probs));
    expect(msg.client_probs).toEqual(probs);
  });

  it('omits client_probs key entirely when null', () => {
    const lms = [{ x: 0.5, y: 0.5, z: 0.0, visibility: 0.9 }];
    const msg = JSON.parse(buildPoseMessage(lms, 300, null));
    expect('client_probs' in msg).toBe(false);
  });
});
