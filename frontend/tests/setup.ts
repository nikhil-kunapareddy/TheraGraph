import '@testing-library/jest-dom/vitest';
import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// jsdom has no canvas, and the structure renderer draws to one. Stubbing the context
// keeps molecule components mountable in tests without pulling in a canvas polyfill.
HTMLCanvasElement.prototype.getContext = vi.fn(() => null) as never;

// Recharts measures its container; jsdom reports zero, which suppresses rendering.
Object.defineProperty(HTMLElement.prototype, 'offsetWidth', { configurable: true, value: 800 });
Object.defineProperty(HTMLElement.prototype, 'offsetHeight', { configurable: true, value: 400 });
global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as never;
