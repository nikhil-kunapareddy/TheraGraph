/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Chart surfaces and ink, from the validated reference palette (dark steps).
        surface: {
          DEFAULT: '#1a1a19',
          raised: '#222221',
          sunken: '#0d0d0d',
          border: '#33332f',
        },
        ink: {
          DEFAULT: '#ffffff',
          secondary: '#c3c2b7',
          muted: '#8a897e',
        },
        // Categorical slots, assigned in fixed order and never cycled.
        series: {
          1: '#3987e5',
          2: '#d95926',
          3: '#199e70',
          4: '#c98500',
          5: '#d55181',
          6: '#008300',
          7: '#9085e9',
          8: '#e66767',
        },
        status: {
          good: '#0ca30c',
          warning: '#fab219',
          serious: '#ec835a',
          critical: '#d03b3b',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
    },
  },
  plugins: [],
};
