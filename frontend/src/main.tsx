import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';
import './index.css';
import { ApiClientError } from './lib/api';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Scientific data changes only when ingestion runs, so a minute of staleness is
      // correct and saves a lot of round-trips while a researcher navigates.
      staleTime: 60_000,
      gcTime: 10 * 60_000,
      refetchOnWindowFocus: false,
      retry: (failureCount, error) => {
        // Never retry a 404 or a validation error: the answer will not change.
        if (error instanceof ApiClientError && !error.retryable) return false;
        return failureCount < 2;
      },
    },
  },
});

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
