import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import {
  useSessions,
  useCurrentSession,
  useNewSession,
  useSwitchSession,
  useDeleteSession,
  useSessionHistory,
} from './api'
import * as useApiModule from '../hooks/useApi'

// ── Mock apiFetch ───────────────────────────────────────────────────
vi.mock('../hooks/useApi', () => ({
  apiFetch: vi.fn(),
  getToken: vi.fn(() => null),
  setToken: vi.fn(),
}))

const mockedApiFetch = vi.mocked(useApiModule.apiFetch)

// ── Wrapper ─────────────────────────────────────────────────────────
function createWrapper() {
  const qc = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  })
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  }
}

// ── Tests ──────────────────────────────────────────────────────────

beforeEach(() => {
  vi.clearAllMocks()
})

describe('useSessions', () => {
  it('has correct query key', () => {
    const { result } = renderHook(() => useSessions(), { wrapper: createWrapper() })
    expect(result.current.data).toBeUndefined() // hasn't fetched yet
  })

  it('fetches from /session/list', async () => {
    mockedApiFetch.mockResolvedValueOnce({
      type: 'session_response',
      payload: { action: 'list', sessions: [] },
    })

    const { result } = renderHook(() => useSessions(), { wrapper: createWrapper() })

    await waitFor(() => {
      expect(result.current.data).toBeDefined()
    })

    expect(mockedApiFetch).toHaveBeenCalledWith('/session/list')
  })
})

describe('useCurrentSession', () => {
  it('fetches from /session/current', async () => {
    mockedApiFetch.mockResolvedValueOnce({
      type: 'session_response',
      payload: {
        action: 'current',
        session: { thread_id: 'ws-test', label: 'test', message_count: 0, is_current: true },
      },
    })

    const { result } = renderHook(() => useCurrentSession(), { wrapper: createWrapper() })

    await waitFor(() => {
      expect(result.current.data).toBeDefined()
    })

    expect(mockedApiFetch).toHaveBeenCalledWith('/session/current')
  })
})

describe('useNewSession', () => {
  it('posts to /session with action=new', async () => {
    mockedApiFetch.mockResolvedValueOnce({
      type: 'session_response',
      payload: { action: 'new', thread_id: 'ws-new123' },
    })

    const { result } = renderHook(() => useNewSession(), { wrapper: createWrapper() })

    result.current.mutate(undefined)

    await waitFor(() => {
      expect(mockedApiFetch).toHaveBeenCalledWith('/session', expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ action: 'new' }),
      }))
    })
  })

  it('invalidates sessions and session queries on success', async () => {
    mockedApiFetch.mockResolvedValueOnce({
      type: 'session_response',
      payload: { action: 'new', thread_id: 'ws-new' },
    })

    const wrapper = createWrapper()
    const { result } = renderHook(() => useNewSession(), { wrapper })

    // Set up mock queries that should be invalidated
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const invalidateSpy = vi.spyOn(qc, 'invalidateQueries')

    // Need to test with a real QueryClient — use a separate render
    // with a spied QueryClient
    const qc2 = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const spy = vi.spyOn(qc2, 'invalidateQueries')

    function Wrapper2({ children }: { children: ReactNode }) {
      return <QueryClientProvider client={qc2}>{children}</QueryClientProvider>
    }

    const { result: r2 } = renderHook(() => useNewSession(), { wrapper: Wrapper2 })
    r2.current.mutate(undefined)

    await waitFor(() => {
      expect(mockedApiFetch).toHaveBeenCalled()
    })

    // After success, should invalidate both query keys
    await waitFor(() => {
      expect(spy).toHaveBeenCalledWith({ queryKey: ['sessions'] })
      expect(spy).toHaveBeenCalledWith({ queryKey: ['session'] })
    })
  })
})

describe('useSwitchSession', () => {
  it('posts to /session with action=switch and thread_id', async () => {
    mockedApiFetch.mockResolvedValueOnce({
      type: 'session_response',
      payload: { action: 'switch', thread_id: 'ws-switch' },
    })

    const { result } = renderHook(() => useSwitchSession(), { wrapper: createWrapper() })

    result.current.mutate('ws-switch')

    await waitFor(() => {
      expect(mockedApiFetch).toHaveBeenCalledWith('/session', expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ action: 'switch', thread_id: 'ws-switch' }),
      }))
    })
  })
})

describe('useDeleteSession', () => {
  it('posts to /session with action=delete and thread_id', async () => {
    mockedApiFetch.mockResolvedValueOnce({
      type: 'session_response',
      payload: { action: 'delete', thread_id: 'ws-del', deleted: true },
    })

    const { result } = renderHook(() => useDeleteSession(), { wrapper: createWrapper() })

    result.current.mutate('ws-del')

    await waitFor(() => {
      expect(mockedApiFetch).toHaveBeenCalledWith('/session', expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ action: 'delete', thread_id: 'ws-del' }),
      }))
    })
  })
})

describe('useSessionHistory', () => {
  it('does not fetch when threadId is null', () => {
    renderHook(() => useSessionHistory(null), { wrapper: createWrapper() })
    expect(mockedApiFetch).not.toHaveBeenCalled()
  })

  it('fetches with thread_id query param', async () => {
    mockedApiFetch.mockResolvedValueOnce({
      type: 'session_response',
      payload: { action: 'history', thread_id: 'ws-abc', messages: [] },
    })

    const { result } = renderHook(() => useSessionHistory('ws-abc'), { wrapper: createWrapper() })

    await waitFor(() => {
      expect(result.current.data).toBeDefined()
    })

    expect(mockedApiFetch).toHaveBeenCalledWith('/session/history?thread_id=ws-abc')
  })
})
