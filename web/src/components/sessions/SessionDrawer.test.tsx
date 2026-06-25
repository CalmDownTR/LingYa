import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SessionDrawer } from './SessionDrawer'
import * as api from '../../lib/api'
import type { SessionInfo } from '../../types'

// ── Mock the api hooks module ───────────────────────────────────────
vi.mock('../../lib/api', () => ({
  useSessions: vi.fn(),
  useNewSession: vi.fn(),
  useSwitchSession: vi.fn(),
  useDeleteSession: vi.fn(),
}))

// ── Helpers ─────────────────────────────────────────────────────────

const mockSessionA: SessionInfo = {
  thread_id: 'ws-aaa11111',
  label: '会话 aaa11111',
  message_count: 5,
  is_current: true,
}

const mockSessionB: SessionInfo = {
  thread_id: 'ws-bbb22222',
  label: '会话 bbb22222',
  message_count: 3,
  is_current: false,
}

function renderDrawer(open: boolean, onClose = vi.fn(), onSessionChange = vi.fn()) {
  const qc = new QueryClient()
  return {
    qc,
    onClose,
    onSessionChange,
    ...render(
      <QueryClientProvider client={qc}>
        <SessionDrawer open={open} onClose={onClose} onSessionChange={onSessionChange} />
      </QueryClientProvider>,
    ),
  }
}

// ── Tests ──────────────────────────────────────────────────────────

describe('SessionDrawer', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Default mocks for mutations
    vi.mocked(api.useNewSession).mockReturnValue({ mutate: vi.fn() } as never)
    vi.mocked(api.useSwitchSession).mockReturnValue({ mutate: vi.fn() } as never)
    vi.mocked(api.useDeleteSession).mockReturnValue({ mutate: vi.fn() } as never)
  })

  // ── Visibility ──────────────────────────────────────────────────

  it('when open=false, drawer is hidden via CSS (slide-out + pointer-events none)', () => {
    vi.mocked(api.useSessions).mockReturnValue({
      data: { type: 'session_response', payload: { action: 'list', sessions: [] } },
      isLoading: false,
      isError: false,
    } as never)

    renderDrawer(false)
    // Drawer is mounted (so transitions can play) but visually hidden.
    // The header label "会话" is present in the DOM but the drawer is
    // translated off-screen and ignores pointer events.
    const drawer = screen.getByRole('dialog', { hidden: true })
    expect(drawer.className).toContain('-translate-x-full')
    expect(drawer.className).toContain('pointer-events-none')
    expect(drawer.getAttribute('aria-hidden')).toBe('true')
  })

  it('when open=true, drawer is visible (slide-in + pointer-events auto)', () => {
    vi.mocked(api.useSessions).mockReturnValue({
      data: { type: 'session_response', payload: { action: 'list', sessions: [] } },
      isLoading: false,
      isError: false,
    } as never)

    renderDrawer(true)
    const drawer = screen.getByRole('dialog')
    expect(drawer.className).toContain('translate-x-0')
    expect(drawer.className).not.toContain('pointer-events-none')
    expect(drawer.getAttribute('aria-hidden')).toBe('false')
  })

  // ── Loading state ───────────────────────────────────────────────

  it('shows loading indicator while sessions are loading', () => {
    vi.mocked(api.useSessions).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
    } as never)

    renderDrawer(true)
    // Should show loading, NOT "还没有会话"
    expect(screen.getByText(/加载中/)).toBeInTheDocument()
    expect(screen.queryByText('还没有会话 — 开始对话后会自动创建')).not.toBeInTheDocument()
  })

  // ── Error state ─────────────────────────────────────────────────

  it('shows error message when sessions fail to load', () => {
    vi.mocked(api.useSessions).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error('Network error'),
    } as never)

    renderDrawer(true)
    // Should show error, NOT "还没有会话"
    expect(screen.getByText(/加载失败/)).toBeInTheDocument()
    expect(screen.queryByText('还没有会话 — 开始对话后会自动创建')).not.toBeInTheDocument()
  })

  // ── Empty state ─────────────────────────────────────────────────

  it('shows empty message when there are no sessions', () => {
    vi.mocked(api.useSessions).mockReturnValue({
      data: { type: 'session_response', payload: { action: 'list', sessions: [] } },
      isLoading: false,
      isError: false,
    } as never)

    renderDrawer(true)
    expect(screen.getByText('还没有会话 — 开始对话后会自动创建')).toBeInTheDocument()
  })

  // ── Session list ────────────────────────────────────────────────

  it('renders all sessions from the list', () => {
    vi.mocked(api.useSessions).mockReturnValue({
      data: {
        type: 'session_response',
        payload: { action: 'list', sessions: [mockSessionA, mockSessionB] },
      },
      isLoading: false,
      isError: false,
    } as never)

    renderDrawer(true)
    expect(screen.getByText('会话 aaa11111')).toBeInTheDocument()
    expect(screen.getByText('会话 bbb22222')).toBeInTheDocument()
    expect(screen.getByText('5 条消息')).toBeInTheDocument()
    expect(screen.getByText('3 条消息')).toBeInTheDocument()
  })

  it('highlights the current session', () => {
    vi.mocked(api.useSessions).mockReturnValue({
      data: {
        type: 'session_response',
        payload: { action: 'list', sessions: [mockSessionA, mockSessionB] },
      },
      isLoading: false,
      isError: false,
    } as never)

    renderDrawer(true)
    // Current session row has border-accent on the outermost div
    const currentLabel = screen.getByText('会话 aaa11111')
    const currentRow = currentLabel.closest('[class*="border-l-[3px]"]')
    expect(currentRow?.className).toContain('border-accent')
  })

  // ── Switch session ──────────────────────────────────────────────

  it('switches to another session on click', async () => {
    const user = userEvent.setup()
    const mockMutate = vi.fn()
    vi.mocked(api.useSwitchSession).mockReturnValue({ mutate: mockMutate } as never)
    vi.mocked(api.useSessions).mockReturnValue({
      data: {
        type: 'session_response',
        payload: { action: 'list', sessions: [mockSessionA, mockSessionB] },
      },
      isLoading: false,
      isError: false,
    } as never)

    const onSessionChange = vi.fn()
    const onClose = vi.fn()
    renderDrawer(true, onClose, onSessionChange)

    // Click session B
    await user.click(screen.getByText('会话 bbb22222'))

    // mutate should be called with thread_id and onSuccess/onError callbacks
    expect(mockMutate).toHaveBeenCalledWith('ws-bbb22222', expect.objectContaining({
      onSuccess: expect.any(Function),
    }))
  })

  it('calls onSessionChange and onClose on successful switch', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
    const mockMutate = vi.fn()
    vi.mocked(api.useSwitchSession).mockReturnValue({ mutate: mockMutate } as never)
    vi.mocked(api.useSessions).mockReturnValue({
      data: {
        type: 'session_response',
        payload: { action: 'list', sessions: [mockSessionA, mockSessionB] },
      },
      isLoading: false,
      isError: false,
    } as never)

    const onSessionChange = vi.fn()
    const onClose = vi.fn()
    renderDrawer(true, onClose, onSessionChange)

    await user.click(screen.getByText('会话 bbb22222'))

    // Simulate onSuccess callback
    const onSuccess = mockMutate.mock.calls[0][1].onSuccess
    onSuccess()

    // onSessionChange fires immediately
    expect(onSessionChange).toHaveBeenCalledWith('ws-bbb22222')
    // onClose fires after a short delay for selection highlight transition
    expect(onClose).not.toHaveBeenCalled()
    vi.advanceTimersByTime(200)
    expect(onClose).toHaveBeenCalled()

    vi.useRealTimers()
  })

  // BUG #2 (fixed): Optimistic update rolls back on failed switch
  it('rolls back optimistic update and shows error on failed switch', async () => {
    const user = userEvent.setup()
    const mockMutate = vi.fn()
    vi.mocked(api.useSwitchSession).mockReturnValue({ mutate: mockMutate } as never)

    const initialData = {
      type: 'session_response',
      payload: { action: 'list', sessions: [mockSessionA, mockSessionB] },
    }
    vi.mocked(api.useSessions).mockReturnValue({
      data: initialData,
      isLoading: false,
      isError: false,
    } as never)

    const onSessionChange = vi.fn()
    const onClose = vi.fn()
    const { qc } = renderDrawer(true, onClose, onSessionChange)

    // Pre-populate the cache so handleSwitch's snapshot is truthy
    qc.setQueryData(['sessions'], initialData)
    const setQueryDataSpy = vi.spyOn(qc, 'setQueryData')

    // Cache should have the original data before click
    const originalCache = qc.getQueryData(['sessions'])

    await user.click(screen.getByText('会话 bbb22222'))

    // Optimistic update sets B as current
    expect(setQueryDataSpy).toHaveBeenCalledTimes(1)

    const onError = mockMutate.mock.calls[0][1].onError
    expect(onError).toBeDefined()
    act(() => {
      onError(new Error('Switch failed'))
    })

    // Rollback: setQueryData called again with original cache
    expect(setQueryDataSpy).toHaveBeenCalledTimes(2)
    expect(qc.getQueryData(['sessions'])).toEqual(originalCache)
    expect(onSessionChange).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()
    // Error message comes from err.message when available
    expect(screen.getByText(/Switch failed/)).toBeInTheDocument()
  })

  // ── New session ─────────────────────────────────────────────────

  it('creates a new session on button click', async () => {
    const user = userEvent.setup()
    const mockMutate = vi.fn()
    vi.mocked(api.useNewSession).mockReturnValue({ mutate: mockMutate } as never)
    vi.mocked(api.useSessions).mockReturnValue({
      data: {
        type: 'session_response',
        payload: { action: 'list', sessions: [mockSessionA] },
      },
      isLoading: false,
      isError: false,
    } as never)

    renderDrawer(true)
    await user.click(screen.getByText('新建'))

    expect(mockMutate).toHaveBeenCalledWith(undefined, expect.objectContaining({
      onSuccess: expect.any(Function),
    }))
  })

  it('switches to new session on successful creation', async () => {
    const user = userEvent.setup()
    const mockMutate = vi.fn()
    vi.mocked(api.useNewSession).mockReturnValue({ mutate: mockMutate } as never)
    vi.mocked(api.useSessions).mockReturnValue({
      data: {
        type: 'session_response',
        payload: { action: 'list', sessions: [mockSessionA] },
      },
      isLoading: false,
      isError: false,
    } as never)

    const onSessionChange = vi.fn()
    const onClose = vi.fn()
    renderDrawer(true, onClose, onSessionChange)

    await user.click(screen.getByText('新建'))

    // Simulate onSuccess with response data
    const onSuccess = mockMutate.mock.calls[0][1].onSuccess
    onSuccess({ payload: { thread_id: 'ws-new12345' } })

    expect(onSessionChange).toHaveBeenCalledWith('ws-new12345')
    expect(onClose).toHaveBeenCalled()
  })

  // BUG #4 (fixed): Error feedback when new session creation fails
  it('shows error when new session creation fails', async () => {
    const user = userEvent.setup()
    const mockMutate = vi.fn()
    vi.mocked(api.useNewSession).mockReturnValue({ mutate: mockMutate } as never)
    vi.mocked(api.useSessions).mockReturnValue({
      data: {
        type: 'session_response',
        payload: { action: 'list', sessions: [mockSessionA] },
      },
      isLoading: false,
      isError: false,
    } as never)

    const onSessionChange = vi.fn()
    const onClose = vi.fn()
    renderDrawer(true, onClose, onSessionChange)

    await user.click(screen.getByText('新建'))

    const callOpts = mockMutate.mock.calls[0][1]
    expect(callOpts.onError).toBeDefined()
    act(() => {
      callOpts.onError(new Error('Create failed'))
    })

    expect(onSessionChange).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()
    // Error message comes from err.message when available
    expect(screen.getByText(/Create failed/)).toBeInTheDocument()
  })

  // ── Delete session ──────────────────────────────────────────────

  it('deletes a non-current session', async () => {
    const user = userEvent.setup()
    const mockMutate = vi.fn()
    vi.mocked(api.useDeleteSession).mockReturnValue({ mutate: mockMutate } as never)
    vi.mocked(api.useSessions).mockReturnValue({
      data: {
        type: 'session_response',
        payload: { action: 'list', sessions: [mockSessionA, mockSessionB] },
      },
      isLoading: false,
      isError: false,
    } as never)

    renderDrawer(true)

    // Session B row contains the label + delete button (only for non-current)
    // The delete button has .lucide-trash-2 icon
    const trashButtons = document.querySelectorAll('.lucide-trash-2')
    expect(trashButtons.length).toBe(1) // Only session B has delete

    await user.click(trashButtons[0] as Element)
    expect(mockMutate).toHaveBeenCalledWith('ws-bbb22222')
  })

  it('hides delete button for current session', () => {
    vi.mocked(api.useSessions).mockReturnValue({
      data: {
        type: 'session_response',
        payload: { action: 'list', sessions: [mockSessionA] },
      },
      isLoading: false,
      isError: false,
    } as never)

    renderDrawer(true)
    // Current session should not have a delete button
    // The Trash2 icon is conditionally rendered: {!s.is_current && (<button>Trash2</button>)}
    const trashIcons = document.querySelectorAll('.lucide-trash-2')
    expect(trashIcons.length).toBe(0)
  })

  // ── Close ───────────────────────────────────────────────────────

  it('closes when backdrop is clicked', async () => {
    const user = userEvent.setup()
    vi.mocked(api.useSessions).mockReturnValue({
      data: { type: 'session_response', payload: { action: 'list', sessions: [] } },
      isLoading: false,
      isError: false,
    } as never)

    const onClose = vi.fn()
    renderDrawer(true, onClose)

    // Click the backdrop (first child of the fragment, z-40)
    const backdrop = document.querySelector('[class*="z-40"]')
    expect(backdrop).toBeTruthy()
    await user.click(backdrop!)
    expect(onClose).toHaveBeenCalled()
  })
})
