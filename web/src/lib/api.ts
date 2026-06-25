import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from '../hooks/useApi'
import type { SettingsResponse, OceanRequest, SessionInfo } from '../types'

// ── Settings ────────────────────────────────────────────────────────

export function useSettings() {
  return useQuery<SettingsResponse>({
    queryKey: ['settings'],
    queryFn: () => apiFetch('/settings'),
  })
}

export function useUpdateOcean() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (ocean: OceanRequest) =>
      apiFetch('/settings/ocean', {
        method: 'PUT',
        body: JSON.stringify(ocean),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['settings'] }),
  })
}

export function useUpdateIdentity() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (identity: { identity?: string; core_belief?: string }) =>
      apiFetch('/settings/identity', {
        method: 'PUT',
        body: JSON.stringify(identity),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['settings'] }),
  })
}

export function useUpdateTone() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (preset: string) =>
      apiFetch('/settings/tone', {
        method: 'PUT',
        body: JSON.stringify({ preset }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['settings'] }),
  })
}

export function useResetSettings() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => apiFetch('/settings/reset', { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['settings'] }),
  })
}

// ── Sessions ────────────────────────────────────────────────────────

interface SessionActionResponse {
  type: string
  payload: { action: string; thread_id: string }
}

export function useSessions() {
  return useQuery<{ type: string; payload: { action: string; sessions: SessionInfo[] } }>({
    queryKey: ['sessions'],
    queryFn: () => apiFetch('/session/list'),
    refetchInterval: 30_000,
  })
}

export function useCurrentSession() {
  return useQuery<{ type: string; payload: { action: string; session: SessionInfo } }>({
    queryKey: ['session', 'current'],
    queryFn: () => apiFetch('/session/current'),
  })
}

export function useNewSession() {
  const qc = useQueryClient()
  return useMutation<SessionActionResponse>({
    mutationFn: () =>
      apiFetch<SessionActionResponse>('/session', {
        method: 'POST',
        body: JSON.stringify({ action: 'new' }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['sessions'] })
      qc.invalidateQueries({ queryKey: ['session'] })
    },
  })
}

export function useSwitchSession() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (thread_id: string) =>
      apiFetch('/session', {
        method: 'POST',
        body: JSON.stringify({ action: 'switch', thread_id }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['sessions'] })
      qc.invalidateQueries({ queryKey: ['session'] })
    },
  })
}

export function useDeleteSession() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (thread_id: string) =>
      apiFetch('/session', {
        method: 'POST',
        body: JSON.stringify({ action: 'delete', thread_id }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['sessions'] }),
  })
}

// ── Chat History ──────────────────────────────────────────────────────

interface HistoryMessage {
  role: 'user' | 'her'
  content: string
}

interface HistoryResponse {
  type: string
  payload: { action: string; thread_id: string; messages: HistoryMessage[] }
}

export function useSessionHistory(threadId: string | null) {
  return useQuery<HistoryResponse>({
    queryKey: ['session', 'history', threadId],
    queryFn: () =>
      apiFetch<HistoryResponse>(
        `/session/history?thread_id=${encodeURIComponent(threadId!)}`,
      ),
    enabled: threadId !== null,
    staleTime: 0,
  })
}
