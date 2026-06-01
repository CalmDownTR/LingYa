"""Test WebSocket protocol helpers — unit tests, no real server."""

from __future__ import annotations

import asyncio
import json
import struct

import pytest

from lingya.gateway.protocol import (
    OP_CLOSE,
    OP_PING,
    OP_PONG,
    OP_TEXT,
    _encode_frame,
    _encode_masked_frame,
    _generate_accept_key,
    _read_frame,
    _read_http_request,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_reader(data: bytes) -> asyncio.StreamReader:
    """Feed bytes into a StreamReader for protocol-level testing."""
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


# ── _generate_accept_key tests ──────────────────────────────────────


class TestGenerateAcceptKey:
    def test_rfc_example(self):
        """RFC 6455 Section 4.2.2 example."""
        key = "dGhlIHNhbXBsZSBub25jZQ=="
        expected = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        assert _generate_accept_key(key) == expected

    def test_empty_key(self):
        """Even an empty key should produce a valid base64 output."""
        result = _generate_accept_key("")
        assert len(result) > 0
        # Should be valid base64
        import base64

        base64.b64decode(result)  # Should not raise

    def test_different_keys_produce_different_outputs(self):
        """Different keys should produce different accept values."""
        a = _generate_accept_key("keyA==")
        b = _generate_accept_key("keyB==")
        assert a != b

    def test_output_is_base64_encoded_sha1(self):
        """Output length should match base64-encoded SHA-1 (28 chars)."""
        import base64
        import hashlib

        key = "test-key-12345"
        result = _generate_accept_key(key)
        # SHA-1 produces 20 bytes, base64 of 20 bytes = 28 chars
        assert len(result) == 28
        # Decode and verify it's 20 bytes
        decoded = base64.b64decode(result)
        assert len(decoded) == 20


# ── _encode_frame tests ─────────────────────────────────────────────


class TestEncodeFrame:
    def test_small_payload(self):
        """Payload < 126 bytes: length fits in 7 bits."""
        payload = b"hello"
        frame = _encode_frame(OP_TEXT, payload)

        # Byte 0: FIN(1) + RSV(0) + opcode(0x1) = 0x81
        assert frame[0] == 0x81
        # Byte 1: MASK(0) + length(5) = 0x05
        assert frame[1] == 0x05
        # Payload follows directly
        assert frame[2:] == b"hello"
        assert len(frame) == 2 + 5  # header + payload

    def test_medium_payload_126_bytes(self):
        """Payload of exactly 126 bytes uses extended 16-bit length."""
        payload = b"x" * 126
        frame = _encode_frame(OP_TEXT, payload)

        assert frame[0] == 0x81
        assert frame[1] == 126  # Extended length marker
        # Next 2 bytes: 126 in big-endian
        assert struct.unpack(">H", frame[2:4])[0] == 126
        assert frame[4:] == payload
        assert len(frame) == 4 + 126

    def test_medium_payload_65535_bytes(self):
        """Payload of 65535 bytes uses extended 16-bit length."""
        payload = b"x" * 65535
        frame = _encode_frame(OP_TEXT, payload)

        assert frame[0] == 0x81
        assert frame[1] == 126
        assert struct.unpack(">H", frame[2:4])[0] == 65535
        assert len(frame) == 4 + 65535

    def test_large_payload(self):
        """Payload >= 65536 bytes uses extended 64-bit length."""
        payload = b"x" * 65536
        frame = _encode_frame(OP_TEXT, payload)

        assert frame[0] == 0x81
        assert frame[1] == 127  # 64-bit extended length marker
        assert struct.unpack(">Q", frame[2:10])[0] == 65536
        assert len(frame) == 10 + 65536

    def test_opcode_is_preserved_in_header(self):
        """Different opcodes produce correct first byte."""
        text_frame = _encode_frame(OP_TEXT, b"x")
        close_frame = _encode_frame(OP_CLOSE, b"")
        ping_frame = _encode_frame(OP_PING, b"ping")
        pong_frame = _encode_frame(OP_PONG, b"pong")

        assert text_frame[0] == 0x81  # 0x80 | 0x1
        assert close_frame[0] == 0x88  # 0x80 | 0x8
        assert ping_frame[0] == 0x89  # 0x80 | 0x9
        assert pong_frame[0] == 0x8A  # 0x80 | 0xA

    def test_zero_length_payload(self):
        """Empty payload: length is 0."""
        frame = _encode_frame(OP_CLOSE, b"")
        assert frame[0] == 0x88
        assert frame[1] == 0x00
        assert len(frame) == 2

    def test_server_frames_are_not_masked(self):
        """Server->client frames must NOT have the MASK bit set."""
        payload = b"unmasked"
        frame = _encode_frame(OP_TEXT, payload)
        assert (frame[1] & 0x80) == 0


# ── _encode_masked_frame tests ──────────────────────────────────────


class TestEncodeMaskedFrame:
    def test_default_mask_is_random(self):
        """When no mask is provided, one is generated automatically."""
        frame = _encode_masked_frame(OP_TEXT, b"hello")
        # Frame has MASK bit set
        assert (frame[1] & 0x80) != 0

        # Two calls should use different masks (probabilistic)
        frame2 = _encode_masked_frame(OP_TEXT, b"hello")
        # The mask bytes (bytes 2-5 for small payload) should differ
        mask1 = frame[2:6]
        mask2 = frame2[2:6]
        # Extremely unlikely to be the same
        assert mask1 != mask2

    def test_explicit_mask(self):
        """Providing an explicit mask uses it."""
        payload = b"hello"
        mask = b"\x01\x02\x03\x04"
        frame = _encode_masked_frame(OP_TEXT, payload, mask)

        # Frame: [0x81, 0x85, mask(4), masked_payload(5)]
        assert frame[0] == 0x81  # FIN + text opcode
        assert frame[1] == 0x85  # MASK=1 + len=5
        assert frame[2:6] == mask
        # Verify payload is correctly masked
        expected_masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        assert frame[6:] == expected_masked

    @pytest.mark.asyncio
    async def test_masked_frame_round_trip(self):
        """Masked frame can be read back by _read_frame."""
        payload = b"Hello, masked world!"
        frame = _encode_masked_frame(OP_TEXT, payload)

        reader = _make_reader(frame)
        opcode, result = await _read_frame(reader)

        assert opcode == OP_TEXT
        assert result == payload

    @pytest.mark.asyncio
    async def test_masked_close_frame(self):
        """Close frame can be masked."""
        frame = _encode_masked_frame(OP_CLOSE, b"", b"\xAA\xBB\xCC\xDD")

        reader = _make_reader(frame)
        opcode, payload = await _read_frame(reader)

        assert opcode == OP_CLOSE
        assert payload == b""

    @pytest.mark.asyncio
    async def test_masked_large_payload(self):
        """Large masked payload round-trips correctly."""
        payload = b"x" * 300
        frame = _encode_masked_frame(OP_TEXT, payload)

        reader = _make_reader(frame)
        opcode, result = await _read_frame(reader)

        assert opcode == OP_TEXT
        assert result == payload

    def test_opcode_preserved_in_masked_frame(self):
        """Opcode is set correctly even when masked."""
        text_frame = _encode_masked_frame(OP_TEXT, b"x")
        close_frame = _encode_masked_frame(OP_CLOSE, b"")
        ping_frame = _encode_masked_frame(OP_PING, b"ping")
        pong_frame = _encode_masked_frame(OP_PONG, b"pong")

        assert text_frame[0] == 0x81
        assert close_frame[0] == 0x88
        assert ping_frame[0] == 0x89
        assert pong_frame[0] == 0x8A


# ── _read_frame tests (via round-trip) ──────────────────────────────


@pytest.mark.asyncio
class TestReadFrameRoundTrip:
    async def test_text_frame_round_trip(self):
        """Encode a text frame, then read it back."""
        original = b"Hello, WebSocket!"
        encoded = _encode_frame(OP_TEXT, original)
        reader = _make_reader(encoded)
        opcode, payload = await _read_frame(reader)

        assert opcode == OP_TEXT
        assert payload == original

    async def test_ping_frame_round_trip(self):
        """Ping opcode is preserved."""
        original = b"are you there?"
        encoded = _encode_frame(OP_PING, original)
        reader = _make_reader(encoded)
        opcode, payload = await _read_frame(reader)

        assert opcode == OP_PING
        assert payload == original

    async def test_pong_frame_round_trip(self):
        """Pong opcode is preserved."""
        original = b"yes"
        encoded = _encode_frame(OP_PONG, original)
        reader = _make_reader(encoded)
        opcode, payload = await _read_frame(reader)

        assert opcode == OP_PONG
        assert payload == original

    async def test_close_frame_round_trip(self):
        """Close opcode is preserved."""
        encoded = _encode_frame(OP_CLOSE, b"")
        reader = _make_reader(encoded)
        opcode, payload = await _read_frame(reader)

        assert opcode == OP_CLOSE
        assert payload == b""

    async def test_large_payload_round_trip(self):
        """Round-trip works for payloads > 126 bytes."""
        original = b"x" * 200
        encoded = _encode_frame(OP_TEXT, original)
        reader = _make_reader(encoded)
        opcode, payload = await _read_frame(reader)

        assert opcode == OP_TEXT
        assert payload == original

    async def test_json_text_frame(self):
        """A realistic JSON message round-trips correctly."""
        message = json.dumps({"type": "ping", "payload": {}}).encode("utf-8")
        encoded = _encode_frame(OP_TEXT, message)
        reader = _make_reader(encoded)
        opcode, payload = await _read_frame(reader)

        assert opcode == OP_TEXT
        assert json.loads(payload) == {"type": "ping", "payload": {}}


# ── Masked frame tests (client->server) ─────────────────────────────


@pytest.mark.asyncio
class TestMaskedFrame:
    async def test_masked_frame_unmasked_correctly(self):
        """Client->server masked frames are properly unmasked."""
        payload = b"masked message"
        mask = b"\x01\x02\x03\x04"
        frame = _encode_masked_frame(OP_TEXT, payload, mask)

        reader = _make_reader(frame)
        opcode, result = await _read_frame(reader)

        assert opcode == OP_TEXT
        assert result == payload

    async def test_masked_frame_with_zero_mask(self):
        """Mask of all zeros is a no-op."""
        payload = b"visible"
        mask = b"\x00\x00\x00\x00"
        frame = _encode_masked_frame(OP_TEXT, payload, mask)

        reader = _make_reader(frame)
        opcode, result = await _read_frame(reader)

        assert result == payload

    async def test_masked_frame_with_all_ones_mask(self):
        """Mask of all 0xFF flips all bits in transit, _read_frame unmasks."""
        payload = b"\x00\x01\x02\x03"
        mask = b"\xFF\xFF\xFF\xFF"

        frame = _encode_masked_frame(OP_TEXT, payload, mask)
        # The masked payload in the frame should be bit-flipped
        in_transit_payload = frame[6:]
        expected_masked = bytes(b ^ 0xFF for b in payload)
        assert in_transit_payload == expected_masked

        # _read_frame should unmask it back to original
        reader = _make_reader(frame)
        opcode, result = await _read_frame(reader)
        assert result == payload

    async def test_masked_close_frame(self):
        """Close frame can also be masked."""
        mask = b"\xAA\xBB\xCC\xDD"
        frame = _encode_masked_frame(OP_CLOSE, b"", mask)

        reader = _make_reader(frame)
        opcode, payload = await _read_frame(reader)

        assert opcode == OP_CLOSE
        assert payload == b""

    async def test_masked_ping_frame(self):
        """Ping from client can be masked."""
        ping_data = b"client-ping"
        mask = b"\x37\xFA\x21\x3D"
        frame = _encode_masked_frame(OP_PING, ping_data, mask)

        reader = _make_reader(frame)
        opcode, payload = await _read_frame(reader)

        assert opcode == OP_PING
        assert payload == ping_data

    async def test_large_masked_frame(self):
        """Masked frame > 125 bytes works correctly."""
        payload = b"x" * 300
        mask = b"\x12\x34\x56\x78"
        frame = _encode_masked_frame(OP_TEXT, payload, mask)

        reader = _make_reader(frame)
        opcode, result = await _read_frame(reader)

        assert result == payload


# ── _read_http_request tests ────────────────────────────────────────


@pytest.mark.asyncio
class TestReadHttpRequest:
    async def test_parses_get_request_with_headers(self):
        """Standard GET request with WebSocket upgrade headers."""
        request = (
            b"GET /ws HTTP/1.1\r\n"
            b"Host: localhost:8765\r\n"
            b"Upgrade: websocket\r\n"
            b"Connection: Upgrade\r\n"
            b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            b"Sec-WebSocket-Version: 13\r\n"
            b"\r\n"
        )
        reader = _make_reader(request)
        result = await _read_http_request(reader)

        assert result["method"] == "GET"
        assert result["path"] == "/ws"
        assert result["headers"]["host"] == "localhost:8765"
        assert result["headers"]["upgrade"] == "websocket"
        assert result["headers"]["connection"] == "Upgrade"
        assert result["headers"]["sec-websocket-key"] == "dGhlIHNhbXBsZSBub25jZQ=="
        assert result["headers"]["sec-websocket-version"] == "13"

    async def test_empty_request(self):
        """Empty request returns empty dict."""
        reader = _make_reader(b"\r\n")
        result = await _read_http_request(reader)

        assert result["method"] == ""
        assert result["path"] == ""
        assert result["headers"] == {}

    async def test_request_without_headers(self):
        """Request line without headers."""
        reader = _make_reader(b"GET / HTTP/1.1\r\n\r\n")
        result = await _read_http_request(reader)

        assert result["method"] == "GET"
        assert result["path"] == "/"
        assert result["headers"] == {}

    async def test_header_keys_are_lowercase(self):
        """Header keys are normalized to lowercase."""
        request = (
            b"GET / HTTP/1.1\r\n"
            b"Content-Type: application/json\r\n"
            b"X-Custom-Header: value\r\n"
            b"\r\n"
        )
        reader = _make_reader(request)
        result = await _read_http_request(reader)

        assert "content-type" in result["headers"]
        assert "x-custom-header" in result["headers"]
        assert result["headers"]["content-type"] == "application/json"
