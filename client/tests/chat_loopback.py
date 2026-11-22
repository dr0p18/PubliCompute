#!/usr/bin/env python3
"""Test the CLI against the web chat fake: python3 -B client/tests/chat_loopback.py [binary]."""

import contextlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import pty
import select
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import termios
import threading
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("finto", ROOT / "web/chat/finto/server.py")
FAKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FAKE)
BINARY = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else ROOT / "target/release/publicompute"
EMAIL = "cli@example.test"
MODEL = "gemma-4-12b"
QWEN = "qwen3.8-flash-next"
MAX_REQUEST = 256 * 1024


def read_until(fd, marker, timeout=6):
    end = time.monotonic() + timeout
    output = bytearray()
    while marker not in output:
        ready, _, _ = select.select([fd], [], [], max(0, end - time.monotonic()))
        if not ready:
            raise AssertionError("timed out waiting for expected CLI output")
        chunk = os.read(fd, 4096)
        if not chunk:
            raise AssertionError("CLI closed before expected output")
        output.extend(chunk)
    return bytes(output)


def read_available(fd):
    output = bytearray()
    while select.select([fd], [], [], 0)[0]:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


def delta(text):
    return {"choices": [{"delta": {"content": text}}]}


def finished():
    yield {"choices": [{"finish_reason": "stop"}], "publicompute": {"free_seat": True}}
    yield "[DONE]"


class ChatLoopback(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.session = self.directory / "publicompute/session"
        self.server = FAKE.crea_server(port=0, quota=0, saldo=2345, fast=True)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        self.port = self.server.server_address[1]
        self.api = f"http://127.0.0.1:{self.port}"
        self.env = dict(os.environ, XDG_CONFIG_HOME=str(self.directory), PUBLICOMPUTE_API=self.api)
        status, raw = self.request("POST", "/v1/auth/invite", {"code": "PC-BETA-0001", "email": EMAIL})
        self.assertEqual(status, 200)
        self.key = json.loads(raw)["access_key"]

    def request(self, method, path, body=None, cookie=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        headers = {"Cookie": cookie} if cookie else {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            connection.request(method, path, json.dumps(body) if body is not None else None, headers)
            reply = connection.getresponse()
            return reply.status, reply.read()
        finally:
            connection.close()

    @contextlib.contextmanager
    def tty_command(self, *args, answer_to_tty=False, stdout=subprocess.PIPE):
        master, slave = pty.openpty()
        # Give the real binary a controlling TTY while keeping diagnostics and answers separate.
        stub = "import fcntl,os,sys,termios; fcntl.ioctl(0,termios.TIOCSCTTY,0); os.execv(sys.argv[1],sys.argv[1:])"
        cmd = [sys.executable, "-c", stub, str(BINARY), *args]
        try:
            with subprocess.Popen(cmd, env=self.env, start_new_session=True,
                                  stdin=slave, stdout=slave if answer_to_tty else stdout, stderr=subprocess.PIPE) as child:
                try:
                    yield child, master, slave
                finally:
                    if child.poll() is None:
                        child.kill()
        finally:
            os.close(master)
            os.close(slave)

    def login(self, key=None):
        key = self.key if key is None else key
        with self.tty_command("login") as (child, master, slave):
            read_until(master, b"Email: ")
            os.write(master, EMAIL.encode() + b"\n")
            read_until(master, b"Access key: ")
            os.write(master, key.encode() + b"\n")
            out, err = child.communicate(timeout=6)
            ready, _, _ = select.select([master], [], [], 0)
            terminal = os.read(master, 4096) if ready else b""
            self.assertFalse(key.encode() in out + err + terminal, "access key appeared in CLI output")
            self.assertTrue(termios.tcgetattr(slave)[3] & termios.ECHO, "terminal echo was not restored")
            self.assertEqual(out, b"")
            return child.returncode, err

    def stream_fixture(self, events):
        def send_stream(handler, _session, _model, body, _words, _last):
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.close_connection = True
            for event in events(body):
                data = json.dumps(event) if isinstance(event, dict) else event
                handler.wfile.write(f"data: {data}\n\n".encode())
                handler.wfile.flush()
        return patch.object(FAKE.Handler, "_stream", send_stream)

    def authenticate(self):
        self.assertEqual(self.login(), (0, b"Logged in.\n"))

    def command(self, *args, env=None, input=None):
        return subprocess.run([BINARY, *args], env=self.env if env is None else env, input=input,
                              capture_output=True, timeout=8, check=False)

    def chat(self, question, model=MODEL):
        return self.command("chat", "--model", model, question)

    def assert_error(self, result, text):
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(text, result.stderr)

    def answer_for(self, question):
        return FAKE.scegli_risposta([{"role": "user", "content": question}]).encode()

    def test_login_success(self):
        self.authenticate()
        session = json.loads(self.session.read_text())
        self.assertEqual(set(session), {"api", "expires_at", "token"})
        self.assertEqual(session["api"], self.api)
        self.assertGreater(session["expires_at"], time.time())
        self.assertEqual(stat.S_IMODE(self.session.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.session.parent.stat().st_mode), 0o700)
        self.assertFalse(self.key in self.session.read_text(), "access key was stored")

    def test_login_invalid_credentials(self):
        code, err = self.login("wrong-fixture-key")
        self.assertNotEqual(code, 0)
        self.assertIn(b"invalid_credentials", err)
        self.assertFalse(self.session.exists())

    def test_login_rate_limited(self):
        for _ in range(10):
            code, err = self.login("wrong-fixture-key")
            self.assertNotEqual(code, 0)
            self.assertIn(b"invalid_credentials", err)
        code, err = self.login()
        self.assertNotEqual(code, 0)
        self.assertIn(b"rate_limited", err)
        self.assertIn(b"3600 seconds", err)
        self.assertFalse(self.session.exists())

    def test_login_ctrl_c_restores_terminal_echo(self):
        with self.tty_command("login") as (child, master, slave):
            read_until(master, b"Email: ")
            os.write(master, EMAIL.encode() + b"\n")
            read_until(master, b"Access key: ")
            self.assertFalse(termios.tcgetattr(slave)[3] & termios.ECHO)
            os.write(master, b"partial-fixture-key\x03")
            out, err = child.communicate(timeout=6)
            self.assertNotEqual(child.returncode, 0)
            self.assertEqual(out, b"")
            self.assertIn(b"login cancelled", err)
            self.assertTrue(termios.tcgetattr(slave)[3] & termios.ECHO)
            self.assertFalse(self.session.exists())

    def test_login_accepts_pasted_credentials_without_echo(self):
        with self.tty_command("login") as (child, master, slave):
            read_until(master, b"Email: ")
            self.assertFalse(termios.tcgetattr(slave)[3] & termios.ECHO)
            os.write(master, f"{EMAIL}\n{self.key}\n".encode())
            out, err = child.communicate(timeout=6)
            self.assertFalse(self.key.encode() in read_available(master) + out + err, "access key appeared in output")
            self.assertEqual((child.returncode, out, err), (0, b"", b"Logged in.\n"))
            self.assertTrue(termios.tcgetattr(slave)[3] & termios.ECHO)

    def test_login_termination_signals_restore_terminal_echo(self):
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
            with self.subTest(signal=sig), self.tty_command("login") as (child, master, slave):
                read_until(master, b"Email: ")
                os.write(master, EMAIL.encode() + b"\n")
                read_until(master, b"Access key: ")
                os.write(master, b"unfinished-fixture-key")
                child.send_signal(sig)
                out, err = child.communicate(timeout=6)
                self.assertNotEqual(child.returncode, 0)
                self.assertEqual(out, b"")
                self.assertIn(b"login cancelled", err)
                self.assertTrue(termios.tcgetattr(slave)[3] & termios.ECHO)
                self.assertFalse(self.session.exists())
                os.write(master, b"after login\n")
                self.assertTrue(read_until(slave, b"after login\n") == b"after login\n", "credential input was left queued")

    def test_chat_uses_free_seat(self):
        self.authenticate()
        self.server.stato.free_seat = True
        result = self.chat("hello", QWEN)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, self.answer_for("hello"))
        self.assertNotIn(b"contribution seat not used", result.stderr)
        self.assertEqual(self.server.stato.utenti[EMAIL]["saldo"], 2345)

    def test_paid_chat_reports_updated_balance(self):
        self.authenticate()
        env = dict(self.env, PUBLICOMPUTE_API="http://invalid.example.test")
        result = self.command("chat", "--model", MODEL, "--api", self.api, "hello", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, self.answer_for("hello"))
        balance = self.server.stato.utenti[EMAIL]["saldo"]
        self.assertLess(balance, 2345)
        self.assertIn(f"contribution seat not used, balance {balance / 1000:.3f} credits".encode(), result.stderr)
        self.assertIn(b"tok/s", result.stderr)

    def test_chat_401(self):
        self.authenticate()
        with self.server.stato.lock:
            self.server.stato.sessioni.clear()
        self.assert_error(self.chat("hello"), b"session expired, run publicompute login")

    def test_chat_402(self):
        self.authenticate()
        self.assert_error(self.chat("/402"), b"insufficient_credits")

    def test_chat_model_and_admission_errors(self):
        self.authenticate()
        result = self.chat("hello", "missing-model")
        self.assert_error(result, b"model_not_found: use --model ID")
        self.assertIn(MODEL.encode(), result.stderr)
        for question, error in [
            ("/429", b"too_many_chats"), ("/network_full", b"network_full"), ("/network_busy", b"network_busy"),
        ]:
            with self.subTest(question=question):
                self.assert_error(self.chat(question), error)

    def test_chat_stream_error_keeps_partial_stdout(self):
        self.authenticate()
        result = self.chat("/sse-error")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stdout and self.answer_for("/sse-error").startswith(result.stdout))
        self.assertIn(b"Simulated stream failure", result.stderr)
        self.assertNotIn(b"contribution seat not used", result.stderr)

    def test_chat_truncated_stream(self):
        self.authenticate()
        for fixture in ("/truncated", "/missing-done"):
            with self.subTest(fixture=fixture):
                result = self.chat(fixture)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stdout and self.answer_for(fixture).startswith(result.stdout))
                self.assertIn(b"incomplete chat stream", result.stderr)
                self.assertNotIn(b"contribution seat not used", result.stderr)

    def test_answers_escape_controls_only_on_terminal_and_diagnostics_escape_them_always(self):
        self.authenticate()
        control = "\x1b]52;c;data\x07\x9b\x7f\r\n\tend"

        def events(body):
            if body["messages"][-1]["content"] == "error":
                yield {"error": {"message": control}}
            else:
                yield delta(control)
                yield from finished()

        with self.stream_fixture(events):
            pipe = self.chat("answer")
            self.assertEqual((pipe.returncode, pipe.stdout), (0, control.encode()))
            with self.tty_command("chat", "--model", MODEL, "answer", answer_to_tty=True) as (child, master, _):
                _, err = child.communicate(timeout=6)
                self.assertEqual(child.returncode, 0, err)
                self.assertEqual(read_available(master), b"\\u{1b}]52;c;data\\u{7}\\u{9b}\\u{7f}\\r\r\n\tend")
            error = self.chat("error")
            self.assert_error(error, b"\\u{1b}]52;c;data\\u{7}\\u{9b}\\u{7f}\\r\\n\\tend")
            self.assertNotIn(b"\x1b", error.stderr)

        orig_get = FAKE.Handler.do_GET

        def models(handler):
            if handler.path == "/v1/models":
                handler._json(200, {"data": [{"id": control}]})
            else:
                orig_get(handler)

        with patch.object(FAKE.Handler, "do_GET", models):
            result = self.chat("hello", "missing-model")
            self.assert_error(result, b"\\u{1b}]52;c;data\\u{7}\\u{9b}\\u{7f}\\r\\n\\tend")
            self.assertNotIn(b"\x1b", result.stderr)

    def test_chat_exact_request_body_limit(self):
        self.authenticate()
        sizes = []
        orig_chat = FAKE.Handler._chat

        def record(handler):
            sizes.append(int(handler.headers["Content-Length"]))
            orig_chat(handler)

        empty = {"model": MODEL, "stream": True, "messages": [{"role": "user", "content": ""}]}
        question = b"a" * (MAX_REQUEST - len(json.dumps(empty, separators=(",", ":")).encode()))
        with patch.object(FAKE.Handler, "_chat", record):
            result = self.command("chat", "--model", MODEL, input=question + b"\n/exit\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout)
            extra = self.command("chat", "--model", MODEL, input=question + b"a\n/exit\n")
            self.assertEqual(extra.stdout, b"")
            self.assertIn(b"exceeds the 256 KiB chat request limit", extra.stderr)
        self.assertEqual(sizes, [MAX_REQUEST])

    def test_chat_uses_agent_serving_model_and_lists_fallback(self):
        self.authenticate()
        path = self.directory / "agent.sock"
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(path))
            listener.listen()
            listener.settimeout(5)

            def reply_to_status():
                with listener.accept()[0] as agent, agent.makefile("rb") as lines:
                    lines.readline()
                    agent.sendall(json.dumps({"ok": True, "status": {
                        "connected": True, "paused": False, "contribution": {"serving": QWEN},
                    }}).encode() + b"\n")

            worker = threading.Thread(target=reply_to_status, daemon=True)
            worker.start()
            result = self.command("--socket", str(path), "chat", "hello")
            worker.join(5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, self.answer_for("hello"))

        result = self.command("--socket", str(path), "chat", "hello")
        self.assert_error(result, b"use --model ID")
        self.assertIn(MODEL.encode(), result.stderr)

    def test_chat_enforces_origin_and_refuses_redirect(self):
        self.authenticate()
        result = self.command("chat", "--model", MODEL, "--api", f"http://localhost:{self.port}", "hello")
        self.assert_error(result, b"session belongs to a different API")

        def redirect(handler):
            handler._corpo_json()
            handler._json(307, {}, {"Location": "/v1/auth/logout"})

        with patch.object(FAKE.Handler, "_chat", redirect):
            self.assert_error(self.chat("hello"), b"HTTP 307")
        self.assertEqual(self.chat("hello").returncode, 0, "session was sent to the redirect target")

    def test_logout_revokes_bound_session(self):
        self.env["PUBLICOMPUTE_API"] = f"http://localhost:{self.port}"
        self.authenticate()
        token = json.loads(self.session.read_text())["token"]
        result = self.command("logout", env=dict(self.env, PUBLICOMPUTE_API="http://invalid.example.test"))
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"", b"Logged out.\n"))
        self.assertFalse(self.session.exists())
        self.assertEqual(self.request("GET", "/v1/me", cookie=f"pc_session={token}")[0], 401)

    def test_logout_removes_session_when_revocation_fails(self):
        self.authenticate()
        self.server.shutdown()
        self.server.server_close()
        result = self.command("logout")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"server revocation could not be confirmed", result.stderr)
        self.assertFalse(self.session.exists())

    def test_interactive_ctrl_c_stops_answer_and_keeps_conversation(self):
        self.authenticate()
        calls = []
        deterministic_reply = FAKE.scegli_risposta

        def remember(messages):
            calls.append(messages)
            return deterministic_reply(messages)

        with patch.object(FAKE, "scegli_risposta", side_effect=remember), self.tty_command("chat", "--model", QWEN) as (child, master, _):
            err = child.stderr.fileno()
            out = child.stdout.fileno()
            read_until(err, b"> ")
            os.write(master, b"first question\n")
            read_until(err, b"tok/s\n> ")
            read_until(out, deterministic_reply([{"role": "user", "content": "first question"}]).encode())

            self.server.stato.fast = False
            os.write(master, b"interrupt this answer\n")
            expected = deterministic_reply([{"role": "user", "content": "interrupt this answer"}])
            read_until(out, expected.split()[0].encode())
            os.write(master, b"\x03")
            read_until(err, b"answer stopped\n> ", timeout=1)
            self.assertIsNone(child.poll())

            self.server.stato.fast = True
            os.write(master, b"third question\n")
            read_until(err, b"tok/s\n> ")
            os.write(master, b"\x04")
            child.communicate(timeout=6)
            self.assertEqual(child.returncode, 0)

        self.assertEqual(calls[-1], [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": self.answer_for("first question").decode()},
            {"role": "user", "content": "third question"},
        ])

    def test_failed_or_cancelled_trim_candidate_keeps_committed_turn(self):
        self.authenticate()
        calls = []
        initial_answer = "a" * (MAX_REQUEST - 500)
        waiting = threading.Event()
        finish_cancelled = threading.Event()

        def events(body):
            messages = body["messages"]
            calls.append(messages)
            question = messages[-1]["content"]
            if question.startswith("fail "):
                yield {"error": {"message": "fixture failed"}}
                return
            if question.startswith("cancel "):
                yield delta("</think>partial")
                waiting.set()
                finish_cancelled.wait(timeout=6)
                return
            yield delta(initial_answer if question == "first" else "last answer")
            yield from finished()

        try:
            with self.stream_fixture(events), self.tty_command("chat", "--model", MODEL, stdout=subprocess.DEVNULL) as (child, master, _):
                err = child.stderr.fileno()
                read_until(err, b"> ")
                os.write(master, b"first\n")
                read_until(err, b"> ")
                os.write(master, b"fail " + b"b" * 1000 + b"\n")
                failure = read_until(err, b"fixture failed\n> ")
                self.assertNotIn(b"dropped oldest turns", failure)
                os.write(master, b"cancel " + b"c" * 1000 + b"\n")
                self.assertTrue(waiting.wait(3))
                os.write(master, b"\x03")
                stopped = read_until(err, b"answer stopped\n> ")
                self.assertNotIn(b"dropped oldest turns", stopped)
                os.write(master, b"last\n/exit\n")
                _, err = child.communicate(timeout=6)
                self.assertEqual(child.returncode, 0, err)
        finally:
            finish_cancelled.set()

        self.assertEqual([len(messages) for messages in calls], [1, 1, 1, 3])
        self.assertEqual([message["role"] for message in calls[-1]], ["user", "assistant", "user"])
        self.assertEqual(calls[-1][0]["content"], "first")
        self.assertTrue(calls[-1][1]["content"] == initial_answer)
        self.assertEqual(calls[-1][2]["content"], "last")

    def test_non_qwen_answer_streams_before_the_stream_ends(self):
        # Only Qwen opens reasoning in the prompt; other models must stream at once, as the web chat does.
        self.authenticate()
        calls = []
        first_sent = threading.Event()
        send_rest = threading.Event()

        def events(body):
            messages = body["messages"]
            calls.append(messages)
            if messages[-1]["content"] == "first":
                yield delta("streamed ")
                first_sent.set()
                send_rest.wait(timeout=6)
                yield delta("answer")
            else:
                yield delta("second answer")
            yield from finished()

        try:
            with self.stream_fixture(events), self.tty_command("chat", "--model", MODEL) as (child, master, _):
                read_until(child.stderr.fileno(), b"> ")
                os.write(master, b"first\n")
                self.assertTrue(first_sent.wait(3))
                self.assertTrue(read_until(child.stdout.fileno(), b"streamed").startswith(b"streamed"))
                send_rest.set()
                os.write(master, b"second\n/exit\n")
                out, err = child.communicate(timeout=6)
                self.assertEqual(child.returncode, 0, err)
                self.assertIn(b"second answer", out)
        finally:
            send_rest.set()

        self.assertEqual(calls[-1][1], {"role": "assistant", "content": "streamed answer"})

    def test_ctrl_c_cancels_write_to_full_stdout_pipe(self):
        self.authenticate()
        sent = threading.Event()

        def events(_body):
            yield delta("</think>answer")
            yield from finished()
            sent.set()

        read_fd, write_fd = os.pipe()
        try:
            os.set_blocking(write_fd, False)
            try:
                while True:
                    os.write(write_fd, b"x" * 4096)
            except BlockingIOError:
                pass
            os.set_blocking(write_fd, True)
            with self.stream_fixture(events), self.tty_command("chat", "--model", MODEL, stdout=write_fd) as (child, master, _):
                err = child.stderr.fileno()
                read_until(err, b"> ")
                os.write(master, b"question\n")
                self.assertTrue(sent.wait(3))
                self.assertFalse(select.select([child.stderr], [], [], 0.2)[0], "answer did not wait for space in the pipe")
                os.write(master, b"\x03")
                read_until(err, b"answer stopped\n> ", timeout=1)
                self.assertTrue(os.get_blocking(write_fd), "stdout flags were not restored")
                os.write(master, b"/exit\n")
                child.communicate(timeout=6)
                self.assertEqual(child.returncode, 0)
        finally:
            os.close(write_fd)
            os.close(read_fd)

    def test_interactive_session_expiry_prevents_next_request(self):
        self.authenticate()
        stored = json.loads(self.session.read_text())
        stored["expires_at"] = int(time.time()) + 3
        self.session.write_text(json.dumps(stored))
        requests = []

        def events(body):
            requests.append(body)
            yield delta("answer")
            yield from finished()

        with self.stream_fixture(events), self.tty_command("chat", "--model", MODEL) as (child, master, _):
            err = child.stderr.fileno()
            read_until(err, b"> ")
            os.write(master, b"first question\n")
            read_until(err, b"> ")
            time.sleep(max(0, stored["expires_at"] - time.time()))
            os.write(master, b"expired question\n/exit\n")
            out, err = child.communicate(timeout=6)
            self.assertNotEqual(child.returncode, 0)
            self.assertEqual(out, b"answer")
            self.assertIn(b"session expired, run publicompute login", err)
        self.assertEqual(len(requests), 1)

    def test_interactive_exit_from_pipe(self):
        self.authenticate()
        result = self.command("chat", "--model", MODEL, input=b"/exit\n")
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, b"", b""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
