"""Bounded authenticated inference admission and backend-to-writer handoff."""

import queue
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


class AdmissionError(RuntimeError):
    def __init__(self, event: str):
        self.event = event
        super().__init__(event.replace("_", " "))


@dataclass(eq=False)
class InferenceTicket:
    deadline: float
    worker: bool = False
    queued_bytes: int = 0
    cancelled: threading.Event = field(default_factory=threading.Event)
    released: bool = False
    reader_started: bool = False
    reader_done: bool = False
    writer_done: bool = False


class InferenceAdmission:
    """FIFO through body validation and lease acquisition; never leases a waiter.

    Worker reservations bound active bodies/readers. Waiting bodies have their
    own aggregate byte budget. A ticket stays at the head until its lease is
    acquired, so arrivals cannot overtake a slow body or a model activation.
    """

    def __init__(self, workers: int, capacity: int, body_bytes: int, timeout: float):
        self.slots = threading.BoundedSemaphore(workers)
        self.capacity = capacity
        self.body_limit = body_bytes
        self.timeout = timeout
        self.pending: deque[InferenceTicket] = deque()
        self.tickets: set[InferenceTicket] = set()
        self.body_bytes = 0
        self.condition = threading.Condition()
        self.stopped = False

    def reserve(self, length: int, *, quiescing: bool) -> InferenceTicket:
        with self.condition:
            if self.stopped:
                raise AdmissionError("server_shutdown")
            worker = not self.pending and not quiescing and self.slots.acquire(False)
            if not worker:
                waiting = sum(not ticket.worker for ticket in self.pending)
                if waiting >= self.capacity or self.body_bytes + length > self.body_limit:
                    raise AdmissionError("queue_full")
            ticket = InferenceTicket(time.monotonic() + self.timeout,
                                     worker=worker, queued_bytes=0 if worker else length)
            self.body_bytes += ticket.queued_bytes
            self.pending.append(ticket)
            self.tickets.add(ticket)
            return ticket

    def wait_turn(self, ticket: InferenceTicket, ready: Callable[[], bool]) -> None:
        with self.condition:
            while True:
                if self.stopped:
                    raise AdmissionError("server_shutdown")
                if ticket.cancelled.is_set():
                    raise AdmissionError("client_disconnect")
                remaining = ticket.deadline - time.monotonic()
                if remaining <= 0:
                    raise AdmissionError("queue_expired")
                if self.pending[0] is ticket and ready():
                    if ticket.worker or self.slots.acquire(False):
                        ticket.worker = True
                        return
                self.condition.wait(min(0.05, remaining))

    def started(self, ticket: InferenceTicket) -> None:
        with self.condition:
            assert self.pending[0] is ticket
            self.pending.popleft()
            self.body_bytes -= ticket.queued_bytes
            ticket.queued_bytes = 0
            self.condition.notify_all()

    def finish_reader(self, ticket: InferenceTicket) -> None:
        with self.condition:
            ticket.reader_done = True
            if ticket.writer_done:
                self._release(ticket)

    def finish_writer(self, ticket: InferenceTicket) -> None:
        with self.condition:
            ticket.writer_done = True
            if not ticket.reader_started or ticket.reader_done:
                self._release(ticket)

    def _release(self, ticket: InferenceTicket) -> None:
        # Both halves must finish before another connection can use the slot.
        # Otherwise fast readers plus blocked downstream writers evade the cap.
        with self.condition:
            if ticket.released:
                return
            ticket.released = True
            if ticket in self.pending:
                self.pending.remove(ticket)
            self.body_bytes -= ticket.queued_bytes
            ticket.queued_bytes = 0
            self.tickets.discard(ticket)
            if ticket.worker:
                self.slots.release()
            self.condition.notify_all()

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            for ticket in self.tickets:
                ticket.cancelled.set()
            self.condition.notify_all()


class BackendPipe:
    """Only the reader touches HTTPResponse; only the handler writes downstream."""

    def __init__(self, ticket: InferenceTicket):
        self.ticket = ticket
        self.messages: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=8)
        self.lock = threading.Lock()
        self.socket: socket.socket | None = None
        self.deadline: float | None = None

    def put(self, kind: str, value: Any = None) -> None:
        while not self.ticket.cancelled.is_set():
            try:
                self.messages.put((kind, value), timeout=0.05)
                return
            except queue.Full:
                pass

    def attach(self, sock: socket.socket | None) -> None:
        with self.lock:
            self.socket = sock
            if self.ticket.cancelled.is_set():
                self._shutdown_socket()

    def _shutdown_socket(self) -> None:
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def abort(self) -> None:
        self.ticket.cancelled.set()
        with self.lock:
            # Closing HTTPResponse here can deadlock on its buffered reader's
            # lock. Socket shutdown wakes that reader; it owns all final closes.
            self._shutdown_socket()
