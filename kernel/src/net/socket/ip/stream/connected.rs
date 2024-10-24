// SPDX-License-Identifier: MPL-2.0

use alloc::sync::Weak;

use aster_bigtcp::{
    errors::tcp::{RecvError, SendError},
    socket::{SocketEventObserver, TcpState},
    wire::IpEndpoint,
};

use crate::{
    events::IoEvents,
    net::{
        iface::BoundTcpSocket,
        socket::util::{send_recv_flags::SendRecvFlags, shutdown_cmd::SockShutdownCmd},
    },
    prelude::*,
    process::signal::Pollee,
    util::{MultiRead, MultiWrite},
};

pub struct ConnectedStream {
    bound_socket: BoundTcpSocket,
    remote_endpoint: IpEndpoint,
    /// Indicates whether this connection is "new" in a `connect()` system call.
    ///
    /// If the connection is not new, `connect()` will fail with the error code `EISCONN`,
    /// otherwise it will succeed. This means that `connect()` will succeed _exactly_ once,
    /// regardless of whether the connection is established synchronously or asynchronously.
    ///
    /// If the connection is established synchronously, the synchronous `connect()` will succeed
    /// and any subsequent `connect()` will fail; otherwise, the first `connect()` after the
    /// connection is established asynchronously will succeed and any subsequent `connect()` will
    /// fail.
    is_new_connection: bool,
}

impl ConnectedStream {
    pub fn new(
        bound_socket: BoundTcpSocket,
        remote_endpoint: IpEndpoint,
        is_new_connection: bool,
    ) -> Self {
        Self {
            bound_socket,
            remote_endpoint,
            is_new_connection,
        }
    }

    pub fn shutdown(&self, _cmd: SockShutdownCmd) -> Result<()> {
        // TODO: deal with cmd
        self.bound_socket.close();
        Ok(())
    }

    pub fn try_recv(&self, writer: &mut dyn MultiWrite, _flags: SendRecvFlags) -> Result<usize> {
        let result = self.bound_socket.recv(|socket_buffer| {
            match writer.write(&mut VmReader::from(&*socket_buffer)) {
                Ok(len) => (len, Ok(len)),
                Err(e) => (0, Err(e)),
            }
        });

        match result {
            Ok(Ok(0)) => return_errno_with_message!(Errno::EAGAIN, "the receive buffer is empty"),
            Ok(Ok(recv_bytes)) => Ok(recv_bytes),
            Ok(Err(e)) => Err(e),
            Err(RecvError::Finished) => Ok(0),
            Err(RecvError::InvalidState) => {
                if self.before_established() {
                    return_errno_with_message!(Errno::EAGAIN, "the connection is not established");
                }
                return_errno_with_message!(Errno::ECONNRESET, "the connection is reset")
            }
        }
    }

    pub fn try_send(&self, reader: &mut dyn MultiRead, _flags: SendRecvFlags) -> Result<usize> {
        let result = self.bound_socket.send(|socket_buffer| {
            match reader.read(&mut VmWriter::from(socket_buffer)) {
                Ok(len) => (len, Ok(len)),
                Err(e) => (0, Err(e)),
            }
        });

        match result {
            Ok(Ok(0)) => return_errno_with_message!(Errno::EAGAIN, "the send buffer is full"),
            Ok(Ok(sent_bytes)) => Ok(sent_bytes),
            Ok(Err(e)) => Err(e),
            Err(SendError::InvalidState) => {
                if self.before_established() {
                    return_errno_with_message!(Errno::EAGAIN, "the connection is not established");
                }
                // FIXME: `EPIPE` is another possibility, which means that the socket is shut down
                // for writing. In that case, we should also trigger a `SIGPIPE` if `MSG_NOSIGNAL`
                // is not specified.
                return_errno_with_message!(Errno::ECONNRESET, "the connection is reset");
            }
        }
    }

    pub fn local_endpoint(&self) -> IpEndpoint {
        self.bound_socket.local_endpoint().unwrap()
    }

    pub fn remote_endpoint(&self) -> IpEndpoint {
        self.remote_endpoint
    }

    pub fn check_new(&mut self) -> Result<()> {
        if !self.is_new_connection {
            return_errno_with_message!(Errno::EISCONN, "the socket is already connected");
        }

        self.is_new_connection = false;
        Ok(())
    }

    pub(super) fn init_pollee(&self, pollee: &Pollee) {
        pollee.reset_events();
        self.update_io_events(pollee);
    }

    pub(super) fn update_io_events(&self, pollee: &Pollee) {
        self.bound_socket.raw_with(|socket| {
            if socket.can_recv() {
                pollee.add_events(IoEvents::IN);
            } else {
                pollee.del_events(IoEvents::IN);
            }

            if socket.can_send() {
                pollee.add_events(IoEvents::OUT);
            } else {
                pollee.del_events(IoEvents::OUT);
            }
        });
    }

    pub(super) fn set_observer(&self, observer: Weak<dyn SocketEventObserver>) {
        self.bound_socket.set_observer(observer)
    }

    /// Returns whether the connection is before established.
    ///
    /// Note that a newly accepted socket may not yet be in the [`TcpState::Established`] state.
    /// The accept syscall only verifies that a connection request is incoming by ensuring
    /// that the backlog socket is not in the [`TcpState::Listen`] state.
    /// However, the socket might still be waiting for further ACKs to complete the establishment process.
    /// Therefore, it could be in either the [`TcpState::SynSent`] or [`TcpState::SynReceived`] states.
    /// We must wait until the socket reaches the established state before it can send and receive data.
    ///
    /// FIXME: Should we check established state in accept or here?
    fn before_established(&self) -> bool {
        self.bound_socket.raw_with(|socket| {
            socket.state() == TcpState::SynSent || socket.state() == TcpState::SynReceived
        })
    }
}
