import logging
import multiprocessing
import subprocess
import socket
import time

from distributed import Client

_logger = logging.getLogger(__name__)


# TODO: fix worker ports
# TOOD: allow to set executables
class LocalCluster(object):
    """Start a local ``distributed`` cluster.

    The cluster object may be used as a context manager.
    The cluster is started before entering the ``with`` statement and stopped
    when exiting the cluster.
    For example::

        with LocalCluster() as cluster:
            ...

    is roughly equivalent to::

        cluster = LocalCluster()
        cluster.start()

        ...

        cluster.end()

    :param int workers:
        the number of workers to start. If ``worker <= 0``, the local cluster
        will contain as many workers as the CPUs are available.

    :param int scheduler_port:
        the port of the scheduler to use.

    :ivar client:
        after the cluster is started a distributed client connected to the
        scheduler.

    :ivar get:
        after the cluster is started, a reference to the ``get`` method of the
        client.
    """
    def __init__(
            self,
            workers=-1,
            scheduler_port=8786,
            subprocess=subprocess,
            client_class=Client,
            wait_for_server=None,
    ):
        if workers <= 0:  # pragma: no cover
            workers = multiprocessing.cpu_count()

        if wait_for_server is None:  # pragma: no cover
            wait_for_server = _wait_for_server

        self.client_class = client_class
        self.subprocess = subprocess
        self.wait_for_server = wait_for_server

        self.scheduler_port = scheduler_port
        self.scheduler = None
        self.workers = []
        self.n_workers = workers

    def start(self):
        """Start the cluster by starting all required processes.

        If the cluster is already running, stop it.
        """
        scheduler_address = '127.0.0.1:{}'.format(self.scheduler_port)

        self.stop()

        try:
            _logger.info("start scheduler")
            self.scheduler = self._popen('dask-scheduler', '--port', str(self.scheduler_port))

            _logger.info("wait for scheduler")
            self.wait_for_server(scheduler_address)

            for _ in range(self.n_workers):
                _logger.info("start worker")
                self.workers.append(self._popen('dask-worker', '--nthreads', '1', scheduler_address))

            self.client = self.client_class('127.0.0.1:{}'.format(self.scheduler_port))
            self.get = self.client.get

        except:
            self.stop()
            raise

    def _popen(self, *args):
        return self.subprocess.Popen(args)

    def stop(self):
        """Stop the cluster by killing any external processes.
        """
        for worker in self.workers:
            worker.kill()

        if self.scheduler:
            self.scheduler.kill()

        self.workers = []
        self.scheduler = None
        self.client = None
        self.get = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def _wait_for_server(address, sleep=1, retries=10):  # pragma: no cover
    for _ in range(retries):
        if _check_server_is_listening(address):
            break

        time.sleep(sleep)

    else:
        raise RuntimeError('could not connect to scheduler')


def _check_server_is_listening(address):  # pragma: no cover
    address, port = address.split(':')
    port = int(port)

    s = socket.socket()
    try:
        s.connect((address, port))

    except socket.error:
        return False

    else:
        return True

    finally:
        s.close()
