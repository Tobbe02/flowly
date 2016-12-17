import logging
import multiprocessing
import os
import random
import subprocess
import socket
import sys
import time

import psutil

from distributed import Client

_logger = logging.getLogger(__name__)


_dask_scheduler_cmd = [sys.executable, '-m', 'distributed.cli.dask_scheduler']
_dask_worker_cmd = [sys.executable, '-m', 'distributed.cli.dask_worker']


# TODO: fix worker ports
# TOOD: allow to set executables
class LocalCluster(object):
    """A 'local' ``distributed`` cluster.

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

        cluster.stop()

    For long running usage of the cluster, explicitly calling ``start`` and
    ``stop`` may be the preferred option.

    :param int workers:
        the number of workers to start. If ``worker <= 0``, the local cluster
        will contain as many workers as the CPUs are available.

    :param int scheduler_port:
        the port of the scheduler to use.

    :param Optional[int] hash_seed:
        to get reliable hasing across workers, the ``PYTHONHASHSEED``
        environment variable will be set. If given, this parameter specifies
        the value of said variable. Otherwise, the pre existing value of
        ``PYTHONHASHSEED`` will be used if it is a number, otherwise a random
        numer is chosen.

    :ivar client:
        after the cluster is started a distributed client connected to the
        scheduler.

    :ivar get:
        after the cluster is started, a reference to the ``get`` method of the
        client. It can be used as an argument to the ``compute`` method of dask
        objects.
    """
    def __init__(
            self,
            workers=-1,
            scheduler_port=8786,
            hash_seed=None,
    ):
        if workers <= 0:
            workers = multiprocessing.cpu_count()

        if hash_seed is None:
            hash_seed = _get_hash_seed(os.environ)

        self.hash_seed = hash_seed
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
            self.scheduler = self._popen(
                _dask_scheduler_cmd + [
                    '--port', str(self.scheduler_port),
                    '--host', '127.0.0.1',
                ]
            )

            _logger.info("wait for scheduler")
            _wait_for_server(scheduler_address)

            for _ in range(self.n_workers):
                _logger.info("start worker")
                self.workers.append(self._popen(
                    _dask_worker_cmd + ['--nthreads', '1', scheduler_address]
                ))

            self.client = Client(scheduler_address)
            self.get = self.client.get

        except:
            self.stop()
            raise

    def _popen(self, args):
        # TODO: keep hashseed, if its a number
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = str(self.hash_seed)

        _logger.info('start process %s', args)
        return subprocess.Popen(args, env=env, shell=False)

    def stop(self):
        """Stop the cluster by killing any external processes.
        """
        if self.scheduler:
            _logger.info("kill scheduler")
            _kill(self.scheduler)

        for worker in self.workers:
            _logger.info("kill worker %s", worker)
            _kill(worker)

        self.workers = []
        self.scheduler = None
        self.client = None
        self.get = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def _get_hash_seed(env):
    try:
        return int(env['PYTHONHASHSEED'])

    except (TypeError, KeyError):
        # TODO: is this range fixed or platform dependent?
        return random.randint(0, 4294967295)


def _kill(proc):
    """copied from http://stackoverflow.com/a/25134985
    """
    process = psutil.Process(proc.pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


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
