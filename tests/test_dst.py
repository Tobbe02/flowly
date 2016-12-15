from flowly.dst import LocalCluster
from flowly.tz import raise_

import pytest


def test_context_manager():
    mock_subprocess = MockSubprocess()
    with LocalCluster(
            workers=10,
            client_class=MockClient,
            subprocess=mock_subprocess,
            wait_for_server=lambda _: None
    ) as cluster:
        assert hasattr(cluster, 'get')
        assert hasattr(cluster, 'client')

        assert len(mock_subprocess.processes) == 11
        assert not any(proc.killed for proc in mock_subprocess.processes)

    assert all(proc.killed for proc in mock_subprocess.processes)


def test_context_manager_exceptions():
    mock_subprocess = MockSubprocess()

    with pytest.raises(RuntimeError):
        with LocalCluster(
                workers=10,
                client_class=lambda _: raise_(RuntimeError),
                subprocess=mock_subprocess,
                wait_for_server=lambda _: None
        ):
            pass

    assert all(proc.killed for proc in mock_subprocess.processes)


class MockClient(object):
    def __init__(self, addresses):
        pass

    def get(self, *args):
        pass


class MockSubprocess(object):
    def __init__(self):
        self.processes = []

    def Popen(self, args):
        proc = MockProcess(args)
        self.processes.append(proc)
        return proc


class MockProcess(object):
    def __init__(self, args):
        self.args = args
        self.killed = False

    def kill(self):
        self.killed = True
