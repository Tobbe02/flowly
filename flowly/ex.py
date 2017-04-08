"""Tools for executors.
"""
from contextlib import contextmanager
import functools as ft
import inspect
import itertools as it
import threading
import uuid

try:
    from ipywidgets.widgets import DOMWidget
    from IPython.display import display_javascript, Javascript
    from traitlets import Unicode, List, Dict, Bool
    _ipy_available = True

except ImportError:
    _ipy_available = False

from .tz import try_call, raise_


def get(f):
    return try_call(f.result) if f.done() else try_call(raise_, NotDone)


class NotDone(Exception):
    pass


def submit(*args, **kwargs):
    """Submit a job to an executor.

    Usage::

        # submit the function to an executor
        f = submit(ex, func, *args, **kwargs)

        # submit with a tag
        f = submit(ex, tag, func, *args, **kwargs)

        # as a decorator
        @submit(ex, tag)
        def func():
            pass

        # submit a generator yielding status messages
        def gen(*arguments):
            solution = 0
            for step in range(100):
                solution = update(solution, *arguments)
                yield 'step {} / 100'.format(step + 1)

            return solution

        f = submit(ex, optimize, gen, *arguments)

    """
    ex, func_or_tag, args = args[0], args[1], args[2:]

    if not callable(func_or_tag) and not args:
        return _create_decorator(ex, func_or_tag, kwargs)

    elif not callable(func_or_tag):
        func, args = args[0], args[1:]

        if not inspect.isgeneratorfunction(func):
            return ex.submit_tagged(func_or_tag, func, *args, **kwargs)

        else:
            return submit_status(ex, func_or_tag, func, *args, *kwargs)

    else:
        return submit(func_or_tag, *args, **kwargs)


def _create_decorator(ex, tag, kwargs):
    def impl(func):
        submit(ex, tag, func, **kwargs)
        return func

    return impl


def submit_status(client, tag, genfunc, *args, **kwargs):
    """Submit a function that yields regular status updates.

    :param client:
        must be a ``DashboardExecutor`` that wraps a distributed client.
    """
    assert inspect.isgeneratorfunction(genfunc)

    key = str(uuid.uuid4())
    tag, future = client._submit_tagged(tag, _run_generator, key, genfunc, *args, **kwargs)
    channel = client.channel('generator-{}'.format(key))
    _run_in_background(_update_message, client=client, tag=tag, channel=channel)
    return future


def _update_message(client, tag, channel):
    _set_message = getattr(client, '_set_message', lambda tag, message: None)

    for message in channel:
        _set_message(tag, message)


def submit_generator(client, func, *args, **kwargs):
    """Submit a generator and yield its items as they become available

    A typical use case may be status messages of an optimization. Note, the
    items returned should be small for performance reasons.

    :param client:
        must be a distributed client

    """
    assert inspect.isgeneratorfunction(func)

    key = str(uuid.uuid4())
    future = client.submit(_run_generator, key, func, *args, **kwargs)
    channel = client.channel('generator-{}'.format(key))

    return _SubmittedGeneratorIterator(channel, future)


class DashboardExecutor(object):
    def __init__(self, executor):
        self.executor = executor
        self.ids = iter('job{}'.format(id) for id in it.count())
        self._dashboard = None

    def channel(self, *args, **kwargs):
        return self.executor.channel(*args, **kwargs)

    @property
    def dashboard(self):
        self.init_js()

        if self._dashboard is not None:
            return self._dashboard

        self._dashboard = Dashboard()
        return self._dashboard

    def submit_tagged(self, tag, func, *args, **kwargs):
        _, future = self._submit_tagged(tag, func, *args, **kwargs)
        return future

    def _submit_tagged(self, tag, func, *args, **kwargs):
        future = self.submit(func, *args, **kwargs)

        tag = '{}/{}'.format(next(self.ids), tag)
        self.dashboard.add(tag)
        future.add_done_callback(ft.partial(self._update_job, tag))

        return tag, future

    def submit(self, func, *args, **kwargs):
        return self.executor.submit(func, *args, **kwargs)

    def _update_job(self, tag, future):
        self.dashboard.update(
            tag,
            state='done' if future.exception() is None else 'failed'
        )

    def _set_message(self, tag, message):
        self.dashboard.update(tag, message=str(message))

    @classmethod
    def init_js(cls):
        display_javascript(Javascript(_dashboard_javascript))


if _ipy_available:
    class Dashboard(DOMWidget):
        _view_name = Unicode('FutureList').tag(sync=True)
        _view_module = Unicode('flowly').tag(sync=True)

        futures = List(Dict()).tag(sync=True)
        notify = Bool().tag(sync=True)

        def add(self, tag, state='running'):
            self.futures = list(self.futures) + [{'tag': tag, 'state': state}]

        def update(self, tag, **items):
            self.futures = [
                dict(job, **items) if job['tag'] == tag else job
                for job in self.futures
            ]


class _SubmittedGeneratorIterator(object):
    def __init__(self, channel, future):
        self.channel = channel
        self.future = future

        self.iter = iter(self.channel)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iter)

        except StopIteration:
            pass

        # NOTE: this will also re-reaise remote exceptions
        result = self.future.result()
        raise StopIteration(result)


def _run_in_background(func, *args, **kwargs):
    t = threading.Thread(target=func, args=args, kwargs=kwargs)
    t.start()


@contextmanager
def worker_client(timeout=3):
    from distributed.utils import sync
    from distributed.worker import thread_state
    from distributed.worker_client import WorkerClient

    address = thread_state.execution_state['scheduler']
    worker = thread_state.execution_state['worker']

    with WorkerClient(address, loop=worker.loop) as wc:
        # Make sure connection errors are bubbled to the caller
        sync(wc.loop, wc._start, timeout=timeout)
        assert wc.status == 'running'
        yield wc


def _run_generator(key, func, *args, **kwargs):
    with worker_client() as client:
        channel = client.channel('generator-{}'.format(key))

        try:
            gen = func(*args, **kwargs)

            while True:
                try:
                    channel.append(next(gen))

                except StopIteration as exc:
                    # use getattr for python2 compat
                    return getattr(exc, 'value', None)

        finally:
            channel.stop()


_dashboard_javascript = """
require.undef('flowly');

define('flowly', ["jupyter-js-widgets"], function(widgets) {
    return {
        FutureList: widgets.DOMWidgetView.extend({
            render: function() {
                var header = document.createElement('div');
                header.textContent = 'Jobs';
                header.style.fontSize = '1em';
                header.style.fontWeight = 'bold';

                var listNode = document.createElement('ul');

                this.el.appendChild(header);
                this.el.appendChild(listNode);

                this.on_change();
                this.model.on('change', this.on_change, this);
            },

            removeJob: function(tag) {
                var futures = this.model.get('futures');
                futures = futures.filter(function(job) {
                    return job['tag'] != tag;
                });

                this.model.set('futures', futures);
                this.touch();
            },

            notifyOnStateChange: function() {
                this.states = this.states || {};

                this.model.get('futures').forEach(function(job) {
                    if(this.states[job['tag']] == undefined) {
                        this.states[job['tag']] = job['state'];
                        return;
                    }
                    var changed = this.states[job['tag']] != job['state'];
                    this.states[job['tag']] = job['state'];

                    if(changed) {
                        this.on_state_change(job);
                    }
                }, this);
            },

            on_state_change: function(job) {
                if(this.model.get('notify')) {
                    var n = new Notification(job['tag'] + ': ' + job['state'], {'tag': 'dashboad'});
                    n.onclick = n.close.bind(n);
                }
            },

            on_change: function() {
                var this_ = this;
                var futures = this.model.get('futures');

                var listNode = this.el.lastChild;
                while(listNode.firstChild) {
                    listNode.removeChild(listNode.firstChild);
                }

                futures.forEach(function(job) {
                    var li = document.createElement('li');

                    if(job['state'] == 'done') {
                        li.style.color = '#00bb00';
                    }
                    else if(job['state'] == 'failed') {
                        li.style.color = '#bb0000';
                    }

                    var button = document.createElement('button');
                    button.textContent = '[x]';
                    button.style.cssText = "background: none; border: 0px; padding: 0px; margin: 0px 0.2em;";
                    button.addEventListener('click', this_.removeJob.bind(this_, job['tag']));

                    var text = document.createElement('span');

                    if(job['message']) {
                        text.textContent = job['tag'] + ': ' + job['state'] + ' - ' + job['message'];
                    }
                    else {
                        text.textContent = job['tag'] + ': ' + job['state'];
                    }

                    li.appendChild(button);
                    li.appendChild(text);
                    listNode.appendChild(li);
                });

                this.notifyOnStateChange();
            },
        })
    };
});
"""
