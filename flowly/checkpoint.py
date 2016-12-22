from __future__ import print_function, division, absolute_import

import logging

from ._dispatch import Dispatch
from .hashing import (
    base_system,
    composite_hash,
    compute_hash,
    functional_system,
)

_logger = logging.getLogger()


__all__ = [
    'add_checkpoints',
    'checkpoint',
    'with_checkpoint',
    'rewrite_checkpoints',
]


class with_checkpoint(object):
    """Apply a transformation with checkpointing.
    """
    def __init__(self, checkpoint, transform):
        self.checkpoint = checkpoint
        self.transform = transform

    def __call__(self, obj):
        key = self._compute_key(obj)

        if key in self.checkpoint:
            _logger.info('restore object at %r', self.checkpoint.tag)
            return self.checkpoint[key]

        _logger.info('compute, then store object at %r', self.checkpoint.tag)
        obj = self.transform(obj)
        self.checkpoint[key] = obj
        return obj

    def _compute_key(self, obj):
        return (
            compute_hash(functional_system, obj),
            compute_hash(functional_system, self.transform)
        )

    def __repr__(self):
        return 'with_checkpoint({}, {})'.format(self.checkpoint, self.transform)


@base_system.bind(with_checkpoint)
@composite_hash
def base_system_with_checkpoint(obj, _):
    return type(obj), obj.checkpoint, obj.transform


class checkpoint(object):
    def __init__(self, target, tag='<untagged>'):
        self.target = target
        self.tag = tag

    def __call__(self, obj):
        return obj

    def __contains__(self, key):
        return key in self.target

    def __getitem__(self, key):
        return self.target[key]

    def __setitem__(self, key, result):
        self.target[key] = result


@base_system.bind(checkpoint)
@composite_hash
def base_system_checkpoint(obj, _):
    return type(obj), id(obj.target), obj.tag


def add_checkpoints(transform):
    return rewrite_checkpoints(transform, rewrite_checkpoints)


rewrite_checkpoints = Dispatch()


@rewrite_checkpoints.default
def rewrite_checkpoints_skip(obj, _):
    return obj
