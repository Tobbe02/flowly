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
    'clear_checkpoints',
    'checkpoint',
    'with_checkpoint',
    'rewrite_checkpoints',
]


class with_checkpoint(object):
    """Apply a transformation with checkpointing.

    Before the transform is executued, the given checkpoint will be checked
    for prior invocations of the transform with the passed argument.
    To check for prior invocations :func:`flowly.hashing.functional_hash` will
    be used.

    :param flowly.checkpoint.checkpoint checkpoint:
        a checkpoint object that supports store and restore for
        object, transform pairs.

    :param Callable[Any,Any] transform:
        the transform that will be applied.
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


def clear_checkpoints(target, object=None, tag=None):
    """Clear all checkpoints from target that match certain conditions.

    :param Optional[Any] object:
        if given, remove all checkpoints where ``obj`` was used as an input.

    :param Optional[str] tag:
        if given , remove all checkpoints for the given tag.
    """
    predicates = []

    if object is not None:
        obj_hash = compute_hash(functional_system, object)
        predicates.append(lambda c_obj_hash, _1, _2: c_obj_hash == obj_hash)

    if tag is not None:
        predicates.append(lambda _1, c_tag, _2: c_tag == tag)

    if not predicates:
        target.clear()
        return

    keys_to_delete = [
        key
        for key in target.keys()
        if all(predicate(*key) for predicate in predicates)
    ]

    for key in keys_to_delete:
        del target[key]


class checkpoint(object):
    """A mapping like object to support tracking function execution.

    A checkpoint can also double as a callable in which case it acts as the
    identity function.
    Therefore, a checkpoint can be placed in a pipeline without changing the
    results.

    To add the actual checkpointing support, apply
    :func:`flowly.checkpoint.add_checkpoints` to the transform before executing
    it.

    :param Mapping[Tuple[str,str,str],Any] target:
        a mapping that will be used to track executions.
        The keys will be `hash(obj), tag, hash(transform)` triples and the
        values the result of applying the transform to the object.

    :param str tag:
        a tag to identify the checkpoint.
        It will be used in logging.
    """
    def __init__(self, target, tag='<untagged>'):
        self.target = target
        self.tag = tag

    def __call__(self, obj):
        return obj

    def _key(self, key):
        obj_hash, transform_hash = key
        return obj_hash, self.tag, transform_hash

    def __contains__(self, key):
        return self._key(key) in self.target

    def __getitem__(self, key):
        return self.target[self._key(key)]

    def __setitem__(self, key, result):
        self.target[self._key(key)] = result


@base_system.bind(checkpoint)
@composite_hash
def base_system_checkpoint(obj, _):
    # NOTE: skip the contents of the target, otherwise changes in the target
    #       will change the checkpoint.
    return type(obj), id(obj.target), obj.tag


def add_checkpoints(transform):
    """Rewrite a transformation to add checkpointing support.

    It can be used as a rewrite rule for :func:`flowly.tz.apply` and
    :func:`flowly.dsk.apply`.
    For example::

        apply(transform, obj, rewrites=[add_checkpoints])

    For example::

        add_checkpoints(
            chained(
                func1,
                func2,
                checkpoint(target=_checkpoints, tag='step 1'),
                func3,
                checkpoint(target=_checkpoints, tag='step 1'),
                func4
            )
        )

    is equivalent to::

        chained(
            with_checkpoint(
                checkpoint(target=_checkpoints, tag='step 1'),
                chained(
                    with_checkpoint(
                        checkpoint(target=_checkpoints, tag='step 1'),
                        chained(
                            func1,
                            func2,
                        )
                    ),
                    func3,
                ),
            ),
            func4,
        )
    """
    return rewrite_checkpoints(transform, rewrite_checkpoints)


rewrite_checkpoints = Dispatch()


@rewrite_checkpoints.default
def rewrite_checkpoints_skip(obj, _):
    return obj
