from typing import Mapping, Any, List
from collections import OrderedDict
from torch.nn.modules.module import _IncompatibleKeys, _EXTRA_STATE_KEY_SUFFIX
import itertools
from torch.nn import Module
import torch

# this patch is for adding the `assign` key to load_state_dict.
# the code is in pytorch source for version 2.1

def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs):
    r"""Copies parameters and buffers from :attr:`state_dict` into only
    this module, but not its descendants. This is called on every submodule
    in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
    module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
    For state dicts without metadata, :attr:`local_metadata` is empty.
    Subclasses can achieve class-specific backward compatible loading using
    the version number at `local_metadata.get("version", None)`.
    Additionally, :attr:`local_metadata` can also contain the key
    `assign_to_params_buffers` that indicates whether keys should be
    assigned their corresponding tensor in the state_dict.

    .. note::
        :attr:`state_dict` is not the same object as the input
        :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
        it can be modified.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        prefix (str): the prefix for parameters and buffers used in this
            module
        local_metadata (dict): a dict containing the metadata for this module.
            See
        strict (bool): whether to strictly enforce that the keys in
            :attr:`state_dict` with :attr:`prefix` match the names of
            parameters and buffers in this module
        missing_keys (list of str): if ``strict=True``, add missing keys to
            this list
        unexpected_keys (list of str): if ``strict=True``, add unexpected
            keys to this list
        error_msgs (list of str): error messages should be added to this
            list, and will be reported together in
            :meth:`~torch.nn.Module.load_state_dict`
    """
    for hook in self._load_state_dict_pre_hooks.values():
        hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
    local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}
    assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)

    for name, param in local_state.items():
        key = prefix + name
        if key in state_dict:
            input_param = state_dict[key]
            if not torch.overrides.is_tensor_like(input_param):
                error_msgs.append('While copying the parameter named "{}", '
                                  'expected torch.Tensor or Tensor-like object from checkpoint but '
                                  'received {}'
                                  .format(key, type(input_param)))
                continue

            # This is used to avoid copying uninitialized parameters into
            # non-lazy modules, since they dont have the hook to do the checks
            # in such case, it will error when accessing the .shape attribute.
            is_param_lazy = torch.nn.parameter.is_lazy(param)
            # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
            if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
                input_param = input_param[0]

            if not is_param_lazy and input_param.shape != param.shape:
                # local shape should match the one in checkpoint
                error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                  'the shape in current model is {}.'
                                  .format(key, input_param.shape, param.shape))
                continue
            try:
                with torch.no_grad():
                    if assign_to_params_buffers:
                        # Shape checks are already done above
                        if (isinstance(param, torch.nn.Parameter) and
                                not isinstance(input_param, torch.nn.Parameter)):
                            setattr(self, name, torch.nn.Parameter(input_param))
                        else:
                            setattr(self, name, input_param)
                    else:
                        param.copy_(input_param)
            except Exception as ex:
                error_msgs.append('While copying the parameter named "{}", '
                                  'whose dimensions in the model are {} and '
                                  'whose dimensions in the checkpoint are {}, '
                                  'an exception occurred : {}.'
                                  .format(key, param.size(), input_param.size(), ex.args))
        elif strict:
            missing_keys.append(key)

    extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
    if getattr(self.__class__, "set_extra_state", Module.set_extra_state) is not Module.set_extra_state:
        if extra_state_key in state_dict:
            self.set_extra_state(state_dict[extra_state_key])
        elif strict:
            missing_keys.append(extra_state_key)
    elif strict and (extra_state_key in state_dict):
        unexpected_keys.append(extra_state_key)

    if strict:
        for key in state_dict.keys():
            if key.startswith(prefix) and key != extra_state_key:
                input_name = key[len(prefix):]
                input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                if input_name not in self._modules and input_name not in local_state:
                    unexpected_keys.append(key)

def load_state_dict(self, state_dict: Mapping[str, Any],
                    strict: bool = True, assign: bool = False):
    r"""Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.

    .. warning::
        If :attr:`assign` is ``True`` the optimizer must be created after
        the call to :attr:`load_state_dict`.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        assign (bool, optional): whether to assign items in the state
            dictionary to their corresponding keys in the module instead
            of copying them inplace into the module's current parameters and buffers.
            When ``False``, the properties of the tensors in the current
            module are preserved while when ``True``, the properties of the
            Tensors in the state dict are preserved.
            Default: ``False``

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    Note:
        If a parameter or buffer is registered as ``None`` and its corresponding key
        exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
        ``RuntimeError``.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected state_dict to be dict-like, got {}.".format(type(state_dict)))

    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    error_msgs: List[str] = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = OrderedDict(state_dict)
    if metadata is not None:
        # mypy isn't aware that "_metadata" exists in state_dict
        state_dict._metadata = metadata  # type: ignore[attr-defined]

    def load(module, local_state_dict, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        if assign:
            local_metadata['assign_to_params_buffers'] = assign
        module._load_from_state_dict(
            local_state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                load(child, child_state_dict, child_prefix)

        # Note that the hook can modify missing_keys and unexpected_keys.
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        for hook in module._load_state_dict_post_hooks.values():
            out = hook(module, incompatible_keys)
            assert out is None, (
                "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                "expected to return new values, if incompatible_keys need to be modified,"
                "it should be done inplace."
            )

    load(self, state_dict)
    del load

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))

    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                           self.__class__.__name__, "\n\t".join(error_msgs)))
    return _IncompatibleKeys(missing_keys, unexpected_keys)

if [int(x) for x in torch.__version__.split('.')[0:2]] < [2, 1]:
    Module._load_from_state_dict = _load_from_state_dict
    Module.load_state_dict = load_state_dict
from typing import Mapping, Any, List
from collections import OrderedDict
from torch.nn.modules.module import _IncompatibleKeys, _EXTRA_STATE_KEY_SUFFIX
import itertools
from torch.nn import Module
import torch

# this patch is for adding the `assign` key to load_state_dict.
# the code is in pytorch source for version 2.1

def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs):
    r"""Copies parameters and buffers from :attr:`state_dict` into only
    this module, but not its descendants. This is called on every submodule
    in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
    module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
    For state dicts without metadata, :attr:`local_metadata` is empty.
    Subclasses can achieve class-specific backward compatible loading using
    the version number at `local_metadata.get("version", None)`.
    Additionally, :attr:`local_metadata` can also contain the key
    `assign_to_params_buffers` that indicates whether keys should be
    assigned their corresponding tensor in the state_dict.

    .. note::
        :attr:`state_dict` is not the same object as the input
        :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
        it can be modified.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        prefix (str): the prefix for parameters and buffers used in this
            module
        local_metadata (dict): a dict containing the metadata for this module.
            See
        strict (bool): whether to strictly enforce that the keys in
            :attr:`state_dict` with :attr:`prefix` match the names of
            parameters and buffers in this module
        missing_keys (list of str): if ``strict=True``, add missing keys to
            this list
        unexpected_keys (list of str): if ``strict=True``, add unexpected
            keys to this list
        error_msgs (list of str): error messages should be added to this
            list, and will be reported together in
            :meth:`~torch.nn.Module.load_state_dict`
    """
    for hook in self._load_state_dict_pre_hooks.values():
        hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
    local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}
    assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)

    for name, param in local_state.items():
        key = prefix + name
        if key in state_dict:
            input_param = state_dict[key]
            if not torch.overrides.is_tensor_like(input_param):
                error_msgs.append('While copying the parameter named "{}", '
                                  'expected torch.Tensor or Tensor-like object from checkpoint but '
                                  'received {}'
                                  .format(key, type(input_param)))
                continue

            # This is used to avoid copying uninitialized parameters into
            # non-lazy modules, since they dont have the hook to do the checks
            # in such case, it will error when accessing the .shape attribute.
            is_param_lazy = torch.nn.parameter.is_lazy(param)
            # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
            if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
                input_param = input_param[0]

            if not is_param_lazy and input_param.shape != param.shape:
                # local shape should match the one in checkpoint
                error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                  'the shape in current model is {}.'
                                  .format(key, input_param.shape, param.shape))
                continue
            try:
                with torch.no_grad():
                    if assign_to_params_buffers:
                        # Shape checks are already done above
                        if (isinstance(param, torch.nn.Parameter) and
                                not isinstance(input_param, torch.nn.Parameter)):
                            setattr(self, name, torch.nn.Parameter(input_param))
                        else:
                            setattr(self, name, input_param)
                    else:
                        param.copy_(input_param)
            except Exception as ex:
                error_msgs.append('While copying the parameter named "{}", '
                                  'whose dimensions in the model are {} and '
                                  'whose dimensions in the checkpoint are {}, '
                                  'an exception occurred : {}.'
                                  .format(key, param.size(), input_param.size(), ex.args))
        elif strict:
            missing_keys.append(key)

    extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
    if getattr(self.__class__, "set_extra_state", Module.set_extra_state) is not Module.set_extra_state:
        if extra_state_key in state_dict:
            self.set_extra_state(state_dict[extra_state_key])
        elif strict:
            missing_keys.append(extra_state_key)
    elif strict and (extra_state_key in state_dict):
        unexpected_keys.append(extra_state_key)

    if strict:
        for key in state_dict.keys():
            if key.startswith(prefix) and key != extra_state_key:
                input_name = key[len(prefix):]
                input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                if input_name not in self._modules and input_name not in local_state:
                    unexpected_keys.append(key)

def load_state_dict(self, state_dict: Mapping[str, Any],
                    strict: bool = True, assign: bool = False):
    r"""Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.

    .. warning::
        If :attr:`assign` is ``True`` the optimizer must be created after
        the call to :attr:`load_state_dict`.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        assign (bool, optional): whether to assign items in the state
            dictionary to their corresponding keys in the module instead
            of copying them inplace into the module's current parameters and buffers.
            When ``False``, the properties of the tensors in the current
            module are preserved while when ``True``, the properties of the
            Tensors in the state dict are preserved.
            Default: ``False``

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    Note:
        If a parameter or buffer is registered as ``None`` and its corresponding key
        exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
        ``RuntimeError``.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected state_dict to be dict-like, got {}.".format(type(state_dict)))

    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    error_msgs: List[str] = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = OrderedDict(state_dict)
    if metadata is not None:
        # mypy isn't aware that "_metadata" exists in state_dict
        state_dict._metadata = metadata  # type: ignore[attr-defined]

    def load(module, local_state_dict, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        if assign:
            local_metadata['assign_to_params_buffers'] = assign
        module._load_from_state_dict(
            local_state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                load(child, child_state_dict, child_prefix)

        # Note that the hook can modify missing_keys and unexpected_keys.
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        for hook in module._load_state_dict_post_hooks.values():
            out = hook(module, incompatible_keys)
            assert out is None, (
                "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                "expected to return new values, if incompatible_keys need to be modified,"
                "it should be done inplace."
            )

    load(self, state_dict)
    del load

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))

    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                           self.__class__.__name__, "\n\t".join(error_msgs)))
    return _IncompatibleKeys(missing_keys, unexpected_keys)

if [int(x) for x in torch.__version__.split('.')[0:2]] < [2, 1]:
    Module._load_from_state_dict = _load_from_state_dict
    Module.load_state_dict = load_state_dict
