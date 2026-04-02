"""Static plugin registry for the Automated Swing Trader plugin system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Type, TypeVar

from common.exceptions import ConfigurationError, CriticalError, ValidationError
from common.interface import Result, ResultStatus

from .interface import PluginBase, PluginCategory, PluginMetadata

__all__ = ["PluginRegistry"]

PluginType = TypeVar("PluginType", bound=PluginBase[Any, Any, Any])


@dataclass(frozen=True, slots=True)
class _RegisteredPlugin:
    """Internal representation mapping plugin names to implementations."""

    plugin_class: Type[PluginType]
    metadata: PluginMetadata


class PluginRegistry:
    """In-memory registry tracking statically imported plugin classes."""

    _MODULE_NAME = "plugins.registry"
    _DUPLICATE_REASON = "PLUGIN_ALREADY_REGISTERED"
    _UNKNOWN_PLUGIN_REASON = "PLUGIN_NOT_REGISTERED"
    _INVALID_PLUGIN_REASON = "PLUGIN_INVALID_DEFINITION"
    _CONFIG_INVALID_REASON = "PLUGIN_CONFIG_INVALID"

    _CATEGORY_KEYS: Dict[str, PluginCategory] = {
        "data_sources": PluginCategory.DATA_SOURCE,
        "scanners": PluginCategory.SCANNER,
        "signals": PluginCategory.SIGNAL,
        "strategies": PluginCategory.STRATEGY,
        "risk_policies": PluginCategory.RISK_POLICY,
    }

    def __init__(self) -> None:
        self._registry: MutableMapping[str, _RegisteredPlugin] = {}

    def register_plugin(self, name: str, plugin_class: Type[PluginType]) -> None:
        """Register ``plugin_class`` under ``name`` with metadata validation."""

        if name in self._registry:
            error = ConfigurationError(
                f"Plugin {name!r} already registered.",
                module=self._MODULE_NAME,
                reason_code=self._DUPLICATE_REASON,
            )
            raise error

        metadata = getattr(plugin_class, "metadata", None)
        if not isinstance(metadata, PluginMetadata):
            error = ConfigurationError(
                f"Plugin {plugin_class!r} missing PluginMetadata.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_PLUGIN_REASON,
                details={"name": name},
            )
            raise error

        if metadata.name != name:
            error = ConfigurationError(
                "Plugin metadata.name must match registry name.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_PLUGIN_REASON,
                details={"registry_name": name, "metadata_name": metadata.name},
            )
            raise error

        self._registry[name] = _RegisteredPlugin(plugin_class=plugin_class, metadata=metadata)

    def register(
        self,
        name: str,
        plugin_class: Type[PluginType],
        *,
        metadata: PluginMetadata | None = None,
    ) -> None:
        """Register ``plugin_class`` under ``name``, optionally overriding metadata.

        Notes:
            This is a convenience wrapper around :meth:`register_plugin` that enables
            the same plugin implementation to be registered under multiple names by
            supplying a different :class:`~plugins.interface.PluginMetadata`.
        """

        if metadata is None:
            self.register_plugin(name, plugin_class)
            return

        if not isinstance(metadata, PluginMetadata):
            raise TypeError("metadata must be a PluginMetadata instance.")
        if metadata.name != name:
            error = ConfigurationError(
                "Plugin metadata.name must match registry name.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_PLUGIN_REASON,
                details={"registry_name": name, "metadata_name": metadata.name},
            )
            raise error

        wrapped_class = type(
            f"{plugin_class.__name__}__{name}",
            (plugin_class,),
            {"metadata": metadata},
        )
        self.register_plugin(name, wrapped_class)

    def get_plugin(self, name: str) -> Type[PluginType]:
        """Return the registered plugin class for ``name`` or raise error."""

        try:
            return self._registry[name].plugin_class
        except KeyError as exc:
            error = ConfigurationError(
                f"Plugin {name!r} is not registered.",
                module=self._MODULE_NAME,
                reason_code=self._UNKNOWN_PLUGIN_REASON,
            )
            raise error from exc

    def list_plugins(
        self,
        *,
        category: PluginCategory | None = None,
        enabled_only: bool = True,
    ) -> list[PluginMetadata]:
        """Return metadata for all registered plugins, optionally filtered."""

        records_iter: Iterable[_RegisteredPlugin] = self._registry.values()
        if category is not None:
            records_iter = [record for record in records_iter if record.metadata.category is category]
        if enabled_only:
            records_iter = [record for record in records_iter if record.metadata.enabled]
        return [record.metadata for record in records_iter]

    def load_from_config(
        self,
        config_dict: Mapping[str, Any],
    ) -> Result[dict[str, PluginBase[Any, Any, Any]]]:
        """Instantiate enabled plugins using the supplied configuration mapping."""

        plugins_section = config_dict.get("plugins")
        if plugins_section is None:
            return Result.success({})

        if not isinstance(plugins_section, Mapping):
            error = ConfigurationError(
                "plugins section must be a mapping of categories.",
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        instances: dict[str, PluginBase[Any, Any, Any]] = {}

        for category_key, category_value in plugins_section.items():
            expected_category = self._CATEGORY_KEYS.get(category_key)
            if expected_category is None:
                error = ConfigurationError(
                    f"Unknown plugin category {category_key!r}.",
                    module=self._MODULE_NAME,
                    reason_code=self._CONFIG_INVALID_REASON,
                )
                return error.to_result(data=None, status=ResultStatus.FAILED)

            if not isinstance(category_value, Iterable):
                error = ConfigurationError(
                    f"Plugin category {category_key!r} must be iterable.",
                    module=self._MODULE_NAME,
                    reason_code=self._CONFIG_INVALID_REASON,
                )
                return error.to_result(data=None, status=ResultStatus.FAILED)

            for raw_entry in category_value:
                if not isinstance(raw_entry, Mapping):
                    error = ConfigurationError(
                        "Plugin entry must be a mapping.",
                        module=self._MODULE_NAME,
                        reason_code=self._CONFIG_INVALID_REASON,
                        details={"category": category_key, "entry": repr(raw_entry)},
                    )
                    return error.to_result(data=None, status=ResultStatus.FAILED)

                name = raw_entry.get("name")
                if not isinstance(name, str):
                    error = ConfigurationError(
                        "Plugin entry missing string 'name'.",
                        module=self._MODULE_NAME,
                        reason_code=self._CONFIG_INVALID_REASON,
                        details={"category": category_key},
                    )
                    return error.to_result(data=None, status=ResultStatus.FAILED)

                record = self._registry.get(name)
                if record is None:
                    error = ConfigurationError(
                        f"Plugin {name!r} referenced in config but not registered.",
                        module=self._MODULE_NAME,
                        reason_code=self._UNKNOWN_PLUGIN_REASON,
                        details={"category": category_key},
                    )
                    return error.to_result(data=None, status=ResultStatus.FAILED)

                if record.metadata.category is not expected_category:
                    error = ConfigurationError(
                        (
                            f"Plugin {name!r} registered under category "
                            f"{record.metadata.category.value!r} but config references {category_key!r}."
                        ),
                        module=self._MODULE_NAME,
                        reason_code=self._CONFIG_INVALID_REASON,
                        details={"registry_category": record.metadata.category.value},
                    )
                    return error.to_result(data=None, status=ResultStatus.FAILED)

                enabled_flag = bool(raw_entry.get("enabled", True))
                effective_enabled = record.metadata.enabled and enabled_flag
                if not effective_enabled:
                    continue

                plugin_class = record.plugin_class
                raw_config = raw_entry.get("config", {})

                try:
                    try:
                        plugin_instance = plugin_class(config=raw_config)  # type: ignore[arg-type]
                    except TypeError:
                        plugin_instance = plugin_class()  # type: ignore[call-arg]
                except PermissionError as exc:
                    error = CriticalError.from_error(
                        exc,
                        module=self._MODULE_NAME,
                        reason_code="PLUGIN_INIT_PERMISSION_DENIED",
                        details={"plugin": name},
                    )
                    return Result.failed(error, error.reason_code)
                except Exception as exc:  # noqa: BLE001 - bubble as configuration failure.
                    error = ConfigurationError.from_error(
                        exc,
                        module=self._MODULE_NAME,
                        reason_code="PLUGIN_INIT_FAILURE",
                        details={"plugin": name},
                    )
                    return Result.failed(error, error.reason_code)

                try:
                    validation_result = plugin_instance.validate_config(raw_config)  # type: ignore[arg-type]
                except PermissionError as exc:
                    error = CriticalError.from_error(
                        exc,
                        module=self._MODULE_NAME,
                        reason_code="PLUGIN_CONFIG_PERMISSION_DENIED",
                        details={"plugin": name},
                    )
                    return Result.failed(error, error.reason_code)
                except ValidationError as exc:
                    error = ValidationError.from_error(
                        exc,
                        module=self._MODULE_NAME,
                        reason_code=self._CONFIG_INVALID_REASON,
                        details={"plugin": name},
                    )
                    return Result.failed(error, error.reason_code)
                except Exception as exc:  # noqa: BLE001 - normalise into validation failure.
                    error = ValidationError.from_error(
                        exc,
                        module=self._MODULE_NAME,
                        reason_code=self._CONFIG_INVALID_REASON,
                        details={"plugin": name},
                    )
                    return Result.failed(error, error.reason_code)

                if validation_result.status is not ResultStatus.SUCCESS:
                    error = validation_result.error or ValidationError(
                        "Plugin configuration validation failed.",
                        module=self._MODULE_NAME,
                        reason_code=self._CONFIG_INVALID_REASON,
                        details={"plugin": name},
                    )
                    reason = validation_result.reason_code or self._CONFIG_INVALID_REASON
                    return Result.failed(error, reason)

                if validation_result.data is not None:
                    setattr(plugin_instance, "config", validation_result.data)

                instances[name] = plugin_instance

        return Result.success(instances)
