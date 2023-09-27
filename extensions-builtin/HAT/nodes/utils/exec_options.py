from __future__ import annotations

from typing import Dict

SettingsJson = Dict[str, object]
JsonExecutionOptions = Dict[str, SettingsJson]


class ExecutionOptions:
    def __init__(
        self,
        backend_settings: JsonExecutionOptions,
    ) -> None:
        self.__settings = backend_settings

    @staticmethod
    def parse(json: JsonExecutionOptions) -> ExecutionOptions:
        return ExecutionOptions(backend_settings=json)

    def get_package_settings(self, package_id: str) -> SettingsJson:
        return self.__settings.get(package_id, {})


__global_exec_options = ExecutionOptions({})


def get_execution_options() -> ExecutionOptions:
    return __global_exec_options


def set_execution_options(value: ExecutionOptions):
    # TODO: Make the mutable global state unnecessary
    # pylint: disable=global-statement
    global __global_exec_options
    __global_exec_options = value
