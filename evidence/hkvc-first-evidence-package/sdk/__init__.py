"""Public SDK boundary for the HKVC evidence package.

This public repository intentionally does not ship Catalyst algorithm
implementations. Runtime features are resolved from the closed/freemium
``catalyst-brain`` SDK at execution time.
"""

from sdk.sdk_proxy import CatalystSDKFeatureUnavailable, call_sdk_feature, load_sdk_feature


def __getattr__(name: str):
    return load_sdk_feature(name)


__all__ = ["CatalystSDKFeatureUnavailable", "call_sdk_feature", "load_sdk_feature"]
