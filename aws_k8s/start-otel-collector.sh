#!/bin/sh
echo "Starting OpenTelemetry Collector"
echo "Environment Variables:"
env
echo "Configuration File:"
cat /etc/otel-collector-config.yaml
otelcol --config=/etc/otel-collector-config.yaml
