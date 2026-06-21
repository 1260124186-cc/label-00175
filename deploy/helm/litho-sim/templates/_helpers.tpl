{{/*
Expand the name of the chart.
*/}}
{{- define "litho-sim.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "litho-sim.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "litho-sim.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "litho-sim.labels" -}}
helm.sh/chart: {{ include "litho-sim.chart" . }}
app.kubernetes.io/name: {{ include "litho-sim.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.commonLabels }}
{{- toYaml . | nindent 0 }}
{{- end }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "litho-sim.selectorLabels" -}}
app.kubernetes.io/name: {{ include "litho-sim.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Component labels
*/}}
{{- define "litho-sim.componentLabels" -}}
{{- include "litho-sim.selectorLabels" . }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{/*
Create the name of the service account to use
*/}}
{{- define "litho-sim.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
    {{ default (include "litho-sim.fullname" .) .Values.serviceAccount.name }}
{{- else -}}
    {{ default "default" .Values.serviceAccount.name }}
{{- end -}}
{{- end -}}

{{/*
Redis connection info helper
*/}}
{{- define "litho-sim.redis.host" -}}
{{- if .Values.redis.useExternal -}}
{{- .Values.redis.external.host -}}
{{- else -}}
{{- printf "%s-%s" (include "litho-sim.fullname" .) "redis" -}}
{{- end -}}
{{- end -}}

{{- define "litho-sim.redis.port" -}}
{{- if .Values.redis.useExternal -}}
{{- .Values.redis.external.port | default 6379 -}}
{{- else -}}
{{- .Values.redis.service.port | default 6379 -}}
{{- end -}}
{{- end -}}

{{- define "litho-sim.redis.password" -}}
{{- if .Values.redis.useExternal -}}
{{- .Values.redis.external.password | default "" -}}
{{- else -}}
{{- .Values.redis.auth.password | default "" -}}
{{- end -}}
{{- end -}}

{{/*
Image registry helper
*/}}
{{- define "litho-sim.image" -}}
{{- $registry := .Values.global.imageRegistry -}}
{{- $repository := .image.repository -}}
{{- $tag := .image.tag -}}
{{- if hasPrefix "docker.io/" $repository -}}
{{- printf "%s:%s" $repository $tag -}}
{{- else -}}
{{- printf "%s%s:%s" $registry $repository $tag -}}
{{- end -}}
{{- end -}}

{{/*
Pull policy helper
*/}}
{{- define "litho-sim.imagePullPolicy" -}}
{{- default .Values.global.imagePullPolicy .image.pullPolicy -}}
{{- end -}}

{{/*
GPU affinity helper
*/}}
{{- define "litho-sim.gpu.affinity" -}}
{{- if and .Values.celeryWorker.gpu.enabled .Values.celeryWorker.gpu.nodeSelector -}}
nodeAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
    nodeSelectorTerms:
    - matchExpressions:
      {{- range $key, $value := .Values.celeryWorker.gpu.nodeSelector }}
      - key: {{ $key | quote }}
        operator: In
        values:
        - {{ $value | quote }}
      {{- end }}
{{- end -}}
{{- end -}}

{{/*
GPU tolerations helper
*/}}
{{- define "litho-sim.gpu.tolerations" -}}
{{- if .Values.celeryWorker.gpu.enabled }}
{{- toYaml .Values.celeryWorker.gpu.tolerations | nindent 0 }}
{{- end -}}
{{- end -}}

{{/*
GPU resources helper
*/}}
{{- define "litho-sim.gpu.resources" -}}
{{- if .Values.celeryWorker.gpu.enabled }}
{{- toYaml .Values.celeryWorker.gpu.resources | nindent 0 }}
{{- end -}}
{{- end -}}

{{/*
Security context helper
*/}}
{{- define "litho-sim.podSecurityContext" -}}
{{- toYaml .podSecurityContext | nindent 0 }}
{{- end -}}

{{- define "litho-sim.containerSecurityContext" -}}
{{- toYaml .securityContext | nindent 0 }}
{{- end -}}
