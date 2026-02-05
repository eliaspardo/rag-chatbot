# flake8: noqa
import importlib
import importlib.metadata
import logging
import os
import shlex
import sys
import tempfile
import textwrap
import types
import warnings

_logger = logging.getLogger("mlflow.server")

from flask import Flask, Response, request, send_from_directory
from packaging.version import Version

from mlflow.environment_variables import (
    _MLFLOW_SGI_NAME,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_SERVER_ENABLE_JOB_EXECUTION,
)
from mlflow.exceptions import MlflowException
from mlflow.server import handlers
from mlflow.server.constants import (
    ARTIFACT_ROOT_ENV_VAR,
    ARTIFACTS_DESTINATION_ENV_VAR,
    ARTIFACTS_ONLY_ENV_VAR,
    BACKEND_STORE_URI_ENV_VAR,
    HUEY_STORAGE_PATH_ENV_VAR,
    PROMETHEUS_EXPORTER_ENV_VAR,
    REGISTRY_STORE_URI_ENV_VAR,
    SECRETS_CACHE_MAX_SIZE_ENV_VAR,
    SECRETS_CACHE_TTL_ENV_VAR,
    SERVE_ARTIFACTS_ENV_VAR,
)
from mlflow.server.handlers import (
    STATIC_PREFIX_ENV_VAR,
    _add_static_prefix,
    _search_datasets_handler,
    create_promptlab_run_handler,
    gateway_proxy_handler,
    get_artifact_handler,
    get_logged_model_artifact_handler,
    get_metric_history_bulk_handler,
    get_metric_history_bulk_interval_handler,
    get_model_version_artifact_handler,
    get_trace_artifact_handler,
    get_ui_telemetry_handler,
    post_ui_telemetry_handler,
    upload_artifact_handler,
)
from mlflow.utils.os import is_windows
from mlflow.utils.plugins import get_entry_points
from mlflow.utils.process import _exec_cmd
from mlflow.version import VERSION

REL_STATIC_DIR = "js/build"

app = Flask(__name__, static_folder=REL_STATIC_DIR)
IS_FLASK_V1 = Version(importlib.metadata.version("flask")) < Version("2.0")

is_running_as_server = (
    "gunicorn" in sys.modules
    or "uvicorn" in sys.modules
    or "waitress" in sys.modules
    or os.getenv(BACKEND_STORE_URI_ENV_VAR)
    or os.getenv(SERVE_ARTIFACTS_ENV_VAR)
)


@app.after_request
def _compare_parents_csp(response):
    if request.path.endswith("/compare-parents") or request.path.endswith(
        "/compare-parents.js"
    ):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' data: blob:; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:; "
            "style-src 'self' 'unsafe-inline' data: blob:; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' data: blob:; "
            "font-src 'self' data: blob:;"
        )
    return response


if is_running_as_server:
    from mlflow.server import security

    security.init_security_middleware(app)

for http_path, handler, methods in handlers.get_endpoints():
    app.add_url_rule(http_path, handler.__name__, handler, methods=methods)

if os.getenv(PROMETHEUS_EXPORTER_ENV_VAR):
    from mlflow.server.prometheus_exporter import activate_prometheus_exporter

    prometheus_metrics_path = os.getenv(PROMETHEUS_EXPORTER_ENV_VAR)
    if not os.path.exists(prometheus_metrics_path):
        os.makedirs(prometheus_metrics_path)
    activate_prometheus_exporter(app)


# Provide a health check endpoint to ensure the application is responsive
@app.route(_add_static_prefix("/health"))
def health():
    return "OK", 200


# Provide an endpoint to query the version of mlflow running on the server
@app.route(_add_static_prefix("/version"))
def version():
    return VERSION, 200


# Serve the "get-artifact" route.
@app.route(_add_static_prefix("/get-artifact"))
def serve_artifacts():
    return get_artifact_handler()


# Serve the "model-versions/get-artifact" route.
@app.route(_add_static_prefix("/model-versions/get-artifact"))
def serve_model_version_artifact():
    return get_model_version_artifact_handler()


# Serve the "metrics/get-history-bulk" route.
@app.route(_add_static_prefix("/ajax-api/2.0/mlflow/metrics/get-history-bulk"))
def serve_get_metric_history_bulk():
    return get_metric_history_bulk_handler()


# Serve the "metrics/get-history-bulk-interval" route.
@app.route(_add_static_prefix("/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval"))
def serve_get_metric_history_bulk_interval():
    return get_metric_history_bulk_interval_handler()


# Serve the "experiments/search-datasets" route.
@app.route(
    _add_static_prefix("/ajax-api/2.0/mlflow/experiments/search-datasets"),
    methods=["POST"],
)
def serve_search_datasets():
    return _search_datasets_handler()


# Serve the "runs/create-promptlab-run" route.
@app.route(
    _add_static_prefix("/ajax-api/2.0/mlflow/runs/create-promptlab-run"),
    methods=["POST"],
)
def serve_create_promptlab_run():
    return create_promptlab_run_handler()


@app.route(
    _add_static_prefix("/ajax-api/2.0/mlflow/gateway-proxy"), methods=["POST", "GET"]
)
def serve_gateway_proxy():
    return gateway_proxy_handler()


@app.route(_add_static_prefix("/ajax-api/2.0/mlflow/upload-artifact"), methods=["POST"])
def serve_upload_artifact():
    return upload_artifact_handler()


# Serve the "/get-trace-artifact" route to allow frontend to fetch trace artifacts
# and render them in the Trace UI. The request body should contain the request_id
# of the trace.
@app.route(
    _add_static_prefix("/ajax-api/2.0/mlflow/get-trace-artifact"), methods=["GET"]
)
def serve_get_trace_artifact():
    return get_trace_artifact_handler()


@app.route(
    _add_static_prefix("/ajax-api/2.0/mlflow/logged-models/<model_id>/artifacts/files"),
    methods=["GET"],
)
def serve_get_logged_model_artifact(model_id: str):
    return get_logged_model_artifact_handler(model_id)


@app.route(_add_static_prefix("/ajax-api/3.0/mlflow/ui-telemetry"), methods=["GET"])
def serve_get_ui_telemetry():
    return get_ui_telemetry_handler()


@app.route(_add_static_prefix("/ajax-api/3.0/mlflow/ui-telemetry"), methods=["POST"])
def serve_post_ui_telemetry():
    return post_ui_telemetry_handler()


_COMPARE_PARENTS_JS = textwrap.dedent(
    """
              window.__compareParentsLoaded = true;
              const defaultMetrics = ["Completeness_GEval", "Grounding_GEval", "Reasoning_GEval"];
              const palette = [
                "#4d9de0",
                "#f07167",
                "#8ac926",
                "#ffca3a",
                "#9b5de5",
                "#43aa8b",
                "#ff924c",
                "#577590",
              ];

              function getParam(name, fallback = "") {
                const params = new URLSearchParams(window.location.search);
                return params.get(name) || fallback;
              }

              function parseList(value) {
                if (!value) return [];
                return value
                  .split(/[\\n,]+/)
                  .map((item) => item.trim())
                  .filter(Boolean);
              }

              function setInputsFromUrl() {
                document.getElementById("experimentId").value = getParam("experiment_id");
                const parentsParam = getParam("parents");
                const metricsParam = getParam("metrics");
                const parentA = getParam("parent_a");
                const parentB = getParam("parent_b");
                const metric1 = getParam("metric1");
                const metric2 = getParam("metric2");
                const metric3 = getParam("metric3");
                const parentsFallback = [parentA, parentB].filter(Boolean).join(", ");
                const metricsFallback = [metric1, metric2, metric3].filter(Boolean).join(", ");
                document.getElementById("parentsInput").value = parentsParam || parentsFallback;
                document.getElementById("metricsInput").value =
                  metricsParam || metricsFallback || defaultMetrics.join(", ");
              }

              function updateUrlFromInputs() {
                const params = new URLSearchParams();
                const experimentId = document.getElementById("experimentId").value.trim();
                const parents = parseList(document.getElementById("parentsInput").value);
                const metrics = parseList(document.getElementById("metricsInput").value);
                if (experimentId) {
                  params.set("experiment_id", experimentId);
                }
                if (parents.length) {
                  params.set("parents", parents.join(","));
                  if (parents[0]) params.set("parent_a", parents[0]);
                  if (parents[1]) params.set("parent_b", parents[1]);
                }
                if (metrics.length) {
                  params.set("metrics", metrics.join(","));
                  if (metrics[0]) params.set("metric1", metrics[0]);
                  if (metrics[1]) params.set("metric2", metrics[1]);
                  if (metrics[2]) params.set("metric3", metrics[2]);
                }
                window.history.replaceState({}, "", `${window.location.pathname}?${params.toString()}`);
              }

              async function searchRuns(experimentId, parentRunId) {
                const body = {
                  experiment_ids: [experimentId],
                  filter: `tags.mlflow.parentRunId = '${parentRunId}'`,
                  max_results: 10000,
                };
                const res = await fetch("/ajax-api/2.0/mlflow/runs/search", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(body),
                });
                const data = await res.json();
                return data.runs || [];
              }

              function isLikelyRunId(value) {
                return /^[0-9a-f]{32}$/i.test(value);
              }

              async function resolveParentRunId(experimentId, parentInput) {
                if (!parentInput) return "";
                const body = {
                  experiment_ids: [experimentId],
                  filter: `attributes.run_name = '${parentInput}'`,
                  max_results: 1,
                };
                const res = await fetch("/ajax-api/2.0/mlflow/runs/search", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(body),
                });
                const data = await res.json();
                const run = (data.runs || [])[0];
                return run ? (run.info.run_id || run.info.run_uuid) : "";
              }

              async function getRunById(experimentId, runId) {
                if (!runId) return null;
                const body = {
                  experiment_ids: [experimentId],
                  filter: `attributes.run_id = '${runId}'`,
                  max_results: 1,
                };
                const res = await fetch("/ajax-api/2.0/mlflow/runs/search", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(body),
                });
                const data = await res.json();
                return (data.runs || [])[0] || null;
              }

              function parseRun(run) {
                const params = (run.data.params || []).reduce((acc, p) => {
                  acc[p.key] = p.value;
                  return acc;
                }, {});
                const metrics = (run.data.metrics || []).reduce((acc, m) => {
                  acc[m.key] = m.value;
                  return acc;
                }, {});
                const tags = (run.data.tags || []).reduce((acc, t) => {
                  acc[t.key] = t.value;
                  return acc;
                }, {});
                return {
                  runId: run.info.run_id || run.info.run_uuid,
                  runName: run.info.run_name,
                  params,
                  metrics,
                  tags,
                };
              }

              function numericOrNull(value) {
                if (value === undefined || value === null) return null;
                const n = Number(value);
                return Number.isFinite(n) ? n : null;
              }

              function normalizeQuestionId(value) {
                if (value === undefined || value === null) return null;
                const num = numericOrNull(value);
                if (num !== null) return num;
                return String(value);
              }

              function buildPoints(runs, parentKey, metricKey) {
                return runs
                  .map((r) => {
                    const qid = normalizeQuestionId(r.params.question_id);
                    const y = numericOrNull(r.metrics[metricKey]);
                    if (qid === null || y === null) return null;
                    return { parentKey, qid, y, runName: r.runName || r.runId, run: r };
                  })
                  .filter(Boolean);
              }

              function resizeCanvas(canvas, height) {
                const cssWidth = canvas.clientWidth || 800;
                const ratio = window.devicePixelRatio || 1;
                canvas.style.height = `${height}px`;
                canvas.width = Math.floor(cssWidth * ratio);
                canvas.height = Math.floor(height * ratio);
                const ctx = canvas.getContext("2d");
                ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
              }

              function getPlotHeight(canvas) {
                const panel = canvas.closest(".plot-panel") || canvas.parentElement;
                const panelHeight = panel ? panel.clientHeight : 300;
                return Math.max(200, Math.min(600, panelHeight - 48));
              }

              function renderTooltipContent(point, metricKey, parentLabels) {
                const label = parentLabels[point.parentKey] || point.parentKey;
                const run = point.run;
                return [
                  `<h4>Question ID: ${point.qid}</h4>`,
                  `<div class="kv">${label}</div>`,
                ].join("");
              }

              function renderMultiTooltipContent(qid, metricKey, runs, parentLabels) {
                const sections = runs
                  .map((point) => {
                    const label = parentLabels[point.parentKey] || point.parentKey;
                    return [
                      `<div class="kv" style="margin-top:8px;">${label}</div>`,
                    ].join("");
                  })
                  .join("");
                return [`<h4>Question ID: ${qid}</h4>`, sections].join("");
              }

              function formatKeyValues(obj) {
                if (!obj) return "<span class='muted'>n/a</span>";
                const keys = Object.keys(obj);
                if (!keys.length) return "<span class='muted'>n/a</span>";
                return keys.map((k) => `<b>${k}</b>: ${obj[k]}`).join("<br/>");
              }

              function renderSidebarContent(qid, row, parentLabels, metricKeys, parentOrder) {
                const cards = parentOrder.map((parentKey) => {
                  const run = row?.[parentKey];
                  const label = parentLabels[parentKey] || parentKey;
                  const metricList = metricKeys
                    .map((k) => `<b>${k}</b>: ${run?.metrics?.[k] ?? "n/a"}`)
                    .join("<br/>");
                  const runJson = run ? JSON.stringify(run, null, 2) : "n/a";
                  return `
                    <div class="card">
                      <h4>${label}</h4>
                      <div class="kv"><b>run_name</b>: ${run?.runName ?? "n/a"}<br/><b>run_id</b>: ${run?.runId ?? "n/a"}</div>
                      <div class="kv" style="margin-top:6px;"><b>scores</b><br/>${metricList}</div>
                      <details>
                        <summary>Params</summary>
                        <div class="kv">${formatKeyValues(run?.params)}</div>
                      </details>
                      <details>
                        <summary>Raw JSON</summary>
                        <pre>${runJson}</pre>
                      </details>
                    </div>
                  `;
                });
                return `<div class="cols">${cards.join("")}</div>`;
              }

              function drawScatter(canvas, points, titleEl, metricKey, colors, runsByQid, parentLabels, parentOrder) {
                resizeCanvas(canvas, getPlotHeight(canvas));
                const ctx = canvas.getContext("2d");
                ctx.clearRect(0, 0, canvas.clientWidth || canvas.width, canvas.clientHeight || canvas.height);
                titleEl.textContent = metricKey;

                if (!points.length) {
                  ctx.fillStyle = "#9aa4b2";
                  ctx.fillText("No data", 16, 20);
                  return;
                }

                const padding = { left: 50, right: 20, top: 20, bottom: 40 };
                const plotWidth = canvas.width - padding.left - padding.right;
                const plotHeight = canvas.height - padding.top - padding.bottom;

                const isNumeric = points.every((p) => typeof p.qid === "number");
                let xValues = points.map((p) => p.qid);
                let xLabelMin = "";
                let xLabelMax = "";
                if (!isNumeric) {
                  const unique = Array.from(new Set(xValues.map(String)));
                  unique.sort();
                  const indexByLabel = new Map(unique.map((v, i) => [v, i]));
                  xValues = points.map((p) => indexByLabel.get(String(p.qid)));
                  xLabelMin = unique[0] || "";
                  xLabelMax = unique[unique.length - 1] || "";
                }
                const yValues = points.map((p) => p.y);
                const xMin = Math.min(...xValues);
                const xMax = Math.max(...xValues);
                const yMin = Math.min(...yValues);
                const yMax = Math.max(...yValues);
                const xRange = xMax - xMin || 1;
                const yRange = yMax - yMin || 1;

                ctx.strokeStyle = "rgba(255,255,255,0.08)";
                ctx.lineWidth = 1;
                for (let i = 0; i <= 5; i++) {
                  const y = padding.top + (plotHeight * i) / 5;
                  ctx.beginPath();
                  ctx.moveTo(padding.left, y);
                  ctx.lineTo(padding.left + plotWidth, y);
                  ctx.stroke();
                }

                ctx.fillStyle = "#9aa4b2";
                ctx.font = "12px sans-serif";
                if (isNumeric) {
                  const ticks = [];
                  const tickStep = 5;
                  const first = Math.ceil(xMin / tickStep) * tickStep;
                  for (let t = first; t <= xMax; t += tickStep) {
                    ticks.push(t);
                  }
                  if (!ticks.includes(xMin)) ticks.unshift(xMin);
                  if (!ticks.includes(xMax)) ticks.push(xMax);
                  ticks.forEach((tick) => {
                    const x = padding.left + ((tick - xMin) / xRange) * plotWidth;
                    ctx.fillText(String(tick), x - 6, padding.top + plotHeight + 24);
                    ctx.beginPath();
                    ctx.moveTo(x, padding.top + plotHeight);
                    ctx.lineTo(x, padding.top + plotHeight + 6);
                    ctx.strokeStyle = "rgba(255,255,255,0.2)";
                    ctx.stroke();
                  });
                } else {
                  ctx.fillText(xLabelMin, padding.left, padding.top + plotHeight + 24);
                  ctx.fillText(xLabelMax, padding.left + plotWidth - 60, padding.top + plotHeight + 24);
                }
                ctx.fillText(String(yMax.toFixed(2)), 8, padding.top + 10);
                ctx.fillText(String(yMin.toFixed(2)), 8, padding.top + plotHeight);

                const screenPoints = points.map((p, idx) => {
                  const xVal = isNumeric ? p.qid : xValues[idx];
                  const x = padding.left + ((xVal - xMin) / xRange) * plotWidth;
                  const y = padding.top + (1 - (p.y - yMin) / yRange) * plotHeight;
                  return { ...p, x, y };
                });

                points.forEach((p, idx) => {
                  const x = screenPoints[idx].x;
                  const y = screenPoints[idx].y;
                  ctx.fillStyle = colors[p.parentKey];
                  ctx.beginPath();
                  ctx.arc(x, y, 4, 0, Math.PI * 2);
                  ctx.fill();
                });

                const tooltip = document.getElementById("tooltip");
                const sidebar = document.getElementById("sidebar");
                const sidebarContent = document.getElementById("sidebarContent");
                const sidebarTitle = document.getElementById("sidebarTitle");
                const sidebarClose = document.getElementById("sidebarClose");
                const sidebarResize = document.getElementById("sidebarResize");
                let tooltipLocked = false;
                let lastHoverKey = "";
                let hideTimer = null;
                const positionTooltip = (anchorX, anchorY) => {
                  const tooltipRect = tooltip.getBoundingClientRect();
                  const offset = 12;
                  let left = anchorX + offset;
                  let top = anchorY + offset;
                  if (left + tooltipRect.width > window.innerWidth - 8) {
                    left = anchorX - tooltipRect.width - offset;
                  }
                  if (left < 8) left = 8;
                  if (top + tooltipRect.height > window.innerHeight - 8) {
                    top = Math.max(8, window.innerHeight - tooltipRect.height - 8);
                  }
                  tooltip.style.left = `${left}px`;
                  tooltip.style.top = `${top}px`;
                };
                const hideTooltip = () => {
                  if (tooltipLocked || tooltip.matches(":hover")) {
                    return;
                  }
                  tooltip.style.display = "none";
                  tooltip.scrollTop = 0;
                  lastHoverKey = "";
                };
                sidebarClose.onclick = () => {
                  sidebar.style.display = "none";
                };
                sidebarResize.onmousedown = (evt) => {
                  evt.preventDefault();
                  const startX = evt.clientX;
                  const startWidth = sidebar.getBoundingClientRect().width;
                  const onMove = (moveEvt) => {
                    const delta = startX - moveEvt.clientX;
                    const newWidth = Math.min(
                      window.innerWidth * 0.9,
                      Math.max(520, startWidth + delta)
                    );
                    sidebar.style.width = `${newWidth}px`;
                  };
                  const onUp = () => {
                    window.removeEventListener("mousemove", onMove);
                    window.removeEventListener("mouseup", onUp);
                  };
                  window.addEventListener("mousemove", onMove);
                  window.addEventListener("mouseup", onUp);
                };
                tooltip.onmouseenter = (evt) => {
                  evt.stopPropagation();
                  tooltipLocked = true;
                  if (hideTimer) {
                    clearTimeout(hideTimer);
                    hideTimer = null;
                  }
                };
                tooltip.onmouseleave = () => {
                  tooltipLocked = false;
                  hideTimer = window.setTimeout(() => {
                    hideTooltip();
                    hideTimer = null;
                  }, 120);
                };
                tooltip.onfocusin = () => {
                  tooltipLocked = true;
                  if (hideTimer) {
                    clearTimeout(hideTimer);
                    hideTimer = null;
                  }
                };
                tooltip.onfocusout = () => {
                  tooltipLocked = false;
                  hideTimer = window.setTimeout(() => {
                    hideTooltip();
                    hideTimer = null;
                  }, 120);
                };
                tooltip.onwheel = (evt) => {
                  evt.preventDefault();
                  evt.stopPropagation();
                  tooltip.scrollTop += evt.deltaY;
                };
                const hitRadius = 14;
                canvas.onmousemove = (evt) => {
                  const rect = canvas.getBoundingClientRect();
                  const mx = evt.clientX - rect.left;
                  const my = evt.clientY - rect.top;
                  const hits = screenPoints.filter((p) => {
                    const dx = mx - p.x;
                    const dy = my - p.y;
                    return dx * dx + dy * dy <= hitRadius * hitRadius;
                  });
                  if (!hits.length) {
                    return;
                  }
                  const best = hits[0];
                  const hoverKey = `${best.qid}:${metricKey}:${hits.length}`;
                  if (hoverKey !== lastHoverKey) {
                    tooltip.innerHTML =
                      hits.length > 1
                        ? renderMultiTooltipContent(best.qid, metricKey, hits, parentLabels)
                        : renderTooltipContent(best, metricKey, parentLabels);
                    tooltip.style.display = "block";
                    const anchorX = rect.left + best.x;
                    const anchorY = rect.top + best.y;
                    positionTooltip(anchorX, anchorY);
                    tooltip.tabIndex = 0;
                    tooltip.focus({ preventScroll: true });
                    lastHoverKey = hoverKey;
                    if (hideTimer) {
                      clearTimeout(hideTimer);
                      hideTimer = null;
                    }
                  } else if (!tooltipLocked) {
                    const anchorX = rect.left + best.x;
                    const anchorY = rect.top + best.y;
                    positionTooltip(anchorX, anchorY);
                  }
                };
                canvas.onmouseleave = () => {
                  if (!tooltipLocked) {
                    hideTooltip();
                  }
                };
                canvas.onclick = (evt) => {
                  const rect = canvas.getBoundingClientRect();
                  const mx = evt.clientX - rect.left;
                  const my = evt.clientY - rect.top;
                  let best = null;
                  let bestDist = hitRadius * hitRadius;
                  screenPoints.forEach((p) => {
                    const dx = mx - p.x;
                    const dy = my - p.y;
                    const d2 = dx * dx + dy * dy;
                    if (d2 <= bestDist) {
                      bestDist = d2;
                      best = p;
                    }
                  });
                  if (!best) return;
                  const row = runsByQid.get(String(best.qid));
                  sidebarTitle.textContent = `question_id: ${best.qid} (${metricKey})`;
                  sidebarContent.innerHTML = renderSidebarContent(
                    best.qid,
                    row,
                    parentLabels,
                    window.__metricKeys || [metricKey],
                    parentOrder
                  );
                  sidebar.style.display = "block";
                };
              }

              let __compareParentsObservers = [];
              let __compareParentsState = null;

              function renderPlots(state) {
                const { metricKeys, parentInputs, parsedByParent, rows, parentLabels } = state;
                const parentKeys = parentInputs.map((_, idx) => `p${idx}`);
                const colors = parentKeys.reduce((acc, key, idx) => {
                  acc[key] = palette[idx % palette.length];
                  return acc;
                }, {});

                const headerHeight = document.querySelector("header")?.offsetHeight || 0;
                const controlsHeight = document.querySelector(".controls")?.parentElement?.offsetHeight || 0;
                const legendHeight = document.getElementById("legend")?.parentElement?.offsetHeight || 0;
                const available = window.innerHeight - headerHeight - controlsHeight - legendHeight - 80;
                const defaultPlotHeight = Math.max(220, Math.floor(available / 3));

                __compareParentsObservers.forEach((observer) => observer.disconnect());
                __compareParentsObservers = [];

                const plotsContainer = document.getElementById("plotsContainer");
                plotsContainer.innerHTML = "";
                window.__metricKeys = metricKeys.slice();

                metricKeys.forEach((metricKey) => {
                  const panel = document.createElement("div");
                  panel.className = "panel plot-panel";
                  panel.style.height = `${defaultPlotHeight}px`;
                  const title = document.createElement("div");
                  title.className = "subtle";
                  title.textContent = metricKey;
                  const canvas = document.createElement("canvas");
                  canvas.width = 1200;
                  canvas.height = 300;
                  panel.appendChild(title);
                  panel.appendChild(canvas);
                  plotsContainer.appendChild(panel);

                  const points = [];
                  parsedByParent.forEach((runs, idx) => {
                    points.push(...buildPoints(runs, `p${idx}`, metricKey));
                  });

                  const redraw = () => {
                    drawScatter(canvas, points, title, metricKey, colors, rows, parentLabels, parentKeys);
                  };

                  redraw();

                  const observer = new ResizeObserver(() => {
                    redraw();
                  });
                  observer.observe(panel);
                  __compareParentsObservers.push(observer);
                });
              }

              async function loadData() {
                const statusEl = document.getElementById("status");
                statusEl.textContent = "";
                try {
                  updateUrlFromInputs();
                  const experimentId = document.getElementById("experimentId").value.trim();
                  let parentInputs = parseList(document.getElementById("parentsInput").value);
                  let metricKeys = parseList(document.getElementById("metricsInput").value);
                  if (!metricKeys.length) {
                    metricKeys = defaultMetrics.slice();
                  }

                  if (!experimentId || parentInputs.length < 1) {
                    statusEl.textContent = "Provide experiment_id and at least one parent run.";
                    return;
                  }

                  statusEl.textContent = "Loading parent runs...";
                parentInputs = await Promise.all(
                  parentInputs.map(async (parentInput) => {
                    if (!isLikelyRunId(parentInput)) {
                      const resolved = await resolveParentRunId(experimentId, parentInput);
                      return resolved || parentInput;
                    }
                    return parentInput;
                  })
                );
                  parentInputs = parentInputs.filter(Boolean);
                  if (!parentInputs.length) {
                    statusEl.textContent = "No parent runs resolved.";
                    return;
                  }

                  const parentRuns = await Promise.all(
                    parentInputs.map((parentId) => getRunById(experimentId, parentId))
                  );
                  const parentLabels = parentInputs.reduce((acc, parentId, idx) => {
                    acc[`p${idx}`] = parentRuns[idx]?.info?.run_name || parentId;
                    return acc;
                  }, {});

                  const legend = document.getElementById("legend");
                  legend.innerHTML = "";
                  parentInputs.forEach((parentId, idx) => {
                    const key = `p${idx}`;
                    const color = palette[idx % palette.length];
                    const span = document.createElement("span");
                    span.innerHTML = `<span class="dot" style="background: ${color};"></span>${parentLabels[key]}`;
                    legend.appendChild(span);
                  });

                  statusEl.textContent = "Loading child runs...";
                  const allRuns = await Promise.all(
                    parentInputs.map((parentId) => searchRuns(experimentId, parentId))
                  );

                  const parsedByParent = allRuns.map((runs) => runs.map(parseRun));

                  const rows = new Map();
                  function upsert(run, key) {
                    const qid = run.params.question_id;
                    if (!qid) return;
                    if (!rows.has(qid)) rows.set(qid, {});
                    rows.get(qid)[key] = run;
                  }
                  function upsertRuns(runs, key) {
                    runs.forEach((r) => upsert(r, key));
                  }
                  parsedByParent.forEach((runs, idx) => upsertRuns(runs, `p${idx}`));

                  __compareParentsState = {
                    metricKeys,
                    parentInputs,
                    parsedByParent,
                    rows,
                    parentLabels,
                  };
                  renderPlots(__compareParentsState);

                  const totalRuns = parsedByParent.reduce((acc, runs) => acc + runs.length, 0);
                  statusEl.textContent = `Loaded ${parentInputs.length} parent(s), ${totalRuns} child run(s).`;
                } catch (err) {
                  console.error("Compare parents error", err);
                  statusEl.textContent = `Error: ${err?.message || err}`;
                }
              }

              document.getElementById("loadBtn").addEventListener("click", loadData);
              setInputsFromUrl();
              const statusEl = document.getElementById("status");
              if (statusEl && !statusEl.textContent) {
                statusEl.textContent = "Ready.";
              }
              loadData();
              window.addEventListener("resize", () => {
                if (__compareParentsState) {
                  renderPlots(__compareParentsState);
                }
              });
            
    """
)


@app.route(_add_static_prefix("/compare-parents.js"))
def serve_compare_parents_js():
    return Response(_COMPARE_PARENTS_JS, mimetype="application/javascript")


@app.route(_add_static_prefix("/compare-parents"))
def serve_compare_parents():
    html = textwrap.dedent(
        """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>MLflow Parent Run Compare</title>
            <style>
              :root {
                color-scheme: light dark;
                --bg: #0e1116;
                --panel: #151a21;
                --muted: #9aa4b2;
                --text: #e6e9ef;
                --accent-a: #4d9de0;
                --accent-b: #f07167;
                --grid: rgba(255, 255, 255, 0.08);
              }
              body {
                margin: 0;
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
                background: var(--bg);
                color: var(--text);
              }
              header {
                padding: 16px 20px;
                border-bottom: 1px solid var(--grid);
                background: linear-gradient(180deg, #12161d, #0e1116);
              }
              h1 {
                margin: 0 0 6px 0;
                font-size: 18px;
                font-weight: 600;
              }
              .subtle {
                color: var(--muted);
                font-size: 13px;
              }
              .container {
                padding: 16px 20px 28px 20px;
                display: grid;
                gap: 16px;
              }
              .panel {
                background: var(--panel);
                border: 1px solid var(--grid);
                border-radius: 10px;
                padding: 12px;
              }
              .controls {
                display: grid;
                gap: 10px;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                align-items: start;
              }
              .controls > div {
                display: flex;
                flex-direction: column;
              }
              .button-col {
                display: flex;
                flex-direction: column;
              }
              .button-col label {
                visibility: hidden;
                height: 12px;
                margin-bottom: 6px;
              }
              label {
                display: block;
                font-size: 12px;
                color: var(--muted);
                margin-bottom: 6px;
              }
              input {
                width: 100%;
                box-sizing: border-box;
                padding: 8px 10px;
                border-radius: 6px;
                border: 1px solid var(--grid);
                background: #0f1319;
                color: var(--text);
                font-size: 13px;
              }
              button {
                padding: 9px 12px;
                border-radius: 6px;
                border: 1px solid var(--grid);
                background: #1b2230;
                color: var(--text);
                cursor: pointer;
                font-size: 13px;
                font-weight: 600;
              }
              button:hover {
                background: #222b3d;
              }
              .legend {
                display: flex;
                gap: 16px;
                flex-wrap: wrap;
                align-items: center;
                font-size: 12px;
                color: var(--muted);
              }
              .tooltip {
                position: fixed;
                pointer-events: auto;
                z-index: 9999;
                background: #0f1319;
                border: 1px solid var(--grid);
                border-radius: 8px;
                padding: 10px;
                max-width: 560px;
                max-height: 420px;
                overflow: auto;
                overscroll-behavior: contain;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
                display: none;
              }
              .sidebar {
                position: fixed;
                right: 16px;
                top: 16px;
                bottom: 16px;
                width: 520px;
                background: #0f1319;
                border: 1px solid var(--grid);
                border-radius: 10px;
                padding: 12px;
                overflow: auto;
                display: none;
                z-index: 10000;
                box-shadow: 0 16px 36px rgba(0, 0, 0, 0.45);
                font-size: 13px;
                min-width: 520px;
                max-width: 90vw;
              }
              .sidebar-resize {
                position: absolute;
                left: -6px;
                top: 0;
                bottom: 0;
                width: 12px;
                cursor: ew-resize;
                background: rgba(255, 255, 255, 0.04);
                border-left: 1px solid rgba(255, 255, 255, 0.08);
                z-index: 1;
                pointer-events: auto;
              }
              .sidebar h3 {
                margin: 0 0 8px 0;
                font-size: 14px;
                font-weight: 600;
              }
              .sidebar .cols {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 10px;
                margin-bottom: 12px;
              }
              .sidebar .card {
                border: 1px solid var(--grid);
                border-radius: 8px;
                padding: 8px;
                background: #10151d;
              }
              .sidebar .card h4 {
                margin: 0 0 6px 0;
                font-size: 13px;
                font-weight: 600;
                color: var(--text);
              }
              .kv {
                font-size: 12px;
                color: var(--muted);
                line-height: 1.5;
              }
              .kv b {
                color: var(--text);
                font-weight: 600;
              }
              details {
                margin-top: 10px;
                border: 1px solid var(--grid);
                border-radius: 8px;
                padding: 6px 8px;
                background: #0f1319;
              }
              details summary {
                cursor: pointer;
                color: var(--muted);
                font-size: 11px;
                user-select: none;
              }
              .sidebar .close {
                position: absolute;
                top: 10px;
                right: 10px;
                border: 1px solid var(--grid);
                background: #141923;
                color: var(--text);
                border-radius: 6px;
                padding: 4px 8px;
                cursor: pointer;
                font-size: 12px;
              }
              .tooltip h4 {
                margin: 0 0 6px 0;
                font-size: 12px;
                font-weight: 600;
                color: var(--text);
              }
              .tooltip pre {
                margin: 0;
                white-space: pre-wrap;
                word-break: break-word;
                color: var(--muted);
                font-size: 11px;
              }
              .dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 6px;
              }
              .plot-grid {
                display: grid;
                gap: 16px;
              }
              .help {
                font-size: 11px;
                color: var(--muted);
                margin-top: 6px;
              }
              .status {
                font-size: 11px;
                color: var(--muted);
                margin-top: 6px;
              }
              .plot-panel {
                resize: vertical;
                overflow: hidden;
                min-height: 220px;
              }
              canvas {
                width: 100%;
                height: 240px;
                background: #0f1319;
                border-radius: 8px;
              }
              table {
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
              }
              th, td {
                border-bottom: 1px solid var(--grid);
                padding: 6px 8px;
                text-align: left;
                vertical-align: top;
              }
              th {
                color: var(--muted);
                font-weight: 600;
              }
              .muted {
                color: var(--muted);
              }
            </style>
          </head>
          <body>
            <header>
              <h1>Parent Run Comparison</h1>
            </header>
            <div class="container">
              <div class="panel">
                <div class="controls">
                  <div>
                    <label>Experiment ID</label>
                    <input id="experimentId" placeholder="e.g. 2" />
                    <div class="status" id="status"></div>
                  </div>
                  <div>
                    <label>Parent Runs</label>
                    <input id="parentsInput" placeholder="parent_a, parent_b, parent_c" />
                    <div class="help">Comma or newline separated run names or IDs.</div>
                  </div>
                  <div>
                    <label>Metrics</label>
                    <input id="metricsInput" placeholder="Completeness_GEval, Grounding_GEval, Reasoning_GEval" />
                    <div class="help">Comma or newline separated metric keys.</div>
                  </div>
                  <div class="button-col">
                    <label>Load</label>
                    <button id="loadBtn">Load</button>
                  </div>
                </div>
              </div>
              <div class="panel">
                <div class="legend" id="legend"></div>
              </div>
              <div class="plot-grid" id="plotsContainer"></div>
            </div>
            <div class="tooltip" id="tooltip"></div>
            <div class="sidebar" id="sidebar">
              <div class="sidebar-resize" id="sidebarResize"></div>
              <button class="close" id="sidebarClose">Close</button>
              <h3 id="sidebarTitle">Details</h3>
              <div id="sidebarContent"></div>
            </div>
            <script src="/compare-parents.js"></script>
          </body>
        </html>
        """
    )
    response = Response(html, mimetype="text/html")
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' data: blob:; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:; "
        "style-src 'self' 'unsafe-inline' data: blob:; "
        "img-src 'self' data: blob:; "
        "connect-src 'self' data: blob:; "
        "font-src 'self' data: blob:;"
    )
    return response


# We expect the react app to be built assuming it is hosted at /static-files, so that requests for
# CSS/JS resources will be made to e.g. /static-files/main.css and we can handle them here.
# The files are hashed based on source code, so ok to send Cache-Control headers via max_age.
@app.route(_add_static_prefix("/static-files/<path:path>"))
def serve_static_file(path):
    if IS_FLASK_V1:
        return send_from_directory(app.static_folder, path, cache_timeout=2419200)
    else:
        return send_from_directory(app.static_folder, path, max_age=2419200)


# Serve the index.html for the React App for all other routes.
@app.route(_add_static_prefix("/"))
def serve():
    if os.path.exists(os.path.join(app.static_folder, "index.html")):
        return send_from_directory(app.static_folder, "index.html")

    text = textwrap.dedent(
        """
    Unable to display MLflow UI - landing page (index.html) not found.

    You are very likely running the MLflow server using a source installation of the Python MLflow
    package.

    If you are a developer making MLflow source code changes and intentionally running a source
    installation of MLflow, you can view the UI by running the Javascript dev server:
    https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md#running-the-javascript-dev-server

    Otherwise, uninstall MLflow via 'pip uninstall mlflow', reinstall an official MLflow release
    from PyPI via 'pip install mlflow', and rerun the MLflow server.
    """
    )
    return Response(text, mimetype="text/plain")


def _find_app(app_name: str) -> str:
    apps = get_entry_points("mlflow.app")
    for app in apps:
        if app.name == app_name:
            return app.value

    raise MlflowException(
        f"Failed to find app '{app_name}'. Available apps: {[a.name for a in apps]}"
    )


def _is_factory(app: str) -> bool:
    """
    Returns True if the given app is a factory function, False otherwise.

    Args:
        app: The app to check, e.g. "mlflow.server.app:app
    """
    module, obj_name = app.rsplit(":", 1)
    mod = importlib.import_module(module)
    obj = getattr(mod, obj_name)
    return isinstance(obj, types.FunctionType)


def get_app_client(app_name: str, *args, **kwargs):
    """
    Instantiate a client provided by an app.

    Args:
        app_name: The app name defined in `setup.py`, e.g., "basic-auth".
        args: Additional arguments passed to the app client constructor.
        kwargs: Additional keyword arguments passed to the app client constructor.

    Returns:
        An app client instance.
    """
    clients = get_entry_points("mlflow.app.client")
    for client in clients:
        if client.name == app_name:
            cls = client.load()
            return cls(*args, **kwargs)

    raise MlflowException(
        f"Failed to find client for '{app_name}'. Available clients: {[c.name for c in clients]}"
    )


def _build_waitress_command(waitress_opts, host, port, app_name, is_factory):
    opts = shlex.split(waitress_opts) if waitress_opts else []
    return [
        sys.executable,
        "-m",
        "waitress",
        *opts,
        f"--host={host}",
        f"--port={port}",
        "--ident=mlflow",
        *(["--call"] if is_factory else []),
        app_name,
    ]


def _build_gunicorn_command(gunicorn_opts, host, port, workers, app_name):
    bind_address = f"{host}:{port}"
    opts = shlex.split(gunicorn_opts) if gunicorn_opts else []
    return [
        sys.executable,
        "-m",
        "gunicorn",
        *opts,
        "-b",
        bind_address,
        "-w",
        str(workers),
        app_name,
    ]


def _build_uvicorn_command(uvicorn_opts, host, port, workers, app_name, env_file=None):
    """Build command to run uvicorn server."""
    opts = shlex.split(uvicorn_opts) if uvicorn_opts else []
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        *opts,
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        str(workers),
    ]
    if env_file:
        cmd.extend(["--env-file", env_file])
    cmd.append(app_name)
    return cmd


def _run_server(
    *,
    file_store_path,
    registry_store_uri,
    default_artifact_root,
    serve_artifacts,
    artifacts_only,
    artifacts_destination,
    host,
    port,
    static_prefix=None,
    workers=None,
    gunicorn_opts=None,
    waitress_opts=None,
    expose_prometheus=None,
    app_name=None,
    uvicorn_opts=None,
    env_file=None,
    secrets_cache_ttl=None,
    secrets_cache_max_size=None,
):
    """
    Run the MLflow server, wrapping it in gunicorn, uvicorn, or waitress on windows

    Args:
        static_prefix: If set, the index.html asset will be served from the path static_prefix.
                       If left None, the index.html asset will be served from the root path.
        uvicorn_opts: Additional options for uvicorn server.

    Returns:
        None
    """
    env_map = {}
    if file_store_path:
        env_map[BACKEND_STORE_URI_ENV_VAR] = file_store_path
    if registry_store_uri:
        env_map[REGISTRY_STORE_URI_ENV_VAR] = registry_store_uri
    if default_artifact_root:
        env_map[ARTIFACT_ROOT_ENV_VAR] = default_artifact_root
    if serve_artifacts:
        env_map[SERVE_ARTIFACTS_ENV_VAR] = "true"
    if artifacts_only:
        env_map[ARTIFACTS_ONLY_ENV_VAR] = "true"
    if artifacts_destination:
        env_map[ARTIFACTS_DESTINATION_ENV_VAR] = artifacts_destination
    if static_prefix:
        env_map[STATIC_PREFIX_ENV_VAR] = static_prefix

    if expose_prometheus:
        env_map[PROMETHEUS_EXPORTER_ENV_VAR] = expose_prometheus

    if secrets_cache_ttl is not None:
        env_map[SECRETS_CACHE_TTL_ENV_VAR] = str(secrets_cache_ttl)
    if secrets_cache_max_size is not None:
        env_map[SECRETS_CACHE_MAX_SIZE_ENV_VAR] = str(secrets_cache_max_size)

    if secret_key := MLFLOW_FLASK_SERVER_SECRET_KEY.get():
        env_map[MLFLOW_FLASK_SERVER_SECRET_KEY.name] = secret_key

    # Determine which server we're using (only one should be true)
    using_gunicorn = gunicorn_opts is not None
    using_waitress = waitress_opts is not None
    using_uvicorn = not using_gunicorn and not using_waitress

    if using_uvicorn:
        env_map[_MLFLOW_SGI_NAME.name] = "uvicorn"
    elif using_waitress:
        env_map[_MLFLOW_SGI_NAME.name] = "waitress"
    elif using_gunicorn:
        env_map[_MLFLOW_SGI_NAME.name] = "gunicorn"

    if app_name is None:
        is_factory = False
        # For uvicorn, use the FastAPI app; for gunicorn/waitress, use the Flask app
        app = "mlflow.server.fastapi_app:app" if using_uvicorn else f"{__name__}:app"
    else:
        app = _find_app(app_name)
        is_factory = _is_factory(app)
        # `waitress` doesn't support `()` syntax for factory functions.
        # Instead, we need to use the `--call` flag.
        # Don't use () syntax if we're using uvicorn
        use_factory_syntax = not is_windows() and is_factory and not using_uvicorn
        app = f"{app}()" if use_factory_syntax else app

    # Determine which server to use
    if using_uvicorn:
        # Use uvicorn (default when no specific server options are provided)
        full_command = _build_uvicorn_command(
            uvicorn_opts, host, port, workers or 4, app, env_file
        )
    elif using_waitress:
        # Use waitress if explicitly requested
        warnings.warn(
            "We recommend using uvicorn for improved performance. "
            "Please use uvicorn by default or specify '--uvicorn-opts' "
            "instead of '--waitress-opts'.",
            FutureWarning,
            stacklevel=2,
        )
        full_command = _build_waitress_command(
            waitress_opts, host, port, app, is_factory
        )
    elif using_gunicorn:
        # Use gunicorn if explicitly requested
        if sys.platform == "win32":
            raise MlflowException(
                "Gunicorn is not supported on Windows. "
                "Please use uvicorn (default) or specify '--waitress-opts'."
            )
        warnings.warn(
            "We recommend using uvicorn for improved performance. "
            "Please use uvicorn by default or specify '--uvicorn-opts' "
            "instead of '--gunicorn-opts'.",
            FutureWarning,
            stacklevel=2,
        )
        full_command = _build_gunicorn_command(
            gunicorn_opts, host, port, workers or 4, app
        )
    else:
        # This shouldn't happen given the logic in CLI, but handle it just in case
        raise MlflowException("No server configuration specified.")

    if MLFLOW_SERVER_ENABLE_JOB_EXECUTION.get():
        # The `HUEY_STORAGE_PATH_ENV_VAR` is used by both MLflow server handler workers and
        # huey job runner (huey_consumer).
        env_map[HUEY_STORAGE_PATH_ENV_VAR] = (
            tempfile.mkdtemp(dir="/dev/shm")  # Use in-memory file system if possible
            if os.path.exists("/dev/shm")
            else tempfile.mkdtemp()
        )

    if MLFLOW_SERVER_ENABLE_JOB_EXECUTION.get():
        from mlflow.server.jobs.utils import _check_requirements

        try:
            _check_requirements(file_store_path)
        except Exception as e:
            raise MlflowException(
                f"MLflow job runner requirements checking failed (root error: {e!s}). "
                "If you don't need MLflow job runner, you can disable it by setting "
                "environment variable 'MLFLOW_SERVER_ENABLE_JOB_EXECUTION' to 'false'."
            )

    server_proc = _exec_cmd(
        full_command, extra_env=env_map, capture_output=False, synchronous=False
    )

    if MLFLOW_SERVER_ENABLE_JOB_EXECUTION.get():
        from mlflow.environment_variables import MLFLOW_TRACKING_URI
        from mlflow.server.jobs.utils import _launch_job_runner

        _launch_job_runner(
            {
                **env_map,
                # Set tracking URI environment variable for job runner
                # so that all job processes inherits it.
                MLFLOW_TRACKING_URI.name: f"http://{host}:{port}",
            },
            server_proc.pid,
        )

    server_proc.wait()
