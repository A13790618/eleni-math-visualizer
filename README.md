---
title: Eleni Math Visualizer
emoji: 📐
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Eleni Math Visualization API

Backend for a Coze custom plugin that creates Greek-labelled geometry diagrams,
solid-geometry projections, and function plots.

## Endpoints

- `GET /api/health`
- `GET /openapi.json`
- `POST /api/geometry`
- `POST /api/plot`
- `GET /images/<filename>`

The service runs without authentication when `API_KEY` is unset. For a public
deployment, set a private `API_KEY` secret and configure the same value in Coze
as either an `X-API-Key` header or a Bearer token.

Generated images use unguessable names and remain on the service for 24 hours.
Interactive viewer links carry a signed, tamper-resistant model description;
they survive service restarts and remain valid for seven days. Rotating the
owner secret invalidates previous viewer links. The service stores no student
data or database.

See `openapi.json` for the complete request contract and examples.
