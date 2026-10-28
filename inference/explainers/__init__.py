# Explainability helpers — Milestone 2.
#
# Planned:
#   shap_explainer.py   — SHAP TreeExplainer for XGBoost / RF (per-feature contributions)
#   ae_explainer.py     — per-feature reconstruction error (x − x̂)² already embedded in
#                         core/detectors/mlp_ae.py; this module will format it for the UI
#
# Both are inference-time, not training-time — they run inside /api/routers/predict.py
# after Stage 2 and Stage 3 have produced their scores.
