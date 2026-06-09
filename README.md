# 🏙️ Smart City Surveillance Analytics & Modeling Engine

[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](https://streamlit.io)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-ready **Smart City Infrastructure & Public Safety Dashboard** powered by a specialized **Multi-Agent Analytical Framework** and Machine Learning. 

This application ingests municipal spatial infrastructure data, transforms urban parameters, clusters districts into strategic tiers using **Unsupervised learning ($K$-Means)**, and estimates sensor deployment requirements utilizing **Supervised ensembles (Random Forest)**.

---

## 🤖 Multi-Agent Architecture

Unlike traditional monolithic scripts, this system decouples operational concerns into distinct, cooperating algorithmic agents:

```text
       ┌────────────────────────────────────────────────────────┐
       │     v10_Addresses_camera_installation.csv (Raw Data)  │
       └───────────────────────────┬────────────────────────────┘
                                   ▼
                   ┌───────────────────────────────┐
                   │    Data Ingestion Agent       │ ── Processing & Tokenizing
                   └───────────────┬───────────────┘
                                   ▼
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Analytics Agent │       │ Visual Eng Agent│       │    ML Agent     │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ • Macro Metrics │       │ • Plotly Theme  │       │ • Unsupervised  │
│ • Aggregations  │       │ • Distribution  │       │   K-Means Clust │
│ • Leaderboards  │       │   Pie & Bar     │       │ • Supervised    │
│                 │       │   Graphics      │       │   Random Forest │
└─────────────────┘       └─────────────────┘       └─────────────────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   ▼
                  ┌────────────────────────────────┐
                  │ Streamlit UI Presentation Layer│ ── Web Application
                  └────────────────────────────────┘
