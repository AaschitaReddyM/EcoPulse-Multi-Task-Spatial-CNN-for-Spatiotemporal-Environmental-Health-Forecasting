# EcoPulse-Multi-Task-Spatial-CNN-for-Spatiotemporal-Environmental-Health-Forecasting
EcoPulse is an enterprise-grade multi-task spatial CNN pipeline built in PyTorch to forecast respiratory, cardiovascular, and metabolic clinical risks simultaneously by combining longitudinal data with hyper-local spatiotemporal environmental feeds. Includes INT8 weight quantization for resource-constrained embedded edge hardware.

Academic context
Developed as an advanced Data Science semester project during the Master of Science in Data Science program at Pace University, New York. The engine is designed to comply strictly with the federally mandated SMART on FHIR interoperability standards for native integration inside enterprise EHR charting screens

## 🧠 Architectural Overview

EcoPulse treats hyper-local geographic weather and pollution data as 2D spatial matrices[cite: 2]. Features are extracted via a shared Convolutional Neural Network (CNN) core before passing into distinct multi-task prediction heads[cite: 2]. This allows simultaneous training on parallel healthcare risk vectors while utilizing specialized clinical priority multipliers ($\alpha, \beta, \gamma$) to balance model penalty parameters[cite: 2].

### Key Features
* **Multi-Task Deep Learning Framework:** Splittable neural architecture optimized for concurrent, independent clinical risk outputs[cite: 2].
* **Spatial Feature Extraction:** Utilizes a custom 2D CNN core to parse grid-based environmental data feeds (e.g., PM2.5, ozone concentration, and surface temperature arrays)[cite: 2].
* **Resource Constraint Optimization:** Features an implementation of post-training dynamic quantization to map FP32 weights to signed INT8 configuration, validating low-latency edge device inference constraints[cite: 2].

---

## 🛠️ Production Technical Stack

* **Machine Learning Core:** PyTorch, PyTorch Forecasting[cite: 2]
* **Geospatial Processing:** Uber H3 (Spatial Hexagonal Indexing)[cite: 2]
* **Data Interoperability:** Fast Healthcare Interoperability Resources (HL7 FHIR v4/v5)[cite: 2]
* **Database Tier:** TimescaleDB + HIPAA-Compliant PostgreSQL[cite: 2]

---

## 💻 Code Structure & Execution

The core pipeline features a synthetic spatiotemporal grid generator to emulate multi-channel environmental hazard layers without external dependencies. 

### Prerequisites
```bash
pip install torch torchvision numpy


