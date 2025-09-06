# Architecture Diagrams

## 🏗️ System Architecture Overview

### **High-Level Architecture**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Organic Agriculture Agentic AI               │
│                         System Architecture                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   AI Agents     │    │   Output Layer  │
│                 │    │                 │    │                 │
│ • Weather API   │───▶│ • Weather Agent │───▶│ • Dashboard     │
│ • Market API    │───▶│ • Market Agent  │───▶│ • Notifications │
│ • IoT Sensors   │───▶│ • Pest Agent    │───▶│ • Alerts        │
│ • Historical    │───▶│ • Prediction    │───▶│ • Reports       │
│   Data          │    │ • Decision      │    │                 │
│                 │    │ • Notification  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Processing Layer                        │
│ • MongoDB (Data Storage)                                       │
│ • LangGraph (Agent Orchestration)                              │
│ • FastAPI (API Gateway)                                        │
│ • Kafka (Message Streaming)                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Architecture

### **Complete Data Flow**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Weather   │    │   Market    │    │    IoT      │    │ Historical  │
│     API     │    │     API     │    │  Sensors    │    │    Data     │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Weather    │    │   Market    │    │    Pest     │    │   Data      │
│   Agent     │    │    Agent    │    │   Agent     │    │ Processor   │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │                  │
       └──────────────────┼──────────────────┼──────────────────┘
                          │                  │
                          ▼                  ▼
                ┌─────────────────────────────────┐
                │     Supply Chain Risk           │
                │        Predictor                │
                │    (CatBoost/XGBoost)           │
                └─────────────────┬───────────────┘
                                  │
                                  ▼
                ┌─────────────────────────────────┐
                │      Action Recommender         │
                │     (Business Rules Engine)     │
                └─────────────────┬───────────────┘
                                  │
                                  ▼
                ┌─────────────────────────────────┐
                │     Notification Agent          │
                │  (Multi-channel Alerts)        │
                └─────────────────┬───────────────┘
                                  │
                                  ▼
                ┌─────────────────────────────────┐
                │        Output Layer             │
                │ • Dashboard (React)             │
                │ • Email/SMS/WhatsApp            │
                │ • API Responses                 │
                └─────────────────────────────────┘
```

---

## 🤖 Agent Architecture

### **Agent Interaction Diagram**
```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Agent Orchestration                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Weather   │    │   Market    │    │    Pest     │
│   Agent     │    │   Agent     │    │   Agent     │
│             │    │             │    │             │
│ • Fetch     │    │ • Scrape    │    │ • Process   │
│   Weather   │    │   Prices    │    │   IoT Data  │
│ • Detect    │    │ • Analyze   │    │ • Detect    │
│   Anomalies │    │   Trends    │    │   Outbreaks │
│ • Forecast  │    │ • Predict   │    │ • Assess    │
│   Patterns  │    │   Demand    │    │   Risk      │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
                          ▼
                ┌─────────────────────────────────┐
                │     Risk Prediction Agent       │
                │                                 │
                │ • Aggregate Data                │
                │ • Feature Engineering           │
                │ • ML Model Inference            │
                │ • Risk Scoring                  │
                │ • Confidence Calculation        │
                └─────────────────┬───────────────┘
                                  │
                                  ▼
                ┌─────────────────────────────────┐
                │     Decision Agent              │
                │                                 │
                │ • Business Rules Engine         │
                │ • Priority Scoring              │
                │ • Action Generation             │
                │ • Cost-Benefit Analysis         │
                └─────────────────┬───────────────┘
                                  │
                                  ▼
                ┌─────────────────────────────────┐
                │   Notification Agent            │
                │                                 │
                │ • Multi-channel Delivery        │
                │ • Template Management            │
                │ • Delivery Tracking             │
                │ • Feedback Collection            │
                └─────────────────────────────────┘
```

---

## 🗄️ Database Architecture

### **MongoDB Collections Structure**
```
┌─────────────────────────────────────────────────────────────────┐
│                        MongoDB Database                         │
│                    (organic_agriculture)                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Weather      │    │     Market      │    │      Pest       │
│   Collection    │    │   Collection    │    │   Collection    │
│                 │    │                 │    │                 │
│ • farm_id       │    │ • crop          │    │ • farm_id       │
│ • timestamp     │    │ • region        │    │ • crop          │
│ • temperature   │    │ • price_current │    │ • pest_risk     │
│ • rainfall      │    │ • price_forecast│    │ • disease_alert │
│ • humidity      │    │ • demand_trend  │    │ • pest_types    │
│ • wind_speed    │    │ • supply_level  │    │ • severity      │
│ • storm_alert   │    │ • volatility    │    │ • confidence    │
└─────────────────┘    └─────────────────┘    └─────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Risk        │    │    Actions      │    │  Notifications  │
│  Collection     │    │  Collection     │    │  Collection     │
│                 │    │                 │    │                 │
│ • farm_id       │    │ • farm_id       │    │ • user_id       │
│ • crop          │    │ • action_id     │    │ • farm_id       │
│ • risk_score    │    │ • action_type   │    │ • type          │
│ • risk_type     │    │ • description   │    │ • title         │
│ • confidence    │    │ • priority      │    │ • message       │
│ • factors       │    │ • deadline      │    │ • priority      │
│ • timestamp     │    │ • status        │    │ • status        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🌐 API Architecture

### **FastAPI Backend Structure**
```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Weather       │    │     Market      │    │      Pest       │
│   Endpoints     │    │   Endpoints     │    │   Endpoints     │
│                 │    │                 │    │                 │
│ • GET /weather/ │    │ • GET /market/  │    │ • GET /pest/    │
│   current       │    │   prices        │    │   risk          │
│ • GET /weather/ │    │ • GET /market/  │    │ • POST /pest/   │
│   historical    │    │   demand        │    │   report        │
│ • WebSocket     │    │ • WebSocket     │    │ • WebSocket     │
│   /ws/weather   │    │   /ws/market    │    │   /ws/pest      │
└─────────────────┘    └─────────────────┘    └─────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prediction    │    │   Decision      │    │  Notification   │
│   Endpoints     │    │   Endpoints     │    │   Endpoints     │
│                 │    │                 │    │                 │
│ • POST /predict/│    │ • POST /recommend│   │ • GET /notifications│
│   risk-score    │    │   /actions      │    │ • POST /notifications│
│ • GET /predict/ │    │ • POST /recommend│   │   /send         │
│   batch         │    │   /validate     │    │ • WebSocket     │
│ • WebSocket     │    │ • WebSocket     │    │   /ws/notifications│
│   /ws/predict   │    │   /ws/decision  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🎨 Frontend Architecture

### **React Dashboard Structure**
```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │    │   Risk          │    │   Market        │
│   Components    │    │   Components    │    │   Components    │
│                 │    │                 │    │                 │
│ • Farm Overview │    │ • Risk Heatmap  │    │ • Price Charts  │
│ • Alert Panel   │    │ • Risk Timeline │    │ • Trend Analysis│
│ • Navigation    │    │ • Risk Details  │    │ • Demand Forecast│
│ • User Profile  │    │ • Risk Alerts   │    │ • Market Alerts │
└─────────────────┘    └─────────────────┘    └─────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Weather       │    │   Actions       │    │   Settings      │
│   Components    │    │   Components    │    │   Components    │
│                 │    │                 │    │                 │
│ • Weather Map   │    │ • Action List   │    │ • User Settings │
│ • Forecast      │    │ • Action Details│    │ • Farm Settings │
│ • Alerts        │    │ • Action History│    │ • Notification  │
│ • Historical    │    │ • Action Status │    │   Preferences   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🔄 Real-time Data Flow

### **WebSocket Communication**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Real-time Data Flow                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Weather   │    │   Market    │    │    IoT      │
│   Updates   │    │   Updates   │    │  Updates    │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WebSocket Gateway                           │
│ • Authentication & Authorization                               │
│ • Message Routing                                              │
│ • Rate Limiting                                                │
│ • Connection Management                                        │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend WebSocket                          │
│ • Real-time Updates                                            │
│ • State Management                                             │
│ • Error Handling                                               │
│ • Reconnection Logic                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Deployment Architecture

### **Production Deployment**
```
┌─────────────────────────────────────────────────────────────────┐
│                        Cloud Infrastructure                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load          │    │   Application   │    │   Database      │
│   Balancer      │    │   Servers       │    │   Cluster       │
│                 │    │                 │    │                 │
│ • SSL/TLS       │    │ • FastAPI       │    │ • MongoDB       │
│ • Rate Limiting │    │ • LangGraph     │    │   Replica Set   │
│ • Health Checks │    │ • ML Models     │    │ • Sharding      │
│ • Auto Scaling  │    │ • Auto Scaling  │    │ • Backup        │
└─────────────────┘    └─────────────────┘    └─────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Message       │    │   Monitoring    │    │   Storage       │
│   Queue         │    │   & Logging     │    │                 │
│                 │    │                 │    │                 │
│ • Kafka         │    │ • Prometheus    │    │ • S3/Blob       │
│ • Event         │    │ • Grafana       │    │   Storage       │
│   Streaming     │    │ • ELK Stack     │    │ • Model         │
│ • Dead Letter   │    │ • Alerting      │    │   Artifacts     │
│   Queue         │    │ • Dashboards    │    │ • Data          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 📊 Monitoring Architecture

### **Observability Stack**
```
┌─────────────────────────────────────────────────────────────────┐
│                        Monitoring Stack                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   Infrastructure│    │   Business      │
│   Metrics       │    │   Metrics       │    │   Metrics       │
│                 │    │                 │    │                 │
│ • API Latency   │    │ • CPU Usage     │    │ • Risk Accuracy │
│ • Error Rates   │    │ • Memory Usage  │    │ • User Activity │
│ • Throughput    │    │ • Disk Usage    │    │ • Revenue       │
│ • Agent         │    │ • Network I/O   │    │ • Cost Savings  │
│   Performance   │    │ • Database      │    │ • User          │
│                 │    │   Performance   │    │   Satisfaction  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Centralized Monitoring                      │
│ • Prometheus (Metrics Collection)                              │
│ • Grafana (Visualization & Dashboards)                         │
│ • ELK Stack (Logs & Search)                                    │
│ • AlertManager (Alerting)                                      │
│ • Jaeger (Distributed Tracing)                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**Last Updated**: [Auto-updated by system]  
**Version**: 1.0  
**Maintainer**: Architecture Team
