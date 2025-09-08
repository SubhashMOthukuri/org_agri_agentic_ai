# 🎨 Dashboard Visual Mockups

**FAANG-Level Visual Specifications** for Agricultural AI Dashboard

---

## 🖥️ **Desktop Layout (1920x1080)**

### **Main Dashboard View**
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🌾 AgriAI Dashboard                    🔍 [Search farms, crops...]                    🔔(3) 👤 John Doe ▼      │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 🏠 Dashboard │ 🌤️ Weather │ 📈 Market │ 🐛 Pest │ 🌾 Yield │ ⚙️ Settings                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐         │
│  │ 🌡️ Temperature  │ │ 💧 Humidity     │ │ 🌧️ Precipitation│ │ 💨 Wind Speed   │ │ ☀️ UV Index     │         │
│  │     25°C        │ │      65%        │ │      2.3mm      │ │     12 km/h     │ │      6.2        │         │
│  │   ↗️ +2.1°C     │ │   ↘️ -3.2%      │ │   ↗️ +0.8mm     │ │   ↗️ +1.2 km/h  │ │   ↗️ +0.5       │         │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘         │
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ 📊 Real-Time Weather Chart (D3.js)                                                                      │   │
│  │                                                                                                         │   │
│  │   30°C ┤                                                                                                │   │
│  │         │     ╭─╮                                                                                       │   │
│  │   25°C ┤     ╱   ╲     ╭─╮                                                                             │   │
│  │         │   ╱     ╲   ╱   ╲                                                                             │   │
│  │   20°C ┤ ╱         ╲╱     ╲                                                                             │   │
│  │         │                                                                                               │   │
│  │   15°C ┤                                                                                                │   │
│  │         └─────────────────────────────────────────────────────────────────────────────────────────────│   │
│  │         06:00  09:00  12:00  15:00  18:00  21:00  00:00                                                │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                                 │
│  ┌─────────────────────────────────┐ ┌─────────────────────────────────┐ ┌─────────────────────────────────┐   │
│  │ 🚨 Pest Detection Alerts        │ │ 📈 Market Price Trends          │ │ 🤖 AI Recommendations          │   │
│  │                                 │ │                                 │ │                                 │   │
│  │ 🔴 High Risk: Aphids detected   │ │ 🌾 Wheat: $245.50/ton          │ │ 💡 Optimal planting time:       │   │
│  │    Location: Field A-3          │ │    ↗️ +$12.30 (5.3%)           │ │    Next 3 days                  │   │
│  │    Confidence: 94%              │ │                                 │ │                                 │   │
│  │                                 │ │ 🌽 Corn: $198.75/ton           │ │ 🌱 Recommended fertilizer:      │   │
│  │ 🟡 Medium: Fungus spotted       │ │    ↘️ -$3.20 (1.6%)            │ │    NPK 15-15-15, 50kg/acre     │   │
│  │    Location: Field B-1          │ │                                 │ │                                 │   │
│  │    Confidence: 78%              │ │ 📊 6-Month Trend: ↗️ +15.2%    │ │ 💧 Irrigation schedule:         │   │
│  │                                 │ │                                 │ │    Every 2 days, 2 hours        │   │
│  │ 🟢 Low: Soil moisture optimal   │ │                                 │ │                                 │   │
│  │    Location: Field C-2          │ │                                 │ │                                 │   │
│  │    Confidence: 89%              │ │                                 │ │                                 │   │
│  └─────────────────────────────────┘ └─────────────────────────────────┘ └─────────────────────────────────┘   │
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ 🗺️ Interactive Farm Map (Mapbox + D3.js)                                                                 │   │
│  │                                                                                                         │   │
│  │     Field A-1    Field A-2    Field A-3                                                                │   │
│  │   ┌─────────┐   ┌─────────┐   ┌─────────┐                                                              │   │
│  │   │  🌾🌾🌾  │   │  🌾🌾🌾  │   │  🐛🌾🌾  │                                                              │   │
│  │   │  🌾🌾🌾  │   │  🌾🌾🌾  │   │  🌾🌾🌾  │                                                              │   │
│  │   │  🌾🌾🌾  │   │  🌾🌾🌾  │   │  🌾🌾🌾  │                                                              │   │
│  │   └─────────┘   └─────────┘   └─────────┘                                                              │   │
│  │                                                                                                         │   │
│  │     Field B-1    Field B-2    Field B-3                                                                │   │
│  │   ┌─────────┐   ┌─────────┐   ┌─────────┐                                                              │   │
│  │   │  🌾🌾🌾  │   │  🌾🌾🌾  │   │  🌾🌾🌾  │                                                              │   │
│  │   │  🌾🌾🌾  │   │  🌾🌾🌾  │   │  🌾🌾🌾  │                                                              │   │
│  │   │  🌾🌾🌾  │   │  🌾🌾🌾  │   │  🌾🌾🌾  │                                                              │   │
│  │   └─────────┘   └─────────┘   └─────────┘                                                              │   │
│  │                                                                                                         │   │
│  │ Legend: 🌾 Healthy  🐛 Pest Alert  💧 Irrigation  🚜 Equipment  📊 Sensor                               │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 🌾 AgriAI Dashboard | 📊 System Status: All Green | 🔄 Last Update: 2s ago | © 2024 | Privacy | Terms | Support │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📱 **Mobile Layout (375x812)**

### **Mobile Dashboard View**
```
┌─────────────────────────────────┐
│ 🌾 AgriAI    🔔(3) 👤 John ▼   │
├─────────────────────────────────┤
│ 🏠 🌤️ 📈 🐛 🌾 ⚙️              │
├─────────────────────────────────┤
│                                 │
│ ┌─────────────────────────────┐ │
│ │ 🌡️ Temperature: 25°C ↗️+2.1°C│ │
│ └─────────────────────────────┘ │
│                                 │
│ ┌─────────────────────────────┐ │
│ │ 💧 Humidity: 65% ↘️-3.2%    │ │
│ └─────────────────────────────┘ │
│                                 │
│ ┌─────────────────────────────┐ │
│ │ 🌧️ Precipitation: 2.3mm    │ │
│ │    ↗️+0.8mm                 │ │
│ └─────────────────────────────┘ │
│                                 │
│ ┌─────────────────────────────┐ │
│ │ 🚨 Pest Alerts (3)          │ │
│ │ 🔴 High: Aphids Field A-3   │ │
│ │ 🟡 Med: Fungus Field B-1    │ │
│ │ 🟢 Low: Soil Field C-2      │ │
│ └─────────────────────────────┘ │
│                                 │
│ ┌─────────────────────────────┐ │
│ │ 📈 Market Prices            │ │
│ │ 🌾 Wheat: $245.50 ↗️+5.3%   │ │
│ │ 🌽 Corn: $198.75 ↘️-1.6%    │ │
│ └─────────────────────────────┘ │
│                                 │
│ ┌─────────────────────────────┐ │
│ │ 🤖 AI Recommendations       │ │
│ │ 💡 Plant in 3 days          │ │
│ │ 🌱 Use NPK 15-15-15         │ │
│ │ 💧 Irrigate every 2 days    │ │
│ └─────────────────────────────┘ │
│                                 │
├─────────────────────────────────┤
│ 📊 All Green | 🔄 2s ago        │
└─────────────────────────────────┘
```

---

## 🎨 **Component Visual Specifications**

### **1. Header Components**

#### **Logo Design**
```
Desktop:  🌾 AgriAI
          Agricultural Intelligence

Mobile:   🌾 AgriAI
```

#### **Search Bar Design**
```
Desktop:  [🔍 Search farms, crops, weather...                    ] [×]
Mobile:   [🔍 Search...] [×]
```

#### **Notification Badge**
```
🔔(3)  →  Red circle with white "3"
🔔(12) →  Red circle with white "12"
🔔(99+)→  Red circle with white "99+"
```

### **2. Metric Cards**

#### **Temperature Card**
```
┌─────────────────────────┐
│ 🌡️ Temperature          │
│                         │
│        25°C             │
│                         │
│      ↗️ +2.1°C          │
│                         │
│    [Temperature Icon]   │
└─────────────────────────┘
```

#### **Humidity Card**
```
┌─────────────────────────┐
│ 💧 Humidity             │
│                         │
│        65%              │
│                         │
│      ↘️ -3.2%           │
│                         │
│    [Humidity Icon]      │
└─────────────────────────┘
```

### **3. Chart Components**

#### **Real-Time Weather Chart**
```
Temperature Over Time (24h)
   30°C ┤
         │     ╭─╮
   25°C ┤     ╱   ╲     ╭─╮
         │   ╱     ╲   ╱   ╲
   20°C ┤ ╱         ╲╱     ╲
         │
   15°C ┤
         └─────────────────
         06:00  12:00  18:00
```

#### **Market Price Chart**
```
Price Trends (6 months)
$300 ┤
     │     ╭─╮
$250 ┤   ╱     ╲   ╭─╮
     │ ╱         ╲╱   ╲
$200 ┤╱
     └─────────────────
     Jan  Apr  Jul  Oct
```

### **4. Alert Components**

#### **Pest Alert**
```
🚨 Pest Detection Alert
┌─────────────────────────┐
│ 🔴 HIGH RISK            │
│                         │
│ Aphids detected         │
│ Location: Field A-3     │
│ Confidence: 94%         │
│                         │
│ [View Details] [Dismiss]│
└─────────────────────────┘
```

#### **Weather Alert**
```
⚠️ Weather Warning
┌─────────────────────────┐
│ 🟡 MODERATE RISK        │
│                         │
│ Heavy rain expected     │
│ Time: 2-4 PM today      │
│ Confidence: 87%         │
│                         │
│ [View Details] [Dismiss]│
└─────────────────────────┘
```

---

## 🎯 **Color Specifications**

### **Primary Colors**
- **Green (Primary):** #22c55e (Success, Growth, Agriculture)
- **Blue (Secondary):** #3b82f6 (Information, Sky, Water)
- **Yellow (Warning):** #eab308 (Caution, Sun, Energy)
- **Red (Danger):** #ef4444 (Alerts, Pests, Critical)
- **Purple (AI):** #8b5cf6 (Intelligence, Technology)

### **Neutral Colors**
- **Gray 50:** #f9fafb (Background)
- **Gray 100:** #f3f4f6 (Card Background)
- **Gray 200:** #e5e7eb (Borders)
- **Gray 500:** #6b7280 (Secondary Text)
- **Gray 900:** #111827 (Primary Text)

### **Status Colors**
- **Success:** Green 500 (#22c55e)
- **Warning:** Yellow 500 (#eab308)
- **Error:** Red 500 (#ef4444)
- **Info:** Blue 500 (#3b82f6)

---

## 🎭 **Animation Specifications**

### **Hover Effects**
- **Cards:** Lift 2px, shadow increase
- **Buttons:** Scale 1.02x
- **Icons:** Scale 1.1x, color change

### **Loading States**
- **Spinner:** 360° rotation, 1s duration
- **Skeleton:** Pulse animation, 2s duration
- **Progress:** Smooth width transition

### **Page Transitions**
- **Fade In:** 0.3s ease-out
- **Slide Up:** 0.3s ease-out
- **Scale:** 0.2s ease-out

---

## 📐 **Spacing & Layout**

### **Grid System**
- **Desktop:** 4 columns (3fr each)
- **Tablet:** 2 columns (1fr each)
- **Mobile:** 1 column (1fr)

### **Spacing Scale**
- **xs:** 4px (0.25rem)
- **sm:** 8px (0.5rem)
- **md:** 16px (1rem)
- **lg:** 24px (1.5rem)
- **xl:** 32px (2rem)
- **2xl:** 48px (3rem)

### **Border Radius**
- **sm:** 4px (0.25rem)
- **md:** 8px (0.5rem)
- **lg:** 12px (0.75rem)
- **xl:** 16px (1rem)

---

## 🚀 **Ready to Build!**

This visual specification provides:
- ✅ **Exact pixel-perfect designs** for all components
- ✅ **Responsive layouts** for desktop, tablet, and mobile
- ✅ **Consistent color palette** with agricultural theme
- ✅ **Smooth animations** and micro-interactions
- ✅ **Accessibility considerations** with proper contrast
- ✅ **FAANG-level polish** that will impress any tech company

**Tomorrow we'll implement these designs and create the most beautiful agricultural AI dashboard! 🌾✨**
