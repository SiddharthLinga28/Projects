# 🚗 Spot Finder – Smart Parking System Using Wireless Sensor Network

### 📌 Developed By:
- **Siddharth Linga** – Dept. of Computer Science, Texas A&M University - Corpus Christi  
  📧 slinga1@islander.tamucc.edu  
---

## 📝 Abstract

**Spot Finder** is a Smart Parking Management System (SPMS) that uses **Wireless Sensor Networks (WSN)** and **IoT** to provide real-time updates on parking spot availability in urban environments. The system integrates sensors, cloud computing, and mobile applications to reduce traffic congestion, lower emissions, and enhance parking efficiency. A simulator built using **Python** and **Matplotlib** validates the system’s effectiveness and revenue generation.

---

## 🚀 Features

- 🚦 Real-time parking availability
- 📱 Mobile app with live updates & secure payment
- ⚙️ Differentiated pricing for VIP and regular users
- 🌐 IoT-based communication using **MQTT/CoAP**
- ☁️ Cloud-based backend (AWS IoT Core integration)
- 💡 Energy-efficient sensor nodes and WSN mesh
- 📊 Python-based simulator for statistical analysis

---

## 🎯 Project Objectives

- Develop a WSN-powered parking system with real-time space detection.
- Provide a tiered pricing model for VIP and regular users.
- Implement a secure, mobile-first interface for parking spot updates.
- Reduce parking search time and improve urban traffic flow.
- Validate the system through a scalable Python simulation.

---

## 🏗️ System Architecture

### Components:
- **Sensors:** Ultrasonic/IR sensors detect car presence.
- **WSN Nodes:** Form a mesh network for data relay.
- **IoT Protocols:** MQTT & CoAP for lightweight, real-time communication.
- **Cloud Server:** Stores data, applies pricing logic, and handles user interactions.
- **Mobile App:** Displays availability, cost estimates, and supports payments.
- **Python Simulator:** Mimics up to 100 parking slots for performance testing.

---

## 📱 Mobile App Features

- Realtime spot availability map (Green – available, Red – occupied, Blue – VIP).
- Notifications before session expiry.
- Transparent cost breakdown and secure payments.
- Integrated with cloud via IoT protocols for low-latency updates.

---

## 🧠 Simulator Overview

Developed in **Python** with **Matplotlib**, the simulator:
- Models 100 parking spaces
- Differentiates between VIP and regular customers
- Calculates cost, parking duration, and penalty if applicable

### Sample Pricing:
- **VIP:** $2/hour (no penalty)
- **Regular:**
  - 1st hour: $4  
  - 2nd hour: +$3  
  - 3rd hour onward: +$2/hr  
  - >5 hours: +$25 penalty

---

## 📈 Results Summary

- **Total Vehicles Processed:** 70
- **Successful Parking Rate:** 95%
- **Average Parking Time:** 2.5 hours
- **Revenue Generated:** ~$400 per run
- **VIP Revenue Share:** ~$80
- **System Insights:**
  - High scalability & fault tolerance
  - Energy-efficient architecture
  - Effective penalty enforcement

---

## 🆚 System Comparison

| Feature             | RFID System | Camera System | Proposed WSN System |
|---------------------|-------------|---------------|----------------------|
| Real-time Updates   | ❌          | ✅            | ✅                   |
| Scalability         | Medium      | High          | High                 |
| Cost Efficiency     | High        | Low           | High                 |
| Energy Efficiency   | Medium      | Low           | High                 |
| User Differentiation| ❌          | ❌            | ✅                   |

---

## 🛠️ Tech Stack

- **Frontend:** React Native (Mobile App)
- **Backend:** Flask / Django
- **Cloud:** AWS IoT Core
- **Protocols:** MQTT / CoAP
- **Hardware:** Arduino / Raspberry Pi, IR/Ultrasonic Sensors
- **Simulation:** Python, Matplotlib

---

## 📦 Folder Structure

You can find our idea in form of a simulation and caluclations of fair is also done in it saved as Simulation.ipynb in the folder.
