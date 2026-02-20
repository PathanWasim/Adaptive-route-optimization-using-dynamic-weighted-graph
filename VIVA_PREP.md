# 🎓 DAA Project Viva Prep & Cheat Sheet

## 🚀 30-Second Elevator Pitch
"This project is a **Disaster Evacuation Routing System** that models cities as dynamic weighted graphs. Unlike standard maps that just give the shortest path by distance, our system adapts to **real-time disaster scenarios** (fires, floods, earthquakes) by dynamically modifying edge weights based on risk and proximity. We implemented **Dijkstra, A*, and Bellman-Ford** algorithms from scratch to compare their efficiency in these scenarios, proving that **A* with a Haversine heuristic** offers the best balance of speed and optimality for this domain."

---

## 🔑 Key Features to Highlight
1.  **Dynamic Graph Modeling:** Weights = $Distance \times (1 + RiskFactor + Congestion)$.
2.  **Real-World Data:** Uses **OSMnx** to extract real street networks (not just random graphs).
3.  **Algorithmic Depth:** Implemented **Dijkstra**, **A***, and **Bellman-Ford**.
4.  **Academic Rigor:** Includes time complexity analysis ($O(E \log V)$) and correctness proofs.
5.  **Interactive Viz:** Step-by-step algorithm animation showing node relaxation.

---

## ❓ Probable Viva Questions & Answers

### Q1: Why did you choose Dijkstra's Algorithm?
**A:** "Dijkstra is the industry standard for finding shortest paths in graphs with non-negative weights. Since road distances and risk factors are always non-negative, Dijkstra guarantees the optimal path. We used a **Binary Heap (Min-Heap)** priority queue to optimize it to **$O(E \log V)$**, which is scalable for city-sized graphs."

### Q2: How is A* different from Dijkstra?
**A:** "A* is an informed search algorithm. It uses a **heuristic function** (we used **Haversine Distance**) to estimate the cost to the target. This guides the search *towards* the destination, exploring fewer nodes than Dijkstra (which explores radially). In our benchmarks, A* visited **30-60% fewer nodes** while still guaranteeing the optimal path because our heuristic is **admissible** (straight line $\le$ road distance)."

### Q3: Why implement Bellman-Ford if Dijkstra is faster?
**A:** "We implemented Bellman-Ford mainly for **comparative analysis**. It runs in **$O(V \cdot E)$** time, which is much slower (quadratic vs log-linear). However, it validates our decision to use Dijkstra by showing the performance gap. Also, unlike Dijkstra, Bellman-Ford can handle **negative edge weights**, though that doesn't apply to our road network model."

### Q4: What is the time complexity of your system?
**A:**
*   **Graph Construction:** $O(V + E)$
*   **Dijkstra/A* Pathfinding:** $O(E \log V)$
*   **Bellman-Ford:** $O(V \cdot E)$
*   **Space Complexity:** $O(V + E)$ (Adjacency List)

### Q5: How do you model the disaster?
**A:** "We model disasters as circular zones with an **epicenter** and **radius**.
*   **Edges inside the zone** get a weight penalty: `new_weight = weight * (1 + severity)`.
*   **Edges very close to the center** (high severity) are marked as `blocked` (infinite weight).
*   This forces the algorithm to route *around* the danger zone."

### Q6: How does the backend work?
**A:** "We use **Flask** (Python) for the backend. The graph is stored in memory as an adjacency list. The frontend (Leaflet.js) sends the source/target IDs. The backend computes the path using our custom engine and returns a JSON list of coordinates. We strictly **do not use Google Maps API** for routing; we only use Leaflet for rendering the tiles."

---

## 💻 Code Explanation Cheat Sheet

### Dynamic Weight Calculation (`weight_calculator.py`)
```python
def calculate_dynamic_weight(edge, disaster):
    base_weight = edge.distance
    risk_multi = 0
    if disaster and is_in_zone(edge, disaster):
        risk_multi = disaster.severity * (1 - dist_to_center / radius)
    return base_weight * (1 + risk_multi + edge.congestion)
```
*   **Logic:** The closer to the disaster, the higher the weight (cost), making the path less likely to be chosen.

### Priority Queue Pattern (`priority_queue = [(0.0, source)]`)
*   We use Python's `heapq` module.
*   It's a **Min-Heap**, so `heappop` always gives the smallest element in $O(\log N)$.
*   We push tuples `(cost, node_id)`. Python compares the first element (cost) by default.

### A* Heuristic (`astar_engine.py`)
```python
f_score = g_score + heuristic(neighbor, target)
```
*   `g_score`: Actual cost from start to current node.
*   `heuristic`: Estimated cost from current to target (Haversine/Crow-flies distance).

---

## 📊 Performance Stats (for Reference)
*   **Small Graph (50 nodes):** Dijkstra ~0.5ms, Bellman-Ford ~2ms
*   **Medium Graph (500 nodes):** Dijkstra ~5ms, Bellman-Ford ~150ms
*   **Large Graph (2000 nodes):** Dijkstra ~20ms, Bellman-Ford ~2000ms+
*   **A* Improvement:** Typically 1.5x - 2x faster than Dijkstra on average.
