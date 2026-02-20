import sys
import os
import json

# Add current directory to path
sys.path.append(os.getcwd())

from disaster_evacuation.benchmarks.benchmark_runner import BenchmarkRunner

def main():
    print("Running pathfinding benchmarks...")
    print("Algorithms: Dijkstra, A*, Bellman-Ford")
    print("Graph Sizes: 50, 100, 200, 400 nodes")
    print("-" * 50)
    
    runner = BenchmarkRunner()
    
    # Run smaller set for quick checking, but large enough to show trends
    results = runner.run_benchmarks(sizes=[50, 100, 200, 400], runs_per_size=3)
    
    print("\nBenchmark Results Summary:")
    print(runner.get_summary_table())
    
    # Save to JSON
    output_file = "benchmark_results.json"
    runner.save_results(output_file)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
