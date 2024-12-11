from z3 import *
import time

class SimpleSymbolicExecutor:
    def __init__(self):
        self.solver = Solver()
        
    def analyze_simple_path(self):
        """Simple path condition example"""
        print("\n=== Simple Path Analysis ===")
        
        # Create symbolic variables
        x = Int('x')
        y = Int('y')
        
        # Add constraints
        self.solver.add(x > 0)           # x must be positive
        self.solver.add(y == x + 5)      # y is x + 5
        self.solver.add(y < 10)          # y must be less than 10
        
        print("Constraints:")
        print("  x > 0")
        print("  y == x + 5")
        print("  y < 10")
        
        # Check if constraints are satisfiable
        if self.solver.check() == sat:
            model = self.solver.model()
            print("\nSolution found:")
            print(f"  x = {model[x]}")
            print(f"  y = {model[y]}")
        else:
            print("\nNo solution exists")
            
        self.solver.reset()

    def analyze_max_function(self):
        """Analyze a max function symbolically"""
        print("\n=== Max Function Analysis ===")
        
        # Create symbolic variables
        a = Int('a')
        b = Int('b')
        result = Int('result')
        
        # Encode max function logic
        path1 = And(a >= b, result == a)  # if a >= b: return a
        path2 = And(a < b, result == b)   # if a < b: return b
        
        # Try different paths
        print("Checking path where a >= b:")
        self.solver.add(path1)
        self.solver.add(result > 10)  # Additional constraint: result must be > 10
        
        if self.solver.check() == sat:
            model = self.solver.model()
            print(f"  Found solution: a = {model[a]}, b = {model[b]}, max = {model[result]}")
        else:
            print("  No solution for this path")
            
        self.solver.reset()
        
        print("\nChecking path where a < b:")
        self.solver.add(path2)
        self.solver.add(result > 10)  # Same additional constraint
        
        if self.solver.check() == sat:
            model = self.solver.model()
            print(f"  Found solution: a = {model[a]}, b = {model[b]}, max = {model[result]}")
        else:
            print("  No solution for this path")
            
        self.solver.reset()

    def analyze_array_access(self):
        """Analyze array bounds and access"""
        print("\n=== Array Access Analysis ===")
        
        # Create array and index
        arr = Array('arr', IntSort(), IntSort())
        idx = Int('idx')
        length = Int('length')
        
        # Add constraints
        self.solver.add(length == 5)                  # Array length is 5
        self.solver.add(idx >= 0)                     # Index must be non-negative
        self.solver.add(idx < length)                 # Index must be within bounds
        self.solver.add(Select(arr, idx) > 10)        # Value at index must be > 10
        
        print("Constraints:")
        print("  length == 5")
        print("  0 <= idx < length")
        print("  arr[idx] > 10")
        
        if self.solver.check() == sat:
            model = self.solver.model()
            idx_val = model[idx].as_long()  # Get the concrete integer value
            print("\nPossible solution:")
            print(f"  idx = {idx_val}")
            print(f"  arr[{idx_val}] = {model[arr][idx_val]}")  # Use the evaluated model of arr
        else:
            print("\nNo valid array access possible")
            
        self.solver.reset()

    def find_bug_example(self):
        """Find a bug in a simple function"""
        print("\n=== Bug Finding Example ===")
        
        # Symbolic variables for function: def abs_plus_one(x): return x + 1 if x >= 0 else -x + 1
        x = Int('x')
        result = Int('result')
        
        # Function implementation
        path1 = And(x >= 0, result == x + 1)      # Positive path
        path2 = And(x < 0, result == -x + 1)      # Negative path
        
        # Property we want to verify: result should always be positive
        property_violated = result <= 0
        
        print("Checking if abs_plus_one(x) can ever return non-positive number")
        print("Function: return x + 1 if x >= 0 else -x + 1")
        
        # Check positive path
        self.solver.add(path1)
        self.solver.add(property_violated)
        
        if self.solver.check() == sat:
            model = self.solver.model()
            print(f"\nBug found in positive path:")
            print(f"  x = {model[x]}")
            print(f"  result = {model[result]}")
        else:
            print("\nNo bugs in positive path")
            
        self.solver.reset()
        
        # Check negative path
        self.solver.add(path2)
        self.solver.add(property_violated)
        
        if self.solver.check() == sat:
            model = self.solver.model()
            print(f"\nBug found in negative path:")
            print(f"  x = {model[x]}")
            print(f"  result = {model[result]}")
        else:
            print("\nNo bugs in negative path")
            
        self.solver.reset()

def main():
    executor = SimpleSymbolicExecutor()
    
    # Run all analyses
    executor.analyze_simple_path()
    executor.analyze_max_function()
    executor.analyze_array_access()
    executor.find_bug_example()

if __name__ == "__main__":
    main()