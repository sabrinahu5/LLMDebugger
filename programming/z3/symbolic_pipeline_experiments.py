import z3
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from dataclasses import dataclass
import traceback

@dataclass
class TestCase:
    inputs: List[Any]
    expected_output: Any

@dataclass
class ProblemSpec:
    description: str
    test_cases: List[TestCase]
    function_signature: str

# Test problems
test_problems = [
    ProblemSpec(
        description="Write a function that returns the sum of squares of two numbers",
        test_cases=[
            TestCase(inputs=[2, 3], expected_output=13),
            TestCase(inputs=[0, 0], expected_output=0),
            TestCase(inputs=[-1, 1], expected_output=2)
        ],
        function_signature="def sum_squares(a: int, b: int) -> int:"
    ),
    ProblemSpec(
        description="Write a function that returns the factorial of a number",
        test_cases=[
            TestCase(inputs=[5], expected_output=120),
            TestCase(inputs=[0], expected_output=1),
            TestCase(inputs=[3], expected_output=6)
        ],
        function_signature="def factorial(n: int) -> int:"
    )
]

incorrect_solutions = [
    """
def sum_squares(a: int, b: int) -> int:
    return a + b  # Incorrect implementation
""",
    """
def factorial(n: int) -> int:
    if n == 1:  # Wrong base case
        return 1
    return n * factorial(n - 1)
"""
]

class SymbolicExecutor:
    def __init__(self):
        self.solver = z3.Solver()
        self.symbolic_vars = {}
        self.path_conditions = []
        
    def create_symbolic_var(self, name: str, var_type: type) -> z3.ExprRef:
        """Create symbolic variables based on type"""
        if var_type == int:
            return z3.Int(name)
        elif var_type == float:
            return z3.Real(name)
        elif var_type == bool:
            return z3.Bool(name)
        else:
            raise ValueError(f"Unsupported type: {var_type}")

    def analyze_ast(self, node, symbolic_locals: dict) -> Tuple[z3.ExprRef, list]:
        """Recursively analyze AST nodes and build constraints"""
        import ast
        
        if isinstance(node, ast.Num):
            return z3.IntVal(node.n), []
        
        elif isinstance(node, ast.Name):
            if node.id in symbolic_locals:
                return symbolic_locals[node.id], []
            raise ValueError(f"Unknown variable: {node.id}")
            
        elif isinstance(node, ast.BinOp):
            left, left_constraints = self.analyze_ast(node.left, symbolic_locals)
            right, right_constraints = self.analyze_ast(node.right, symbolic_locals)
            
            if isinstance(node.op, ast.Add):
                return left + right, left_constraints + right_constraints
            elif isinstance(node.op, ast.Sub):
                return left - right, left_constraints + right_constraints
            elif isinstance(node.op, ast.Mult):
                return left * right, left_constraints + right_constraints
            elif isinstance(node.op, ast.Div):
                return left / right, left_constraints + right_constraints + [right != 0]
            
        elif isinstance(node, ast.Compare):
            left, left_constraints = self.analyze_ast(node.left, symbolic_locals)
            constraints = left_constraints
            
            for op, comparator in zip(node.ops, node.comparators):
                right, right_constraints = self.analyze_ast(comparator, symbolic_locals)
                constraints.extend(right_constraints)
                
                if isinstance(op, ast.Lt):
                    constraints.append(left < right)
                elif isinstance(op, ast.LtE):
                    constraints.append(left <= right)
                elif isinstance(op, ast.Gt):
                    constraints.append(left > right)
                elif isinstance(op, ast.GtE):
                    constraints.append(left >= right)
                elif isinstance(op, ast.Eq):
                    constraints.append(left == right)
                elif isinstance(op, ast.NotEq):
                    constraints.append(left != right)
                    
            return z3.Bool(f"cond_{len(self.path_conditions)}"), constraints
            
        elif isinstance(node, ast.If):
            test_expr, test_constraints = self.analyze_ast(node.test, symbolic_locals)
            
            # Analyze both branches
            then_constraints = []
            else_constraints = []
            
            # Then branch
            self.path_conditions.append(test_expr)
            for stmt in node.body:
                _, new_constraints = self.analyze_ast(stmt, symbolic_locals)
                then_constraints.extend(new_constraints)
            self.path_conditions.pop()
            
            # Else branch
            if node.orelse:
                self.path_conditions.append(z3.Not(test_expr))
                for stmt in node.orelse:
                    _, new_constraints = self.analyze_ast(stmt, symbolic_locals)
                    else_constraints.extend(new_constraints)
                self.path_conditions.pop()
                
            return None, test_constraints + then_constraints + else_constraints
            
        return None, []

    def get_symbolic_trace(self, code: str, function_name: str) -> str:
        """
        Generate symbolic execution trace for the given code.
        Returns constraints and path conditions as a string.
        """
        import ast
        import inspect
        
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Find the function definition
            function_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    function_def = node
                    break
                    
            if not function_def:
                return "Error: Function not found in code"

            # Create symbolic variables for parameters
            symbolic_locals = {}
            for arg in function_def.args.args:
                # Get type hint if available
                arg_type = int  # default to int if no type hint
                if arg.annotation and isinstance(arg.annotation, ast.Name):
                    type_str = arg.annotation.id
                    arg_type = {'int': int, 'float': float, 'bool': bool}.get(type_str, int)
                
                symbolic_vars = self.create_symbolic_var(arg.arg, arg_type)
                symbolic_locals[arg.arg] = symbolic_vars
                self.symbolic_vars[arg.arg] = symbolic_vars

            # Analyze the function body
            constraints = []
            for node in function_def.body:
                _, node_constraints = self.analyze_ast(node, symbolic_locals)
                constraints.extend(node_constraints)

            # Generate trace output
            trace = "Symbolic Analysis:\n"
            trace += "Variables:\n"
            for name, var in self.symbolic_vars.items():
                trace += f"  {name}: {var}\n"
            
            trace += "\nPath Constraints:\n"
            for i, constraint in enumerate(constraints, 1):
                trace += f"  {i}. {constraint}\n"
            
            # Check satisfiability of constraints
            self.solver.push()
            for constraint in constraints:
                self.solver.add(constraint)
            
            trace += "\nConstraint Satisfiability:\n"
            if self.solver.check() == z3.sat:
                model = self.solver.model()
                trace += "  Constraints are satisfiable\n"
                trace += "  Example values:\n"
                for var_name, var in self.symbolic_vars.items():
                    if var in model:
                        trace += f"    {var_name} = {model[var]}\n"
            else:
                trace += "  Constraints are unsatisfiable\n"
            
            self.solver.pop()
            return trace
            
        except Exception as e:
            return f"Symbolic execution failed: {str(e)}\n{traceback.format_exc()}"

class ConcreteExecutor:
    def get_execution_trace(self, code: str, test_case: TestCase) -> str:
        """
        Enhanced concrete execution trace that includes variable states
        """
        trace = []
        variable_states = {}
        
        def trace_function(frame, event, arg):
            if event == 'line':
                # Get local variables
                locals_dict = frame.f_locals.copy()
                vars_str = ", ".join(f"{k}={v}" for k, v in locals_dict.items()
                                   if not k.startswith('__'))
                
                trace.append(f"Line {frame.f_lineno}: {frame.f_code.co_name}")
                if vars_str:
                    trace.append(f"Variables: {vars_str}")
                    variable_states[frame.f_lineno] = locals_dict
            return trace_function
            
        try:
            locals_dict = {}
            exec(code, globals(), locals_dict)
            func = list(locals_dict.values())[0]  # Get the defined function
            
            import sys
            sys.settrace(trace_function)
            result = func(*test_case.inputs)
            sys.settrace(None)
            
            trace.append(f"\nFunction returned: {result}")
            trace.append(f"Expected output: {test_case.expected_output}")
            
            return "\n".join(trace)
        except Exception as e:
            return f"Execution failed: {str(e)}\n{traceback.format_exc()}"

class LLMDebugger:
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.symbolic_executor = SymbolicExecutor()
        self.concrete_executor = ConcreteExecutor()
        self.client = OpenAI(base_url=base_url, api_key="dummy")
        
    def debug_with_symbolic(self, problem: ProblemSpec, incorrect_code: str) -> str:
        """
        Debug code using enhanced symbolic execution feedback
        """
        symbolic_trace = self.symbolic_executor.get_symbolic_trace(
            incorrect_code, 
            problem.function_signature.split()[1].split('(')[0]
        )
        
        # Enhanced prompt that better utilizes symbolic analysis
        prompt = f"""
        I need help fixing this incorrect code implementation:
        ```python
        {incorrect_code}
        ```
        
        Problem Description:
        {problem.description}
        
        Test Cases:
        {self._format_test_cases(problem.test_cases)}
        
        The symbolic execution analysis revealed:
        {symbolic_trace}
        
        Based on the symbolic analysis above, please:
        1. Identify any path constraints that might be violated
        2. Check if the constraints align with the test cases
        3. Provide a corrected implementation that satisfies both the symbolic constraints and test cases
        4. Explain the key changes made
        
        Please provide only the corrected code implementation without explanation.
        """
        
        response = self.client.completions.create(
            model="bigcode/starcoder",
            prompt=prompt,
            max_tokens=1000
        )
        return response.choices[0].text

    def debug_with_trace(self, problem: ProblemSpec, incorrect_code: str, test_case: TestCase) -> str:
        """
        Debug code using enhanced concrete execution trace feedback combined with symbolic insights
        """
        execution_trace = self.concrete_executor.get_execution_trace(incorrect_code, test_case)
        symbolic_trace = self.symbolic_executor.get_symbolic_trace(
            incorrect_code,
            problem.function_signature.split()[1].split('(')[0]
        )
        
        prompt = f"""
        I need help fixing this incorrect code implementation:
        ```python
        {incorrect_code}
        ```
        
        Problem Description:
        {problem.description}
        
        Concrete Execution Trace (failing test case {test_case.inputs} → expected {test_case.expected_output}):
        {execution_trace}
        
        Symbolic Analysis:
        {symbolic_trace}
        
        Using both the concrete execution trace and symbolic analysis:
        1. Identify where the execution path diverges from expected behavior
        2. Check if any symbolic constraints are violated
        3. Provide a corrected implementation
        
        Please provide only the corrected code implementation without explanation.
        """
        
        response = self.client.completions.create(
            model="bigcode/starcoder",
            prompt=prompt,
            max_tokens=1000
        )
        return response.choices[0].text

    def _format_test_cases(self, test_cases: List[TestCase]) -> str:
        """Helper method to format test cases for prompts"""
        return "\n".join([
            f"Input: {tc.inputs} → Expected: {tc.expected_output}"
            for tc in test_cases
        ])

def evaluate_debugging_approaches(problems: List[ProblemSpec], incorrect_solutions: List[str], base_url: str = "http://localhost:8000/v1") -> Dict:
    """
    Enhanced evaluation that provides more detailed results
    """
    debugger = LLMDebugger(base_url)
    results = {
        "symbolic": {"successes": 0, "failures": 0, "details": []},
        "trace": {"successes": 0, "failures": 0, "details": []}
    }
    
    for i, (problem, incorrect_code) in enumerate(zip(problems, incorrect_solutions)):
        print(f"\nEvaluating Problem {i+1}:")
        print(f"Description: {problem.description}")
        print("Original incorrect code:")
        print(incorrect_code)
        
        # Try symbolic debugging
        print("\nAttempting symbolic debugging...")
        fixed_symbolic = debugger.debug_with_symbolic(problem, incorrect_code)
        symbolic_success = test_solution(fixed_symbolic, problem.test_cases)
        results["symbolic"]["successes" if symbolic_success else "failures"] += 1
        results["symbolic"]["details"].append({
            "problem_index": i,
            "success": symbolic_success,
            "original_code": incorrect_code,
            "fixed_code": fixed_symbolic
        })
        print(f"Symbolic debugging {'succeeded' if symbolic_success else 'failed'}")
        print("Fixed code (symbolic):")
        print(fixed_symbolic)
        
        # Try trace-based debugging
        print("\nAttempting trace-based debugging...")
        fixed_trace = debugger.debug_with_trace(problem, incorrect_code, problem.test_cases[0])
        trace_success = test_solution(fixed_trace, problem.test_cases)
        results["trace"]["successes" if trace_success else "failures"] += 1
        results["trace"]["details"].append({
            "problem_index": i,
            "success": trace_success,
            "original_code": incorrect_code,
            "fixed_code": fixed_trace
        })
        print(f"Trace-based debugging {'succeeded' if trace_success else 'failed'}")
        print("Fixed code (trace-based):")
        print(fixed_trace)
    
    print("\nFinal Results:")
    print(f"Symbolic debugging: {results['symbolic']['successes']} successes, {results['symbolic']['failures']} failures")
    print(f"Trace-based debugging: {results['trace']['successes']} successes, {results['trace']['failures']} failures")
    
    return results

def test_solution(code: str, test_cases: List[TestCase]) -> bool:
    """
    Enhanced test solution with better error handling and reporting
    """
    try:
        locals_dict = {}
        exec(code, globals(), locals_dict)
        func = list(locals_dict.values())[0]
        
        for test_case in test_cases:
            result = func(*test_case.inputs)
            if result != test_case.expected_output:
                print(f"Test case failed: inputs={test_case.inputs}, "
                      f"expected={test_case.expected_output}, got={result}")
                return False
        return True
    except Exception as e:
        print(f"Code execution failed: {str(e)}")
        return False

# Run evaluation on test problems
if __name__ == "__main__":
    base_url = "http://localhost:8000/v1"  # Using local vllm OpenAI-compatible server
    print("Starting evaluation of debugging approaches...")
    results = evaluate_debugging_approaches(test_problems, incorrect_solutions, base_url)