from z3 import *
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import operator
import ast

@dataclass
class ProgramState:
    """Represents the state of program variables at a point in execution"""
    variables: Dict[str, int]

@dataclass
class HoleProgram:
    """Represents a program with a hole"""
    variables: List[str]
    pre_condition: Optional[str] = None   # e.g., "x >= 0"
    post_condition: Optional[str] = None  # e.g., "x * x + 2 * x + 1"

class ExpressionEvaluator(ast.NodeVisitor):
    """Converts Python expressions to Z3 expressions"""
    def __init__(self, variables):
        self.variables = variables
        self.ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.USub: operator.neg,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Eq: operator.eq,
        }

    def visit_Name(self, node):
        if node.id == 'If':
            return If  # Return Z3's If function directly
        if node.id in self.variables:
            return self.variables[node.id]
        raise ValueError(f"Unknown variable: {node.id}")

    def visit_Num(self, node):
        return node.n

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = type(node.op)
        if op in self.ops:
            return self.ops[op](left, right)
        raise ValueError(f"Unsupported binary operator: {op}")

    def visit_Compare(self, node):
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        op = type(node.ops[0])
        if op in self.ops:
            return self.ops[op](left, right)
        raise ValueError(f"Unsupported comparison operator: {op}")

    def visit_NameConstant(self, node):  # For Python < 3.8
        if isinstance(node.value, bool):
            return BoolVal(node.value)
        return node.value

    def visit_Constant(self, node):  # For Python >= 3.8
        if isinstance(node.value, bool):
            return BoolVal(node.value)
        return node.value

    def visit_Call(self, node):
        if node.func.id == 'If':
            # Handle Z3's If function
            if len(node.args) != 3:
                raise ValueError("If requires exactly 3 arguments")
            condition = self.visit(node.args[0])
            then_expr = self.visit(node.args[1])
            else_expr = self.visit(node.args[2])
            return If(condition, then_expr, else_expr)
        raise ValueError(f"Unknown function call: {node.func.id}")

def python_expr_to_z3(expr_str: str, variables: Dict[str, Union[int, ArithRef]]) -> ExprRef:
    """Converts a Python expression string to a Z3 expression"""
    tree = ast.parse(expr_str, mode='eval')
    evaluator = ExpressionEvaluator(variables)
    return evaluator.visit(tree.body)

def synthesize_hole(program: HoleProgram) -> Optional[str]:
    """
    Attempts to synthesize code to fill a hole in a program using universal quantifiers.
    """
    s = Solver()
    
    print("\nDebug - Creating constraints:")
    
    # Create symbolic variables
    symbolic_vars = {}
    for var_name in program.variables:
        symbolic_vars[var_name] = Int(var_name)
        print(f"Created symbolic variable: {var_name}")
    
    # Create hole as a function
    hole = Function('hole', *([IntSort()] * len(program.variables)), IntSort())
    print("Created hole function")
    
    # Create the universal quantifier constraints
    vars_list = [symbolic_vars[v] for v in program.variables]
    
    # Convert pre and post conditions to Z3 expressions
    pre_cond = (python_expr_to_z3(program.pre_condition, symbolic_vars) 
                if program.pre_condition else BoolVal(True))
    post_cond = python_expr_to_z3(program.post_condition, symbolic_vars)
    
    # Create the main constraint
    constraint = ForAll(vars_list,
        Implies(
            pre_cond,
            hole(*vars_list) == post_cond
        )
    )
    s.add(constraint)
    
    print(f"Added constraint: ∀{program.variables}. "
          f"{program.pre_condition if program.pre_condition else 'True'} → "
          f"hole({', '.join(program.variables)}) == {program.post_condition}")
    
    print("\nDebug - Checking for solution:")
    if s.check() == sat:
        print("Found satisfiable model")
        m = s.model()
        result = interpret_model(m, hole, program.variables)
        return result
    else:
        print("No solution found")
        return None

def interpret_model(model, hole_func, variables):
    """Attempts to interpret Z3's solution into readable code"""
    print("\nDebug - Model interpretation:")
    try:
        # Test with sample inputs
        sample_inputs = [(i,) for i in range(1, 6)]
        if len(variables) > 1:
            # For multiple variables, test combinations
            sample_inputs = [(i, j) for i in range(1, 4) for j in range(1, 4)]
        
        print("Testing hole function with sample inputs:")
        for inputs in sample_inputs:
            args = [Int(str(x)) for x in inputs]
            result = model.evaluate(hole_func(*args))
            input_str = ', '.join(str(x) for x in inputs)
            print(f"hole({input_str}) = {result}")
        
        # Get the general expression
        symbolic_inputs = [Int(v) for v in variables]
        expression = model.evaluate(hole_func(*symbolic_inputs))
        return f"return {expression}"
        
    except Exception as e:
        print(f"Debug - Error during interpretation: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Could not interpret model"

def example_usage():
    # Example 1: Quadratic function
    program1 = HoleProgram(
        variables=['x'],
        pre_condition="True",
        post_condition="x * x + 2 * x + 1"  # x² + 2x + 1
    )
    
    # Example 2: Maximum of two numbers
    program2 = HoleProgram(
        variables=['x', 'y'],
        pre_condition="True",
        post_condition="If(x > y, x, y)"
    )
    
    print("Testing quadratic function synthesis:")
    result1 = synthesize_hole(program1)
    if result1:
        print(f"\nSynthesized quadratic function: {result1}")
    
    print("\nTesting max function synthesis:")
    result2 = synthesize_hole(program2)
    if result2:
        print(f"\nSynthesized max function: {result2}")

if __name__ == "__main__":
    example_usage()
