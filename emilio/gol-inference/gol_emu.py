#!/usr/bin/env python3
"""
ISA Emulator for the scalable GOL computer.
Verifies program correctness before running on actual GOL hardware.

Instruction set matches scalable-gol-computer/assembly.py
"""

import re
import math


class GOLComputerEmulator:
    """Software emulator of the GOL computer's instruction set."""
    
    def __init__(self, bits=8, num_vars=8, max_lines=16, debug=False):
        self.bits = bits
        self.mask = (1 << bits) - 1  # 0xFF for 8-bit
        self.num_vars = num_vars
        self.max_lines = max_lines
        self.debug = debug
        self.reset()
    
    def reset(self):
        # a0 = program counter, a1..aN = general purpose
        self.vars = [0] * self.num_vars
        self.pc = 0  # also stored as vars[0]
        self.output = []
        self.steps = 0
        self.halted = False
    
    def _val(self, v):
        """Mask to bits width, handle unsigned."""
        return v & self.mask
    
    def _signed(self, v):
        """Interpret as signed value."""
        v = v & self.mask
        if v >= (1 << (self.bits - 1)):
            return v - (1 << self.bits)
        return v
    
    def _get(self, name):
        """Get variable value by name (e.g. 'a3')."""
        idx = int(name[1:])
        return self.vars[idx]
    
    def _set(self, name, value):
        """Set variable value by name."""
        idx = int(name[1:])
        self.vars[idx] = self._val(value)
    
    def execute_line(self, instruction):
        """Execute one preprocessed instruction line."""
        parts = instruction.strip().split()
        op = parts[0]
        
        if op == 'write':
            # write dest value
            dest, val = parts[1], int(parts[2])
            self._set(dest, val)
            if dest == 'a0':
                self.pc = self._get('a0')
                return  # goto behavior
                
        elif op == 'goto':
            target = int(parts[1])
            self.pc = target
            return
            
        elif op == 'move':
            # move dest src -> dest = src
            dest, src = parts[1], parts[2]
            self._set(dest, self._get(src))
            
        elif op == 'jump':
            # jump src -> skip src lines (hardware: a0 = a0 + src, then auto-increment)
            src = parts[1]
            self.pc = self.pc + self._get(src) + 1
            return
            
        elif op == 'print':
            src = parts[1]
            val = self._get(src)
            self.output.append(val)
            if self.debug:
                print(f"  OUTPUT: {val} (signed: {self._signed(val)})")
            
        elif op == '+':
            # + dest src1 src2 -> dest = src1 + src2
            dest, src1, src2 = parts[1], parts[2], parts[3]
            self._set(dest, self._get(src1) + self._get(src2))
            
        elif op == '-':
            # - dest src1 src2 -> dest = src1 - src2
            dest, src1, src2 = parts[1], parts[2], parts[3]
            self._set(dest, self._get(src1) - self._get(src2))
            
        elif op == '++':
            # ++ dest src -> dest = src + 1
            dest, src = parts[1], parts[2]
            self._set(dest, self._get(src) + 1)
            
        elif op == '*-':
            # *- dest src -> dest = -src
            dest, src = parts[1], parts[2]
            self._set(dest, -self._get(src))
            
        elif op == 'or':
            dest, src1, src2 = parts[1], parts[2], parts[3]
            self._set(dest, self._get(src1) | self._get(src2))
            
        elif op == 'and':
            dest, src1, src2 = parts[1], parts[2], parts[3]
            self._set(dest, self._get(src1) & self._get(src2))
            
        elif op == 'xor':
            dest, src1, src2 = parts[1], parts[2], parts[3]
            self._set(dest, self._get(src1) ^ self._get(src2))
            
        elif op == 'not':
            dest, src = parts[1], parts[2]
            self._set(dest, ~self._get(src))
            
        elif op == '>>':
            dest, src = parts[1], parts[2]
            self._set(dest, self._get(src) >> 1)
            
        elif op == '<<':
            dest, src = parts[1], parts[2]
            self._set(dest, self._get(src) << 1)
            
        elif op == 'rr':
            dest, src = parts[1], parts[2]
            v = self._get(src)
            self._set(dest, ((v >> 1) | ((v & 1) << (self.bits - 1))))
            
        elif op == 'rl':
            dest, src = parts[1], parts[2]
            v = self._get(src)
            msb = (v >> (self.bits - 1)) & 1
            self._set(dest, ((v << 1) | msb))
            
        elif op == '=0':
            dest, src = parts[1], parts[2]
            self._set(dest, 1 if self._get(src) == 0 else 0)
            
        elif op == '!=0':
            dest, src = parts[1], parts[2]
            self._set(dest, 1 if self._get(src) != 0 else 0)
            
        elif op == 'less':
            dest, src = parts[1], parts[2]
            self._set(dest, self._get(src) & 1)  # LSB
            
        elif op == 'most':
            dest, src = parts[1], parts[2]
            self._set(dest, (self._get(src) >> (self.bits - 1)) & 1)  # MSB
            
        elif op == 'rfb':
            # rfb dest index -> dest = vars[index]
            dest, idx_var = parts[1], parts[2]
            idx = self._get(idx_var)
            if idx < self.num_vars:
                self._set(dest, self.vars[idx])
                
        elif op == 'wfb':
            # wfb src index -> vars[index] = src
            src, idx_var = parts[1], parts[2]
            idx = self._get(idx_var)
            if idx < self.num_vars:
                self.vars[idx] = self._val(self._get(src))
                
        elif op == 'disp':
            pass  # ignore display
            
        elif op == 'erase':
            pass  # ignore display
            
        else:
            raise ValueError(f"Unknown instruction: {op}")
        
        # Normal PC advance
        self.pc += 1
    
    def run(self, program_lines, max_steps=1000):
        """Run program until halt or max_steps reached."""
        self.reset()
        
        while self.steps < max_steps and not self.halted:
            if self.pc < 0 or self.pc >= len(program_lines):
                self.halted = True
                break
            
            line = program_lines[self.pc]
            if self.debug:
                state = " ".join(f"a{i}={self.vars[i]}" for i in range(min(8, self.num_vars)))
                print(f"[step {self.steps:3d}] pc={self.pc:2d}: {line:30s} | {state}")
            
            self.execute_line(line)
            self.vars[0] = self._val(self.pc)  # sync PC register
            self.steps += 1
        
        return self.output


def parse_program(asm_text):
    """Parse assembly text into list of instruction lines."""
    lines = []
    for line in asm_text.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            lines.append(line)
    return lines


# ═══════════════════════════════════════════════════════════════════
# Test programs
# ═══════════════════════════════════════════════════════════════════

# Multiply via repeated addition: a1 * a2 → a3
MULTIPLY_SCALABLE = """
write a1 3
write a2 5
write a3 0
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
goto 7
"""
# a4 defaults to 0, so "- a2 a2 a4" doesn't work. Need to subtract 1.
# Let me fix: use "write a4 1" then subtract.

MULTIPLY_V2 = """
write a1 3
write a2 5
write a3 0
write a4 1
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
goto 10
goto 4
print a3
goto 11
"""

# 2x2 Matrix multiply:
# A = [[a,b],[c,d]], B = [[e,f],[g,h]]
# C = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
# For simplicity: A=[[1,2],[3,4]], B=[[5,6],[7,8]]
# C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
#   = [[5+14, 6+16], [15+28, 18+32]]
#   = [[19, 22], [43, 50]]

# Single multiply subroutine using the scalable computer:
# multiply(a1, a2) -> a3
# Uses a4 as constant 1, a5 as counter, a6 as temp
MULTIPLY_SUB = """
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
"""

# Full matmul program for 2x2:
# Store matrix elements, compute each output element
MATMUL_2x2 = """
write a1 1
write a2 2
write a3 0
write a4 1
write a5 5
write a6 0
+ a6 a6 a1
- a5 a5 a4
!=0 a7 a5
jump a7
goto 10
write a5 7
+ a6 a6 a2
- a5 a5 a4
!=0 a7 a5
jump a7
goto 15
print a6
goto 16
"""


def test_fibonacci():
    """Test Fibonacci program (original Loizeau computer)."""
    print("=== Fibonacci (original computer ISA) ===")
    # Original computer uses a,b,c,...,h variables
    # Map to a0=h(PC), a1=g, a2=f, a3=e, a4=d, a5=c, a6=b, a7=a
    # Actually, the original uses different naming. Let me use the original ISA.
    # write a 1 → write a7 1 (a=111 → variable 7)
    # write b 1 → write a6 1 (b=110 → variable 6)
    # add a b a → + a7 a7 a6
    # print a → print a7
    # add a b b → + a6 a7 a6
    # print b → print a6
    # goto 2
    
    program = parse_program("""
write a7 1
write a6 1
+ a7 a7 a6
print a7
+ a6 a7 a6
print a6
goto 2
    """)
    
    emu = GOLComputerEmulator(bits=8, num_vars=8, debug=False)
    output = emu.run(program, max_steps=100)
    
    print(f"Output: {output}")
    # Expected Fibonacci: 2, 3, 5, 8, 13, 21, 34, 55, 89, 144(→overflow)
    expected = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144 & 0xFF]
    match = output[:len(expected)] == expected
    print(f"Expected: {expected}")
    print(f"Match: {match}")
    print()


def test_multiply():
    """Test multiply via repeated addition."""
    print("=== Multiply 3 * 5 ===")
    program = parse_program(MULTIPLY_V2)
    
    emu = GOLComputerEmulator(bits=8, num_vars=8, debug=True)
    output = emu.run(program, max_steps=100)
    
    print(f"\nOutput: {output}")
    print(f"Expected: [15]")
    print(f"Correct: {output == [15]}")
    print()


def test_matmul_element():
    """Test computing one element of a 2x2 matmul."""
    print("=== Matmul element: 1*5 + 2*7 = 19 ===")
    
    # Compute: a*e + b*g where a=1, b=2, e=5, g=7
    # Strategy: 
    #   1. Compute a*e via repeated addition → result1
    #   2. Compute b*g via repeated addition → result2
    #   3. Add result1 + result2
    
    # Using variables:
    # a1 = multiplicand (first a, then b)
    # a2 = multiplier counter (first e, then g)
    # a3 = accumulator (running total for the output element)
    # a4 = constant 1
    # a5 = temp for !=0
    
    program = parse_program("""
write a3 0
write a4 1
write a1 1
write a2 5
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
goto 10
goto 4
write a1 2
write a2 7
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
goto 18
goto 12
print a3
goto 19
    """)
    
    emu = GOLComputerEmulator(bits=8, num_vars=8, debug=True)
    output = emu.run(program, max_steps=200)
    
    print(f"\nOutput: {output}")
    print(f"Expected: [19]  (1*5 + 2*7)")
    print(f"Correct: {output == [19]}")
    print()


def test_full_matmul():
    """Test full 2x2 matrix multiplication."""
    print("=== Full 2x2 Matmul ===")
    print("A = [[1,2],[3,4]], B = [[5,6],[7,8]]")
    print("C = [[19,22],[43,50]]")
    print()
    
    # We need to compute 4 output elements.
    # Each element = sum of 2 products.
    # For 8-bit with only 8 variables, we need to reuse variables.
    #
    # Strategy: compute each element one at a time, print it.
    # Variables: a1=multiplicand, a2=counter, a3=accumulator, a4=1, a5=temp, a6/a7=spare
    #
    # For C[0][0] = 1*5 + 2*7 = 19
    # For C[0][1] = 1*6 + 2*8 = 22
    # For C[1][0] = 3*5 + 4*7 = 43
    # For C[1][1] = 3*6 + 4*8 = 50
    
    # Using "goto" to jump between multiply subroutines
    # Each multiply block: set a1, set a2, loop: add/dec/check/jump/exit/back
    # Block pattern (6 lines per multiply after setup):
    #   + a3 a3 a1     (add)
    #   - a2 a2 a4     (dec)
    #   !=0 a5 a2      (check)
    #   jump a5         (skip exit if nonzero)
    #   goto EXIT       (done with this multiply)
    #   goto LOOP_START (back to add)
    
    program = parse_program("""
write a3 0
write a4 1
write a1 1
write a2 5
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
goto 10
goto 4
write a1 2
write a2 7
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
goto 18
goto 12
print a3
write a3 0
write a1 1
write a2 6
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
goto 28
goto 22
write a1 2
write a2 8
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
goto 36
goto 30
print a3
write a3 0
write a1 3
write a2 5
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
goto 46
goto 40
write a1 4
write a2 7
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
goto 54
goto 48
print a3
write a3 0
write a1 3
write a2 6
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
goto 64
goto 58
write a1 4
write a2 8
+ a3 a3 a1
- a2 a2 a4
!=0 a5 a2
jump a5
goto 72
goto 66
print a3
goto 73
    """)
    
    emu = GOLComputerEmulator(bits=8, num_vars=8, debug=False)
    output = emu.run(program, max_steps=500)
    
    print(f"Output: {output}")
    expected = [19, 22, 43, 50]
    print(f"Expected: {expected}")
    print(f"Correct: {output == expected}")
    
    if output == expected:
        print("\n✓ 2x2 matrix multiplication verified!")
        print("  [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]")
    print()
    
    return program, output == expected


if __name__ == '__main__':
    test_fibonacci()
    test_multiply()
    test_matmul_element()
    test_full_matmul()
