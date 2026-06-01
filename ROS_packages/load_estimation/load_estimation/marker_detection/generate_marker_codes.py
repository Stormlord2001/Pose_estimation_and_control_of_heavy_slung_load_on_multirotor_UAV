
def getRingCodes(bits: int, transitions: int):
    assert bits % 2 == 0, "Bits must be even"
    assert transitions > 0 and transitions < bits/2, "Transitions must be between 1 and bits/2"

    codes = []
    half_bits = bitsrl(bits, 1)

    for i in range(pow(2, bits-2)-1):
        
        code = bitsll(i, 1) + 1
        code = findSmallestRotation(code, bits)

        diff = (code & pow(2, half_bits)-1) & (bitsrl(code & bitsll(pow(2, half_bits)-1, half_bits), half_bits))
        parity = calcParity(code)

        bit_transitions = countBitTransitions(code)

        if diff > 0 and parity and bit_transitions == transitions and code not in codes:

            codes.append(code)

    codes.sort()
    return codes

def findSmallestRotation(code: int, bits:int) -> int:
    smallest = code
    for i in range(1, bits):
        smallest = min(smallest, bitwiseRotateLeft(code, i, bits))
    return smallest

def bitwiseRotateLeft(val: int, bits: int, total_bits: int) -> int:
    return ((val << bits) | (val >> (total_bits - bits))) & ((1 << total_bits) - 1)

def calcParity(code: int) -> bool:
    parity = True
    while code:
        parity = not parity
        code = code & (code - 1)
    return parity

def countBitTransitions(code: int) -> int:
    transitions = 0
    prev_bit = 0

    while code:
        new_bit = code & 1
        # Larger due to transitions being counted as 0->1 only
        if new_bit > prev_bit:
            transitions += 1
        prev_bit = new_bit
        code >>= 1
    return transitions

def bitsrl(bits: int, n: int) -> int:
    return bits >> n

def bitsll(bits: int, n: int) -> int:
    return bits << n




