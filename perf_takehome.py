"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        """
        Pack multiple slots into VLIW instruction bundles.
        Each instruction can contain multiple engine slots executing in parallel.

        This version respects RAW (Read-After-Write) dependencies by tracking which
        scratch locations are written in the current instruction.
        """
        instrs = []
        i = 0
        n = len(slots)

        while i < n:
            instr = {}
            # Track used slots per engine
            used_slots = {}
            # Track written scratch locations in this instruction (for dependency checking)
            written = set()

            j = i
            # Greedy packing: add slots until we hit a limit or dependency
            while j < n:
                engine, slot = slots[j]
                used = used_slots.get(engine, 0)

                if used >= SLOT_LIMITS.get(engine, 1):
                    # This engine is full, end this instruction
                    break

                # Check for RAW dependencies: does this slot read anything written earlier?
                can_add = True
                sources = []
                written_dest = None

                if engine == "alu":
                    # Format: (op, dest, src1, src2)
                    sources = [slot[2], slot[3]]
                    written_dest = slot[1]
                elif engine == "load":
                    if slot[0] == "const":
                        # ("const", dest, val) - val is immediate, not scratch
                        sources = []
                    else:
                        # ("load", dest, addr) - addr is scratch
                        sources = [slot[2]]
                    written_dest = slot[1]
                elif engine == "store":
                    # Format: ("store", addr, src)
                    sources = [slot[1], slot[2]]
                    written_dest = None
                elif engine == "debug":
                    # Format: ("compare", loc, key)
                    sources = [slot[1]]  # loc
                    written_dest = None
                elif engine == "flow":
                    op = slot[0]
                    if op == "select":
                        # ("select", dest, cond, a, b)
                        sources = [slot[2], slot[3], slot[4]]
                        written_dest = slot[1]
                    elif op == "add_imm":
                        # ("add_imm", dest, a, imm) - imm is immediate
                        sources = [slot[2]]
                        written_dest = slot[1]
                    elif op == "vselect":
                        # ("vselect", dest, cond, a, b)
                        sources = [slot[2], slot[3], slot[4]]
                        written_dest = slot[1]
                    else:
                        # Other flow ops (jump, halt, pause, etc.)
                        sources = []
                        written_dest = None
                else:
                    # Unknown engine
                    sources = []
                    written_dest = None

                # Check if any source was written earlier in this instruction
                for src in sources:
                    if src in written:
                        can_add = False
                        break

                if not can_add:
                    break

                # Add this slot
                if engine not in instr:
                    instr[engine] = []
                instr[engine].append(slot)
                used_slots[engine] = used + 1

                # Track the destination as written
                if written_dest is not None:
                    written.add(written_dest)

                j += 1

                # Check if any engine can still accept more slots
                can_add_more = False
                k = j
                while k < n:
                    next_engine, _ = slots[k]
                    if used_slots.get(next_engine, 0) < SLOT_LIMITS.get(next_engine, 1):
                        can_add_more = True
                        break
                    k += 1

                if not can_add_more:
                    break

            if instr:
                instrs.append(instr)
            i = j

        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_hash_vec(self, val_vec, tmp1_vec, tmp2_vec):
        """
        Vectorized hash computation using valu instructions.
        Process 8 hash computations in parallel.
        """
        slots = []

        for op1, val1, op2, op3, val3 in HASH_STAGES:
            # tmp1_vec = val_vec op1 val1
            slots.append(("valu", (op1, tmp1_vec, val_vec, self.scratch_const(val1))))
            # tmp2_vec = val_vec op3 val3 (note: val3 is scalar, broadcast)
            slots.append(("alu", ("+", tmp1_vec, val_vec, self.scratch_const(val3))))
            # Actually for hash we need the intermediate values
            # Let me rewrite this properly

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized implementation using VLIW SIMD with valu (vector ALU).
        Process 8 elements in parallel using valu instructions.
        """
        # Vector registers
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_hash_tmp = self.alloc_scratch("v_hash_tmp", VLEN)
        v_even = self.alloc_scratch("v_even", VLEN)
        v_offset = self.alloc_scratch("v_offset", VLEN)
        v_cmp = self.alloc_scratch("v_cmp", VLEN)

        # Constant vectors (broadcast via vbroadcast)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)

        # Hash constant vectors
        hconsts = []
        for _, val1, _, val3, _ in HASH_STAGES:
            vc1 = self.alloc_scratch(f"hconst1_{val1}", VLEN)
            vc3 = self.alloc_scratch(f"hconst3_{val3}", VLEN)
            hconsts.append((vc1, vc3))

        # Scalar registers
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp0 = self.alloc_scratch("tmp0")
        tmp1 = self.alloc_scratch("tmp1")

        # Initialize scalar variables
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        init_slots = []
        for i, v in enumerate(init_vars):
            init_slots.append(("load", ("const", tmp0, i)))
            init_slots.append(("load", ("load", self.scratch[v], tmp0)))

        # Initialize vector constants using vbroadcast
        zero = self.scratch_const(0)
        one = self.scratch_const(1)
        two = self.scratch_const(2)
        init_slots.append(("valu", ("vbroadcast", v_zero, zero)))
        init_slots.append(("valu", ("vbroadcast", v_one, one)))
        init_slots.append(("valu", ("vbroadcast", v_two, two)))

        # Initialize hash constants
        for (vc1, vc3), (op1, val1, op2, op3, val3) in zip(hconsts, HASH_STAGES):
            c1 = self.scratch_const(val1)
            c3 = self.scratch_const(val3)
            init_slots.append(("valu", ("vbroadcast", vc1, c1)))
            init_slots.append(("valu", ("vbroadcast", vc3, c3)))

        self.instrs.extend(self.build(init_slots))
        self.add("flow", ("pause",))

        n_vectors = batch_size // VLEN

        # Create list of hash stages for easier iteration
        hash_stages = list(zip(HASH_STAGES, hconsts))

        for round in range(rounds):
            for vec_idx in range(n_vectors):
                offset = vec_idx * VLEN

                # Load idx/val vectors - pack as much as possible
                slots = [
                    ("flow", ("add_imm", tmp_addr, self.scratch["inp_indices_p"], offset)),
                    ("load", ("vload", v_idx, tmp_addr)),
                ]
                self.instrs.extend(self.build(slots))
                slots = [
                    ("flow", ("add_imm", tmp_addr, self.scratch["inp_values_p"], offset)),
                    ("load", ("vload", v_val, tmp_addr)),
                ]
                self.instrs.extend(self.build(slots))

                # Gather node_vals - don't pack due to tmp_addr/tmp0 dependencies
                for i in range(VLEN):
                    idx_pos = v_idx + i
                    node_pos = v_node_val + i
                    self.add("alu", ("+", tmp_addr, self.scratch["forest_values_p"], idx_pos))
                    self.add("load", ("load", tmp0, tmp_addr))
                    self.add("flow", ("add_imm", node_pos, tmp0, 0))

                # XOR
                self.add("valu", ("^", v_val, v_val, v_node_val))

                # Hash: 6 stages
                for (op1, val1, op2, op3, val3), (vc1, vc3) in hash_stages:
                    self.add("valu", (op1, v_hash_tmp, v_val, vc1))
                    self.add("valu", (op3, v_node_val, v_val, vc3))
                    self.add("valu", (op2, v_val, v_hash_tmp, v_node_val))

                # Compute next idx
                self.add("valu", ("%", v_even, v_val, v_two))
                self.add("valu", ("==", v_even, v_even, v_zero))
                self.add("flow", ("vselect", v_offset, v_even, v_one, v_two))
                self.add("valu", ("*", v_idx, v_idx, v_two))
                self.add("valu", ("+", v_idx, v_idx, v_offset))

                # Bounds check and wrap
                for i in range(VLEN):
                    self.add("alu", ("<", v_cmp + i, v_idx + i, self.scratch["n_nodes"]))
                    self.add("flow", ("select", v_idx + i, v_cmp + i, v_idx + i, v_zero))

                # Store results
                self.add("flow", ("add_imm", tmp_addr, self.scratch["inp_indices_p"], offset))
                self.add("store", ("vstore", tmp_addr, v_idx))
                self.add("flow", ("add_imm", tmp_addr, self.scratch["inp_values_p"], offset))
                self.add("store", ("vstore", tmp_addr, v_val))

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
