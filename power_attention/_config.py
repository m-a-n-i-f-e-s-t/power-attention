# Non user-facing configs

# Setting this to True makes attention faster but more numerically unstable, not covered by tests
normal_space = False

# Setting this to True makes attention behaves like flash attention, not covered by tests
flash_equivalent = False

# eps used in attention (log(|S| + eps) in the kernel)
eps = 1e-6