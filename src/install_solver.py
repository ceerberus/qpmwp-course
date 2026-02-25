import subprocess

solvers_to_install = ['clarabel', 'cvxopt', 'daqp', 'ecos', 'gurobi', 'highs', 'mosek', 'osqp', 'piqp', 'proxqp', 'qpalm', 'quadprog', 'scs']

for solver in solvers_to_install:
    print(f"Installing {solver}...")
    subprocess.run(
        ['pip', 'install', f'qpsolvers[{solver}]', '--quiet'],
        check=True
    )
    print(f"{solver} installed successfully.")

print("\nAll solvers installed! Restart kernel if needed.")