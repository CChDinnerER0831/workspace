import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
 
# 使用指定的數據點

i_const = 350 * 10**6
v_i_values = np.array([0.18, 0.168, 0.156, 0.144, 0.132, 0.12, 0.108, 0.096, 0.084, 0.072, 0.06, 0.048, 0.036, 0.024, 0.012])
f_values = np.array([0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.83, 0.79, 0.76, 0.70, 0.63, 0.47])

# 使用插值方法來近似函數f(v/i)
f_interpolated = interp1d(v_i_values, f_values, kind='linear', fill_value='extrapolate')
# 繪製原始數據點和插值曲線
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.scatter(v_i_values, f_values, color='blue', label='離散數據點')
v_i_smooth = np.linspace(min(v_i_values), max(v_i_values), 100)
plt.plot(v_i_smooth, f_interpolated(v_i_smooth), 'r-', label='插值曲線 f(v/i)')
plt.xlabel('v/i')
plt.ylabel('f(v/i)')
plt.title('函數f(v/i)的插值近似')
plt.legend()
plt.grid(True)

# 350 million
def differential_equation(t, v):
    # dv = f(v/i)* sediment inflow is 200,000000 *9.8 n/y /9600 n/y specific weight
    return  -200000000*9.8/9600 * f_interpolated(v/ i_const)

# 初始條件
v0 = 42 * 10**6  # 42 million initial volume
t_span = (0, 1000)  # 時間範圍
t_eval = np.linspace(0, 1000, 500)  # 評估點
print('and')
print()
# 求解微分方程 Runge-Kutta法
solution = solve_ivp(differential_equation, t_span, [v0], method='RK45', t_eval=t_eval)
print(22222)
# 繪製v(t)的解
plt.subplot(2, 1, 2)
plt.plot(solution.t, solution.y[0], 'g-', label='v(t)')
plt.xlabel('Time t')
plt.ylabel('Volume v')
plt.title('Solution of ODE dv = -f(v/i)dt')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 繪製v/i vs t的關係
plt.figure(figsize=(10, 4))
plt.plot(solution.t, solution.y[0] / i_const, 'b-', label='v(t)/i')
plt.xlabel('Time t')
plt.ylabel('v/i')
plt.title('v/i vs. Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 尋找v=0.2v0的時間點
v_target = 0.2 * v0
v_solution = solution.y[0]
t_solution = solution.t

# 增加更穩健的尋找方法
# 檢查是否達到目標值
if min(v_solution) <= v_target:
    # 尋找交叉點
    for i in range(len(t_solution)-1):
        if (v_solution[i] >= v_target and v_solution[i+1] <= v_target) or \
           (v_solution[i] <= v_target and v_solution[i+1] >= v_target):
            # 線性插值
            t1, t2 = t_solution[i], t_solution[i+1]
            v1, v2 = v_solution[i], v_solution[i+1]
            t_v_target = t1 + (t2 - t1) * (v_target - v1) / (v2 - v1)
            print(f"初始體積 v₀ = {v0:.2e}")
            print(f"目標體積 v = 0.2v₀ = {v_target:.2e}")
            print(f"達到目標體積的時間 t = {t_v_target:.4f}")
            
            # 繪製目標點
            plt.figure(figsize=(10, 6))
            plt.plot(solution.t, solution.y[0], 'g-', label='v(t)')
            plt.axhline(y=v_target, color='r', linestyle='--', label='v=0.2v₀')
            plt.axvline(x=t_v_target, color='b', linestyle='--')
            plt.plot(t_v_target, v_target, 'ro', markersize=6)
            plt.annotate(f't = {t_v_target:.2f}', 
                         xy=(t_v_target, v_target),
                         xytext=(t_v_target + 5, v_target * 1.1),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
            plt.xlabel('Time t')
            plt.ylabel('Volume v')
            plt.title('Time to reach v=0.2v₀')
            plt.grid(True)
            plt.legend()
            plt.show()
            break
else:
    print(f"初始體積 v₀ = {v0:.2e}")
    print(f"目標體積 v = 0.2v₀ = {v_target:.2e}")
    print("在模擬時間範圍內未達到目標體積")

