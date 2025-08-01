import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure

class EconomicOptimizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("经济优化模型分析程序")
        self.root.geometry("800x600")
        
        # 创建标签页
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 创建三个标签页
        self.profit_frame = ttk.Frame(self.notebook)
        self.investment_frame = ttk.Frame(self.notebook)
        self.fund_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.profit_frame, text="利润最大化")
        self.notebook.add(self.investment_frame, text="投资收益与风险")
        self.notebook.add(self.fund_frame, text="资金最优使用")
        
        self.setup_profit_tab()
        self.setup_investment_tab()
        self.setup_fund_tab()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    
    def setup_profit_tab(self):
        """设置利润最大化标签页"""
        frame = ttk.LabelFrame(self.profit_frame, text="柯布-道格拉斯生产函数参数")
        frame.pack(fill='x', padx=5, pady=5)
        
        # 参数输入
        ttk.Label(frame, text="常数α:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.alpha_entry = ttk.Entry(frame, width=10)
        self.alpha_entry.insert(0, "1.0")
        self.alpha_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(frame, text="劳动力弹性α:").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.labor_alpha_entry = ttk.Entry(frame, width=10)
        self.labor_alpha_entry.insert(0, "0.6")
        self.labor_alpha_entry.grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(frame, text="资本弹性β:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.beta_entry = ttk.Entry(frame, width=10)
        self.beta_entry.insert(0, "0.4")
        self.beta_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # 收入成本参数
        cost_frame = ttk.LabelFrame(self.profit_frame, text="收入成本参数")
        cost_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(cost_frame, text="b0:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.b0_entry = ttk.Entry(cost_frame, width=10)
        self.b0_entry.insert(0, "100")
        self.b0_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(cost_frame, text="b1:").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.b1_entry = ttk.Entry(cost_frame, width=10)
        self.b1_entry.insert(0, "-0.1")
        self.b1_entry.grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(cost_frame, text="c0:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.c0_entry = ttk.Entry(cost_frame, width=10)
        self.c0_entry.insert(0, "50")
        self.c0_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(cost_frame, text="c1:").grid(row=1, column=2, sticky='w', padx=5, pady=2)
        self.c1_entry = ttk.Entry(cost_frame, width=10)
        self.c1_entry.insert(0, "0.05")
        self.c1_entry.grid(row=1, column=3, padx=5, pady=2)
        
        ttk.Label(cost_frame, text="固定成本M0:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.m0_entry = ttk.Entry(cost_frame, width=10)
        self.m0_entry.insert(0, "1000")
        self.m0_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # 优化选择
        opt_frame = ttk.LabelFrame(self.profit_frame, text="优化目标")
        opt_frame.pack(fill='x', padx=5, pady=5)
        
        self.profit_type = tk.StringVar(value="max_profit")
        ttk.Radiobutton(opt_frame, text="纯利润最大化", variable=self.profit_type, value="max_profit").pack(anchor='w')
        ttk.Radiobutton(opt_frame, text="利润成本比最大化", variable=self.profit_type, value="max_ratio").pack(anchor='w')
        
        # 按钮和结果显示
        ttk.Button(self.profit_frame, text="计算最优解", command=self.solve_profit).pack(pady=10)
        
        self.profit_result = tk.Text(self.profit_frame, height=8, width=80)
        self.profit_result.pack(fill='both', expand=True, padx=5, pady=5)
    
    def setup_investment_tab(self):
        """设置投资收益与风险标签页"""
        param_frame = ttk.LabelFrame(self.investment_frame, text="投资参数")
        param_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(param_frame, text="总资金M:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.total_fund_entry = ttk.Entry(param_frame, width=10)
        self.total_fund_entry.insert(0, "100000")
        self.total_fund_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="风险承受度b:").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.risk_tolerance_entry = ttk.Entry(param_frame, width=10)
        self.risk_tolerance_entry.insert(0, "0.1")
        self.risk_tolerance_entry.grid(row=0, column=3, padx=5, pady=2)
        
        # 投资项目参数
        project_frame = ttk.LabelFrame(self.investment_frame, text="投资项目参数 (用逗号分隔)")
        project_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(project_frame, text="收益率r:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.returns_entry = ttk.Entry(project_frame, width=30)
        self.returns_entry.insert(0, "0.08, 0.12, 0.15, 0.10")
        self.returns_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(project_frame, text="风险损失率q:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.risks_entry = ttk.Entry(project_frame, width=30)
        self.risks_entry.insert(0, "0.02, 0.05, 0.08, 0.03")
        self.risks_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(project_frame, text="交易费率p:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.fees_entry = ttk.Entry(project_frame, width=30)
        self.fees_entry.insert(0, "0.01, 0.015, 0.02, 0.01")
        self.fees_entry.grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Button(self.investment_frame, text="计算最优投资分配", command=self.solve_investment).pack(pady=10)
        
        self.investment_result = tk.Text(self.investment_frame, height=10, width=80)
        self.investment_result.pack(fill='both', expand=True, padx=5, pady=5)
    
    def setup_fund_tab(self):
        """设置资金最优使用标签页"""
        param_frame = ttk.LabelFrame(self.fund_frame, text="资金使用参数")
        param_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(param_frame, text="初始资金:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.initial_fund_entry = ttk.Entry(param_frame, width=10)
        self.initial_fund_entry.insert(0, "100")
        self.initial_fund_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="银行利率:").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.interest_rate_entry = ttk.Entry(param_frame, width=10)
        self.interest_rate_entry.insert(0, "0.1")
        self.interest_rate_entry.grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(param_frame, text="使用年限:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.years_entry = ttk.Entry(param_frame, width=10)
        self.years_entry.insert(0, "4")
        self.years_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Button(self.fund_frame, text="计算最优资金使用", command=self.solve_fund).pack(pady=10)
        
        self.fund_result = tk.Text(self.fund_frame, height=10, width=80)
        self.fund_result.pack(fill='both', expand=True, padx=5, pady=5)
    
    def solve_profit(self):
        """求解利润最大化问题"""
        try:
            # 获取参数
            alpha_const = float(self.alpha_entry.get())
            alpha_labor = float(self.labor_alpha_entry.get())
            beta = float(self.beta_entry.get())
            b0 = float(self.b0_entry.get())
            b1 = float(self.b1_entry.get())
            c0 = float(self.c0_entry.get())
            c1 = float(self.c1_entry.get())
            m0 = float(self.m0_entry.get())
            
            # 定义目标函数
            def objective(x):
                L, K = x
                Q = alpha_const * (L ** alpha_labor) * (K ** beta)
                
                if self.profit_type.get() == "max_profit":
                    # 纯利润最大化（取负值因为要最小化）
                    profit = (b0 - c0) * Q + (b1 - c1) * (Q ** 2) - m0
                    return -profit
                else:
                    # 利润成本比最大化
                    profit = (b0 - c0) * Q + (b1 - c1) * (Q ** 2) - m0
                    cost = m0 + c0 * Q + c1 * (Q ** 2)
                    if cost <= 0:
                        return float('inf')
                    return -profit / cost
            
            # 约束条件
            constraints = [
                {'type': 'ineq', 'fun': lambda x: x[0]},  # L >= 0
                {'type': 'ineq', 'fun': lambda x: x[1]},  # K >= 0
            ]
            
            # 初始猜测
            x0 = [10, 10]
            
            # 求解
            result = minimize(objective, x0, method='SLSQP', constraints=constraints)
            
            if result.success:
                L_opt, K_opt = result.x
                Q_opt = alpha_const * (L_opt ** alpha_labor) * (K_opt ** beta)
                
                # 计算最优值
                profit_opt = (b0 - c0) * Q_opt + (b1 - c1) * (Q_opt ** 2) - m0
                cost_opt = m0 + c0 * Q_opt + c1 * (Q_opt ** 2)
                
                result_text = f"优化结果:\n"
                result_text += f"最优劳动力投入 L*: {L_opt:.4f}\n"
                result_text += f"最优资本投入 K*: {K_opt:.4f}\n"
                result_text += f"最优产出 Q*: {Q_opt:.4f}\n"
                result_text += f"最大利润: {profit_opt:.4f}\n"
                result_text += f"总成本: {cost_opt:.4f}\n"
                result_text += f"利润成本比: {profit_opt/cost_opt:.4f}\n"
                
                self.profit_result.delete(1.0, tk.END)
                self.profit_result.insert(tk.END, result_text)
                
                # 绘制3D图
                self.plot_profit_surface(alpha_const, alpha_labor, beta, b0, b1, c0, c1, m0, L_opt, K_opt)
                
            else:
                messagebox.showerror("错误", "优化失败")
                
        except Exception as e:
            messagebox.showerror("错误", f"参数输入错误: {str(e)}")
    
    def plot_profit_surface(self, alpha_const, alpha_labor, beta, b0, b1, c0, c1, m0, L_opt, K_opt):
        """绘制利润函数3D图"""
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建网格
        L_range = np.linspace(0.1, 20, 30)
        K_range = np.linspace(0.1, 20, 30)
        L_grid, K_grid = np.meshgrid(L_range, K_range)
        
        # 计算利润
        Q_grid = alpha_const * (L_grid ** alpha_labor) * (K_grid ** beta)
        profit_grid = (b0 - c0) * Q_grid + (b1 - c1) * (Q_grid ** 2) - m0
        
        # 绘制曲面
        surf = ax.plot_surface(L_grid, K_grid, profit_grid, cmap='viridis', alpha=0.7)
        
        # 标记最优点
        Q_opt = alpha_const * (L_opt ** alpha_labor) * (K_opt ** beta)
        profit_opt = (b0 - c0) * Q_opt + (b1 - c1) * (Q_opt ** 2) - m0
        ax.scatter([L_opt], [K_opt], [profit_opt], color='red', s=100, label='最优点')
        
        ax.set_xlabel('劳动力 L')
        ax.set_ylabel('资本 K')
        ax.set_zlabel('利润')
        ax.set_title('利润函数3D图')
        ax.legend()
        
        # 显示图形
        plt.figure(figsize=(10, 6))
        plt.show()
    
    def solve_investment(self):
        """求解投资收益与风险问题"""
        try:
            # 获取参数
            M = float(self.total_fund_entry.get())
            b = float(self.risk_tolerance_entry.get())
            
            returns = [float(x.strip()) for x in self.returns_entry.get().split(',')]
            risks = [float(x.strip()) for x in self.risks_entry.get().split(',')]
            fees = [float(x.strip()) for x in self.fees_entry.get().split(',')]
            
            n = len(returns)
            
            # 目标函数系数（最大化净收益，所以取负值）
            c = [-(r - f) for r, f in zip(returns, fees)]
            
            # 不等式约束矩阵 A_ub * x <= b_ub
            A_ub = []
            b_ub = []
            
            # 风险约束
            for i in range(n):
                risk_constraint = [0] * n
                risk_constraint[i] = risks[i] / M
                A_ub.append(risk_constraint)
                b_ub.append(b)
            
            # 等式约束矩阵 A_eq * x = b_eq
            A_eq = [[1 + f for f in fees]]
            b_eq = [M]
            
            # 变量下界
            bounds = [(0, None) for _ in range(n)]
            
            # 求解
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if result.success:
                optimal_x = result.x
                max_return = -result.fun
                
                result_text = "最优投资分配:\n"
                total_investment = 0
                for i in range(n):
                    result_text += f"项目 {i+1}: {optimal_x[i]:.2f} (占比: {optimal_x[i]/M*100:.1f}%)\n"
                    total_investment += optimal_x[i]
                
                result_text += f"\n总投资: {total_investment:.2f}\n"
                result_text += f"最大净收益: {max_return:.2f}\n"
                
                # 计算风险
                total_risk = sum(risks[i] * optimal_x[i] / M for i in range(n))
                result_text += f"总风险: {total_risk:.4f}\n"
                
                self.investment_result.delete(1.0, tk.END)
                self.investment_result.insert(tk.END, result_text)
                
                # 绘制饼图
                self.plot_investment_pie(optimal_x, n)
                
            else:
                messagebox.showerror("错误", "投资优化失败")
                
        except Exception as e:
            messagebox.showerror("错误", f"参数输入错误: {str(e)}")
    
    def plot_investment_pie(self, optimal_x, n):
        """绘制投资分配饼图"""
        labels = [f'项目{i+1}' for i in range(n)]
        sizes = optimal_x
        
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('最优投资分配')
        plt.axis('equal')
        plt.show()
    
    def solve_fund(self):
        """求解资金最优使用问题"""
        try:
            # 获取参数
            initial_fund = float(self.initial_fund_entry.get())
            interest_rate = float(self.interest_rate_entry.get())
            years = int(self.years_entry.get())
            
            # 定义目标函数（取负值因为要最小化）
            def objective(x):
                return -sum(xi ** 0.5 for xi in x)
            
            # 约束条件
            def constraint1(x):
                return initial_fund - x[0]
            
            def constraint2(x):
                return (1 + interest_rate) * (initial_fund - x[0]) - x[1]
            
            def constraint3(x):
                remaining_after_2 = (1 + interest_rate) * ((1 + interest_rate) * (initial_fund - x[0]) - x[1]) - x[2]
                return remaining_after_2
            
            def constraint4(x):
                remaining_after_3 = (1 + interest_rate) * ((1 + interest_rate) * ((1 + interest_rate) * (initial_fund - x[0]) - x[1]) - x[2]) - x[3]
                return remaining_after_3
            
            constraints = [
                {'type': 'ineq', 'fun': constraint1},
                {'type': 'ineq', 'fun': constraint2},
                {'type': 'ineq', 'fun': constraint3},
                {'type': 'ineq', 'fun': constraint4},
            ]
            
            # 变量下界
            bounds = [(0, None) for _ in range(years)]
            
            # 初始猜测
            x0 = [initial_fund / years] * years
            
            # 求解
            result = minimize(objective, x0, method='SLSQP', constraints=constraints, bounds=bounds)
            
            if result.success:
                optimal_x = result.x
                max_benefit = -result.fun
                
                result_text = "最优资金使用方案:\n"
                for i in range(years):
                    result_text += f"第{i+1}年使用: {optimal_x[i]:.2f} 万元\n"
                
                result_text += f"\n总效益: {max_benefit:.4f} 万元\n"
                
                # 计算每年剩余资金
                remaining = initial_fund
                result_text += "\n每年资金流动:\n"
                for i in range(years):
                    result_text += f"第{i+1}年初: {remaining:.2f} 万元\n"
                    result_text += f"第{i+1}年使用: {optimal_x[i]:.2f} 万元\n"
                    remaining = (remaining - optimal_x[i]) * (1 + interest_rate)
                    if i < years - 1:
                        result_text += f"第{i+1}年末存银行: {remaining:.2f} 万元\n"
                
                self.fund_result.delete(1.0, tk.END)
                self.fund_result.insert(tk.END, result_text)
                
                # 绘制柱状图
                self.plot_fund_bar(optimal_x, years)
                
            else:
                messagebox.showerror("错误", "资金优化失败")
                
        except Exception as e:
            messagebox.showerror("错误", f"参数输入错误: {str(e)}")
    
    def plot_fund_bar(self, optimal_x, years):
        """绘制资金使用柱状图"""
        years_labels = [f'第{i+1}年' for i in range(years)]
        
        plt.figure(figsize=(10, 6))
        plt.bar(years_labels, optimal_x, color='skyblue', alpha=0.7)
        plt.xlabel('年份')
        plt.ylabel('使用资金 (万元)')
        plt.title('最优资金使用方案')
        plt.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(optimal_x):
            plt.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
 
    def run(self):
        """运行程序"""
        self.root.mainloop()

# 运行程序
if __name__ == "__main__":
    app = EconomicOptimizer()
    app.run()