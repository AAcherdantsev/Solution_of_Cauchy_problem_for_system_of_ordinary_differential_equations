from sympy import *
import numpy as np
import matplotlib.pyplot as plt

def input_data(file_name):   # как обычно.  Это функция для ввода информации
    with open(file_name) as file:
        file.readline()
        n = int(file.readline())   # число уравнений и начальных условий
        file.readline()
        equations = []   #  это правые части
        for i in range(n):
            equations.append(simplify(file.readline().split("=")[-1]))
        file.readline()
        start_conditions = []  # это начальные условия
        for i in range(n):
            start_conditions.append(float(file.readline().split("=")[-1]))
        file.readline()
        segment = list(map(float, file.readline().split()))   #  это отрезок
        file.readline()
        step = float(file.readline())   #  это шаг
        file.readline()
        eps = float(file.readline())   # это точность
        return n, equations, start_conditions, segment, step, eps


def find_value(equations, value_x, values_Y): # эта функция подставляет value_x и values_Y в equations
    result = [eq for eq in equations]
    for i in range(len(equations)):
        result[i] = result[i].subs("x", value_x)
        for j in range(len(equations)):
            result[i] = result[i].subs("y" + str(j + 1), values_Y[j])
    return result
    

def solve_system_runge_kutta(n, equations, start_conditions, segment, step):  # это решает систему методом Рунге-Кутты 4 порядка точности и заданным шагом
    values_x = [segment[0] + i * step for i in range(int((segment[1] - segment[0]) / step) + 1)]  # задаем значения аргумента
    values_Y = [[start_conditions[i] for i in range(n)]]   # создаем список значений функции.  Пока известно только начальное
                                                           # состояние.
    curr_value_x = values_x[0] # задаем текущее значение аргумента
    curr_value_Y = [start_conditions[i] for i in range(n)]  # задаем текущие значения функций
    for value in values_x[1:]:   # цикл по всем значениям аргумента
        # вычисляем все k
        k1 = list(map(lambda X: step * X, find_value(equations, curr_value_x, curr_value_Y)))
        var = [curr_value_Y[i] + k1[i] / 3 for i in range(n)]
        k2 = list(map(lambda X: step * X, find_value(equations, curr_value_x + step / 3, var)))
        var = [curr_value_Y[i] - (k1[i] / 3) + k2[i] for i in range(n)]
        k3 = list(map(lambda X: step * X, find_value(equations, curr_value_x + (2 * step) / 3, var)))
        var = [curr_value_Y[i] + k1[i] - k2[i] + k3[i] for i in range(n)]
        k4 = list(map(lambda X: step * X, find_value(equations, curr_value_x + step, var)))
        values_Y.append([])
        # вычисляем новые значения функций
        for i in range(n):
            values_Y[-1].append(curr_value_Y[i] + (1 / 8) * (k1[i] + 3 * k2[i] + 3 * k3[i] + k4[i])) 
        #  задаем новые знаяения аргумента и функций
        curr_value_x = value
        curr_value_Y = [values_Y[-1][i] for i in range(n)]
    return values_x, values_Y  # возвращяем список из значений аргумента и списка из списков значений
                               # функций

def solve_system_runge_kutta_auto(n, equations, start_conditions, segment, step):  # решаем с автоматическим шагом
    h = step  # начальный шаг
    eps1 = eps / (2 ** 5)
    flag_less = False  # флаг для опред.  был ли шаг на пред итерации меньше eps1
    flag_change = False  # флаг для опред.  уменьшался ли шаг на текущей итерации
    values_Y = [[start_conditions[i] for i in range(n)]]
    values_x = [segment[0]]
    curr_value_x = values_x[0]
    curr_value_Y = [start_conditions[i] for i in range(n)]
    while curr_value_x <= segment[1]:
        k1 = list(map(lambda X: h * X, find_value(equations, curr_value_x, curr_value_Y)))
        var = [curr_value_Y[i] + k1[i] / 3 for i in range(n)]
        k2 = list(map(lambda X: h * X, find_value(equations, curr_value_x + h / 3, var)))
        var = [curr_value_Y[i] - (k1[i] / 3) + k2[i] for i in range(n)]
        k3 = list(map(lambda X: h * X, find_value(equations, curr_value_x + (2 * h) / 3, var)))
        var = [curr_value_Y[i] + k1[i] - k2[i] + k3[i] for i in range(n)]
        k4 = list(map(lambda X: h * X, find_value(equations, curr_value_x + h, var)))
        values_Y.append([])
        for i in range(n):
            values_Y[-1].append(curr_value_Y[i] + (1 / 8) * (k1[i] + 3 * k2[i] + 3 * k3[i] + k4[i]))             
        # тут будем определять шаг автоматически:
        curr_value_x += h
        values_x.append(curr_value_x)
        flag_change = False
        while True:
             # считаем ошибку по формулам
             k1 = list(map(lambda X: h * X, find_value(equations, curr_value_x, curr_value_Y)))
             var = [curr_value_Y[i] + k1[i] / 3 for i in range(n)]
             k2 = list(map(lambda X: h * X, find_value(equations, curr_value_x + h / 3, var)))
             var = [curr_value_Y[i] - (k1[i] / 3) + k2[i] for i in range(n)]
             k3 = list(map(lambda X: h * X, find_value(equations, curr_value_x + (2 * h) / 3, var)))
             var = [curr_value_Y[i] + k1[i] - k2[i] + k3[i] for i in range(n)]
             k4 = list(map(lambda X: h * X, find_value(equations, curr_value_x + h, var)))
             E = max(list(map(lambda X: (2 / 3) * X, [(k1[i] - k2[i] - k3[i] + k4[i]) for i in range(n)])))
             psi = E / (2 ** 5 - 1)  # 6.46
             M = max(list(map(lambda X: abs(X), values_Y[-1])) + [1])
             err = abs(psi / M)  # 6.47
             if err <= eps or h < step:  # если ошибка меньше eps или зашли за ограничитель то не уменьшаем шаг
                break
             else:
                h = h / 2
                # ставим флаг, что шаг уменьшили
                flag_change = True
        if not flag_change and flag_less:  # если шаг не менялся на этой итерации, а на прошлой h < eps1, то
            h = h * 2
        # ставим флаг
        flag_less = True if err < eps1 else False
        #  обновляем текущие значения функций
        curr_value_Y = [values_Y[-1][i] for i in range(n)]
    return values_x, values_Y


def solve_system_Adams(n, equations, start_conditions, segment, step):  # это решает систему методом  Адамса 4 порядка точности 

    values_x = [segment[0] + i * step for i in range(int((segment[1] - segment[0]) / step) + 1)]  # задаем значения аргумента
    values_Y = [[start_conditions[i] for i in range(n)]]   # создаем список значений функции.  Пока известно только начальное
                                                           # состояние.
    curr_value_x = values_x[0] # задаем текущее значение аргумента
    it_is_first_iterr = True
    while curr_value_x <= segment[1]:  
        # вычисляем 
        if it_is_first_iterr:
            it_is_first_iterr = False
            # Для k = 1
            temp = find_value(equations, curr_value_x, values_Y[-1]) 
            new_Y1 = [values_Y[-1][i] + step * temp[i] for i in range(n)]

            temp = find_value(equations, curr_value_x, new_Y1)
            new_Y1 = [values_Y[-1][i] + step * temp[i] for i in range(n)]

            values_Y.append(new_Y1)
            curr_value_x += step 
            # Для k = 2

            temp_n = find_value(equations, curr_value_x, values_Y[-1])
            temp_n_1 = find_value(equations, curr_value_x, values_Y[-2])
            new_Y1 = [values_Y[-1][i] + (1/2)*step * (3*temp_n[i] - temp_n_1[i]) for i in range(n)]

            temp_n__1 = find_value(equations, curr_value_x, new_Y1)
            temp_n = find_value(equations, curr_value_x, values_Y[-1])
            new_Y1 = [values_Y[-1][i] + (1/2) * step * (temp_n__1[i] + temp_n[i]) for i in range(n)]

            values_Y.append(new_Y1)
            curr_value_x += step
            # Для k = 3

            temp_n = find_value(equations, curr_value_x, values_Y[-1])
            temp_n_1 = find_value(equations, curr_value_x, values_Y[-2])
            temp_n_2 = find_value(equations, curr_value_x, values_Y[-3])
            new_Y1 = [values_Y[-1][i] + (1/12)*step * (23 * temp_n[i] - 16 * temp_n_1[i] + 5 * temp_n_2[i]) for i in range(n)]

            temp__n = find_value(equations, curr_value_x, new_Y1)
            temp_n = find_value(equations, curr_value_x, values_Y[-1])
            temp_n_1 = find_value(equations, curr_value_x, values_Y[-2])
            new_Y1 = [values_Y[-1][i] + (1/12)*step * (5 * temp__n[i] + 8 * temp_n[i] - 1 * temp_n_1[i]) for i in range(n)]

            values_Y.append(new_Y1)
            curr_value_x += step

        # Для k = 4

        temp_n = find_value(equations, curr_value_x, values_Y[-1])
        temp_n_1 = find_value(equations, curr_value_x, values_Y[-2])
        temp_n_2 = find_value(equations, curr_value_x, values_Y[-3])
        temp_n_3 = find_value(equations, curr_value_x, values_Y[-4])
        new_Y1 = [values_Y[-1][i] + (1/24)*step * (55*temp_n[i] - 59 * temp_n_1[i] + 37 * temp_n_2[i] - 9 * temp_n_3[i]) for i in range(n)]


        temp__n = find_value(equations, curr_value_x, new_Y1)
        temp_n = find_value(equations, curr_value_x, values_Y[-1])
        temp_n_1 = find_value(equations, curr_value_x, values_Y[-2])
        temp_n_2 = find_value(equations, curr_value_x, values_Y[-3])
        new_Y1 = [values_Y[-1][i] + (1/24)*step * \
            (9*temp__n[i] + 19*temp_n[i] - 5*temp_n_1[i] + temp_n_2[i]) for i in range(n)]

        values_Y.append(new_Y1)
        curr_value_x += step

    return values_x, values_Y[1:]  # возвращяем список из значений аргумента и списка из списков значений
                               # функций

def show_graph(values_x, values_Y):
    """
    values_U1 = [exp(-30 * x) for x in values_x]
    Y1 = [values_Y[j][0] for j in range(len(values_Y))]
    m1 = [abs(values_U1[i] - Y1[i]) for i in range(len(values_U1))]
    err = max(m1)   
    print("Погрешность: " + str(err))
    """
    for i in range(n):
        value = [values_Y[j][i] for j in range(len(values_Y))]
        plt.plot(values_x, value, label = "y" + str(i + 1) + "(x)")
    plt.grid()
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    n, equations, start_conditions, segment, step, eps = input_data("input.txt")
    values_x, values_Y = solve_system_runge_kutta(n, equations, start_conditions, segment, step)
    #values_x, values_Y = solve_system_runge_kutta_auto(n, equations, start_conditions,\
    #segment, step)
    values_x, values_Y = solve_system_Adams(n, equations, start_conditions, segment, step)
    values_x1, values_Y1 = solve_system_runge_kutta(n, equations, start_conditions, segment, step)

    for i in range(n):
        value = [values_Y[j][i] for j in range(len(values_Y))]
        value1 = [values_Y1[j][i] for j in range(len(values_Y1))]
        plt.plot(values_x, value, label = "y" + str(i + 1) + "(x) (Adams)")
        plt.plot(values_x, value1, label = "y" + str(i + 1) + "(x) (Runge-Kutta)")
    plt.grid()
    plt.legend()
    plt.show()

    #show_graph(values_x, values_Y) 