% Модель раскраски с разрушением симметрии
% Идея: цвета взаимозаменяемы, поэтому фиксируем порядок их появления.
% Вершина 0 всегда получает цвет 1.
% Каждая следующая вершина получает цвет <= max(предыдущих) + 1.

int: n;
int: m;
array[1..m, 1..2] of int: edges;
int: max_colors;

array[1..n] of var 1..max_colors: color;

constraint forall(e in 1..m)(
    color[edges[e, 1]] != color[edges[e, 2]]
);

constraint color[1] = 1;
constraint forall(i in 2..n)(
    color[i] <= max([color[j] | j in 1..i-1]) + 1
);

var int: obj = max(color);

solve :: int_search(color, first_fail, indomain_min)
      minimize obj;

output ["colors = ", show(obj), "\n"] ++
       [show(color[i]) ++ " " | i in 1..n] ++ ["\n"];
