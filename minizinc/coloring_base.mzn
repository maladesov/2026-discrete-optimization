% Базовая модель раскраски графа (без эвристик)
% Стратегия поиска: стандартная

int: n;
int: m;
array[1..m, 1..2] of int: edges;
int: max_colors;

array[1..n] of var 1..max_colors: color;

constraint forall(e in 1..m)(
    color[edges[e, 1]] != color[edges[e, 2]]
);

var int: obj = max(color);

solve minimize obj;

output ["colors = ", show(obj), "\n"] ++
       [show(color[i]) ++ " " | i in 1..n] ++ ["\n"];
