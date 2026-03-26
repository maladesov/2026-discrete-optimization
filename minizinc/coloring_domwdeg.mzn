% Модель раскраски с эвристикой dom_w_deg
% dom_w_deg = domain_size / weighted_degree — адаптивная эвристика,
% которая учитывает историю неудач при поиске

int: n;
int: m;
array[1..m, 1..2] of int: edges;
int: max_colors;

array[1..n] of var 1..max_colors: color;

constraint forall(e in 1..m)(
    color[edges[e, 1]] != color[edges[e, 2]]
);

var int: obj = max(color);

solve :: int_search(color, dom_w_deg, indomain_min)
      minimize obj;

output ["colors = ", show(obj), "\n"] ++
       [show(color[i]) ++ " " | i in 1..n] ++ ["\n"];
