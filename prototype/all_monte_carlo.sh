
cp config_games.py_example config_games.py
python monte_carlo.py --config config_games

cp config_linear.py_example config_linear.py
python monte_carlo.py --config config_linear

cp config_logistic.py_example config_logistic.py
python monte_carlo.py --config config_logistic

cp config_missing_data.py_example config_missing_data.py
python monte_carlo.py --config config_missing_data

python linear_growing_kappa.py
