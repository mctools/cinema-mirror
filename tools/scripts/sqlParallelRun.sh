echo "Spawning 50 processes"
for i in {1..50};
do
    python $CINEMAPATH/scripts/test_BL6_optuna.py > /dev/null 2>&1 &
    done