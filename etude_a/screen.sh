for t in zengarden hacienda cornfield_crossing snowtuxpeak black_forest scotland sandtrack cocoa_temple abyss volcano_island mines xr591; do
  for e in team3 team4 team5; do echo "$t $e"; done
done | xargs -P 8 -L 1 sh -c 'timeout 1200 uv run src/train.py --algo dagger --total-steps 0 --track $0 --expert $1 --exp-name _screen > logs/screen/$0-$1.log 2>&1'
