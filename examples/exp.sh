## yelp human LFs ##
python3 autolfgen.py --dataset="yelp" --codexLF=False --LM="Snorkel" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=False --LM="MV" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=False --LM="WMV" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=False --LM="FS" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=False --LM="DS" --EM="Stop"

## yelp codex LFs ##
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="basic_mission" --LM="Snorkel" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="basic_mission" --LM="MV" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="basic_mission" --LM="WMV" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="basic_mission" --LM="FS" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="basic_mission" --LM="DS" --EM="Stop"

python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="data_example" --LM="Snorkel" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="data_example" --LM="MV" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="data_example" --LM="WMV" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="data_example" --LM="FS" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="data_example" --LM="DS" --EM="Stop"

python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="human_heuristic" --LM="Snorkel" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="human_heuristic" --LM="MV" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="human_heuristic" --LM="WMV" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="human_heuristic" --LM="FS" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="human_heuristic" --LM="DS" --EM="Stop"

python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="human_label_function" --LM="Snorkel" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="human_label_function" --LM="MV" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="human_label_function" --LM="WMV" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="human_label_function" --LM="FS" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="human_label_function" --LM="DS" --EM="Stop"

python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="task_description" --LM="Snorkel" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="task_description" --LM="MV" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="task_description" --LM="WMV" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="task_description" --LM="FS" --EM="Stop"
python3 autolfgen.py --dataset="yelp" --codexLF=True --prompt_type="task_description" --LM="DS" --EM="Stop"