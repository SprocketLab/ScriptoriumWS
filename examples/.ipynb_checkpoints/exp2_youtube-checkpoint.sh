echo "YouTube - BM"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="basic_mission" --LM="Snorkel"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="basic_mission" --LM="WMV"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="basic_mission" --LM="MV"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="basic_mission" --LM="DS"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="basic_mission" --LM="FS"

echo "YouTube - TD"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="task_description" --LM="Snorkel"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="task_description" --LM="WMV"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="task_description" --LM="MV"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="task_description" --LM="DS"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="task_description" --LM="FS"

echo "YouTube - HH"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="human_heuristic" --LM="Snorkel"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="human_heuristic" --LM="WMV"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="human_heuristic" --LM="MV"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="human_heuristic" --LM="DS"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="human_heuristic" --LM="FS"

echo "YouTube - HL"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="human_label_function" --LM="Snorkel"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="human_label_function" --LM="WMV"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="human_label_function" --LM="MV"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="human_label_function" --LM="DS"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="human_label_function" --LM="FS"

echo "YouTube - DE"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="data_example" --LM="Snorkel"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="data_example" --LM="WMV"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="data_example" --LM="MV"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="data_example" --LM="DS"
python3 autolfgen_integration.py --dataset="youtube" --prompt_type="data_example" --LM="FS"