from pathlib import Path
base_folder = str(Path(__file__).parent.parent.parent).rstrip("/") + "/"  # Ensure there's at least a / at the end

spans_separators = ["."]  #maybe also try ['.', ',', ';']

unk_ent_id = "0"

